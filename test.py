import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import os
import io
import torch
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient  # MongoDB async driver

app = FastAPI()
load_dotenv()

# ---------------------------- MongoDB Setup ----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client.legal_assistant
documents_collection = db.documents
queries_collection = db.queries

# ---------------------------- NLP Model Setup ----------------------------
nlp = spacy.load("en_core_web_sm")  # spaCy for text processing
embedding_model = SentenceTransformer('all-mpnet-base-v2')

model_name = "google/flan-t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# ---------------------------- FAISS Setup ----------------------------
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
document_store = {}

def store_document(text: str):
    """Stores document embeddings in FAISS."""
    embedding = embedding_model.encode([text])[0]
    faiss_index.add(np.array([embedding], dtype=np.float32))
    doc_id = len(document_store)
    document_store[doc_id] = text
    
    
async def summarize_text(text: str) -> str:
    """Summarizes text efficiently by processing chunks in parallel."""
    max_input_length = 256
    chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]
    
    async def process_chunk(chunk):
        input_text = f"summarize: {chunk}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        outputs = model.generate(inputs["input_ids"], max_length=500)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    summaries = await asyncio.gather(*(process_chunk(chunk) for chunk in chunks))
    return " ".join(summaries)


async def retrieve_documents(query: str, top_k: int = 3):
    """Retrieves relevant documents using FAISS."""
    if faiss_index.ntotal == 0:
        return []  # Prevent FAISS error when empty

    query_embedding = embedding_model.encode([query])[0]
    _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return [document_store[idx] for idx in indices[0] if idx in document_store]

# ---------------------------- Text Processing Functions ----------------------------
class ChatRequest(BaseModel):
    question: str

def clean_text(text: str) -> str:
    """Cleans and formats extracted text."""
    return re.sub(r'\s+', ' ', text.strip())
class LegalDocumentRequest(BaseModel):
    template: str  # User selects document type (rental, land, building, lease)
    clauses: str   # User provides custom clauses

def clean_text(text: str) -> str:
    """Cleans and formats extracted text using spaCy."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 2]
    return " ".join(sentences)


async def retrieve_similar(query: str, top_k: int = 3) -> str:
    """Retrieves top similar documents from FAISS for context-based answering."""
    if faiss_index.ntotal == 0:
        return ""
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return " ".join([document_store.get(idx, "") for idx in indices[0]])


async def add_document_to_storage(text: str, filename: str):
    """Encodes text, adds to FAISS, and stores in MongoDB."""
    global faiss_index, document_store
    embedding = embedding_model.encode([text], convert_to_numpy=True)
    
    index_id = len(document_store)
    document_store[index_id] = text

    # Efficient FAISS Indexing
    if faiss_index.ntotal == 0:
        faiss_index.add(embedding)
    else:
        faiss_index.add(np.vstack(embedding))

    await documents_collection.insert_one({
        "filename": filename,
        "text": text,
        "embedding": embedding.tolist()
    })


async def generate_legal_document(request: LegalDocumentRequest) -> str:
    """Generates a legally formatted document based on the selected template and user-provided clauses."""
    template = request.template.lower()
    clauses = request.clauses

    document_templates = {
        "rental agreement": """
        **RENTAL AGREEMENT**

        This Rental Agreement is made and entered into on [Effective Date] by and between:

        **Landlord:** [Landlord Name], residing at [Landlord Address].

        **Tenant:** [Tenant Name], residing at [Tenant Address].

        **1. RENTAL PROPERTY**  
        The Landlord agrees to rent to the Tenant the property located at [Rental Address] ("Premises").  

        **2. ADDITIONAL CLAUSES**  
        {clauses}  
        """,

        "land registration": """
        **LAND REGISTRATION AGREEMENT**

        This Land Registration Agreement is made and entered into on [Effective Date] between:

        **Seller:** [Seller Name], residing at [Seller Address].  

        **Buyer:** [Buyer Name], residing at [Buyer Address].  

        **1. LAND DESCRIPTION**  
        - Land Address: [Land Address]  
        - Total Area: [Land Size] acres  

        **2. ADDITIONAL CLAUSES**  
        {clauses}  
        """,

        "building registration": """
        **BUILDING REGISTRATION AGREEMENT**

        This Building Registration Agreement is made on [Effective Date] between:

        **Owner:** [Owner Name], residing at [Owner Address].  

        **Registrar:** [Registrar Name], an official representative of [City/Municipality].  

        **1. PROPERTY DETAILS**  
        - Address: [Building Address]  
        - Type: [Residential/Commercial]  

        **2. ADDITIONAL CLAUSES**  
        {clauses}  
        """,

        "lease agreement": """
        **LEASE AGREEMENT**

        This Lease Agreement is made and entered into as of [Effective Date] by and between:

        **Landlord:** [Landlord Name], residing at [Landlord Address].  

        **Tenant:** [Tenant Name], residing at [Tenant Address].  

        **1. PREMISES**  
        - Address: [Property Address]  

        **2. ADDITIONAL CLAUSES**  
        {clauses}  
        """
    }

    if template not in document_templates:
        raise HTTPException(status_code=400, detail="Invalid template selected.")

    return document_templates[template].format(clauses=clauses)

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extracts text from PDF using PyMuPDF."""
    try:
        with pdf_file.file as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            text = " ".join([page.get_text("text") for page in doc])
        return clean_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def extract_text_from_image(image_file: UploadFile) -> str:
    """Extracts text from images using OCR (pytesseract)."""
    try:
        image = Image.open(io.BytesIO(image_file.file.read()),)
        text = pytesseract.image_to_string(image)
        return clean_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from image: {str(e)}")

# ---------------------------- FastAPI Endpoints ----------------------------
@app.post("/generate-legal-doc/")
async def generate_legal_doc(request: LegalDocumentRequest):
    result = await generate_legal_document(request)
    return jsonable_encoder({"legal_document": result})

@app.post("/upload-legal-doc/")
async def upload_legal_doc(file: UploadFile = File(...)):
    """Uploads a legal document (PDF/Image), extracts text, and stores it."""
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
            summary = await summarize_text(text)
        elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(file)
            summary = await summarize_text(text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
        
        store_document(text)
        return {"message": "Document uploaded successfully", "extracted_text": text[:500], "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chatbot/")
async def answer_legal_question(request: ChatRequest):
    """Answers legal questions using retrieved context from FAISS."""
    try:
        context = await retrieve_similar(request.question)
        prompt = f"Context: {context}\n\nQuestion: {request.question}" if context else request.question
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        outputs = model.generate(inputs["input_ids"], max_length=200)
        
        return {"answer": tokenizer.decode(outputs[0], skip_special_tokens=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot response: {str(e)}")
    
    
    
@app.on_event("shutdown")
async def shutdown_event():
    """Closes MongoDB connection on server shutdown."""
    client.close()
