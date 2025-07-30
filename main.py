import os
import io
import asyncio
from uuid import uuid4
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader
import docx
from pptx import Presentation

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from pymongo.errors import ConnectionFailure

from pinecone import Pinecone, PodSpec
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# --- Load env vars ---
load_dotenv()

# --- FastAPI App ---
app = FastAPI(title="NoteFlow AI", version="1.0.0")

# --- CORS ---
origins = ["http://localhost", "http://127.0.0.1:8081", "exp://*", os.getenv("FRONTEND_URL", "http://example.com")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
MONGO_URI = os.getenv("MONGO_URL")
EMBEDDING_DIMENSION = 384

# --- Globals ---
llm_model_instance: Optional[genai.GenerativeModel] = None
embedding_model_instance: Optional[SentenceTransformer] = None
pinecone_client_instance: Optional[Pinecone] = None
pinecone_index_instance = None
mongo_client_instance = None
db_instance = None
user_collection_instance = None

# --- Models ---
class QueryRequest(BaseModel):
    user_id: str
    document_id: str
    query: str

# --- Startup ---
@app.on_event("startup")
async def startup():
    global llm_model_instance, embedding_model_instance, pinecone_client_instance
    global pinecone_index_instance, mongo_client_instance, db_instance, user_collection_instance

    # MongoDB
    try:
        mongo_client_instance = AsyncIOMotorClient(MONGO_URI)
        db_instance = mongo_client_instance.get_database("test")
        user_collection_instance = db_instance["users"]
        await mongo_client_instance.admin.command("ping")
        print("✅ MongoDB connected")
    except Exception as e:
        print("❌ MongoDB error:", e)
        raise

    # Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model_instance = await asyncio.to_thread(genai.GenerativeModel, model_name="models/gemini-2.0-flash")

    print("✅ Gemini Pro initialized")

    # SentenceTransformer
    embedding_model_instance = await asyncio.to_thread(SentenceTransformer, "all-MiniLM-L6-v2")
    print("✅ Embedding model loaded")

    # Pinecone
    pinecone_client_instance = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pinecone_client_instance.list_indexes().names():
        await asyncio.to_thread(
            pinecone_client_instance.create_index,
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=PodSpec(environment=PINECONE_ENVIRONMENT)
        )
    pinecone_index_instance = pinecone_client_instance.Index(PINECONE_INDEX_NAME)
    print("✅ Pinecone index ready")

@app.on_event("shutdown")
async def shutdown():
    if mongo_client_instance:
        mongo_client_instance.close()
        print("🛑 MongoDB connection closed")

# --- Helper Functions ---
async def extract_text(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()
    stream = io.BytesIO(content)

    if ext == ".pdf":
        reader = await asyncio.to_thread(PdfReader, stream)
        return "\n".join([await asyncio.to_thread(lambda p=p: p.extract_text() or "") for p in reader.pages]).strip()
    elif ext == ".docx":
        doc = await asyncio.to_thread(docx.Document, stream)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    elif ext == ".pptx":
        prs = await asyncio.to_thread(Presentation, stream)
        return "\n".join([
            shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text
        ]).strip()
    elif ext == ".txt":
        return content.decode("utf-8").strip()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def chunk_text(text: str) -> List[str]:
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

async def get_user_labels(user_id: str) -> List[str]:
    user = await user_collection_instance.find_one({"_id": ObjectId(user_id)})
    return user.get("className", []) if user else []

async def embed_and_store(chunks: List[str], user_id: str, doc_id: str):
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = await asyncio.to_thread(embedding_model_instance.encode, chunk)
        vectors.append({
            "id": f"{doc_id}_chunk_{i}",
            "values": emb.tolist(),
            "metadata": {
                "user_id": user_id,
                "file_id": doc_id,
                "chunk_index": i,
                "text": chunk
            }
        })
    await asyncio.to_thread(pinecone_index_instance.upsert, vectors=vectors)

async def classify_with_llm(text: str, label_list: List[str]) -> str:
    prompt = f"""
You are a document classifier. Classify this document into one of the following categories:
{label_list}

Document:
{text[:2000]}

Answer with only one category name from the above list.
"""
    result = await asyncio.to_thread(llm_model_instance.generate_content, prompt)
    return result.text.strip()

# --- Endpoints ---
@app.post("/classify")
async def classify_file(user_id: str = Form(...), file: UploadFile = File(...)):
    text = await extract_text(file)
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="No readable text found")

    chunks = chunk_text(text)
    doc_id = str(uuid4())

    label_list = await get_user_labels(user_id)
    if not label_list:
        raise HTTPException(status_code=404, detail="No classification labels found for user")

    await embed_and_store(chunks, user_id, doc_id)
    label = await classify_with_llm(text, label_list)

    return {"success": True, "document_id": doc_id, "label": label, "chunks_indexed": len(chunks)}

@app.post("/query")
async def query_document(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_vector = await asyncio.to_thread(embedding_model_instance.encode, request.query)
    query_vector = query_vector.tolist()

    response = await asyncio.to_thread(
        pinecone_index_instance.query,
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        filter={"user_id": request.user_id, "file_id": request.document_id}
    )

    matches = response.get("matches", [])
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant content found")

    context = "\n\n".join([m["metadata"]["text"] for m in matches])
    prompt = f"""
You are an intelligent assistant. Answer the question using only the context provided below.

Context:
{context}

Question:
{request.query}

Answer:
"""
    result = await asyncio.to_thread(llm_model_instance.generate_content, prompt)
    return {"success": True, "question": request.query, "answer": result.text.strip(), "context_chunks_used": len(matches)}