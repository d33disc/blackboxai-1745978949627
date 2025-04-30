import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import motor.motor_asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ML Creator and Tools API")

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["ml_tools_db"]
documents_collection = db["documents"]

# Load embedding model for RAG
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Model cache
model_cache = {}

class LoadModelRequest(BaseModel):
    model_name: str

class QueryRequest(BaseModel):
    model_name: str
    query: str
    top_k: Optional[int] = 3

class Document(BaseModel):
    id: Optional[str]
    content: str

@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    model_name = request.model_name
    if model_name in model_cache:
        return {"message": f"Model {model_name} already loaded."}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_cache[model_name] = {"tokenizer": tokenizer, "model": model}
        return {"message": f"Model {model_name} loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add_document")
async def add_document(doc: Document):
    embedding = embedding_model.encode(doc.content, convert_to_tensor=True)
    doc_dict = doc.dict()
    doc_dict["embedding"] = embedding.cpu().numpy().tolist()
    result = await documents_collection.insert_one(doc_dict)
    return {"inserted_id": str(result.inserted_id)}

@app.post("/query")
async def query_model(request: QueryRequest):
    if request.model_name not in model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded")
    # Retrieve relevant documents for RAG
    query_embedding = embedding_model.encode(request.query, convert_to_tensor=True)
    docs_cursor = documents_collection.find({})
    docs = await docs_cursor.to_list(length=100)
    # Compute similarity scores
    doc_embeddings = [doc["embedding"] for doc in docs]
    if not doc_embeddings:
        context = ""
    else:
        import torch
        doc_embeddings_tensor = torch.tensor(doc_embeddings)
        cos_scores = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]
        top_results = torch.topk(cos_scores, k=min(request.top_k, len(docs)))
        context = " ".join([docs[idx]["content"] for idx in top_results.indices])
    # Run model with context + query
    tokenizer = model_cache[request.model_name]["tokenizer"]
    model = model_cache[request.model_name]["model"]
    input_text = f"Context: {context} Query: {request.query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "ML Creator and Tools API is running."}
