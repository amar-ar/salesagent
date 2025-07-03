from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
from datetime import datetime
import logging
from groq import Groq
import fitz  # PyMuPDF for PDF processing
import numpy as np
import pymongo
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urljoin
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ultimate AI Sales Assistant",
    description="AI-powered sales assistant with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test_database")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize services
client = MongoClient(MONGO_URL)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Simple embedding function instead of using SentenceTransformer
def get_embedding(text):
    # This is a very simple embedding function that just returns a hash-based vector
    # In a production environment, you would use a proper embedding model
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    # Convert bytes to a list of floats between -1 and 1
    return [(b / 127.5) - 1 for b in hash_bytes]

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Collections
books_collection = db.books
chunks_collection = db.chunks
conversations_collection = db.conversations

# Pydantic models
class BookInfo(BaseModel):
    title: str
    author: str
    description: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    actions: List[str]
    kpis: List[Dict[str, Any]]
    conversation_id: str

class BookUpload(BaseModel):
    book_id: str
    title: str
    author: str

# Utility functions
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    try:
        # Use our simple embedding function instead of the model
        embeddings = [get_embedding(text) for text in texts]
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

def semantic_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Perform semantic search on document chunks"""
    try:
        # For now, use text search as fallback (can be upgraded to vector search)
        # MongoDB text search
        search_results = chunks_collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        results = []
        for doc in search_results:
            doc['_id'] = str(doc['_id'])
            results.append(doc)
        
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def determine_model_complexity(query: str) -> str:
    """Determine whether to use Scout or Maverick model"""
    complex_keywords = [
        "analyze", "compare", "strategy", "complex", "detailed", "comprehensive",
        "multi-step", "advanced", "sophisticated", "interpret", "visual", "graph", "chart"
    ]
    
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in complex_keywords):
        return "llama-3.2-90b-vision-preview"  # Maverick
    else:
        return "llama-3.1-8b-instant"  # Scout

def get_web_search_results(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Get web search results as fallback"""
    # Placeholder for web search functionality
    # In a real implementation, you'd use a web search API
    return []

def generate_response(query: str, context: str, model: str) -> Dict[str, Any]:
    """Generate response using Groq API"""
    try:
        system_prompt = """You are the Ultimate AI Sales Assistant with deep semantic comprehension of sales strategies and concepts. 
        
        Your task is to:
        1. Analyze the user's query for semantic meaning beyond surface keywords
        2. Use the provided context to generate accurate, actionable responses
        3. Provide structured outputs with clear citations
        4. Include actionable steps and relevant KPIs
        5. Maintain professional sales expertise throughout
        
        Response Format:
        - Direct Answer: Clear, comprehensive response
        - Action Steps: Specific, executable recommendations
        - KPIs: Relevant metrics aligned with sales frameworks
        - Sources: Reference the provided context with specificity
        
        Context: {context}
        
        Provide a professional, structured response that demonstrates deep understanding of both the query and the sales concepts."""
        
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Parse response into structured format
        # This is a simplified parsing - in production, you'd use more sophisticated parsing
        response_data = {
            "answer": content,
            "actions": extract_actions(content),
            "kpis": extract_kpis(content),
            "model_used": model
        }
        
        return response_data
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": "I apologize, but I'm having trouble generating a response at the moment. Please try again.",
            "actions": [],
            "kpis": [],
            "model_used": model
        }

def extract_actions(text: str) -> List[str]:
    """Extract action items from response text"""
    # Simple regex to find action-like statements
    action_patterns = [
        r"(?:Action|Step|Task|Todo|Recommendation):\s*(.+?)(?:\n|$)",
        r"(?:•|\*|\d+\.)\s*([A-Z][^.\n]+\.)",
        r"(?:You should|Consider|Implement|Focus on|Start by)\s+([^.\n]+\.)"
    ]
    
    actions = []
    for pattern in action_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        actions.extend(matches)
    
    return actions[:5]  # Limit to top 5 actions

def extract_kpis(text: str) -> List[Dict[str, Any]]:
    """Extract KPI-related information from response text"""
    kpi_keywords = [
        "conversion rate", "close rate", "pipeline", "revenue", "quota",
        "activity", "calls", "meetings", "demos", "proposals", "win rate"
    ]
    
    kpis = []
    for keyword in kpi_keywords:
        if keyword.lower() in text.lower():
            kpis.append({
                "name": keyword.title(),
                "category": "Sales Performance",
                "framework": "Standard Sales Metrics"
            })
    
    return kpis[:3]  # Limit to top 3 KPIs

# API Routes
@app.get("/")
async def root():
    return {"message": "Ultimate AI Sales Assistant API", "version": "1.0.0"}

@app.post("/api/upload-book")
async def upload_book(file: UploadFile = File(...), title: str = Query(...), author: str = Query(...)):
    """Upload and process a sales book PDF"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read PDF content
        pdf_content = await file.read()
        
        # Extract text
        text = extract_text_from_pdf(pdf_content)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Generate book ID
        book_id = str(uuid.uuid4())
        
        # Store PDF in GridFS
        pdf_id = fs.put(pdf_content, filename=file.filename, book_id=book_id)
        
        # Store book metadata
        book_doc = {
            "_id": book_id,
            "title": title,
            "author": author,
            "filename": file.filename,
            "pdf_id": str(pdf_id),
            "upload_date": datetime.utcnow(),
            "text_length": len(text)
        }
        books_collection.insert_one(book_doc)
        
        # Process text into chunks
        chunks = chunk_text(text)
        
        # Generate embeddings for chunks
        embeddings = generate_embeddings(chunks)
        
        # Store chunks with embeddings
        chunk_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_doc = {
                "_id": str(uuid.uuid4()),
                "book_id": book_id,
                "book_title": title,
                "book_author": author,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }
            chunk_docs.append(chunk_doc)
        
        if chunk_docs:
            chunks_collection.insert_many(chunk_docs)
            
            # Create text index for search
            try:
                chunks_collection.create_index([("text", "text")])
            except Exception as e:
                logger.info(f"Text index might already exist: {e}")
        
        return {
            "message": "Book uploaded and processed successfully",
            "book_id": book_id,
            "title": title,
            "author": author,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading book: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat with the AI sales assistant"""
    try:
        # Generate or use existing conversation ID
        conversation_id = message.conversation_id or str(uuid.uuid4())
        
        # Perform semantic search
        search_results = semantic_search(message.message)
        
        # Build context from search results
        context = ""
        sources = []
        for result in search_results:
            context += f"Source: {result.get('book_title', 'Unknown')} by {result.get('book_author', 'Unknown')}\n"
            context += f"Content: {result.get('text', '')}\n\n"
            
            sources.append({
                "book_title": result.get('book_title', 'Unknown'),
                "book_author": result.get('book_author', 'Unknown'),
                "text_preview": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                "relevance_score": result.get('score', 0)
            })
        
        # If no relevant context found, add web search fallback
        if not context:
            context = "No specific book content found. Drawing from general sales knowledge."
            web_results = get_web_search_results(message.message)
            sources.extend(web_results)
        
        # Determine model complexity
        model = determine_model_complexity(message.message)
        
        # Generate response
        response_data = generate_response(message.message, context, model)
        
        # Store conversation
        conversation_doc = {
            "_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "user_message": message.message,
            "ai_response": response_data["answer"],
            "sources": sources,
            "actions": response_data["actions"],
            "kpis": response_data["kpis"],
            "model_used": model,
            "timestamp": datetime.utcnow()
        }
        conversations_collection.insert_one(conversation_doc)
        
        return ChatResponse(
            response=response_data["answer"],
            sources=sources,
            actions=response_data["actions"],
            kpis=response_data["kpis"],
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books")
async def get_books():
    """Get list of uploaded books"""
    try:
        books = list(books_collection.find({}, {"_id": 1, "title": 1, "author": 1, "upload_date": 1}))
        for book in books:
            book["book_id"] = book["_id"]
            del book["_id"]
        return {"books": books}
    except Exception as e:
        logger.error(f"Error fetching books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        messages = list(conversations_collection.find(
            {"conversation_id": conversation_id},
            {"_id": 0}
        ).sort("timestamp", 1))
        
        return {"conversation_id": conversation_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sample-data")
async def create_sample_data():
    """Create sample sales data for demo purposes"""
    try:
        sample_books = [
            {
                "_id": "sample-book-1",
                "title": "The Sales Acceleration Formula",
                "author": "Mark Roberge",
                "filename": "sales_acceleration_formula.pdf",
                "upload_date": datetime.utcnow(),
                "text_length": 50000
            },
            {
                "_id": "sample-book-2", 
                "title": "Predictably Irrational",
                "author": "Dan Ariely",
                "filename": "predictably_irrational.pdf",
                "upload_date": datetime.utcnow(),
                "text_length": 45000
            }
        ]
        
        sample_chunks = [
            {
                "_id": str(uuid.uuid4()),
                "book_id": "sample-book-1",
                "book_title": "The Sales Acceleration Formula",
                "book_author": "Mark Roberge",
                "chunk_index": 0,
                "text": "The sales acceleration formula is built on three pillars: hiring, training, and managing salespeople. The most successful sales organizations focus on creating predictable, scalable processes that can be measured and optimized over time.",
                "embedding": get_embedding("The sales acceleration formula is built on three pillars: hiring, training, and managing salespeople."),
                "created_at": datetime.utcnow()
            },
            {
                "_id": str(uuid.uuid4()),
                "book_id": "sample-book-1",
                "book_title": "The Sales Acceleration Formula",
                "book_author": "Mark Roberge",
                "chunk_index": 1,
                "text": "Metrics-driven sales management requires tracking key performance indicators like conversion rates, average deal size, and sales cycle length. These metrics help identify bottlenecks in the sales process and optimize for better results.",
                "embedding": get_embedding("Metrics-driven sales management requires tracking key performance indicators like conversion rates, average deal size, and sales cycle length."),
                "created_at": datetime.utcnow()
            },
            {
                "_id": str(uuid.uuid4()),
                "book_id": "sample-book-2",
                "book_title": "Predictably Irrational",
                "book_author": "Dan Ariely",
                "chunk_index": 0,
                "text": "Understanding customer psychology is crucial for sales success. People don't always make rational decisions, and sales professionals who understand behavioral economics can better influence purchasing decisions.",
                "embedding": get_embedding("Understanding customer psychology is crucial for sales success."),
                "created_at": datetime.utcnow()
            }
        ]
        
        # Insert sample data
        books_collection.insert_many(sample_books)
        chunks_collection.insert_many(sample_chunks)
        
        # Create text index
        try:
            chunks_collection.create_index([("text", "text")])
        except Exception as e:
            logger.info(f"Text index might already exist: {e}")
        
        return {"message": "Sample data created successfully", "books": len(sample_books), "chunks": len(sample_chunks)}
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "groq_configured": bool(GROQ_API_KEY),
        "mongo_connected": bool(client.admin.command('ping'))
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "HTTP Exception"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
