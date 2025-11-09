import os
import io
import math
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import db, create_document
from schemas import Document as DocumentSchema, Chunk as ChunkSchema, Conversation as ConversationSchema, Message as MessageSchema
import PyPDF2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight TF-IDF implementation (no heavy deps)
# Cache per document id
_vectorizers: Dict[str, Dict[str, float]] = {}
_doc_vectors: Dict[str, List[Dict[str, float]]] = {}
_doc_norms: Dict[str, List[float]] = {}
_doc_texts: Dict[str, List[str]] = {}

class AskPayload(BaseModel):
    conversation_id: str
    question: str
    top_k: int = 5


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += max(1, chunk_size - overlap)
    return chunks


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w.strip()]


def _build_tfidf(texts: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]], List[float]]:
    # Compute DF
    df: Dict[str, int] = {}
    tokenized = []
    for t in texts:
        toks = list(dict.fromkeys(_tokenize(t)))  # unique per doc
        tokenized.append(_tokenize(t))
        for w in toks:
            df[w] = df.get(w, 0) + 1
    n_docs = max(1, len(texts))
    idf: Dict[str, float] = {w: math.log((n_docs + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    # Build vectors
    vectors: List[Dict[str, float]] = []
    norms: List[float] = []
    for toks in tokenized:
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        vec: Dict[str, float] = {w: tf[w] * idf.get(w, 0.0) for w in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append(vec)
        norms.append(norm)
    return idf, vectors, norms


def _ensure_embeddings_for_document(document_id: str):
    chunks = list(db["chunk"].find({"document_id": document_id}).sort("index", 1))
    texts = [c.get("text", "") for c in chunks]
    if not texts:
        _vectorizers[document_id] = {}
        _doc_vectors[document_id] = []
        _doc_norms[document_id] = []
        _doc_texts[document_id] = []
        return [], []
    idf, vectors, norms = _build_tfidf(texts)
    _vectorizers[document_id] = idf
    _doc_vectors[document_id] = vectors
    _doc_norms[document_id] = norms
    _doc_texts[document_id] = texts
    return texts, chunks


def _vectorize_query(q: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    tf: Dict[str, int] = {}
    for w in _tokenize(q):
        tf[w] = tf.get(w, 0) + 1
    vec = {w: tf[w] * idf.get(w, 0.0) for w in tf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return vec, norm


def _cosine(vec_a: Dict[str, float], norm_a: float, vec_b: Dict[str, float], norm_b: float) -> float:
    # dot product on intersection
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a
    dot = 0.0
    for k, va in vec_a.items():
        vb = vec_b.get(k)
        if vb is not None:
            dot += va * vb
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    if file.content_type not in ["application/pdf", "application/x-pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file")

    pdf_bytes = await file.read()
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes), strict=False)

    num_pages = len(reader.pages)
    full_text = []
    for i in range(num_pages):
        try:
            page = reader.pages[i]
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        full_text.append(txt)
    joined = "\n".join(full_text).strip()

    from bson import ObjectId

    doc = DocumentSchema(
        filename=file.filename,
        content_type=file.content_type or "application/pdf",
        size=len(pdf_bytes),
        title=title,
        num_pages=num_pages,
        num_chunks=0,
    )
    doc_id = create_document("document", doc)

    chunks = _chunk_text(joined)
    for idx, chunk_text in enumerate(chunks):
        ch = ChunkSchema(document_id=doc_id, index=idx, text=chunk_text)
        create_document("chunk", ch)
    db["document"].update_one({"_id": ObjectId(doc_id)}, {"$set": {"num_chunks": len(chunks)}})

    _ensure_embeddings_for_document(doc_id)

    return {"document_id": doc_id, "filename": file.filename, "num_pages": num_pages, "num_chunks": len(chunks)}


@app.post("/start_conversation")
async def start_conversation(document_id: str = Form(...), title: Optional[str] = Form(None)):
    conv = ConversationSchema(document_id=document_id, title=title)
    conv_id = create_document("conversation", conv)
    return {"conversation_id": conv_id}


@app.post("/ask")
async def ask(payload: AskPayload):
    conv = db["conversation"].find_one({"_id": payload.conversation_id})
    if not conv:
        from bson import ObjectId
        try:
            conv = db["conversation"].find_one({"_id": ObjectId(payload.conversation_id)})
        except Exception:
            raise HTTPException(status_code=404, detail="Conversation not found")
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    document_id = conv["document_id"]

    # Make sure vectors exist
    if document_id not in _vectorizers:
        _ensure_embeddings_for_document(document_id)

    idf = _vectorizers.get(document_id, {})
    vectors = _doc_vectors.get(document_id, [])
    norms = _doc_norms.get(document_id, [])
    texts = _doc_texts.get(document_id, [])
    if not vectors:
        raise HTTPException(status_code=400, detail="No text available for this document")

    q_vec, q_norm = _vectorize_query(payload.question, idf)

    sims = []
    for i, vec in enumerate(vectors):
        sim = _cosine(q_vec, q_norm, vec, norms[i])
        sims.append((sim, i))
    sims.sort(reverse=True)

    top_k = max(1, min(payload.top_k, len(sims)))
    top_idx = [i for _, i in sims[:top_k]]
    retrieved = [texts[i] for i in top_idx]

    context = "\n\n".join(retrieved)[:4000]
    answer = _answer_from_context(context, payload.question)

    msg_user = MessageSchema(conversation_id=str(conv["_id"]), role="user", content=payload.question)
    create_document("message", msg_user)
    msg_assistant = MessageSchema(conversation_id=str(conv["_id"]), role="assistant", content=answer)
    create_document("message", msg_assistant)

    return {"answer": answer, "sources": top_idx}


def _answer_from_context(context: str, question: str) -> str:
    if not context.strip():
        return "I couldn't find relevant information in the uploaded PDF."
    import re
    sents = re.split(r"(?<=[.!?])\s+", context)
    q_words = [w.lower() for w in question.split() if len(w) > 2]
    scored = []
    for s in sents:
        score = sum(1 for w in q_words if w in s.lower()) + len(s) * 0.001
        scored.append((score, s))
    scored.sort(reverse=True)
    picked = [s for _, s in scored[:5]]
    summary = " ".join(picked)
    return summary[:1200] if summary else "I couldn't find relevant information in the uploaded PDF."


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
