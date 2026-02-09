"""FastAPI entrypoint for Document QA RAG service."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.ingestion import chunk_text, extract_pdf_text
from app.rag_pipeline import RAGPipeline

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home() -> FileResponse:
    """Serve simple HTML UI."""
    ui_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return FileResponse(ui_path)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload PDF, extract text, chunk, embed, and index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    target_path = settings.upload_dir / file.filename

    try:
        with target_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        text = extract_pdf_text(target_path)
        chunks = chunk_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        stored = pipeline.add_document_chunks(file.filename, chunks)
    except ValueError as exc:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc

    return {
        "message": "Document processed successfully.",
        "document": file.filename,
        "chunks_indexed": stored,
    }


@app.post("/ask")
def ask_question(payload: AskRequest) -> dict:
    """Answer question using indexed documents and LLM."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return pipeline.answer_question(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc


@app.get("/documents")
def list_documents() -> dict:
    """List currently indexed documents."""
    return {"documents": pipeline.list_documents()}
