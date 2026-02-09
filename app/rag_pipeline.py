"""RAG pipeline: embedding, indexing, retrieval, and generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import settings


class RAGPipeline:
    """End-to-end pipeline for indexing and querying document chunks."""

    def __init__(self) -> None:
        self.vectorstore_dir = settings.vectorstore_dir
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vectorstore_dir / "index.faiss"
        self.metadata_path = self.vectorstore_dir / "metadata.json"

        self.embedder = SentenceTransformer(settings.embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()

        self.llm_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url)

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> List[Dict]:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return []

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add_document_chunks(self, document_name: str, chunks: List[str]) -> int:
        """Embed and store chunks in FAISS with metadata."""
        if not chunks:
            raise ValueError("No chunks to add.")

        vectors = self.embedder.encode(chunks, normalize_embeddings=True)
        vectors_np = np.asarray(vectors, dtype="float32")
        self.index.add(vectors_np)

        for chunk in chunks:
            self.metadata.append({"document": document_name, "text": chunk})

        self._persist()
        return len(chunks)

    def list_documents(self) -> List[str]:
        """Return sorted unique document names currently indexed."""
        docs = sorted({item["document"] for item in self.metadata})
        return docs

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict]:
        """Retrieve top-k relevant chunks for a question."""
        if self.index.ntotal == 0:
            return []

        k = top_k or settings.top_k
        q_vec = self.embedder.encode([question], normalize_embeddings=True)
        q_np = np.asarray(q_vec, dtype="float32")

        distances, indices = self.index.search(q_np, k)
        results: List[Dict] = []

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            hit = self.metadata[idx]
            results.append(
                {
                    "document": hit["document"],
                    "text": hit["text"],
                    "score": float(score),
                }
            )

        return results

    def answer_question(self, question: str) -> Dict:
        """Retrieve context and generate answer with OpenAI-compatible API."""
        contexts = self.retrieve(question)
        if not contexts:
            return {
                "answer": "No indexed content found. Please upload and process a document first.",
                "sources": [],
            }

        context_blob = "\n\n".join(
            [f"[{idx + 1}] ({ctx['document']}) {ctx['text']}" for idx, ctx in enumerate(contexts)]
        )

        system_prompt = (
            "You are a helpful assistant for document question answering. "
            "Answer only from the provided context. If unsure, say you don't know."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_blob}\n\n"
            "Provide a concise and accurate answer with source references like [1], [2]."
        )

        completion = self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return {
            "answer": completion.choices[0].message.content,
            "sources": contexts,
        }
