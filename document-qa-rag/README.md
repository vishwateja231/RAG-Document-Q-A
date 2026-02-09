# Document Question Answering System (RAG-based)

A production-oriented Retrieval-Augmented Generation (RAG) project that lets users upload PDF documents, index document chunks in FAISS, retrieve relevant context, and generate answers with an OpenAI-compatible LLM API.

## Features

- Upload PDF documents via API or browser UI
- Parse and chunk document text
- Generate embeddings with Sentence Transformers
- Persist vectors in FAISS
- Ask natural-language questions over uploaded docs
- Retrieve top-k relevant chunks and generate grounded answers

## Architecture

```text
+----------------------+        +--------------------------+
|      Frontend UI     | <----> |      FastAPI Backend     |
|  (HTML + JavaScript) |        |  /upload /ask /documents |
+----------+-----------+        +-----------+--------------+
           |                                |
           |                                v
           |                      +---------+----------+
           |                      |    Ingestion       |
           |                      |  PyPDF + Chunking  |
           |                      +---------+----------+
           |                                |
           |                                v
           |                      +---------+----------+
           |                      | Embedding Model    |
           |                      | SentenceTransform. |
           |                      +---------+----------+
           |                                |
           |                                v
           |                      +---------+----------+
           |                      |  Vector Store      |
           |                      |   FAISS + Metadata |
           |                      +---------+----------+
           |                                |
           |                                v
           |                      +---------+----------+
           +--------------------> |  LLM Generation    |
                                  | OpenAI-compatible  |
                                  +--------------------+
```

## Project Structure

```text
document-qa-rag/
│
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── ingestion.py
│   └── config.py
│
├── data/
│   ├── documents/
│   └── vectorstore/
│
├── static/
│   └── index.html
├── requirements.txt
├── README.md
└── run.sh
```

## Setup and Run

1. Navigate to project folder:

```bash
cd document-qa-rag
```

2. Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Set your OpenAI-compatible key and optional endpoint/model:

```bash
export OPENAI_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
```

4. Start server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or simply:

```bash
./run.sh
```

## API Endpoints

- `POST /upload` — Upload and index a PDF document
- `POST /ask` — Ask a question over indexed documents
- `GET /documents` — List indexed documents
- `GET /health` — Health check
- `GET /` — Browser UI

## Sample cURL Requests

Upload a PDF:

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/document.pdf"
```

Ask a question:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the document?"}'
```

List documents:

```bash
curl "http://localhost:8000/documents"
```

## Example Response (Ask)

```json
{
  "answer": "The report states that revenue increased 24% year-over-year [1].",
  "sources": [
    {
      "document": "report.pdf",
      "text": "...",
      "score": 0.1182
    }
  ]
}
```

## Notes

- Only PDF files are accepted for ingestion.
- FAISS index and metadata are persisted under `data/vectorstore/`.
- Uploaded PDFs are stored under `data/documents/`.
