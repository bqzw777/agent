# AssistAI — Retrieval-Augmented Conversational Agent

This repository contains a minimal, production-oriented scaffold for a Retrieval-Augmented Generation (RAG) conversational agent called AssistAI. It is designed for demos, interviews, and as a starting point for production systems.

Resume-ready summary

- Built "AssistAI", a retrieval-augmented conversational agent (Python, FastAPI, OpenAI, SentenceTransformers, FAISS) that ingests domain documents, performs semantic search, and generates grounded responses with source citations.

Quick setup

Requirements
- Python 3.10+
- An OpenAI API key

Local run
1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. export OPENAI_API_KEY="sk-..."
4. export OPENAI_MODEL="gpt-4o-mini"  # optional; a sensible default is used if omitted
5. python app.py

Endpoints
- POST /ingest (multipart/form-data) file=@path/to/file - Ingests a TXT or PDF and adds chunks to the vector store
- POST /chat  {"query": "..."} - Returns an answer grounded in retrieved context

Demo
- See demo/run_demo.sh for example commands to ingest demo/demo_doc.txt and ask questions.

Files of interest
- app.py — FastAPI app exposing /ingest and /chat
- agent/embeddings.py — VectorStore using SentenceTransformers + FAISS
- agent/agent.py — RAG composition and OpenAI call (reads OPENAI_MODEL env var)
- demo/ — small demo doc and script
- tests/ — small unit test

Security and limitations
- The scaffold sends prompts to OpenAI; do not upload sensitive data without an appropriate contract and safeguards.
- The system uses a simple "say I don't know" system prompt but may still hallucinate if retrieval fails.

License
- MIT
