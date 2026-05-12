from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from agent.embeddings import VectorStore
from agent.agent import Agent
from io import BytesIO
import PyPDF2
import uvicorn
import os

app = FastAPI(title="AssistAI")

DATA_DIR = os.getenv("DATA_DIR", "data")
vs = VectorStore(persist_dir=DATA_DIR)
agent = Agent(vs)

class ChatRequest(BaseModel):
    query: str

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), source: str = Form(None)):
    content = await file.read()
    text = ""
    if file.filename.lower().endswith('.pdf'):
        reader = PyPDF2.PdfReader(BytesIO(content))
        for p in reader.pages:
            text += p.extract_text() or ""
    else:
        text = content.decode('utf-8')

    # chunk text into 2k-character pieces (simple approach)
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000) if text[i:i+2000].strip()]
    metadatas = [{"source": source or file.filename} for _ in chunks]
    vs.add_texts(chunks, metadatas)
    return {"status": "ok", "chunks_added": len(chunks)}

@app.post("/chat")
async def chat(req: ChatRequest):
    out = agent.answer(req.query)
    return out

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
