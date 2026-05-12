import os
import openai
from .embeddings import VectorStore

openai.api_key = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

SYSTEM_PROMPT = """You are an assistant that answers user questions using the provided context. If the answer is not contained in the context, say you don't know. Cite sources where possible."""

class Agent:
    def __init__(self, vs: VectorStore):
        self.vs = vs
        self.model = DEFAULT_MODEL

    def _compose_prompt(self, query, docs):
        context = ''
        for i, d in enumerate(docs):
            text = d.get('text', '')
            src = d.get('source', 'unknown')
            context += f'[{i}] {text} (source: {src})\n'
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely, and include source citation like [0], [1] when you use context."}
        ]
        return prompt

    def answer(self, query, k=4, temperature=0.0):
        docs = self.vs.query(query, k=k)
        prompt = self._compose_prompt(query, docs)
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=512
        )
        text = resp['choices'][0]['message']['content'].strip()
        return {"answer": text, "docs": docs}
