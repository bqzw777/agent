import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

EMB_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

class VectorStore:
    def __init__(self, path='data/index.faiss', dim=384, persist_dir='data'):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.model = SentenceTransformer(EMB_MODEL)
        self.dim = dim
        self.index_path = os.path.join(self.persist_dir, 'index.faiss')
        self.meta_path = os.path.join(self.persist_dir, 'meta.pkl')
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, 'rb') as f:
                    self.meta = pickle.load(f)
            except Exception:
                # fallback to fresh index if corrupted
                self.index = faiss.IndexFlatIP(self.dim)
                self.meta = []
                self._save()
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = []
            self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)

    def _embed(self, texts):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # normalize for inner-product similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-12)
        return embs.astype('float32')

    def add_texts(self, texts, metadatas=None):
        if not texts:
            return
        embs = self._embed(texts)
        self.index.add(embs)
        start_id = len(self.meta)
        for i, t in enumerate(texts):
            m = metadatas[i] if metadatas else {}
            m.setdefault('text', t)
            m.setdefault('id', start_id + i)
            self.meta.append(m)
        self._save()

    def query(self, q, k=5):
        emb = self._embed([q])
        D, I = self.index.search(emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.meta):
                entry = dict(self.meta[idx])
                entry['score'] = float(score)
                results.append(entry)
        return results
