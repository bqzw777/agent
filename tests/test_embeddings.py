import os
import shutil
from agent.embeddings import VectorStore


def test_add_and_query():
    tmp_dir = 'data_test'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    vs = VectorStore(persist_dir=tmp_dir)
    texts = ["This is a test document about cats.", "Another text about dogs."]
    metas = [{"source": "t1"}, {"source": "t2"}]
    vs.add_texts(texts, metas)
    res = vs.query("Tell me about cats", k=2)
    assert len(res) >= 1
    assert any('cats' in r.get('text','').lower() for r in res)
    shutil.rmtree(tmp_dir)
