"""Chunk, embed, and search - a vector store in plain Python.

No framework: chunking is paragraph-aware with overlap, the index is a list of
vectors, and search is exact cosine top-k. At corpus scale that fits in memory
this is not a toy, it is the honest baseline every ANN index approximates.
"""
import math


def chunk(text, size=600, overlap=120):
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:] + "\n\n" + p   # carry overlap forward
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) > 40]


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return num / den


class Index:
    def __init__(self, embedder):
        self.embedder = embedder
        self.rows = []          # (vector, doc_id, chunk_text)

    def add_document(self, doc_id, text):
        for c in chunk(text):
            self.rows.append((self.embedder.embed(c), doc_id, c))

    def search(self, query, k=3):
        qv = self.embedder.embed(query)
        scored = sorted(self.rows, key=lambda r: -cosine(qv, r[0]))
        return [(doc_id, text) for _, doc_id, text in scored[:k]]
