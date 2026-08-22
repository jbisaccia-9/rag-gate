"""Retrieval evaluation: recall@k against hand-labeled queries.

A query counts as a hit if any of the top-k retrieved chunks comes from a
document labeled relevant. Labels were written against the corpus by hand,
including two deliberately hard paraphrase queries with near-zero keyword
overlap - the cases a bag-of-words baseline should miss and a semantic
embedder should catch. That spread is what gives the gate discrimination.
"""
import json
import pathlib

from .index import Index

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name):
    return [json.loads(l) for l in
            (ROOT / "data" / name).read_text().splitlines() if l.strip()]


def build_index(embedder):
    idx = Index(embedder)
    for doc in load("docs.jsonl"):
        idx.add_document(doc["id"], doc["title"] + "\n\n" + doc["text"])
    return idx


def evaluate(embedder, k=3):
    idx = build_index(embedder)
    queries = load("queries.jsonl")
    hits, misses, ranks = 0, [], []
    for q in queries:
        results = idx.search(q["query"], k=k)
        got = [doc_id for doc_id, _ in results]
        hit_rank = next((i + 1 for i, d in enumerate(got) if d in q["relevant"]), None)
        if hit_rank:
            hits += 1
            ranks.append(1.0 / hit_rank)
        else:
            ranks.append(0.0)
            misses.append(f"{q['id']}: {q['query']!r} -> {got}")
    report = {"embedder": embedder.name, "k": k, "queries": len(queries),
              "recall_at_k": round(hits / len(queries), 4),
              "mrr": round(sum(ranks) / len(ranks), 4),
              "missed": misses}
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
