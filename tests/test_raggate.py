from raggate.embed import HashEmbedder
from raggate.eval import evaluate, load
from raggate.gate import check
from raggate.index import chunk, Index


def test_chunking_overlaps_and_filters():
    text = "\n\n".join(f"Paragraph {i} " + "content " * 30 for i in range(6))
    chunks = chunk(text, size=400, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c) > 40 for c in chunks)


def test_corpus_and_labels_integrity():
    docs, queries = load("docs.jsonl"), load("queries.jsonl")
    ids = {d["id"] for d in docs}
    assert len(docs) == 5 and len(queries) == 12
    for q in queries:
        assert set(q["relevant"]) <= ids


def test_index_retrieves_topical_match():
    idx = Index(HashEmbedder())
    idx.add_document("a", "Molten salt stores solar heat in receiver towers.\n\nHeliostats focus the light precisely.")
    idx.add_document("b", "Hydrophones record snapping shrimp on the reef.\n\nQuiet reefs decline further.")
    # Unit test of index MECHANICS - queries share content words with their doc.
    # Semantic hardness (paraphrase, no overlap) lives in the eval set, not here.
    assert idx.search("what stores the heat from sunlight", k=1)[0][0] == "a"
    assert idx.search("record snapping shrimp on a reef", k=1)[0][0] == "b"


def test_hash_baseline_clears_gate():
    report = evaluate(HashEmbedder())
    assert report["recall_at_k"] >= 0.90, report["missed"]
    assert check(report) == 0


def test_gate_refuses_weak_report():
    assert check({"embedder": "x", "k": 3, "queries": 12,
                  "recall_at_k": 0.75, "mrr": 0.6, "missed": []}) == 1


def test_agent_decomposes_and_grounds():
    from raggate.agent import run_agent, grounding_gate, ScriptedModel
    from raggate.eval import build_index
    idx = build_index(HashEmbedder())
    q = "Why do degraded reefs fall quiet and why does warm ice weaken radar returns?"
    answer, retrieved = run_agent(ScriptedModel(), idx, q)
    assert len(retrieved) == 4              # two sub-queries x k=2
    assert grounding_gate(answer, retrieved) == 0


def test_hallucinated_citation_refused():
    from raggate.agent import run_agent, grounding_gate, HallucinatingModel
    from raggate.eval import build_index
    idx = build_index(HashEmbedder())
    answer, retrieved = run_agent(HallucinatingModel(), idx,
                                  "What do hydrophones record on healthy reefs and what makes bioreactor scale-up hard?")
    assert grounding_gate(answer, retrieved) == 1


def test_agentic_coverage_not_worse_than_single_shot():
    from raggate.compare import run_comparison
    r = run_comparison()
    assert r["grounding_failures"] == 0
    assert r["agentic_full_coverage"] >= r["single_shot_full_coverage"]
