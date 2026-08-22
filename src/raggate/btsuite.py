"""Braintrust-shaped eval suite: data -> task -> scorers.

Three scorers cover the repo's three claims: retrieval holds on the labeled
queries (hit_at_3), the agentic mode's coverage never falls below single-shot
on compound questions, and the agent's answers stay grounded. All keyless;
the obs extra pushes the identical suite to hosted Braintrust.
"""
from .compare import run_comparison
from .embed import get_embedder
from .eval import build_index, load


def task(query, index):
    return {"docs": [d for d, _, _ in index.search_ids(query["query"], k=3)]}


def hit_at_3(query, out):
    return 1.0 if any(d in query["relevant"] for d in out["docs"]) else 0.0


def run_local():
    index = build_index(get_embedder("hash"))
    queries = load("queries.jsonl")
    hits = round(sum(hit_at_3(q, task(q, index)) for q in queries) / len(queries), 4)
    comp = run_comparison()
    scores = {
        "hit_at_3": hits,
        "agentic_coverage_holds": 1.0 if comp["agentic_full_coverage"] >= comp["single_shot_full_coverage"] else 0.0,
        "agent_grounding": 1.0 if comp["grounding_failures"] == 0 else 0.0,
    }
    for k, v in scores.items():
        print(f"  {k}: {v}")
    ok = hits >= 0.90 and scores["agentic_coverage_holds"] == 1.0 and scores["agent_grounding"] == 1.0
    print("SUITE: PASS - retrieval and agent hold." if ok else "SUITE: FAIL - regression.")
    return 0 if ok else 1


def push_braintrust():
    import braintrust  # optional extra
    index = build_index(get_embedder("hash"))
    braintrust.Eval("rag-gate",
                    data=lambda: [{"input": q, "expected": q["relevant"]} for q in load("queries.jsonl")],
                    task=lambda q: task(q, index),
                    scores=[lambda input, expected, output:
                            braintrust.Score(name="hit_at_3", score=hit_at_3(input, output))])
