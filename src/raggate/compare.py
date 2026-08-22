"""The measured experiment: single-shot vs agentic retrieval on compound
questions. Coverage = every relevant document appears in what was retrieved.
Numbers over narrative - whatever the table says is what the README reports.
"""
import json
import pathlib

from .agent import run_agent, grounding_gate, ScriptedModel
from .embed import get_embedder
from .eval import build_index

ROOT = pathlib.Path(__file__).resolve().parents[2]


def compound_queries():
    return [json.loads(l) for l in
            (ROOT / "data" / "compound_queries.jsonl").read_text().splitlines() if l.strip()]


def run_comparison(model_factory=ScriptedModel, embedder_kind="hash"):
    index = build_index(get_embedder(embedder_kind))
    rows, gate_fail = [], 0
    for q in compound_queries():
        single = {d for d, _, _ in index.search_ids(q["query"], k=3)}
        answer, retrieved = run_agent(model_factory(), index, q["query"])
        agent_docs = {cid.split("#")[0] for cid in retrieved}
        gate_fail += grounding_gate(answer, retrieved)
        rows.append({"id": q["id"], "relevant": q["relevant"],
                     "single_shot_covered": set(q["relevant"]) <= single,
                     "agentic_covered": set(q["relevant"]) <= agent_docs,
                     "single_docs": sorted(single), "agent_docs": sorted(agent_docs)})
    s = sum(r["single_shot_covered"] for r in rows)
    a = sum(r["agentic_covered"] for r in rows)
    report = {"compound_queries": len(rows), "single_shot_full_coverage": s,
              "agentic_full_coverage": a, "grounding_failures": gate_fail,
              "detail": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "agentic_comparison.json").write_text(json.dumps(report, indent=2))
    for r in rows:
        print(f"  {r['id']}: single-shot {'FULL' if r['single_shot_covered'] else 'partial'} "
              f"{r['single_docs']} | agentic {'FULL' if r['agentic_covered'] else 'partial'} "
              f"{r['agent_docs']}")
    print(f"coverage: single-shot {s}/{len(rows)}, agentic {a}/{len(rows)}; "
          f"grounding failures: {gate_fail}")
    return report
