"""The retrieval gate: an index serves queries only if it clears recall@3 >= 0.90
on the labeled set. An index that cannot find the right document is not a
knowledge base - it is a random-quote generator with confidence."""
import sys

RECALL_MIN = 0.90


def check(report):
    ok = report["recall_at_k"] >= RECALL_MIN
    print(f"  {'PASS' if ok else 'FAIL'}  recall@{report['k']}: "
          f"{report['recall_at_k']} (min {RECALL_MIN})  [embedder: {report['embedder']}]")
    print(f"        mrr: {report['mrr']}")
    for m in report["missed"]:
        print(f"    MISSED: {m}")
    if not ok:
        print("GATE: FAILED - this index must not serve retrieval.")
        return 1
    print("GATE: PASSED - index cleared the retrieval bar.")
    return 0


if __name__ == "__main__":
    from .embed import get_embedder
    from .eval import evaluate
    sys.exit(check(evaluate(get_embedder(sys.argv[1] if len(sys.argv) > 1 else "hash"))))
