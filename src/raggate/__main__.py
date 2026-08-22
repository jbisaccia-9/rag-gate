"""CLI:  python -m raggate eval|gate [hash|nvidia]
        python -m raggate agent [scripted|hallucinating|nvidia-chat]"""
import json
import sys

from .embed import get_embedder
from .eval import evaluate
from .gate import check


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "suite":
        from .btsuite import run_local
        sys.exit(run_local())
    if cmd == "agent":
        from .agent import ScriptedModel, HallucinatingModel, NVIDIAChatModel
        from .compare import run_comparison
        factory = {"scripted": ScriptedModel, "hallucinating": HallucinatingModel,
                   "nvidia-chat": NVIDIAChatModel}[arg or "scripted"]
        report = run_comparison(model_factory=factory)
        ok = (report["grounding_failures"] == 0 and
              report["agentic_full_coverage"] >= report["single_shot_full_coverage"])
        sys.exit(0 if ok else 1)
    report = evaluate(get_embedder(arg or "hash"))
    if cmd == "gate":
        sys.exit(check(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
