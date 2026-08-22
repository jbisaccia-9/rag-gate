"""CLI:  python -m raggate eval [hash|nvidia]  |  gate [hash|nvidia]"""
import json
import sys

from .embed import get_embedder
from .eval import evaluate
from .gate import check


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    kind = sys.argv[2] if len(sys.argv) > 2 else "hash"
    report = evaluate(get_embedder(kind))
    if cmd == "gate":
        sys.exit(check(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
