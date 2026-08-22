# Results

Generated 2026-08-21 by `scripts/make_results.py` — every block below is captured command output, not prose.

## Unit tests

`python -m pytest -q` — exit 0, OK

```
.....                                                                    [100%]
5 passed in 0.03s
```

## Retrieval gate: hash baseline

`python -m raggate gate hash` — exit 0, OK

```
PASS  recall@3: 1.0 (min 0.9)  [embedder: hash-bow-512-nostop]
        mrr: 0.9583
GATE: PASSED - index cleared the retrieval bar.
```
