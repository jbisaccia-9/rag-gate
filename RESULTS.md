# Results

Generated 2026-08-21 by `scripts/make_results.py` — every block below is captured command output, not prose.

## Unit tests

`python -m pytest -q` — exit 0, OK

```
........                                                                 [100%]
8 passed in 0.03s
```

## Retrieval gate: hash baseline

`python -m raggate gate hash` — exit 0, OK

```
PASS  recall@3: 1.0 (min 0.9)  [embedder: hash-bow-512-nostop]
        mrr: 0.9583
GATE: PASSED - index cleared the retrieval bar.
```

## Retrieval gate: nvidia/nv-embed-v1 (live)

Skipped this regeneration: `NVIDIA_API_KEY` not set. The last recorded run of this section is preserved in git history.

## Agentic mode: decomposition loop + grounding gate + coverage comparison

`python -m raggate agent` — exit 0, OK

```
PASS  answer grounding: 3 citations, 0 never retrieved
ANSWER GATE: PASSED - every citation was actually retrieved.
  PASS  answer grounding: 3 citations, 0 never retrieved
ANSWER GATE: PASSED - every citation was actually retrieved.
  PASS  answer grounding: 2 citations, 0 never retrieved
ANSWER GATE: PASSED - every citation was actually retrieved.
  PASS  answer grounding: 4 citations, 0 never retrieved
ANSWER GATE: PASSED - every citation was actually retrieved.
  c1: single-shot FULL ['glacier-radar', 'night-trains', 'solar-thermal'] | agentic FULL ['glacier-radar', 'night-trains', 'solar-thermal']
  c2: single-shot FULL ['coral-acoustics', 'fermentation', 'solar-thermal'] | agentic FULL ['coral-acoustics', 'fermentation', 'solar-thermal']
  c3: single-shot FULL ['glacier-radar', 'night-trains', 'solar-thermal'] | agentic FULL ['night-trains', 'solar-thermal']
  c4: single-shot FULL ['coral-acoustics', 'glacier-radar', 'night-trains'] | agentic FULL ['coral-acoustics', 'glacier-radar', 'night-trains', 'solar-thermal']
coverage: single-shot 4/4, agentic 4/4; grounding failures: 0
```

## Hallucinating agent: refused

`python -m raggate agent hallucinating` — expected non-zero exit, OK

```
FAIL  answer grounding: 4 citations, 1 never retrieved
    UNGROUNDED: [ghost-doc#9]
ANSWER GATE: FAILED - the answer cites chunks the agent never saw.
  FAIL  answer grounding: 4 citations, 1 never retrieved
    UNGROUNDED: [ghost-doc#9]
ANSWER GATE: FAILED - the answer cites chunks the agent never saw.
  FAIL  answer grounding: 3 citations, 1 never retrieved
    UNGROUNDED: [ghost-doc#9]
ANSWER GATE: FAILED - the answer cites chunks the agent never saw.
  FAIL  answer grounding: 5 citations, 1 never retrieved
    UNGROUNDED: [ghost-doc#9]
ANSWER GATE: FAILED - the answer cites chunks the agent never saw.
  c1: single-shot FULL ['glacier-radar', 'night-trains', 'solar-thermal'] | agentic FULL ['glacier-radar', 'night-trains', 'solar-thermal']
  c2: single-shot FULL ['coral-acoustics', 'fermentation', 'solar-thermal'] | agentic FULL ['coral-acoustics', 'fermentation', 'solar-thermal']
  c3: single-shot FULL ['glacier-radar', 'night-trains', 'solar-thermal'] | agentic FULL ['night-trains', 'solar-thermal']
  c4: single-shot FULL ['coral-acoustics', 'glacier-radar', 'night-trains'] | agentic FULL ['coral-acoustics', 'glacier-radar', 'night-trains', 'solar-thermal']
coverage: single-shot 4/4, agentic 4/4; grounding failures: 4
```
