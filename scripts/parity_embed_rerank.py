#!/usr/bin/env python3
"""Parity harness: two mlx servers, embed + rerank, compared passage by passage.

Read-only on both services. Prints a PASS/FAIL table; exits non-zero on FAIL.

    python3 parity_embed_rerank.py --old http://127.0.0.1:9131 --new http://127.0.0.1:9141 \
        --key-file /path/key --corpus /path/corpus.txt
    python3 parity_embed_rerank.py --self-test          # gate logic, no network

Written for the 2026-11 mlx_api → mlx_inference port, kept for the case it is
actually good at: comparing two builds of the same server across a rollout.

Gates — and where the numbers come from
---------------------------------------
The first version of this file used `cosine >= 0.9999` and exact top-10 order,
"from the plan". Neither survived contact with two independent implementations:

* The 2026-11-15 full-corpus run (400 passages, the port's final measurement)
  gave cosine **min 0.9976175**, mean 0.9999899, and rerank |Δ| **max 0.00138**
  over a 0–0.768 score range. The order flips that failed the gate were all
  swaps of items whose scores differed by less than that delta.
* The new server is not noisy: the same request twice returns byte-identical
  embeddings, and a passage embedded inside a batch of 32 is identical to the
  same passage embedded alone (cosine 1.0000000, measured 2026-11-15 on
  mlx-api-text). So a 0.9976 residual is **cross-implementation numerics**, not
  batching and not sampling — nothing a threshold above it can fix.
* What a threshold above it *would* catch is a real change: mean pooling instead
  of CLS on bge-m3 measured cosine **0.6007** in the pre-fix run.

Hence: embeddings must stay within 0.5 % cosine, which is ~2× the worst
cross-implementation residual observed and ~5× below any pooling/normalisation
mistake; and the rerank order is compared with tie tolerance, so a swap between
items closer than 0.005 is reported as a tie rather than a failure. Anything
else is reported as a number, not a verdict.

The Python `mlx_api` was retired on 2026-11-15, so `--old` now means "the other
build", not necessarily mlx_api.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import urllib.error
import urllib.request

# See the docstring for the measurements behind these two numbers.
COSINE_GATE = 0.995
ORDER_TIE_EPSILON = 0.005


def order_violation(
    old_idx: list[int],
    new_idx: list[int],
    old_scores: dict[int, float],
    new_scores: dict[int, float],
    eps: float = ORDER_TIE_EPSILON,
) -> str | None:
    """Why the new ranking is not an acceptable perturbation of the old one.

    Two kinds of difference are tolerated, both bounded by `eps`:
      * membership — an item at the top-N boundary drops out and another enters,
        with their scores (one from each side) inside the tie band;
      * order — adjacent items swap while their old scores are within `eps`.
    Anything larger is a real reordering, returned as a one-line description.
    """
    dropped = [i for i in old_idx if i not in new_scores]
    added = [i for i in new_idx if i not in old_scores]
    for d in dropped:
        if not added or min(abs(old_scores[d] - new_scores[a]) for a in added) > eps:
            return f"top-{len(old_idx)} membership changed beyond {eps}: {d} dropped out"

    for pos in range(1, len(new_idx)):
        a, b = new_idx[pos - 1], new_idx[pos]
        if a in old_scores and b in old_scores and old_scores[a] < old_scores[b]:
            gap = old_scores[b] - old_scores[a]
            if gap > eps:
                return (
                    f"reordered items scored {gap:.5f} apart on the old server: "
                    f"{b} now above {a}"
                )
    return None


def post(url: str, body: dict, key: str, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_all(
    base: str, passages: list[str], key: str, model: str, batch: int = 8
) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(passages), batch):
        chunk = passages[i : i + batch]
        # mlx_api requires `model`; mlx_inference drops it and lets the engine
        # name the checkpoint it serves, so the same body works on both.
        body = post(f"{base}/v1/embeddings", {"model": model, "input": chunk}, key)
        rows = sorted(body["data"], key=lambda d: d["index"])
        out.extend(row["embedding"] for row in rows)
    return out


def self_test() -> int:
    """Gate logic on fixed rankings — no network, no key."""
    near = {"0": 0.500, "1": 0.500, "2": 0.300, "3": 0.200}
    cases = [
        (
            "identical order",
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            near,
            near,
            None,
        ),
        (
            "tied swap within eps",
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            near,
            near,
            None,
        ),
        (
            "real inversion (0.3 apart)",
            [0, 1, 2, 3],
            [1, 2, 0, 3],
            near,
            near,
            "reordered",
        ),
        (
            "boundary swap beyond eps",
            [0, 1, 2],
            [0, 1, 9],
            {"0": 0.9, "1": 0.8, "2": 0.7},
            {"0": 0.9, "1": 0.8, "9": 0.1},
            "membership",
        ),
    ]
    bad = 0
    for name, old_idx, new_idx, old_s, new_s, want in cases:
        got = order_violation(
            old_idx,
            new_idx,
            {int(k): v for k, v in old_s.items()},
            {int(k): v for k, v in new_s.items()},
        )
        ok = (got is None) if want is None else (got is not None and want in got)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got or 'pass'}")
        bad += 0 if ok else 1
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", help="the reference build's base url")
    ap.add_argument("--new", help="the candidate build's base url")
    ap.add_argument("--key-file")
    ap.add_argument("--corpus")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--docs-per-query", type=int, default=30)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    # mlx_api exposed /v2/rerank; mlx_inference exposes /v1/rerank. Comparing two
    # builds of the same server therefore needs the old side pointed here too.
    ap.add_argument("--old-rerank-path", default="/v2/rerank")
    ap.add_argument("--self-test", action="store_true", help="check the gate logic only")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    missing = [
        flag
        for flag, value in (
            ("--old", args.old),
            ("--new", args.new),
            ("--key-file", args.key_file),
            ("--corpus", args.corpus),
        )
        if not value
    ]
    if missing:
        ap.error(f"missing {', '.join(missing)} (or pass --self-test)")
        return 2

    key = pathlib.Path(args.key_file).read_text().strip()
    passages = [
        line.strip()
        for line in pathlib.Path(args.corpus).read_text().splitlines()
        if len(line.strip()) >= 40
    ]
    if len(passages) < 50:
        print(f"corpus too small: {len(passages)} passages", file=sys.stderr)
        return 2
    passages = passages[:400]

    old_vecs = embed_all(args.old, passages, key, args.embed_model)
    new_vecs = embed_all(args.new, passages, key, args.embed_model)
    sims = [cosine(o, n) for o, n in zip(old_vecs, new_vecs)]
    d_old, d_new = len(old_vecs[0]), len(new_vecs[0])
    worst = min(sims)
    print(f"embeddings: {len(passages)} passages, dims {d_old} vs {d_new}")
    print(f"  cosine min={worst:.7f} mean={sum(sims)/len(sims):.7f}")

    ordered = sorted(range(len(sims)), key=lambda i: sims[i])
    print("  worst 3:")
    for i in ordered[:3]:
        print(f"    {sims[i]:.7f}  {passages[i][:70]!r}")

    embed_ok = worst >= COSINE_GATE and d_old == d_new
    print(f"  emb gate (>= {COSINE_GATE}): {'PASS' if embed_ok else 'FAIL'}")

    order_ok = True
    score_deltas: list[float] = []
    old_min = old_max = new_min = new_max = None
    for q in range(min(args.queries, len(passages))):
        query = passages[q]
        docs = passages[q + 1 : q + 1 + args.docs_per_query]
        if len(docs) < 5:
            continue
        old = post(f"{args.old}{args.old_rerank_path}", {"model": args.rerank_model, "query": query, "documents": docs, "top_n": 10}, key)
        new = post(f"{args.new}/v1/rerank", {"model": args.rerank_model, "query": query, "documents": docs, "top_n": 10}, key)
        old_idx = [r["index"] for r in old["results"]]
        new_idx = [r["index"] for r in new["results"]]
        old_scores = {r["index"]: r["relevance_score"] for r in old["results"]}
        new_scores = {r["index"]: r["relevance_score"] for r in new["results"]}
        if old_idx != new_idx:
            why = order_violation(old_idx, new_idx, old_scores, new_scores)
            if why:
                order_ok = False
                print(f"  ORDER VIOLATION q{q}: {why} (old={old_idx} new={new_idx})")
            else:
                print(f"  tied reorder q{q}: old={old_idx} new={new_idx}")
        for idx, s in old_scores.items():
            if idx in new_scores:
                score_deltas.append(abs(s - new_scores[idx]))
        old_min = min(old_scores.values()) if old_min is None else min(old_min, *old_scores.values())
        old_max = max(old_scores.values()) if old_max is None else max(old_max, *old_scores.values())
        new_min = min(new_scores.values()) if new_min is None else min(new_min, *new_scores.values())
        new_max = max(new_scores.values()) if new_max is None else max(new_max, *new_scores.values())

    if score_deltas:
        print(f"rerank scores: old range [{old_min:.4f}, {old_max:.4f}], "
              f"new range [{new_min:.4f}, {new_max:.4f}]")
        print(f"  |delta| max={max(score_deltas):.5f} mean={sum(score_deltas)/len(score_deltas):.5f}")
    print(f"  order gate: {'PASS' if order_ok else 'FAIL'}")

    return 0 if (embed_ok and order_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
