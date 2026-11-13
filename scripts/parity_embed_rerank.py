#!/usr/bin/env python3
"""Parity harness: mlx_api (old) vs mlx_inference (new) for embed + rerank.

Read-only on both services. Prints a PASS/FAIL table; exits non-zero on FAIL.

    python3 parity_embed_rerank.py --old http://127.0.0.1:9131 --new http://127.0.0.1:9141 \
        --key-file /path/key --corpus /path/corpus.txt

Gates (from the plan):
  * embeddings: cosine >= 0.9999 for every passage
  * rerank: identical top-10 ordering for every query
  * anything else is reported as a number, not a verdict
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import urllib.error
import urllib.request

COSINE_GATE = 0.9999


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="mlx_api base url")
    ap.add_argument("--new", required=True, help="mlx_inference base url")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--docs-per-query", type=int, default=30)
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    args = ap.parse_args()

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
        old = post(f"{args.old}/v2/rerank", {"model": args.rerank_model, "query": query, "documents": docs, "top_n": 10}, key)
        new = post(f"{args.new}/v1/rerank", {"model": args.rerank_model, "query": query, "documents": docs, "top_n": 10}, key)
        old_idx = [r["index"] for r in old["results"]]
        new_idx = [r["index"] for r in new["results"]]
        if old_idx != new_idx:
            order_ok = False
            print(f"  ORDER MISMATCH q{q}: old={old_idx} new={new_idx}")
        old_scores = {r["index"]: r["relevance_score"] for r in old["results"]}
        new_scores = {r["index"]: r["relevance_score"] for r in new["results"]}
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
