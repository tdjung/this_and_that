#!/usr/bin/env python3
"""Smoke test for a running JEV-27B vLLM server.

  python serve/smoke_test.py --url http://localhost:8000 --model-dir /models/JEV-27B

Checks:
  1. /v1/models lists both the base model and the jev-decision LoRA
  2. System 2: a short chat completion (Qwen3.8-27B path)
  3. System 1: the model card's refund (noul) and supplier (choice) examples
  4. System 1 latency (median of N single requests)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevtools.client import JevClient  # noqa: E402

OK, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model-dir", default="./JEV-27B")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--latency-runs", type=int, default=10)
    a = ap.parse_args()
    h = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    failures = 0

    # 1. models
    r = httpx.get(f"{a.url}/v1/models", headers=h, timeout=10, trust_env=False)
    ids = [m["id"] for m in r.json().get("data", [])]
    good = "autotrust/JEV-27B" in ids and "jev-decision" in ids
    failures += not good
    print(f"[1] /v1/models -> {ids}  {OK if good else FAIL}")

    # 2. System 2
    t = time.time()
    r = httpx.post(f"{a.url}/v1/chat/completions", headers=h, timeout=120, trust_env=False, json={
        "model": "autotrust/JEV-27B",
        "messages": [{"role": "user", "content": "In one sentence, what is safety stock?"}],
        "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}})
    text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text
    good = r.status_code == 200 and len(text.strip()) > 0
    failures += not good
    print(f"[2] System 2 ({time.time() - t:.2f}s): {text.strip()[:120]!r}  {OK if good else FAIL}")

    # 3. System 1
    jev = JevClient(a.url, a.model_dir, api_key=a.api_key, timeout=60)
    p = jev.decide("noul", "Customer says the parcel arrived damaged and wants their money back.",
                   "Is the customer asking for a refund?")
    good = p["true"] > 0.9
    failures += not good
    print(f"[3a] noul refund: P(true)={p['true']:.3f} (card ≈ 0.978)  {OK if good else FAIL}")

    p = jev.decide("choice", "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                   "Supplier response for this scenario.",
                   ["issue_warning", "renegotiate", "dual_source", "maintain"])
    best = max(p, key=p.get)
    good = best == "dual_source"
    failures += not good
    print(f"[3b] choice supplier: { {k: round(v, 3) for k, v in p.items()} } "
          f"(card ≈ dual_source 0.62)  {OK if good else FAIL}")

    # 4. latency
    lat = []
    for _ in range(a.latency_runs):
        t = time.time()
        jev.decide("noul", "Fix the typo in README.md", "Does this request require changing code?")
        lat.append((time.time() - t) * 1000)
    print(f"[4] System 1 latency: median {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms "
          f"(card: 137 ms on one B200)")

    print("\nALL PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
