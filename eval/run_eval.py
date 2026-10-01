#!/usr/bin/env python3
"""Classify the 30 evaluation prompts with JEV-27B and compare against expected tiers.

  python eval/run_eval.py --config ~/.claude_auto/config.yaml
  python eval/run_eval.py --config proxy/config.example.yaml --jev-url http://gpu01:8000 --model-dir /models/JEV-27B
  python eval/run_eval.py --config proxy/config.example.yaml --mock      # 서버 없이 스크립트만 점검

Reports, per prompt: JEV raw tier, confidence, final tier after the confidence gate and rules.
Summary: exact / within-one accuracy, under-routing (quality risk) vs over-routing (cost),
per-language breakdown, confusion matrix, and option-order stability (--shuffles).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevtools.client import JevClient  # noqa: E402
from jevtools.routing import (RoutingConfig, build_state, combine, interpret,  # noqa: E402
                              jev_options)


class MockJev:
    """Keyword heuristic standing in for JEV so the pipeline can be tested without a GPU."""
    HINTS = [["번역", "명령어", "what does", "뭐야", "방법"],
             ["테스트 추가", "explain", "바꾸고", "에러", "logging"],
             ["기능을 추가", "ci", "migrate", "refactor", "커버리지", "spec", "스펙"],
             ["rtl", "architecture", "데드락", "latency", "oauth"],
             ["증명", "root cause", "50만", "lock-free", "risc-v", "고안"]]

    def __init__(self, cfg):
        self.canonical = jev_options(cfg)

    def decide(self, kind, state, question, options):
        low = state.lower()
        hints = [self.HINTS[self.canonical.index(o)] for o in options]
        score = [1.0 + 3 * sum(h in low for h in hs) for hs in hints]
        tot = sum(score)
        return {o: s / tot for o, s in zip(options, score)}


def run_one(jev, cfg, prompt, order=None):
    opts = jev_options(cfg)
    if order is not None:
        opts = [opts[i] for i in order]
    state = build_state([prompt], cfg)
    t = time.time()
    p = jev.decide("choice", state, cfg.question, opts)
    ms = (time.time() - t) * 1000
    jt, conf, probs = interpret(p, cfg, order)
    return jt, conf, probs, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/.claude_auto/config.yaml"))
    ap.add_argument("--prompts", default=str(ROOT / "eval" / "prompts.jsonl"))
    ap.add_argument("--jev-url", default=None, help="config의 jev.base_url 대신 사용")
    ap.add_argument("--model-dir", default=None, help="config의 jev.model_dir 대신 사용")
    ap.add_argument("--shuffles", type=int, default=0, help="선택지 순서를 섞어 N번 더 분류 (안정성 확인)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--set", choices=["all", "short", "long"], default="all", help="평가할 프롬프트 묶음")
    ap.add_argument("--budget-level", choices=["normal", "tight", "critical"], default="normal",
                    help="config budget.policy를 이 레벨로 적용했을 때의 최종 티어를 계산")
    a = ap.parse_args()

    raw = yaml.safe_load(Path(a.config).expanduser().read_text())
    cfg = RoutingConfig.from_dict(raw)
    names = [t.name for t in cfg.tiers]
    if a.mock:
        jev = MockJev(cfg)
    else:
        j = raw.get("jev", {})
        key = os.environ.get(j.get("api_key_env", "")) if j.get("api_key_env") else None
        jev = JevClient(a.jev_url or j["base_url"], a.model_dir or j["model_dir"],
                        adapter=j.get("adapter", "jev-decision"), timeout=60, api_key=key)

    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    if a.set != "all":
        rows = [r for r in rows if r.get("set", "short") == a.set]
    policy = ((raw.get("budget") or {}).get("policy") or {}).get(a.budget_level) or {}
    rng = random.Random(0)
    results = []
    print(f"budget level = {a.budget_level}  (policy: {policy or '없음'})\n")
    print(f"{'id':4} {'set':5} {'lang':4} {'len':>5} {'expected':12} {'jev':12} {'conf':>5} {'final':12} {'ms':>6}  prompt")
    print("-" * 120)
    for r in rows:
        exp = names.index(r["expected"])
        jt, conf, probs, ms = run_one(jev, cfg, r["prompt"])
        d = combine(cfg, r["prompt"], jt, conf, probs)
        final, reason = d.tier, d.reason
        rep = policy.get(names[final])
        if rep in names:
            reason += f" | budget:{a.budget_level} {names[final]}->{rep}"
            final = names.index(rep)
        flips = 0
        for _ in range(a.shuffles):
            order = list(range(len(names)))
            rng.shuffle(order)
            if run_one(jev, cfg, r["prompt"], order)[0] != jt:
                flips += 1
        mark = "✓" if final == exp else ("▼" if final < exp else "▲")
        head = " ".join(r["prompt"].split())[:40]
        print(f"{r['id']:4} {r.get('set', 'short'):5} {r['lang']:4} {len(r['prompt']):5} {names[exp]:12} {names[jt]:12} "
              f"{conf:5.2f} {names[final]:12} {ms:6.0f}  {mark} {head}")
        results.append({**r, "set": r.get("set", "short"), "expected_idx": exp, "jev": names[jt], "jev_idx": jt,
                        "confidence": round(conf, 4), "final": names[final], "final_idx": final,
                        "reason": reason, "probs": probs, "latency_ms": round(ms, 1),
                        "shuffle_flips": flips})

    # ---------------- summary
    def summarize(key):
        n = len(results)
        exact = sum(x[key] == x["expected_idx"] for x in results)
        within = sum(abs(x[key] - x["expected_idx"]) <= 1 for x in results)
        under = sum(x[key] < x["expected_idx"] for x in results)
        over = sum(x[key] > x["expected_idx"] for x in results)
        return f"exact {exact}/{n} ({exact / n:.0%}) · ±1 {within}/{n} · under-routed {under} · over-routed {over}"

    print("\n=== 요약 ===")
    print(f"JEV 분류 품질 (기대값 대비)  : {summarize('jev_idx')}")
    print(f"최종 라우팅 (룰+비용정책 적용): {summarize('final_idx')}")
    print("  under-routed = 기대보다 낮은 모델로 감 (품질 위험, 비용정책에 의한 의도적 하향 포함)"
          " / over-routed = 더 높은 모델로 감 (비용)")
    for st in sorted({x["set"] for x in results}):
        sub = [x for x in results if x["set"] == st]
        ex = sum(x["jev_idx"] == x["expected_idx"] for x in sub)
        mc = sum(x["confidence"] for x in sub) / len(sub)
        print(f"  [{st}] JEV exact {ex}/{len(sub)} · 평균 확신도 {mc:.2f}")
    for lang in sorted({x["lang"] for x in results}):
        sub = [x for x in results if x["lang"] == lang]
        ex = sum(x["jev_idx"] == x["expected_idx"] for x in sub)
        mc = sum(x["confidence"] for x in sub) / len(sub)
        print(f"  [{lang}] JEV exact {ex}/{len(sub)} · 평균 확신도 {mc:.2f}")
    lat = sorted(x["latency_ms"] for x in results)
    print(f"  분류 지연: median {lat[len(lat) // 2]:.0f} ms, max {lat[-1]:.0f} ms")
    if a.shuffles:
        f = sum(x["shuffle_flips"] for x in results)
        print(f"  선택지 순서 변경 시 결과가 바뀐 비율: {f}/{len(results) * a.shuffles} "
              f"({f / (len(results) * a.shuffles):.0%})")

    print("\n혼동 행렬 (행=기대, 열=최종)")
    cm = defaultdict(Counter)
    for x in results:
        cm[x["expected_idx"]][x["final_idx"]] += 1
    print(" " * 13 + "".join(f"{n[:10]:>11}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:12]:13}" + "".join(f"{cm[i][j]:>11}" for j in range(len(names))))

    dist = Counter(x["final"] for x in results)
    internal = sum(dist[t.name] for t in cfg.tiers if not t.backend.startswith("anthropic"))
    print(f"\n최종 분포: { {n: dist[n] for n in names} }  → 내부 모델 처리 비율 {internal}/{len(results)}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-mock" if a.mock else "") + f"-{a.set}-{a.budget_level}"
    (out / f"eval-{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    with open(out / f"eval-{stamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "set", "lang", "expected", "jev", "confidence", "final", "reason", "shuffle_flips",
                    "latency_ms", "prompt"] + [f"p_{n}" for n in names])
        for x in results:
            w.writerow([x["id"], x["set"], x["lang"], x["expected"], x["jev"], x["confidence"], x["final"],
                        x["reason"], x["shuffle_flips"], x["latency_ms"], x["prompt"]]
                       + [x["probs"].get(n) for n in names])
    print(f"\n결과 저장: {out}/eval-{stamp}.csv, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
