"""Client for autotrust/JEV-27B System 1 (typed decisions) served by vLLM.

Follows the model card: one prefill pass on the `jev-decision` LoRA module with
`max_tokens=1`, constrained to the option tokens, then decision-head bias and
per-kind temperature are applied client-side.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import httpx

LETTERS = "ABCDEFGHIJKLMNOP"
KINDS = ("noul", "choice", "score")


class JevError(RuntimeError):
    pass


@dataclass
class JevHead:
    ranges: dict          # kind -> [start, end) slot range
    verbalizer_ids: list  # token ids per slot
    bias: list            # head bias per slot
    temperature: dict     # kind -> temperature

    @classmethod
    def load(cls, model_dir: str | Path) -> "JevHead":
        d = Path(model_dir).expanduser()
        dh = json.loads((d / "adapter_vllm" / "decision_head.json").read_text())
        cal = json.loads((d / "calibration.json").read_text())
        return cls(dh["slots"]["ranges"], dh["verbalizer_ids"], dh["bias"], cal["per_kind"])


def _normalize_options(kind: str, options):
    if kind == "noul":
        return ["false", "true"]
    if kind == "score":
        return [str(i) for i in range(6)]
    if kind != "choice":
        raise ValueError(f"unknown kind: {kind}")
    if not options or not 2 <= len(options) <= 16:
        raise ValueError("choice needs 2-16 options")
    return list(options)


class JevClient:
    def __init__(self, base_url: str, model_dir: str | Path, adapter: str = "jev-decision",
                 timeout: float = 5.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.head = JevHead.load(model_dir)
        self.adapter = adapter
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # --- request / response building (shared by sync and async paths) ---
    def build(self, kind: str, state: str, question: str, options=None):
        options = _normalize_options(kind, options)
        lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
        prompt = (f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
                  + "\n".join(lines) + "\n[decision]:")
        s = self.head.ranges[kind][0]
        ids = self.head.verbalizer_ids[s: s + len(options)]
        payload = {
            "model": self.adapter, "prompt": prompt, "max_tokens": 1, "temperature": 1.0,
            "logprobs": len(options), "allowed_token_ids": ids,
            "add_special_tokens": False, "return_tokens_as_token_ids": True,
        }
        return options, s, ids, payload

    def parse(self, kind: str, options, s: int, ids, data: dict) -> dict:
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise JevError(f"unexpected response: {str(data)[:300]}") from e
        lp = {}
        for k, v in top.items():
            try:
                lp[int(str(k).rsplit(":", 1)[-1])] = v
            except ValueError:
                continue
        t = self.head.temperature[kind]
        z = [(lp.get(tid, -1e9) + self.head.bias[s + i]) / t for i, tid in enumerate(ids)]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        tot = sum(e)
        return {o: x / tot for o, x in zip(options, e)}

    # --- sync / async calls ---
    def decide(self, kind: str, state: str, question: str, options=None) -> dict:
        options, s, ids, payload = self.build(kind, state, question, options)
        try:
            r = httpx.post(f"{self.base_url}/v1/completions", json=payload,
                           headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise JevError(str(e)) from e
        return self.parse(kind, options, s, ids, r.json())

    async def adecide(self, client: httpx.AsyncClient, kind: str, state: str, question: str,
                      options=None) -> dict:
        options, s, ids, payload = self.build(kind, state, question, options)
        try:
            r = await client.post(f"{self.base_url}/v1/completions", json=payload,
                                  headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise JevError(str(e)) from e
        return self.parse(kind, options, s, ids, r.json())
