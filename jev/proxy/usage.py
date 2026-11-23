"""Local spend meter: counts the tokens of every external (Anthropic) response that passes through
the proxy and prices them, so the budget level can follow your own monthly allowance
(e.g. USD 180) without any company API.

Only claude_auto traffic is seen. Usage from plain `claude` sessions or other machines is not,
so sync it from your company dashboard with POST /usage {"set_spent_usd": 92.5} when needed.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path

from proxy.budget import period_bounds

log = logging.getLogger("claude_auto.usage")
PRICE_KEYS = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")


def cost_usd(usage: dict, price: dict) -> float:
    """usage: Anthropic usage fields; price: USD per 1M tokens."""
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens", 0) or 0
    w5m = cc.get("ephemeral_5m_input_tokens")
    if w5m is None:
        w5m = max((usage.get("cache_creation_input_tokens") or 0) - w1h, 0)
    return (
        (usage.get("input_tokens") or 0) * price.get("input", 0)
        + (usage.get("output_tokens") or 0) * price.get("output", 0)
        + w5m * price.get("cache_write_5m", price.get("input", 0) * 1.25)
        + w1h * price.get("cache_write_1h", price.get("input", 0) * 2)
        + (usage.get("cache_read_input_tokens") or 0) * price.get("cache_read", price.get("input", 0) * 0.1)
    ) / 1e6


class SSEUsageTap:
    """Reads `usage` out of an Anthropic SSE stream as it is relayed, without buffering it."""

    def __init__(self):
        self.buf = b""
        self.usage: dict = {}

    def _event(self, data: dict):
        t = data.get("type")
        if t == "message_start":
            self.usage.update((data.get("message") or {}).get("usage") or {})
        elif t == "message_delta":
            for k, v in (data.get("usage") or {}).items():
                if v is not None:
                    self.usage[k] = v          # message_delta carries cumulative counts

    def feed(self, chunk: bytes):
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.startswith(b"data:") and b'"usage"' in line:
                try:
                    self._event(json.loads(line[5:].strip()))
                except ValueError:
                    pass

    def feed_json(self, body: bytes):
        try:
            d = json.loads(body)
            self.usage.update(d.get("usage") or {})
        except ValueError:
            pass


class UsageMeter:
    def __init__(self, path: str | Path, reset_day: int = 1, monthly_limit_usd: float | None = None):
        self.path = Path(os.path.expanduser(str(path)))
        self.reset_day = reset_day
        self.limit = monthly_limit_usd
        self.lock = threading.Lock()
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def period_key(self, now: dt.datetime | None = None) -> str:
        start, _ = period_bounds(now or dt.datetime.now().astimezone(), self.reset_day)
        return start.date().isoformat()

    def _period(self) -> dict:
        return self.data.setdefault(self.period_key(), {"spent_usd": 0.0, "adjust_usd": 0.0, "by_model": {}})

    def record(self, model: str, usage: dict, price: dict | None) -> float:
        if not usage or not price:
            return 0.0
        c = cost_usd(usage, price)
        with self.lock:
            p = self._period()
            p["spent_usd"] += c
            m = p["by_model"].setdefault(model, {"requests": 0, "usd": 0.0, "input": 0, "output": 0,
                                                "cache_write": 0, "cache_read": 0})
            m["requests"] += 1
            m["usd"] += c
            m["input"] += usage.get("input_tokens") or 0
            m["output"] += usage.get("output_tokens") or 0
            m["cache_write"] += usage.get("cache_creation_input_tokens") or 0
            m["cache_read"] += usage.get("cache_read_input_tokens") or 0
            self._save()
        return c

    def set_spent(self, usd: float):
        """Sync with an external figure: the period total becomes `usd`."""
        with self.lock:
            p = self._period()
            p["adjust_usd"] = usd - p["spent_usd"]
            self._save()

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            tmp.replace(self.path)
        except OSError as e:
            log.warning("usage save failed: %s", e)

    def spent(self) -> float:
        p = self.data.get(self.period_key(), {})
        return p.get("spent_usd", 0.0) + p.get("adjust_usd", 0.0)

    def payload(self) -> dict | None:
        """Budget payload for Budget.level_from_payload (None without a limit)."""
        if not self.limit:
            return None
        start, end = period_bounds(dt.datetime.now().astimezone(), self.reset_day)
        return {"remaining": self.limit - self.spent(), "limit": self.limit,
                "period_start": start.isoformat(), "period_end": end.isoformat()}

    def status(self) -> dict:
        now = dt.datetime.now().astimezone()
        start, end = period_bounds(now, self.reset_day)
        elapsed = max((now - start).total_seconds() / max((end - start).total_seconds(), 1), 1e-6)
        spent = self.spent()
        p = self.data.get(self.period_key(), {})
        return {"period": f"{start.date()} ~ {end.date()}", "limit_usd": self.limit,
                "spent_usd": round(spent, 2),
                "projected_month_end_usd": round(spent / elapsed, 2) if elapsed > 0.03 else None,
                "pace_ok": (spent / elapsed <= self.limit) if (self.limit and elapsed > 0.03) else None,
                "by_model": {k: {**v, "usd": round(v["usd"], 2)} for k, v in p.get("by_model", {}).items()},
                "adjust_usd": round(p.get("adjust_usd", 0.0), 2)}
