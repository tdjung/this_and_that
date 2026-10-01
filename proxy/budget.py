"""Budget awareness for the external (Anthropic) tiers.

The level is the worst of three signals:

  * source  : an external budget feed polled every `interval_s` — an HTTP URL or a shell command
              that prints JSON. Either {"level": "normal|tight|critical"} or
              {"remaining": 1234, "limit": 10000[, "period_start": ISO, "period_end": ISO]}.
              With remaining/limit the level comes from *pacing*: compare the remaining share
              with the share of the budget period that is still left.
  * headers : `anthropic-ratelimit-unified-*` response headers from the upstream (heuristic:
              a status of "rejected" → critical, a "warning" status or high utilization → tight).
  * manual  : POST /budget {"level": "tight", "minutes": 120}

Policy maps (level → {tier: replacement tier}) say where requests go under pressure.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import time

import httpx

log = logging.getLogger("claude_auto.budget")
LEVELS = ["normal", "tight", "critical"]


def _rank(level: str | None) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


def period_bounds(now: dt.datetime, reset_day: int) -> tuple[dt.datetime, dt.datetime]:
    def at(y, m):
        d = min(reset_day, 28)
        return now.replace(year=y, month=m, day=d, hour=0, minute=0, second=0, microsecond=0)
    y, m = now.year, now.month
    start = at(y, m)
    if now < start:
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        start = at(y, m)
    ny, nm = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    return start, at(ny, nm)


class Budget:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg)
        self.src = cfg.get("source") or {}
        self.interval = float(self.src.get("interval_s", 300))
        self.reset_day = int(cfg.get("reset_day", 1))
        self.tight_pace = float(cfg.get("tight_when_below_pace", 0.8))
        self.critical_ratio = float(cfg.get("critical_when_remaining_below", 0.05))
        self.use_headers = bool(cfg.get("use_ratelimit_headers", True))
        self.util_tight = float(cfg.get("header_utilization_tight", 0.8))
        self.header_ttl = float(cfg.get("header_ttl_minutes", 15)) * 60
        self.policy = cfg.get("policy", {}) or {}
        self.source_level, self.source_detail, self.source_ts = None, None, 0.0
        self.header_level, self.header_ts = None, 0.0
        self.manual_level, self.manual_until = None, 0.0

    # ------------------------------------------------------------ level
    @property
    def level(self) -> str:
        now = time.time()
        cands = [self.source_level if now - self.source_ts < max(3 * self.interval, 900) else None,
                 self.header_level if now - self.header_ts < self.header_ttl else None,
                 self.manual_level if now < self.manual_until else None]
        return max(cands, key=_rank) or "normal"

    def level_from_payload(self, d: dict, now: dt.datetime | None = None) -> tuple[str, dict]:
        if d.get("level") in LEVELS:
            return d["level"], d
        rem, lim = d.get("remaining"), d.get("limit")
        if rem is None or not lim:
            raise ValueError(f"budget payload needs 'level' or 'remaining'+'limit': {d}")
        now = now or dt.datetime.now().astimezone()
        if d.get("period_start") and d.get("period_end"):
            start = dt.datetime.fromisoformat(d["period_start"])
            end = dt.datetime.fromisoformat(d["period_end"])
            if start.tzinfo is None:
                start, end = start.astimezone(), end.astimezone()
        else:
            start, end = period_bounds(now, self.reset_day)
        span = max((end - start).total_seconds(), 1)
        time_left = min(max((end - now).total_seconds() / span, 0.0), 1.0)
        ratio = max(float(rem), 0.0) / float(lim)
        if ratio < self.critical_ratio:
            lvl = "critical"
        elif ratio < time_left * self.tight_pace:
            lvl = "tight"
        else:
            lvl = "normal"
        return lvl, {"remaining_ratio": round(ratio, 3), "period_left_ratio": round(time_left, 3)}

    def update_from_headers(self, headers) -> None:
        if not self.use_headers:
            return
        lvl = None
        for k, v in headers.items():
            k, v = k.lower(), str(v).lower()
            if not k.startswith("anthropic-ratelimit-unified"):
                continue
            if k.endswith("status"):
                if "rejected" in v:
                    lvl = max([lvl, "critical"], key=_rank)
                elif "warning" in v:
                    lvl = max([lvl, "tight"], key=_rank)
            elif "utilization" in k:
                try:
                    u = float(v)
                    u = u / 100 if u > 1.0 else u
                    if u >= self.util_tight:
                        lvl = max([lvl, "tight"], key=_rank)
                except ValueError:
                    pass
        if lvl or self.header_level:
            self.header_level, self.header_ts = (lvl or "normal"), time.time()

    def set_manual(self, level: str | None, minutes: float = 120) -> None:
        if level is None:
            self.manual_level, self.manual_until = None, 0.0
        else:
            if level not in LEVELS:
                raise ValueError(level)
            self.manual_level, self.manual_until = level, time.time() + minutes * 60

    # ------------------------------------------------------------ polling
    async def fetch(self, client: httpx.AsyncClient) -> dict:
        t = self.src.get("type")
        if t == "http":
            r = await client.get(self.src["url"], headers=self.src.get("headers") or {}, timeout=10)
            r.raise_for_status()
            return r.json()
        if t == "command":
            p = await asyncio.create_subprocess_shell(self.src["command"], stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(p.communicate(), timeout=30)
            if p.returncode != 0:
                raise RuntimeError(err.decode()[:300])
            return json.loads(out.decode())
        raise ValueError(f"unknown budget source type: {t}")

    async def poll_once(self, client: httpx.AsyncClient) -> None:
        try:
            d = await self.fetch(client)
            self.source_level, self.source_detail = self.level_from_payload(d)
            self.source_ts = time.time()
        except Exception as e:  # noqa: BLE001
            log.warning("budget source failed: %s", e)

    async def run(self, client: httpx.AsyncClient) -> None:
        if not self.src.get("type"):
            return
        while True:
            await self.poll_once(client)
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------ policy
    def replacement(self, tier_name: str) -> str | None:
        return (self.policy.get(self.level) or {}).get(tier_name)

    def status(self) -> dict:
        now = time.time()
        return {"level": self.level,
                "source": {"level": self.source_level, "detail": self.source_detail,
                           "age_s": round(now - self.source_ts) if self.source_ts else None},
                "headers": {"level": self.header_level,
                            "age_s": round(now - self.header_ts) if self.header_ts else None},
                "manual": {"level": self.manual_level,
                           "remaining_min": round((self.manual_until - now) / 60) if self.manual_until > now else 0}}
