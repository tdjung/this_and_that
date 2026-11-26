"""Backend server pools: several servers per backend, load-aware selection with session affinity.

* Session affinity (weighted rendezvous hashing) keeps a session on the same server, so the
  vLLM prefix cache on that server keeps hitting across the tool loop.
* Load comes from each server's vLLM Prometheus `/metrics` (num_requests_running/waiting),
  polled while the proxy is active. Every developer runs a local proxy, so a local in-flight
  count alone cannot see other people's load; the server metric can.
* A server is "busy" when its waiting queue exceeds `max_waiting` (or it just answered 429/503,
  timed out mid-request, or its health probe was slower than `slow_ms`). Busy servers are still
  used as a last resort.
* A server is "down" (never used) when it is dead or is not the expected service:
  - connection refused / connect timeout (request or health probe),
  - its health probe (`health_path`, default /v1/models) failed `down_after_fails` times in a row,
  - it answers but serves a different model (e.g. the port was reused for another server),
  - it answered 404 to /v1/messages (wrong service on that port).
  A down server comes back only after a successful health probe, not after a fixed timer.
* A server can be taken out by hand (`enabled: false` in config, or POST /backends/drain).
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

METRIC_RE = re.compile(r"^vllm[:_]num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)


@dataclass
class Server:
    url: str
    model: str | None = None      # per-server model name override (e.g. JEV-27B serving as 27b)
    weight: float = 1.0
    running: float = 0.0
    waiting: float = 0.0
    metrics_ts: float = 0.0
    down_until: float = 0.0
    busy_until: float = 0.0
    inflight: int = 0             # requests from *this* proxy only
    drained_until: float = 0.0    # taken out by hand (config enabled: false or POST /backends/drain)
    health_ok: bool | None = None  # None: not probed yet
    health_fails: int = 0         # consecutive failed probes (dead / wrong model)
    slow_fails: int = 0           # consecutive probes that timed out after connecting (alive but stuck)
    health_ts: float = 0.0
    health_ms: float | None = None
    health_reason: str = ""

    def load(self) -> float:
        return (self.running + 2 * self.waiting + self.inflight) / max(self.weight, 1e-6)


def parse_metrics(text: str) -> tuple[float, float]:
    vals = {"running": 0.0, "waiting": 0.0}
    for kind, v in METRIC_RE.findall(text):
        try:
            vals[kind] += float(v)
        except ValueError:
            pass
    return vals["running"], vals["waiting"]


def _score(key: str, url: str, weight: float) -> float:
    h = int.from_bytes(hashlib.md5(f"{key}|{url}".encode()).digest()[:8], "big")
    u = (h + 1) / (2 ** 64 + 1)                  # (0, 1)
    return -weight / math.log(u)                 # weighted rendezvous hashing


class Backend:
    def __init__(self, name: str, cfg: dict):
        self.name = name
        servers = cfg.get("servers") or [{"url": cfg["base_url"]}]
        self.servers = [Server(url=s["url"].rstrip("/"), model=s.get("model"),
                               weight=float(s.get("weight", 1.0)),
                               drained_until=0.0 if s.get("enabled", True) else math.inf)
                        for s in servers]
        self.auth = cfg.get("auth", {"mode": "passthrough"})
        self.extra_headers = cfg.get("extra_headers", {}) or {}
        self.external = bool(cfg.get("external", name.startswith("anthropic")))
        # use http(s)_proxy env vars for this backend? default: only for external backends
        self.use_env_proxy = bool(cfg.get("use_env_proxy", self.external))
        self.metrics = bool(cfg.get("metrics", False))
        self.max_waiting = float(cfg.get("max_waiting", 4))
        self.metrics_ttl = float(cfg.get("metrics_ttl_s", 15))
        # health probe: dead vs slow vs wrong service (internal backends only by default)
        self.health = bool(cfg.get("health_check", not self.external))
        self.health_path = cfg.get("health_path", "/v1/models")
        self.down_after = int(cfg.get("down_after_fails", 2))       # failed probes before "down"
        self.stuck_after = int(cfg.get("down_after_timeouts", 4))   # timed-out probes before "down"
        self.slow_ms = float(cfg.get("slow_ms", 3000))              # probe slower than this -> busy
        self.dead_hold = float(cfg.get("down_hold_s", 600))         # down until a probe succeeds (cap)
        self.expected_models: set[str] = set()                      # filled by the router from tiers

    def busy(self, s: Server, now: float | None = None) -> bool:
        now = now or time.time()
        if s.busy_until > now:
            return True
        return self.metrics and now - s.metrics_ts < self.metrics_ttl and s.waiting > self.max_waiting

    def available(self, s: Server, now: float | None = None) -> bool:
        now = now or time.time()
        return s.down_until <= now and s.drained_until <= now

    def order(self, key: str) -> tuple[list[Server], list[Server]]:
        """(free servers, preferred first then least loaded; busy servers, least loaded first)."""
        now = time.time()
        up = [s for s in self.servers if self.available(s, now)]
        if not up:
            return [], []
        pref = max(up, key=lambda s: _score(key, s.url, s.weight))
        free = [s for s in up if not self.busy(s, now)]
        busy = sorted((s for s in up if self.busy(s, now)), key=Server.load)
        rest = sorted((s for s in free if s is not pref), key=Server.load)
        return ([pref] + rest if pref in free else rest), busy

    def mark_down(self, s: Server, secs: float | None = None, reason: str = ""):
        """Dead or wrong service. Held until a health probe succeeds (or `secs` passes)."""
        hold = secs if secs is not None else (self.dead_hold if self.health else 30)
        s.down_until = time.time() + hold
        s.health_ok = False
        if reason:
            s.health_reason = reason

    def mark_busy(self, s: Server, secs: float = 15):
        s.busy_until = time.time() + secs

    def record_health(self, s: Server, outcome: str, reason: str = "", ms: float | None = None):
        """outcome: "ok" | "slow" (connected, then timed out) | "dead" (refused / wrong model / bad status)."""
        now = time.time()
        s.health_ts, s.health_ms, s.health_reason = now, ms, reason
        if outcome == "ok":
            s.health_ok, s.health_fails, s.slow_fails = True, 0, 0
            s.down_until = 0.0
            if ms is not None and ms > self.slow_ms:
                self.mark_busy(s, self.metrics_ttl)
            return
        if outcome == "slow":
            s.slow_fails += 1
            self.mark_busy(s, self.metrics_ttl)
            if s.slow_fails >= self.stuck_after:
                self.mark_down(s, reason=f"stuck: {reason}")
            return
        s.health_fails += 1
        if s.health_fails >= self.down_after:
            self.mark_down(s, reason=reason)

    def drain(self, s: Server, minutes: float):
        s.drained_until = time.time() + minutes * 60 if minutes > 0 else 0.0

    def status(self) -> list[dict]:
        now = time.time()
        out = []
        for s in self.servers:
            state = ("drained" if s.drained_until > now else "down" if s.down_until > now
                     else "busy" if self.busy(s, now) else "ok")
            out.append({"url": s.url, "state": state, "running": s.running, "waiting": s.waiting,
                        "metrics_age_s": round(now - s.metrics_ts) if s.metrics_ts else None,
                        "busy": self.busy(s, now), "down": s.down_until > now, "inflight": s.inflight,
                        "drained_min": (None if s.drained_until <= now else
                                        "forever" if s.drained_until == math.inf
                                        else round((s.drained_until - now) / 60, 1)),
                        "health": {"ok": s.health_ok, "ms": s.health_ms, "reason": s.health_reason,
                                   "age_s": round(now - s.health_ts) if s.health_ts else None}})
        return out
