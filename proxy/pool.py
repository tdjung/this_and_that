"""Backend server pools: several servers per backend, load-aware selection with session affinity.

* Session affinity (weighted rendezvous hashing) keeps a session on the same server, so the
  vLLM prefix cache on that server keeps hitting across the tool loop.
* Load comes from each server's vLLM Prometheus `/metrics` (num_requests_running/waiting),
  polled while the proxy is active. Every developer runs a local proxy, so a local in-flight
  count alone cannot see other people's load; the server metric can.
* A server is "busy" when its waiting queue exceeds `max_waiting` (or it just answered 429/503),
  and "down" for a short while after a connection error.
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
                               weight=float(s.get("weight", 1.0))) for s in servers]
        self.auth = cfg.get("auth", {"mode": "passthrough"})
        self.extra_headers = cfg.get("extra_headers", {}) or {}
        self.external = bool(cfg.get("external", name.startswith("anthropic")))
        self.metrics = bool(cfg.get("metrics", False))
        self.max_waiting = float(cfg.get("max_waiting", 4))
        self.metrics_ttl = float(cfg.get("metrics_ttl_s", 15))

    def busy(self, s: Server, now: float | None = None) -> bool:
        now = now or time.time()
        if s.busy_until > now:
            return True
        return self.metrics and now - s.metrics_ts < self.metrics_ttl and s.waiting > self.max_waiting

    def order(self, key: str) -> tuple[list[Server], list[Server]]:
        """(free servers, preferred first then least loaded; busy servers, least loaded first)."""
        now = time.time()
        up = [s for s in self.servers if s.down_until <= now]
        if not up:
            return [], []
        pref = max(up, key=lambda s: _score(key, s.url, s.weight))
        free = [s for s in up if not self.busy(s, now)]
        busy = sorted((s for s in up if self.busy(s, now)), key=Server.load)
        rest = sorted((s for s in free if s is not pref), key=Server.load)
        return ([pref] + rest if pref in free else rest), busy

    def mark_down(self, s: Server, secs: float = 30):
        s.down_until = time.time() + secs

    def mark_busy(self, s: Server, secs: float = 15):
        s.busy_until = time.time() + secs

    def status(self) -> list[dict]:
        now = time.time()
        return [{"url": s.url, "running": s.running, "waiting": s.waiting,
                 "metrics_age_s": round(now - s.metrics_ts) if s.metrics_ts else None,
                 "busy": self.busy(s, now), "down": s.down_until > now, "inflight": s.inflight}
                for s in self.servers]
