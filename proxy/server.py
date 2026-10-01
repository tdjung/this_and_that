#!/usr/bin/env python3
"""claude_auto local routing proxy.

Claude Code → (ANTHROPIC_BASE_URL) → this proxy → internal Qwen backends / Anthropic.

For each /v1/messages request the proxy decides a tier:
  * forced model alias (claude-tier-<name>)             → that tier
  * request class / agent type policy (config)          → fixed tier or session tier
  * first request of a new user prompt                  → rules + JEV classification
  * tool-loop continuation of the same prompt           → the session's pinned tier
then applies session policy (upgrade_only), tool-error escalation, context-size fit and
quota-aware fallback, rewrites only `model`, and streams the upstream response back unbuffered.

  python proxy/server.py --config ~/.claude_auto/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevtools.client import JevClient  # noqa: E402
from jevtools.routing import (Decision, RoutingConfig, classify_async, is_tool_result_turn,  # noqa: E402
                              tool_errors, user_prompts)

log = logging.getLogger("claude_auto")

AUTO_ALIASES = {"claude-auto", "auto"}
FORCE_PREFIX = "claude-tier-"
DROP_REQ_HEADERS = {"host", "content-length", "connection", "accept-encoding", "authorization",
                    "x-api-key", "keep-alive", "transfer-encoding", "te", "upgrade"}
DROP_RESP_HEADERS = {"content-length", "transfer-encoding", "connection", "keep-alive",
                     "content-encoding"}


@dataclass
class Session:
    tier: int | None = None
    last_prompt_id: str | None = None
    tool_errors: int = 0
    updated: float = field(default_factory=time.time)


class Router:
    def __init__(self, raw: dict, transport: httpx.AsyncBaseTransport | None = None, jev=None):
        self.raw = raw
        self.cfg = RoutingConfig.from_dict(raw)
        self.mode = raw.get("mode", "route")
        self.shadow_tier = self.cfg.index(raw["shadow_tier"]) if raw.get("shadow_tier") else None
        self.backends = raw["backends"]
        self.rc_policy = raw.get("request_classes", {})
        self.agent_types = raw.get("agent_types", {})
        s = raw.get("session", {})
        self.policy = s.get("policy", "upgrade_only")
        self.reset_on_compaction = s.get("reset_on_compaction", True)
        self.ttl = float(s.get("ttl_minutes", 720)) * 60
        fb = raw.get("fallback", {})
        self.err_threshold = int(fb.get("escalate_on_tool_errors", 3))
        self.escalate_on_backend_error = fb.get("escalate_on_backend_error", True)
        self.downgrade_on_quota = fb.get("downgrade_on_quota", True)
        self.quota_retry_after = float(fb.get("quota_min_retry_after_s", 60))
        self.quota_cooldown = float(fb.get("quota_cooldown_minutes", 30)) * 60
        self.blocked: dict[str, float] = {}          # backend -> blocked until (epoch)
        self.sessions: dict[str, Session] = {}
        self.log_path = Path(os.path.expanduser(raw.get("log_path", "~/.claude_auto/decisions.jsonl")))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(600, connect=10))
        if jev is not None:
            self.jev = jev
        else:
            j = raw.get("jev", {})
            key = os.environ.get(j["api_key_env"]) if j.get("api_key_env") else None
            self.jev = JevClient(j["base_url"], j["model_dir"], adapter=j.get("adapter", "jev-decision"),
                                 timeout=float(j.get("timeout_s", 3.0)), api_key=key)

    # ------------------------------------------------------------ helpers
    @property
    def names(self):
        return [t.name for t in self.cfg.tiers]

    def session(self, key: str) -> Session:
        now = time.time()
        if len(self.sessions) > 2000:
            self.sessions = {k: v for k, v in self.sessions.items() if now - v.updated < self.ttl}
        s = self.sessions.get(key)
        if s is None or now - s.updated > self.ttl:
            s = self.sessions[key] = Session()
        s.updated = now
        return s

    def is_blocked(self, backend: str) -> bool:
        until = self.blocked.get(backend)
        return until is not None and until > time.time()

    def write_log(self, rec: dict):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("log write failed: %s", e)

    def fit_context(self, idx: int, est_tokens: int) -> tuple[int, str | None]:
        start = idx
        while idx < self.cfg.top and self.cfg.tiers[idx].max_context * 0.9 < est_tokens:
            idx += 1
        return idx, (f"context~{est_tokens}tok" if idx != start else None)

    def avoid_blocked(self, idx: int) -> tuple[int, str | None]:
        if not self.is_blocked(self.cfg.tiers[idx].backend):
            return idx, None
        for j in range(idx - 1, -1, -1):
            if not self.is_blocked(self.cfg.tiers[j].backend):
                return j, f"quota_blocked:{self.cfg.tiers[idx].backend}"
        return idx, None

    def policy_for(self, h) -> str:
        rc = h.get("x-claude-code-request-class", "main")
        at = h.get("x-claude-code-agent-type")
        if at and at in self.agent_types:
            return self.agent_types[at]
        return self.rc_policy.get(rc, "route")

    # ------------------------------------------------------------ decision
    async def decide(self, body: dict, h) -> tuple[int, dict]:
        model = str(body.get("model", ""))
        rc = h.get("x-claude-code-request-class", "main")
        sid = h.get("x-claude-code-session-id") or (body.get("metadata") or {}).get("user_id") or "anon"
        agent = h.get("x-claude-code-agent-id")
        key = f"{sid}:{agent}" if agent else sid
        info = {"session": sid[-12:], "agent": agent, "rc": rc, "model_in": model}
        messages = body.get("messages") or []
        est = len(json.dumps(body, ensure_ascii=False)) // 3

        # 1. forced alias from /model
        if model.startswith(FORCE_PREFIX) and model[len(FORCE_PREFIX):] in self.names:
            idx = self.names.index(model[len(FORCE_PREFIX):])
            info.update(kind="forced", reason=f"alias:{model}")
            return self.finalize(idx, est, info, None)

        # 2. request class / agent type policy
        pol = self.policy_for(h)
        sess = self.session(key)
        if h.get("x-claude-code-context-compacted") and self.reset_on_compaction:
            sess.tier = None
        if pol in self.names:
            info.update(kind="policy", reason=f"{rc}->{pol}")
            return self.finalize(self.names.index(pol), est, info, None)
        if pol == "session":
            idx = sess.tier if sess.tier is not None else 0
            info.update(kind="session", reason=f"{rc}->session")
            return self.finalize(idx, est, info, None)

        # 3. route: new prompt or continuation?
        pid = h.get("x-claude-code-prompt-id")
        new_prompt = (pid != sess.last_prompt_id) if pid else not is_tool_result_turn(messages)
        n_res, n_err = tool_errors(messages)
        if n_res:
            sess.tool_errors = sess.tool_errors + n_err if n_err else 0

        if new_prompt or sess.tier is None:
            d: Decision = await classify_async(self.jev, self.client, self.cfg, user_prompts(messages))
            idx = d.tier
            info.update(kind="classified", reason=d.reason, jev=self.names[d.jev_tier] if d.jev_tier is not None else None,
                        confidence=d.confidence, probs=d.probs)
            if self.policy == "upgrade_only" and sess.tier is not None and sess.tier > idx:
                info["reason"] += f" | kept {self.names[sess.tier]} (upgrade_only)"
                idx = sess.tier
            sess.last_prompt_id = pid or sess.last_prompt_id
        else:
            idx = sess.tier
            info.update(kind="continuation", reason="pinned")

        if self.err_threshold and sess.tool_errors >= self.err_threshold and idx < self.cfg.top:
            idx += 1
            info["reason"] += f" | tool_errors={sess.tool_errors}->+1"
            sess.tool_errors = 0

        return self.finalize(idx, est, info, sess)

    def finalize(self, idx: int, est: int, info: dict, sess: Session | None) -> tuple[int, dict]:
        idx, why = self.fit_context(idx, est)
        if why:
            info["reason"] += f" | {why}"
        if sess is not None:
            sess.tier = idx                      # pin before quota adjustments (temporary)
        if self.mode == "shadow" and self.shadow_tier is not None and info.get("kind") in ("classified", "continuation"):
            info["would_route"] = self.names[idx]
            idx = self.shadow_tier
            info["reason"] += " | shadow"
        idx, why = self.avoid_blocked(idx)
        if why:
            info["reason"] += f" | {why}"
        info["est_tokens"] = est
        return idx, info

    # ------------------------------------------------------------ forwarding
    def upstream_headers(self, req_headers, backend: dict) -> dict:
        h = {k: v for k, v in req_headers.items() if k.lower() not in DROP_REQ_HEADERS}
        h["accept-encoding"] = "identity"
        auth = backend.get("auth", {"mode": "passthrough"})
        mode = auth.get("mode", "passthrough")
        if mode == "passthrough":
            for k in ("authorization", "x-api-key"):
                if k in req_headers:
                    h[k] = req_headers[k]
        elif mode in ("bearer", "api_key"):
            val = os.environ.get(auth.get("env", ""), "").strip()
            if not val:
                log.warning("env %s is empty; sending no credential to this backend", auth.get("env"))
            elif mode == "bearer":
                h["authorization"] = f"Bearer {val}"
            else:
                h["x-api-key"] = val
        h.update(backend.get("extra_headers", {}) or {})
        return h

    def is_quota(self, resp: httpx.Response) -> bool:
        if resp.status_code != 429:
            return False
        if any(k.lower().startswith("anthropic-ratelimit-unified") and "rejected" in v.lower()
               for k, v in resp.headers.items()):
            return True
        ra = resp.headers.get("retry-after")
        try:
            return ra is None or float(ra) >= self.quota_retry_after
        except ValueError:
            return True

    async def forward(self, request: Request, path: str, body: dict, idx: int, info: dict,
                      attempt: int = 0) -> Response:
        tier = self.cfg.tiers[idx]
        backend = self.backends[tier.backend]
        payload = dict(body)
        payload["model"] = tier.model
        url = backend["base_url"].rstrip("/") + path
        if request.url.query:
            url += "?" + request.url.query
        headers = self.upstream_headers(request.headers, backend)
        req = self.client.build_request("POST", url, headers=headers,
                                        content=json.dumps(payload, ensure_ascii=False).encode())
        t0 = time.time()
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.TransportError as e:
            if attempt == 0 and self.escalate_on_backend_error and idx < self.cfg.top:
                info["reason"] += f" | {tier.backend} unreachable({type(e).__name__})->+1"
                return await self.forward(request, path, body, idx + 1, info, 1)
            self.log_request(info, idx, 502, t0)
            return JSONResponse({"type": "error", "error": {"type": "api_error",
                                 "message": f"claude_auto: backend {tier.backend} unreachable: {e}"}},
                                status_code=502)

        # retry decisions are only possible before any byte is relayed
        if attempt == 0 and resp.status_code >= 500 and self.escalate_on_backend_error \
                and not tier.backend.startswith("anthropic") and idx < self.cfg.top:
            await resp.aclose()
            info["reason"] += f" | {tier.backend} {resp.status_code}->+1"
            return await self.forward(request, path, body, idx + 1, info, 1)
        if attempt == 0 and self.downgrade_on_quota and self.is_quota(resp):
            self.blocked[tier.backend] = time.time() + self.quota_cooldown
            lower, _ = self.avoid_blocked(idx)
            if lower != idx:
                await resp.aclose()
                info["reason"] += f" | {tier.backend} quota(429)->{self.names[lower]}"
                return await self.forward(request, path, body, lower, info, 1)

        self.log_request(info, idx, resp.status_code, t0)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in DROP_RESP_HEADERS}
        out_headers["x-claude-auto-tier"] = tier.name
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=out_headers,
                                 background=BackgroundTask(resp.aclose))

    def log_request(self, info: dict, idx: int, status: int, t0: float):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **info, "tier": self.names[idx],
               "status": status, "ttfb_ms": round((time.time() - t0) * 1000)}
        self.write_log(rec)
        log.info("%-12s %-12s %-6s %s", rec.get("kind"), rec["tier"], status, rec.get("reason"))


def create_app(raw: dict, transport=None, jev=None) -> FastAPI:
    router = Router(raw, transport=transport, jev=jev)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await router.client.aclose()

    app = FastAPI(title="claude_auto proxy", lifespan=lifespan)
    app.state.router = router

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        idx, info = await router.decide(body, request.headers)
        return await router.forward(request, "/v1/messages", body, idx, info)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        body = await request.json()
        sid = request.headers.get("x-claude-code-session-id") or "anon"
        s = router.sessions.get(sid)
        idx = s.tier if s and s.tier is not None else 0
        tier = router.cfg.tiers[idx]
        backend = router.backends[tier.backend]
        payload = {**body, "model": tier.model}
        try:
            r = await router.client.post(backend["base_url"].rstrip("/") + "/v1/messages/count_tokens",
                                         json=payload, headers=router.upstream_headers(request.headers, backend),
                                         timeout=15)
            if r.status_code == 200:
                return Response(r.content, media_type="application/json")
        except httpx.HTTPError:
            pass
        # Claude Code falls back to a character-based estimate when this endpoint is absent
        return JSONResponse({"type": "error", "error": {"type": "not_found_error",
                             "message": "count_tokens unavailable"}}, status_code=404)

    @app.get("/v1/models")
    async def models():
        data = [{"id": "claude-auto", "display_name": "Auto (JEV routing)",
                 "description": "claude_auto proxy picks the tier per prompt"}]
        data += [{"id": f"{FORCE_PREFIX}{t.name}", "display_name": f"Force {t.name}",
                  "description": f"Always {t.model}"} for t in router.cfg.tiers]
        return {"data": data, "has_more": False}

    @app.api_route("/api/hello", methods=["GET", "HEAD"])
    async def hello():
        return Response(status_code=200)

    @app.get("/health")
    async def health():
        return {"ok": True, "mode": router.mode, "sessions": len(router.sessions),
                "blocked": {k: round(v - time.time()) for k, v in router.blocked.items() if v > time.time()}}

    @app.get("/sessions")
    async def sessions():
        return {k[-20:]: {"tier": router.names[v.tier] if v.tier is not None else None,
                          "tool_errors": v.tool_errors, "idle_s": round(time.time() - v.updated)}
                for k, v in router.sessions.items()}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/.claude_auto/config.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    raw = yaml.safe_load(Path(a.config).expanduser().read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import uvicorn
    lst = raw.get("listen", {})
    uvicorn.run(create_app(raw), host=a.host or lst.get("host", "127.0.0.1"),
                port=a.port or int(lst.get("port", 8787)), log_level="warning")


if __name__ == "__main__":
    main()
