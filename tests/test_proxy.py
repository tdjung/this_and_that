"""Proxy tests with mocked upstreams and a fake JEV (no GPU / network needed).

  python -m pytest -q tests/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from proxy.server import create_app  # noqa: E402

SSE = (b'event: message_start\ndata: {"type":"message_start"}\n\n'
       b'event: ping\ndata: {"type":"ping"}\n\n'
       b'event: message_stop\ndata: {"type":"message_stop"}\n\n')


class FakeJev:
    """Returns the tier set in `.next` (by name) with confidence `.conf`; can raise."""
    def __init__(self, cfg_tiers):
        self.criteria = {t["criteria"]: t["name"] for t in cfg_tiers}
        self.next, self.conf, self.fail, self.calls = "qwen3.8-27b", 0.9, False, 0

    async def adecide(self, client, kind, state, question, options):
        self.calls += 1
        if self.fail:
            raise RuntimeError("jev down")
        rest = (1 - self.conf) / (len(options) - 1)
        return {o: (self.conf if self.criteria[o] == self.next else rest) for o in options}


class _Stream(httpx.AsyncByteStream):
    """Unread async body, like a real streamed upstream response."""
    def __init__(self, data: bytes):
        self.data = data

    async def __aiter__(self):
        for i in range(0, len(self.data), 16):
            yield self.data[i:i + 16]


def resp(status, headers=None, body=b""):
    return httpx.Response(status, headers=headers or {}, stream=_Stream(body))


class Upstream:
    """Records requests; per-backend scripted responses."""
    def __init__(self):
        self.requests, self.script = [], {}

    def __call__(self, request: httpx.Request):
        host = request.url.host
        self.requests.append(request)
        if request.url.path.endswith("count_tokens"):
            return resp(404, {"content-type": "application/json"}, b'{"error":"nope"}')
        q = self.script.get(host)
        if q:
            status, headers = q.pop(0)
            return resp(status, {"content-type": "application/json", **headers},
                        b'{"type":"error","error":{"message":"x"}}')
        return resp(200, {"content-type": "text/event-stream",
                          "anthropic-ratelimit-unified-status": "allowed"}, SSE)

    def last(self):
        r = self.requests[-1]
        return r.url.host, json.loads(r.content), r.headers


@pytest.fixture()
def env(tmp_path, monkeypatch):
    raw = yaml.safe_load((ROOT / "proxy" / "config.example.yaml").read_text())
    raw["log_path"] = str(tmp_path / "log.jsonl")
    monkeypatch.setenv("INTERNAL_LLM_TOKEN", "internal-secret")
    up = Upstream()
    jev = FakeJev(raw["tiers"])
    app = create_app(raw, transport=httpx.MockTransport(up), jev=jev)
    with TestClient(app) as c:
        yield c, up, jev, raw


def msgs_user(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def msgs_tool(text, error=False):
    return msgs_user(text) + [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out",
                                      "is_error": error}]}]


def post(c, messages, sid="s1", model="claude-auto", stream=True, **hdr):
    h = {"x-claude-code-session-id": sid, "anthropic-beta": "oauth-2025-04-20,foo",
         "anthropic-version": "2023-06-01", "authorization": "Bearer client-oauth"}
    h.update({k.replace("_", "-"): v for k, v in hdr.items()})
    return c.post("/v1/messages?beta=true", headers=h,
                  json={"model": model, "max_tokens": 10, "stream": stream, "messages": messages})


def test_new_prompt_routes_by_jev_and_rewrites_model(env):
    c, up, jev, _ = env
    jev.next = "qwen3.8-fn"
    r = post(c, msgs_user("explain foo()"))
    assert r.status_code == 200 and r.content == SSE              # stream relayed intact
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"
    host, body, h = up.last()
    assert host == "qwen-fn.internal" and body["model"] == "qwen3.8-flash-next"
    assert h["authorization"] == "Bearer internal-secret"        # internal creds injected
    assert h["anthropic-beta"] == "oauth-2025-04-20,foo"          # betas forwarded verbatim
    assert up.requests[-1].url.query == b"beta=true"


def test_anthropic_passthrough_auth(env):
    c, up, jev, _ = env
    jev.next = "sonnet"
    post(c, msgs_user("implement upload feature"))
    host, body, h = up.last()
    assert host == "api.anthropic.com" and body["model"] == "claude-sonnet-5-5"
    assert h["authorization"] == "Bearer client-oauth"


def test_continuation_is_pinned_without_classifying(env):
    c, up, jev, _ = env
    jev.next = "sonnet"
    post(c, msgs_user("task"))
    jev.next = "qwen3.8-27b"
    calls = jev.calls
    r = post(c, msgs_tool("task"))
    assert r.headers["x-claude-auto-tier"] == "sonnet" and jev.calls == calls


def test_upgrade_only_keeps_higher_tier(env):
    c, up, jev, _ = env
    jev.next = "opus"
    post(c, msgs_user("design arch"))
    jev.next = "qwen3.8-27b"
    r = post(c, msgs_user("design arch") + [{"role": "assistant", "content": "ok"}] + msgs_user("5*3?"))
    assert r.headers["x-claude-auto-tier"] == "opus"


def test_prompt_id_header_detects_new_prompt(env):
    c, up, jev, _ = env
    jev.next = "qwen3.8-27b"
    post(c, msgs_user("a"), x_claude_code_prompt_id="p1")
    jev.next = "sonnet"
    r = post(c, msgs_tool("a"), x_claude_code_prompt_id="p2")        # tool result but new prompt id
    assert r.headers["x-claude-auto-tier"] == "sonnet"
    calls = jev.calls
    post(c, msgs_tool("a"), x_claude_code_prompt_id="p2")
    assert jev.calls == calls


def test_keyword_rule_overrides_low_jev(env):
    c, up, jev, _ = env
    jev.next = "qwen3.8-27b"
    r = post(c, msgs_user("RTL 코드를 분석한 뒤 새롭게 구성해줘"))
    assert r.headers["x-claude-auto-tier"] == "opus"
    r = post(c, msgs_user("make it shortly"), sid="s2")              # 'rtl' inside a word: no match
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-27b"


def test_low_confidence_goes_up_one(env):
    c, up, jev, _ = env
    jev.next, jev.conf = "qwen3.8-27b", 0.4
    r = post(c, msgs_user("hmm"))
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"


def test_jev_failure_uses_fallback(env):
    c, up, jev, _ = env
    jev.fail = True
    r = post(c, msgs_user("anything"))
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"


def test_auxiliary_and_agent_type_and_forced(env):
    c, up, jev, _ = env
    jev.next = "opus"
    r = post(c, msgs_user("title please"), x_claude_code_request_class="auxiliary")
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-27b"
    r = post(c, msgs_user("find files"), x_claude_code_request_class="subagent",
             x_claude_code_agent_type="Explore", x_claude_code_agent_id="a1")
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"
    r = post(c, msgs_user("hi"), model="claude-tier-fable")
    assert r.headers["x-claude-auto-tier"] == "fable"
    assert up.last()[1]["model"] == "claude-fable-5-1"


def test_subagent_gets_own_session(env):
    c, up, jev, _ = env
    jev.next = "opus"
    post(c, msgs_user("big task"))
    jev.next = "qwen3.8-fn"
    r = post(c, msgs_user("small sub task"), x_claude_code_request_class="subagent", x_claude_code_agent_id="a9")
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"           # not dragged up by the parent pin


def test_tool_errors_escalate(env):
    c, up, jev, _ = env
    jev.next = "qwen3.8-fn"
    post(c, msgs_user("fix"))
    for _ in range(2):
        assert post(c, msgs_tool("fix", error=True)).headers["x-claude-auto-tier"] == "qwen3.8-fn"
    assert post(c, msgs_tool("fix", error=True)).headers["x-claude-auto-tier"] == "sonnet"


def test_internal_5xx_escalates(env):
    c, up, jev, _ = env
    jev.next = "qwen3.8-27b"
    up.script["qwen27b.internal"] = [(503, {})]
    r = post(c, msgs_user("q"))
    assert r.status_code == 200 and r.headers["x-claude-auto-tier"] == "qwen3.8-fn"


def test_quota_429_downgrades_and_blocks(env):
    c, up, jev, _ = env
    jev.next = "opus"
    up.script["api.anthropic.com"] = [(429, {"retry-after": "3600"})]
    r = post(c, msgs_user("design"))
    assert r.status_code == 200 and r.headers["x-claude-auto-tier"] == "qwen3.8-fn"
    r = post(c, msgs_user("another"), sid="s3")                      # backend stays blocked
    assert r.headers["x-claude-auto-tier"] == "qwen3.8-fn"
    assert c.get("/health").json()["blocked"]


def test_short_429_is_passed_through(env):
    c, up, jev, _ = env
    jev.next = "sonnet"
    up.script["api.anthropic.com"] = [(429, {"retry-after": "5", "x-should-retry": "true"})]
    r = post(c, msgs_user("x"))
    assert r.status_code == 429 and r.headers["x-should-retry"] == "true"


def test_context_fit_moves_up(env):
    c, up, jev, raw = env
    jev.next = "qwen3.8-27b"
    big = "x" * (raw["tiers"][0]["max_context"] * 3)
    r = post(c, msgs_user("summarize") + [{"role": "assistant", "content": big}] + msgs_tool("summarize"))
    assert r.headers["x-claude-auto-tier"] == "sonnet"


def test_shadow_mode(env, tmp_path):
    c, up, jev, raw = env
    c.app.state.router.mode = "shadow"
    jev.next = "qwen3.8-27b"
    r = post(c, msgs_user("5*3"))
    assert r.headers["x-claude-auto-tier"] == "sonnet"
    rec = json.loads(Path(raw["log_path"]).read_text().splitlines()[-1])
    assert rec["would_route"] == "qwen3.8-27b"


def test_misc_endpoints(env):
    c, up, jev, _ = env
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "claude-auto" in ids and "claude-tier-opus" in ids
    assert c.head("/api/hello").status_code == 200
    r = c.post("/v1/messages/count_tokens", json={"model": "claude-auto", "messages": msgs_user("hi")})
    assert r.status_code == 404
