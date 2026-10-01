"""Tier definitions, rules and JEV-based classification shared by the proxy and the eval.

Routing for one *new* request (the first request of a user prompt):

  1. rules      : keyword / file-extension rules give a *minimum* tier (upgrade only)
  2. JEV        : a `choice` question over the tier descriptions picks a tier
  3. gate       : if JEV's top probability is below the threshold, go one tier up
  4. final tier : max(rule tier, gated JEV tier)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_QUESTION = ("Which is the least capable model tier that can reliably and correctly "
                    "complete this request?")

REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


@dataclass
class Tier:
    name: str
    backend: str
    model: str
    criteria: str
    max_context: int = 200_000


@dataclass
class RoutingConfig:
    tiers: list[Tier]
    question: str = DEFAULT_QUESTION
    confidence_threshold: float = 0.55
    fallback_tier: int = 1
    max_state_chars: int = 2000
    context_turns: int = 2
    keyword_rules: list[tuple[int, list[str]]] = field(default_factory=list)
    extension_rules: list[tuple[int, list[str]]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, cfg: dict) -> "RoutingConfig":
        tiers = [Tier(**t) for t in cfg["tiers"]]
        names = [t.name for t in tiers]
        jev = cfg.get("jev", {})
        rules = cfg.get("rules", {})

        def idx(name):
            if name not in names:
                raise ValueError(f"unknown tier '{name}' (tiers: {names})")
            return names.index(name)

        return cls(
            tiers=tiers,
            question=jev.get("question", DEFAULT_QUESTION),
            confidence_threshold=float(jev.get("confidence_threshold", 0.55)),
            fallback_tier=idx(jev.get("fallback_tier", names[min(1, len(names) - 1)])),
            max_state_chars=int(jev.get("max_state_chars", 2000)),
            context_turns=int(jev.get("context_turns", 2)),
            keyword_rules=[(idx(r["min_tier"]), [k.lower() for k in r["keywords"]])
                           for r in rules.get("keywords", [])],
            extension_rules=[(idx(r["min_tier"]), [e.lower().lstrip(".") for e in r["extensions"]])
                             for r in rules.get("extensions", [])],
        )

    def index(self, name: str) -> int:
        for i, t in enumerate(self.tiers):
            if t.name == name:
                return i
        raise KeyError(name)

    @property
    def top(self) -> int:
        return len(self.tiers) - 1


@dataclass
class Decision:
    tier: int
    reason: str
    jev_tier: int | None = None
    confidence: float | None = None
    probs: dict | None = None
    rule_tier: int | None = None


# ---------------------------------------------------------------- message parsing
def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join(out)


def clean(text: str) -> str:
    return REMINDER_RE.sub("", text or "").strip()


def is_tool_result_turn(messages: list) -> bool:
    """True when the last user message carries tool results (a tool-loop continuation)."""
    if not messages or messages[-1].get("role") != "user":
        return False
    c = messages[-1].get("content")
    return isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)


def tool_errors(messages: list) -> tuple[int, int]:
    """(#tool_result blocks, #errored) in the last user message."""
    if not messages or messages[-1].get("role") != "user":
        return 0, 0
    c = messages[-1].get("content")
    if not isinstance(c, list):
        return 0, 0
    res = [b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"]
    return len(res), sum(1 for b in res if b.get("is_error"))


def user_prompts(messages: list) -> list[str]:
    """Plain-text user prompts (not tool results), oldest first, system reminders removed."""
    out = []
    for m in messages or []:
        if m.get("role") != "user":
            continue
        t = clean(_block_text(m.get("content")))
        if t:
            out.append(t)
    return out


def build_state(prompts: list[str], cfg: RoutingConfig) -> str:
    if not prompts:
        return ""
    current = prompts[-1][: cfg.max_state_chars]
    state = f"Request from a developer to an AI coding assistant:\n{current}"
    earlier = prompts[-1 - cfg.context_turns:-1] if cfg.context_turns else []
    budget = cfg.max_state_chars - len(current)
    if earlier and budget > 200:
        ctx = "\n".join(f"- {p[:400]}" for p in earlier)[:budget]
        state += f"\n\nEarlier messages in the same session:\n{ctx}"
    return state


# ---------------------------------------------------------------- rules
def _kw_match(text: str, kw: str) -> bool:
    if kw.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text) is not None
    return kw in text


def rule_min_tier(text: str, cfg: RoutingConfig) -> tuple[int | None, str]:
    low = (text or "").lower()
    best, why = None, ""
    for tier, kws in cfg.keyword_rules:
        for kw in kws:
            if _kw_match(low, kw) and (best is None or tier > best):
                best, why = tier, f"keyword:{kw}"
    for tier, exts in cfg.extension_rules:
        pat = re.compile(r"[\w./-]+\.(" + "|".join(map(re.escape, exts)) + r")(?![\w])")
        m = pat.search(low)
        if m and (best is None or tier > best):
            best, why = tier, f"file:{m.group(0)}"
    return best, why


# ---------------------------------------------------------------- JEV
def jev_options(cfg: RoutingConfig) -> list[str]:
    return [t.criteria for t in cfg.tiers]


def interpret(probs_by_option: dict, cfg: RoutingConfig, order: list[int] | None = None):
    """Map JEV option probabilities back to tier indices. `order[i]` = tier of option i."""
    opts = list(probs_by_option.keys())
    order = order or list(range(len(opts)))
    by_tier = {order[i]: probs_by_option[o] for i, o in enumerate(opts)}
    jt = max(by_tier, key=by_tier.get)
    return jt, by_tier[jt], {cfg.tiers[k].name: round(v, 4) for k, v in sorted(by_tier.items())}


def combine(cfg: RoutingConfig, text: str, jev_tier: int | None, conf: float | None,
            probs: dict | None, jev_error: str | None = None) -> Decision:
    rule_t, rule_why = rule_min_tier(text, cfg)
    if jev_tier is None:
        tier, reason = cfg.fallback_tier, f"jev_unavailable({jev_error})"
    elif conf is not None and conf < cfg.confidence_threshold and jev_tier < cfg.top:
        tier, reason = jev_tier + 1, f"jev_low_conf({conf:.2f})->+1"
    else:
        tier, reason = jev_tier, f"jev({conf:.2f})"
    if rule_t is not None and rule_t > tier:
        tier, reason = rule_t, f"{rule_why} (jev said {cfg.tiers[jev_tier].name if jev_tier is not None else '-'})"
    return Decision(tier=tier, reason=reason, jev_tier=jev_tier, confidence=conf, probs=probs,
                    rule_tier=rule_t)


def classify_sync(jev, cfg: RoutingConfig, prompts: list[str]) -> Decision:
    state = build_state(prompts, cfg)
    try:
        p = jev.decide("choice", state, cfg.question, jev_options(cfg))
        jt, conf, probs = interpret(p, cfg)
        return combine(cfg, prompts[-1] if prompts else "", jt, conf, probs)
    except Exception as e:  # noqa: BLE001
        return combine(cfg, prompts[-1] if prompts else "", None, None, None, type(e).__name__)


async def classify_async(jev, client, cfg: RoutingConfig, prompts: list[str]) -> Decision:
    state = build_state(prompts, cfg)
    try:
        p = await jev.adecide(client, "choice", state, cfg.question, jev_options(cfg))
        jt, conf, probs = interpret(p, cfg)
        return combine(cfg, prompts[-1] if prompts else "", jt, conf, probs)
    except Exception as e:  # noqa: BLE001
        return combine(cfg, prompts[-1] if prompts else "", None, None, None, type(e).__name__)
