"""
state_machine.py
Parses conversation history to determine the current state and drives
LLM calls with the right context. Stateless — all context comes from
the message history passed in per request.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.prompt_templates import (
    CATALOG_ITEM_TEMPLATE,
    COMPARE_CONTEXT_PROMPT,
    INJECTION_REFUSAL_REPLY,
    RECOMMENDATION_CONTEXT_PROMPT,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
)

# ── State constants ────────────────────────────────────────────────────────────
STATE_CLARIFY = "CLARIFY"
STATE_RECOMMEND = "RECOMMEND"
STATE_COMPARE = "COMPARE"
STATE_REFINE = "REFINE"
STATE_REFUSE = "REFUSE"
STATE_DONE = "DONE"

MAX_TURNS = 8  # user + assistant combined

# Keywords that signal off-topic or injection attempts
OFF_TOPIC_PATTERNS = re.compile(
    r"\b(salary|compensation|legal|lawsuit|discriminat|gdpr|eeoc|"
    r"ignore previous|disregard|forget instructions|jailbreak|"
    r"as an ai|you are now|pretend you|act as|override|system prompt|"
    r"hiring law|background check|immigration|visa|religion|race|gender)\b",
    re.I,
)

COMPARE_PATTERNS = re.compile(
    r"\b(compar|difference|vs\.?|versus|between|which is better|"
    r"what.s the diff)\b",
    re.I,
)

REFINE_PATTERNS = re.compile(
    r"\b(actually|also add|remove|instead|change|update|modify|"
    r"add personality|add cognitive|exclude|replace)\b",
    re.I,
)

JOB_LEVEL_MAP = {
    "entry": "Entry-Level",
    "junior": "Entry-Level",
    "graduate": "Graduate",
    "grad": "Graduate",
    "fresher": "Graduate",
    "mid": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "lead": "Professional Individual Contributor",
    "staff": "Professional Individual Contributor",
    "manager": "Manager",
    "director": "Director",
    "executive": "Executive",
    "vp": "Executive",
    "c-level": "Executive",
    "ceo": "Executive",
    "cto": "Executive",
    "supervisor": "Manager",
    "team lead": "Manager",
}


@dataclass
class ConversationContext:
    """Extracted context from conversation history."""
    role_description: str = ""
    job_level: Optional[str] = None
    test_type_codes: list[str] = field(default_factory=list)
    max_duration: Optional[int] = None
    remote_only: bool = False
    clarify_count: int = 0
    total_turns: int = 0
    is_vague: bool = True
    compare_names: list[str] = field(default_factory=list)
    is_refine: bool = False
    is_off_topic: bool = False
    is_injection: bool = False


def extract_context(messages: list[dict]) -> ConversationContext:
    """
    Parse full conversation history to extract structured context.
    This is deterministic and does not call the LLM.
    """
    ctx = ConversationContext()
    ctx.total_turns = len(messages)

    all_user_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    )
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Injection / off-topic detection (fast regex, no LLM)
    if _is_injection(last_user):
        ctx.is_injection = True
        return ctx
    if _is_off_topic(last_user):
        ctx.is_off_topic = True
        return ctx

    # Compare intent
    if COMPARE_PATTERNS.search(last_user):
        ctx.compare_names = _extract_product_names(last_user + " " + all_user_text)

    # Refine intent
    if REFINE_PATTERNS.search(last_user) and ctx.total_turns > 2:
        ctx.is_refine = True

    # Job level
    ctx.job_level = _extract_job_level(all_user_text)

    # Test type hints from user text
    ctx.test_type_codes = _extract_type_codes(all_user_text)

    # Role description — full user-contributed context
    ctx.role_description = all_user_text

    # Duration constraint
    dur_m = re.search(r"(\d+)\s*min", all_user_text, re.I)
    if dur_m:
        ctx.max_duration = int(dur_m.group(1))

    # Remote preference
    if re.search(r"\bremote\b", all_user_text, re.I):
        ctx.remote_only = True

    # Vagueness: role description is meaningful if > 20 chars and contains
    # a role/skill keyword beyond just "assessment" or "test"
    ctx.is_vague = _is_vague(all_user_text, ctx.total_turns)

    # Count how many assistant turns asked clarification
    ctx.clarify_count = sum(
        1 for m in messages
        if m["role"] == "assistant"
        and any(q in m["content"].lower() for q in ["?", "clarif", "could you tell", "what level"])
    )

    return ctx


# ── Intent helpers ─────────────────────────────────────────────────────────────

def _is_injection(text: str) -> bool:
    injection_terms = [
        "ignore previous", "disregard", "forget instructions",
        "jailbreak", "you are now", "pretend you are", "act as",
        "override", "system prompt", "new persona",
    ]
    tl = text.lower()
    return any(t in tl for t in injection_terms)


def _is_off_topic(text: str) -> bool:
    off_topic_terms = [
        "salary", "compensation", "legal", "lawsuit", "discriminat",
        "gdpr", "eeoc", "hiring law", "background check",
        "immigration", "visa status",
    ]
    tl = text.lower()
    # Allow if it's framed as "what SHL test covers X"
    if "shl" in tl and ("test" in tl or "assessment" in tl):
        return False
    return any(t in tl for t in off_topic_terms)


def _extract_job_level(text: str) -> Optional[str]:
    tl = text.lower()
    for kw, level in JOB_LEVEL_MAP.items():
        if kw in tl:
            return level
    # year-of-experience heuristic
    yr_m = re.search(r"(\d+)\s*(?:\+)?\s*years?", tl)
    if yr_m:
        yrs = int(yr_m.group(1))
        if yrs <= 1:
            return "Entry-Level"
        elif yrs <= 3:
            return "Graduate"
        elif yrs <= 7:
            return "Mid-Professional"
        else:
            return "Professional Individual Contributor"
    return None


def _extract_type_codes(text: str) -> list[str]:
    codes = []
    tl = text.lower()
    if any(k in tl for k in ["personality", "behaviour", "behavior", "opq", "trait", "motivat"]):
        codes.append("P")
    if any(k in tl for k in ["cognitive", "aptitude", "ability", "numerical", "verbal", "reasoning"]):
        codes.append("A")
    if any(k in tl for k in ["knowledge", "technical", "skill", "coding", "java", "python",
                               "sql", "excel", ".net", "aws", "angular", "react"]):
        codes.append("K")
    if any(k in tl for k in ["simulation", "situational", "sjt", "inbox"]):
        codes.append("S")
        codes.append("B")
    if any(k in tl for k in ["competenc", "360", "leadership", "development"]):
        codes.append("C")
        codes.append("D")
    return list(set(codes))


def _extract_product_names(text: str) -> list[str]:
    """Heuristic: extract quoted names or capitalised multi-word phrases."""
    # Quoted
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted
    # Common SHL product names in text
    known = [
        "OPQ", "OPQ32", "OPQ32r", "MQ", "Verify", "SJT",
        "GSA", "Numerical Reasoning", "Verbal Reasoning",
        "Inductive Reasoning", "Deductive Reasoning",
        "Calculation", "Checking", "CCSQ",
    ]
    found = [n for n in known if n.lower() in text.lower()]
    return found


def _is_vague(text: str, turns: int) -> bool:
    """Return True if the accumulated context is too thin to recommend."""
    if turns >= 4:
        return False  # Force commit after 4 turns
    words = text.split()
    meaningful = [
        w for w in words
        if len(w) > 3 and w.lower() not in
        {"need", "want", "hire", "hiring", "assessment", "test", "help", "looking", "some"}
    ]
    return len(meaningful) < 5


def decide_state(ctx: ConversationContext) -> str:
    """
    Pure function: given extracted context, return the next agent state.
    No LLM call here — keeps latency deterministic.
    """
    if ctx.is_injection:
        return STATE_REFUSE
    if ctx.is_off_topic:
        return STATE_REFUSE
    if ctx.compare_names and not ctx.is_vague:
        return STATE_COMPARE
    if ctx.is_refine:
        return STATE_REFINE
    # Force recommend if too many turns have elapsed
    if ctx.total_turns >= MAX_TURNS - 2:
        return STATE_RECOMMEND
    # Need at least a role description before recommending
    if ctx.is_vague and ctx.clarify_count < 2:
        return STATE_CLARIFY
    return STATE_RECOMMEND


def format_catalog_context(items: list[dict]) -> str:
    """Format catalog items into a prompt-injectable string."""
    parts = []
    for item in items:
        parts.append(
            CATALOG_ITEM_TEMPLATE.format(
                name=item["name"],
                url=item["url"],
                test_type=item["test_type"],
                keys=", ".join(item["keys"]),
                job_levels=", ".join(item["job_levels"]) or "All levels",
                duration=item["duration"] or "N/A",
                remote="Yes" if item["remote"] else "No",
                description=item["description"][:300] + "..."
                if len(item["description"]) > 300
                else item["description"],
            )
        )
    return "\n---\n".join(parts)
