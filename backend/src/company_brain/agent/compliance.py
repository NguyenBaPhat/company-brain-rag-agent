"""Deterministic compliance engine.

The policy text lives in the knowledge base (so the agent can *explain* it with citations) and is
mirrored as machine-readable rules in ``compliance_rules.yaml`` so it can be *enforced* by code.
An LLM is never the only line of defence: the same rules run

* as the ``check_compliance`` tool (the agent self-checks drafts and revises),
* inside ``save_creative_brief`` (the tool refuses to persist non-compliant copy), and
* in the ``after_model_callback`` guardrail (blocks offending copy in the final answer).

Only text inside fenced ```copy blocks is treated as customer-facing copy. This lets the agent
*quote* prohibited phrases while explaining the policy without tripping the scanner.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Severity = Literal["block", "warn"]

_COPY_BLOCK = re.compile(r"```copy[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class Violation(BaseModel):
    rule_id: str
    severity: Severity
    match: str
    start: int
    end: int
    description: str
    suggestion: str
    policy_section: str


class MissingDisclaimer(BaseModel):
    id: str
    description: str
    required_any_of: list[str]
    policy_section: str


class ComplianceReport(BaseModel):
    passed: bool = Field(description="True when there are no blocking violations or missing disclaimers.")
    violations: list[Violation] = Field(default_factory=list)
    missing_disclaimers: list[MissingDisclaimer] = Field(default_factory=list)
    rules_version: str = ""
    scanned_chars: int = 0

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]


class _Rule(BaseModel):
    id: str
    severity: Severity
    description: str
    suggestion: str
    policy_section: str
    patterns: list[re.Pattern[str]]

    model_config = {"arbitrary_types_allowed": True}


class _Disclaimer(BaseModel):
    id: str
    description: str
    policy_section: str
    triggers: list[re.Pattern[str]]
    must_include: list[str]

    model_config = {"arbitrary_types_allowed": True}


def _compile(patterns: dict[str, list[str]] | list[str] | None) -> list[re.Pattern[str]]:
    if not patterns:
        return []
    flat = [p for group in patterns.values() for p in group] if isinstance(patterns, dict) else patterns
    return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in flat]


class ComplianceEngine:
    def __init__(self, rules_path: Path) -> None:
        raw = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.version: str = str(raw.get("version", ""))
        self.rules = [
            _Rule(
                id=r["id"],
                severity=r["severity"],
                description=r["description"],
                suggestion=r.get("suggestion", ""),
                policy_section=r.get("policy_section", ""),
                patterns=_compile(r["patterns"]),
            )
            for r in raw["rules"]
        ]
        self.disclaimers = [
            _Disclaimer(
                id=d["id"],
                description=d["description"],
                policy_section=d.get("policy_section", ""),
                triggers=_compile(d["trigger_patterns"]),
                must_include=[s.lower() for group in d["must_include"].values() for s in group],
            )
            for d in raw.get("required_disclaimers", [])
        ]

    def scan(self, text: str) -> ComplianceReport:
        violations: list[Violation] = []
        for rule in self.rules:
            for pattern in rule.patterns:
                for m in pattern.finditer(text):
                    violations.append(
                        Violation(
                            rule_id=rule.id,
                            severity=rule.severity,
                            match=m.group(0).strip(),
                            start=m.start(),
                            end=m.end(),
                            description=rule.description,
                            suggestion=rule.suggestion,
                            policy_section=rule.policy_section,
                        )
                    )
        lowered = text.lower()
        missing = [
            MissingDisclaimer(
                id=d.id,
                description=d.description,
                required_any_of=d.must_include,
                policy_section=d.policy_section,
            )
            for d in self.disclaimers
            if any(t.search(text) for t in d.triggers)
            and not any(s in lowered for s in d.must_include)
        ]
        blocking = any(v.severity == "block" for v in violations)
        return ComplianceReport(
            passed=not blocking and not missing,
            violations=violations,
            missing_disclaimers=missing,
            rules_version=self.version,
            scanned_chars=len(text),
        )


@lru_cache(maxsize=4)
def get_engine(rules_path: str) -> ComplianceEngine:
    return ComplianceEngine(Path(rules_path))


# --------------------------------------------------------------------------- copy blocks
def extract_copy_blocks(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, content) for every ```copy fenced block."""
    return [(m.start(), m.end(), m.group(1)) for m in _COPY_BLOCK.finditer(text)]


def scan_copy_blocks(engine: ComplianceEngine, text: str) -> ComplianceReport | None:
    """Scan only customer-facing copy blocks. Returns None if the text contains no copy."""
    blocks = extract_copy_blocks(text)
    if not blocks:
        return None
    return engine.scan("\n\n".join(b[2] for b in blocks))


def redact_blocked_copy(engine: ComplianceEngine, text: str) -> tuple[str, list[dict[str, str]]]:
    """Replace each copy block that has blocking violations with an explanatory placeholder."""
    blocked: list[dict[str, str]] = []
    out: list[str] = []
    cursor = 0
    for start, end, content in extract_copy_blocks(text):
        report = engine.scan(content)
        out.append(text[cursor:start])
        if report.blocking:
            distinct: dict[str, Violation] = {}
            for v in report.blocking:  # one entry per rule, keeping the first match as the example
                distinct.setdefault(v.rule_id, v)
            for v in distinct.values():
                blocked.append(
                    {
                        "rule_id": v.rule_id,
                        "match": v.match,
                        "suggestion": v.suggestion,
                        "policy_section": v.policy_section,
                    }
                )
            reasons = "; ".join(f"`{v.rule_id}` (\"{v.match}\")" for v in distinct.values())
            fixes = " ".join(dict.fromkeys(v.suggestion for v in distinct.values() if v.suggestion))
            out.append(f"> **Copy blocked by compliance guardrail** — {reasons}. {fixes}")
        else:
            out.append(text[start:end])
        cursor = end
    out.append(text[cursor:])
    return "".join(out), blocked
