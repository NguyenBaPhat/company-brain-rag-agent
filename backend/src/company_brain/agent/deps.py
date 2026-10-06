"""Dependency container shared by the agent's tools, callbacks and the API layer."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..knowledge.store import KnowledgeStore
from .compliance import ComplianceEngine, get_engine


@dataclass
class AgentDeps:
    settings: Settings
    store: KnowledgeStore
    compliance: ComplianceEngine


def build_deps(settings: Settings, store: KnowledgeStore) -> AgentDeps:
    return AgentDeps(
        settings=settings,
        store=store,
        compliance=get_engine(str(settings.compliance_rules_path)),
    )
