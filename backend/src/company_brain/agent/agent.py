"""Agent + Runner factory (Google ADK)."""

from __future__ import annotations

from google.adk import Agent, Runner
from google.adk.agents import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.sessions import BaseSessionService
from google.genai import types

from .deps import AgentDeps
from .guardrails import make_after_model_callback, make_before_tool_callback
from .mock_llm import MockLlm
from .prompts import instruction_provider
from .tools import build_tools

APP_NAME = "company_brain"
AGENT_NAME = "company_brain"


def _generate_config(settings) -> types.GenerateContentConfig:
    thinking = (
        types.ThinkingConfig(thinking_level=settings.gemini_thinking_level.upper())
        if settings.gemini_thinking_level and settings.llm_mode == "gemini"
        else None
    )
    return types.GenerateContentConfig(max_output_tokens=8192, thinking_config=thinking)


def build_agent(deps: AgentDeps) -> Agent:
    settings = deps.settings
    model = MockLlm() if settings.llm_mode == "mock" else settings.gemini_model
    return Agent(
        name=AGENT_NAME,
        model=model,
        description="Internal knowledge assistant for Lumera Skin: grounded answers, citations, compliance.",
        instruction=instruction_provider,
        tools=build_tools(deps),
        # Sampling parameters are deliberately left at the model's recommended defaults.
        generate_content_config=_generate_config(settings),
        before_tool_callback=make_before_tool_callback(settings),
        after_model_callback=make_after_model_callback(deps.compliance),
    )


def build_runner(deps: AgentDeps, session_service: BaseSessionService) -> Runner:
    return Runner(
        app_name=APP_NAME,
        agent=build_agent(deps),
        session_service=session_service,
        auto_create_session=False,
    )


def build_run_config(deps: AgentDeps) -> RunConfig:
    return RunConfig(
        streaming_mode=StreamingMode.SSE,
        max_llm_calls=deps.settings.max_llm_calls_per_turn,
    )
