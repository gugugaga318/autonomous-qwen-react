"""Centralized Qwen-compatible LLM Gateway with structured JSON contracts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.resources import files
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

import httpx

from yield_rca_core.models import (
    AgentKind,
    AgentMode,
    LLMUsageEvent,
    ModelValidationError,
)


class LLMConfigurationError(RuntimeError):
    """Raised when LLM mode is selected without valid provider configuration."""


def _sanitize_diagnostic_value(value: Any, *, limit: int) -> str | None:
    """Bound provider diagnostics and remove common credential representations."""

    if not isinstance(value, (str, int, float)):
        return None
    sanitized = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not sanitized:
        return None
    sanitized = re.sub(
        r"(?i)\b(bearer)\s+[a-z0-9._~+/=-]+",
        r"\1 [REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(api[_ -]?key|authorization)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        sanitized,
    )
    return sanitized[:limit]


class LLMCallError(RuntimeError):
    """Raised when the configured model request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
        request_id: str | None = None,
        failure_category: str = "llm_call_error",
        call_attempt_count: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = _sanitize_diagnostic_value(provider_code, limit=100)
        self.provider_message = _sanitize_diagnostic_value(
            provider_message,
            limit=500,
        )
        self.request_id = _sanitize_diagnostic_value(request_id, limit=100)
        self.failure_category = (
            _sanitize_diagnostic_value(failure_category, limit=100)
            or "llm_call_error"
        )
        self.call_attempt_count = max(1, call_attempt_count)


class LLMOutputValidationError(ValueError):
    """Raised when a model response violates its structured output contract."""


@dataclass(frozen=True)
class LLMSettings:
    agent_mode: str = AgentMode.DETERMINISTIC.value
    provider: str = "dashscope"
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        try:
            AgentMode(self.agent_mode)
        except ValueError as exc:
            raise LLMConfigurationError(
                "YIELD_RCA_AGENT_MODE must be deterministic, fake, or llm"
            ) from exc
        if not self.model.strip():
            raise LLMConfigurationError("LLM model must not be empty")
        if self.agent_mode == AgentMode.LLM.value and not self.api_key.strip():
            raise LLMConfigurationError("DASHSCOPE_API_KEY is required when agent mode is llm")
        if self.timeout_seconds <= 0:
            raise LLMConfigurationError("LLM timeout must be positive")
        if self.max_retries < 0:
            raise LLMConfigurationError("LLM max retries must be non-negative")

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            agent_mode=os.getenv(
                "YIELD_RCA_AGENT_MODE", AgentMode.DETERMINISTIC.value
            ).strip(),
            provider=os.getenv("YIELD_RCA_LLM_PROVIDER", "dashscope").strip(),
            model=os.getenv("YIELD_RCA_LLM_MODEL", "qwen-plus").strip(),
            base_url=os.getenv(
                "YIELD_RCA_LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip(),
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            timeout_seconds=float(os.getenv("YIELD_RCA_LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("YIELD_RCA_LLM_MAX_RETRIES", "1")),
        )


@dataclass(frozen=True)
class LLMRequest:
    agent: str
    prompt_name: str
    prompt_version: str
    payload: dict[str, Any]
    temperature: float = 0.0

    def __post_init__(self) -> None:
        try:
            AgentKind(self.agent)
        except ValueError as exc:
            raise ModelValidationError(f"invalid LLM request agent: {self.agent}") from exc
        if not self.prompt_name or not self.prompt_version:
            raise ModelValidationError("LLM prompt name and version are required")
        if not 0.0 <= self.temperature <= 2.0:
            raise ModelValidationError("LLM temperature must be between 0 and 2")


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]
    usage: LLMUsageEvent


class LLMClient(Protocol):
    provider: str
    model: str

    def complete_json(self, request: LLMRequest) -> LLMResponse: ...


def llm_calls_remaining(client: LLMClient) -> int | None:
    """Return a wrapper-advertised call budget without extending LLMClient.

    Production/provider clients are intentionally not required to expose a
    budget.  Evaluation wrappers may expose either ``remaining_calls`` or the
    existing ``max_calls``/``call_count`` pair.  Unknown budgets return ``None``
    and preserve the normal provider behavior.
    """

    raw_remaining = getattr(client, "remaining_calls", None)
    if isinstance(raw_remaining, int) and not isinstance(raw_remaining, bool):
        return max(0, raw_remaining)
    raw_max = getattr(client, "max_calls", None)
    raw_count = getattr(client, "call_count", None)
    if (
        isinstance(raw_max, int)
        and not isinstance(raw_max, bool)
        and isinstance(raw_count, int)
        and not isinstance(raw_count, bool)
    ):
        return max(0, raw_max - raw_count)
    return None


def llm_call_budget_available(
    client: LLMClient,
    *,
    required_calls: int = 1,
    reserve_calls: int = 0,
) -> bool:
    """Check an optional bounded-client budget without consuming a call."""

    if required_calls < 0 or reserve_calls < 0:
        raise ValueError("required_calls and reserve_calls must be non-negative")
    remaining = llm_calls_remaining(client)
    return remaining is None or remaining >= required_calls + reserve_calls


_USAGE_SINK: ContextVar[list[LLMUsageEvent] | None] = ContextVar(
    "yield_rca_llm_usage_sink",
    default=None,
)


@contextmanager
def capture_llm_usage() -> Iterator[list[LLMUsageEvent]]:
    events: list[LLMUsageEvent] = []
    parent = _USAGE_SINK.get()
    token = _USAGE_SINK.set(events)
    try:
        yield events
    finally:
        _USAGE_SINK.reset(token)
        if parent is not None:
            parent.extend(events)


def _record_usage(event: LLMUsageEvent) -> None:
    sink = _USAGE_SINK.get()
    if sink is not None:
        sink.append(event)


def load_prompt(prompt_name: str, prompt_version: str) -> str:
    resource_name = f"{prompt_name}_{prompt_version}.md"
    resource = files("yield_rca_core.prompts").joinpath(resource_name)
    if not resource.is_file():
        raise LLMConfigurationError(f"prompt not found: {resource_name}")
    return resource.read_text(encoding="utf-8")


def _parse_json_content(content: str) -> dict[str, Any]:
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        normalized = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise LLMOutputValidationError("model response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LLMOutputValidationError("model response must be a JSON object")
    return value


def _sanitize_provider_value(value: Any, *, api_key: str, limit: int) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None
    sanitized = str(value)
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED]")
    return _sanitize_diagnostic_value(sanitized, limit=limit)


def _extract_provider_error(
    response: httpx.Response,
    *,
    api_key: str,
) -> tuple[str | None, str | None, str | None]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None

    error_payload: dict[str, Any] = {}
    top_level: dict[str, Any] = payload if isinstance(payload, dict) else {}
    if isinstance(top_level.get("error"), dict):
        error_payload = top_level["error"]

    provider_code = _sanitize_provider_value(
        error_payload.get("code", top_level.get("code")),
        api_key=api_key,
        limit=100,
    )
    provider_message = _sanitize_provider_value(
        error_payload.get("message", top_level.get("message")),
        api_key=api_key,
        limit=500,
    )
    request_id = _sanitize_provider_value(
        top_level.get("request_id", top_level.get("requestId"))
        or response.headers.get("x-request-id"),
        api_key=api_key,
        limit=100,
    )
    return provider_code, provider_message, request_id


class DashScopeLLMClient:
    provider = "dashscope"

    def __init__(self, settings: LLMSettings) -> None:
        if settings.agent_mode != AgentMode.LLM.value:
            raise LLMConfigurationError("DashScope client requires llm agent mode")
        self.settings = settings
        self.model = settings.model

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        started = perf_counter()
        call_id = f"LLM_{uuid4().hex.upper()}"
        prompt = load_prompt(request.prompt_name, request.prompt_version)
        endpoint = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        request.payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        last_status_code: int | None = None
        last_provider_code: str | None = None
        last_provider_message: str | None = None
        last_request_id: str | None = None
        failure_category = "transport_error"
        call_attempt_count = 0
        for _ in range(self.settings.max_retries + 1):
            call_attempt_count += 1
            try:
                with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                    response = client.post(endpoint, headers=headers, json=body)
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                last_status_code = exc.response.status_code
                failure_category = "provider_http_error"
                (
                    last_provider_code,
                    last_provider_message,
                    last_request_id,
                ) = _extract_provider_error(
                    exc.response,
                    api_key=self.settings.api_key,
                )
                if 400 <= last_status_code < 500 and last_status_code not in {408, 429}:
                    break
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                failure_category = "transport_error"
                continue

            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("response payload must be a JSON object")
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("response content must be a string")
                usage_payload = dict(payload.get("usage", {}))
                prompt_tokens = int(usage_payload.get("prompt_tokens", 0))
                completion_tokens = int(usage_payload.get("completion_tokens", 0))
                details = dict(usage_payload.get("prompt_tokens_details") or {})
                completion_details = dict(
                    usage_payload.get("completion_tokens_details") or {}
                )
                usage = LLMUsageEvent(
                    call_id=call_id,
                    agent=request.agent,
                    provider=self.provider,
                    model=self.model,
                    prompt_version=request.prompt_version,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=max(
                        int(usage_payload.get("total_tokens", 0)),
                        prompt_tokens + completion_tokens,
                    ),
                    cached_tokens=int(details.get("cached_tokens", 0)),
                    reasoning_tokens=int(completion_details.get("reasoning_tokens", 0)),
                    latency_ms=round((perf_counter() - started) * 1000.0, 3),
                )
                _record_usage(usage)
                return LLMResponse(data=_parse_json_content(content), usage=usage)
            except LLMOutputValidationError:
                raise
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                _record_usage(
                    LLMUsageEvent(
                        call_id=call_id,
                        agent=request.agent,
                        provider=self.provider,
                        model=self.model,
                        prompt_version=request.prompt_version,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=round((perf_counter() - started) * 1000.0, 3),
                        status="failed",
                    )
                )
                raise LLMOutputValidationError(
                    "model response envelope is invalid"
                ) from exc
        _record_usage(
            LLMUsageEvent(
                call_id=call_id,
                agent=request.agent,
                provider=self.provider,
                model=self.model,
                prompt_version=request.prompt_version,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=round((perf_counter() - started) * 1000.0, 3),
                status="failed",
            )
        )
        raise LLMCallError(
            "Qwen request failed after retries",
            status_code=last_status_code,
            provider_code=last_provider_code,
            provider_message=last_provider_message,
            request_id=last_request_id,
            failure_category=failure_category,
            call_attempt_count=call_attempt_count,
        ) from last_error


class FakeLLMClient:
    """Deterministic no-cost client used to test the complete LLM path."""

    provider = "fake"
    model = "fake-qwen-plus"

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        started = perf_counter()
        if request.prompt_name == "planner":
            data = dict(request.payload["fallback_plan"])
        elif request.prompt_name == "intent_planner":
            data = dict(request.payload["deterministic_intent_plan"])
        elif request.prompt_name == "next_action_planner":
            data = dict(request.payload["deterministic_planner_decision"])
        elif request.prompt_name == "specialist_tool_planner":
            data = dict(request.payload["deterministic_specialist_decision"])
        elif request.prompt_name == "specialist_analysis":
            data = dict(request.payload["deterministic_specialist_analysis"])
        elif request.prompt_name == "specialist":
            finding = dict(request.payload["deterministic_finding"])
            data = {
                "summary": finding["summary"],
                "confidence": finding["confidence"],
                "evidence_ids": list(finding["evidence_ids"]),
                "engineering_interpretation": finding["summary"],
            }
        elif request.prompt_name == "hypothesis_candidate_generator":
            data = {
                "candidates": list(
                    request.payload["deterministic_candidate_proposals"]
                ),
                "analysis_summary": (
                    "The fake client preserves deterministic hypothesis behavior."
                ),
            }
        elif request.prompt_name == "causal_candidate_comparator":
            comparison = dict(request.payload.get("python_comparison", {}))
            data = {
                "preferred_candidate_index": comparison.get("preferred_candidate_index"),
                "comparison_explanation": comparison.get(
                    "comparison_explanation",
                    "Python comparison did not identify a unique candidate.",
                ),
                "selected_gap_id": None,
            }
        elif request.prompt_name == "causal_adversarial_challenge":
            data = {
                "challenges": [
                    {
                        "candidate_id": str(candidate.get("candidate_id", "")),
                        "strongest_alternative_lane_id": None,
                        "supporting_evidence_ids": [],
                        "contradicting_evidence_ids": [],
                        "unexplained_precursor_evidence_ids": [],
                        "distinguishing_gap_ids": [],
                        "challenge_explanation": (
                            "The fake client leaves competition unresolved so the "
                            "strict Confirmation Gate cannot over-confirm."
                        ),
                        "status": "unresolved",
                    }
                    for candidate in request.payload.get("candidates", [])
                    if isinstance(candidate, dict)
                ],
                "analysis_summary": "The fake client preserves the adversarial boundary.",
            }
        elif request.prompt_name == "rca_reasoning":
            data = {
                "ranked_candidates": list(request.payload["candidate_catalog"]),
                "analysis_summary": request.payload["deterministic_rationale"],
            }
        elif request.prompt_name == "improvement":
            data = {
                "engineering_summary": (
                    f"{request.payload['incident_summary']} "
                    f"{request.payload['fab_level_summary']}"
                ),
                "recommendation_ids": list(request.payload["recommendation_ids"]),
                "evidence_ids": list(request.payload["evidence_ids"]),
            }
        else:
            raise LLMOutputValidationError(f"unsupported fake prompt: {request.prompt_name}")
        serialized_input = json.dumps(request.payload, ensure_ascii=False)
        serialized_output = json.dumps(data, ensure_ascii=False)
        prompt_tokens = max(1, len(serialized_input) // 4)
        completion_tokens = max(1, len(serialized_output) // 4)
        usage = LLMUsageEvent(
            call_id=f"FAKE_{uuid4().hex.upper()}",
            agent=request.agent,
            provider=self.provider,
            model=self.model,
            prompt_version=request.prompt_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round((perf_counter() - started) * 1000.0, 3),
        )
        _record_usage(usage)
        return LLMResponse(data=data, usage=usage)


def build_llm_client(settings: LLMSettings) -> LLMClient | None:
    if settings.agent_mode == AgentMode.DETERMINISTIC.value:
        return None
    if settings.agent_mode == AgentMode.FAKE.value:
        return FakeLLMClient()
    return DashScopeLLMClient(settings)
