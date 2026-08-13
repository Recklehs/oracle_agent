import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle_agent.agents import resolver
from oracle_agent.agents.resolver import JudgeDeps, _agent
from oracle_agent.agents.searcher import EvidenceBundle
from oracle_agent.models import InvestigationInput, InvestigationResult


models.ALLOW_MODEL_REQUESTS = False

Direction = Literal["YES", "NO"]

# FunctionModel은 기본적으로 native structured output을 지원하지 않는다고 표시되므로
# 판정 Agent와 같은 출력 방식으로 테스트하도록 profile을 켠다.
NATIVE_OUTPUT_PROFILE = {"supports_json_schema_output": True}


def _입력(
    *,
    official_sources: list[str] | None = None,
    resolve_after: datetime | None = None,
) -> InvestigationInput:
    return InvestigationInput(
        prediction_id="prediction-1",
        prediction="사건이 기준일까지 발생한다",
        resolution_criteria="공식 발표가 기준일까지 게시되면 YES, 아니면 NO",
        resolve_after=resolve_after or datetime(2025, 1, 2, tzinfo=UTC),
        official_sources=(
            ["https://example.com/official/"] if official_sources is None else official_sources
        ),
    )


def _증거(
    direction: Literal["YES", "NO", "INCONCLUSIVE"],
    *,
    url: str,
    authority: Literal["official", "high_trust", "other"],
    publisher: str,
) -> dict[str, str]:
    return {
        "url": url,
        "title": f"{publisher} 발표",
        "publisher": publisher,
        "original_publisher": publisher,
        "authority": authority,
        "supports": direction,
        "finding": f"{direction}를 지지하는 사실을 확인했습니다.",
    }


def _번들(evidence: list[dict[str, str]]) -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "summary": "조사 결과를 요약했습니다.",
            "search_queries": [{"query": "사건 공식 결과"}],
            "search_candidates": [
                {
                    "url": item["url"],
                    "title": item["title"],
                    "source_domain": "example.com",
                    "discovered_by": "MARKET_OFFICIAL_SOURCE",
                    "preliminary_authority": item["authority"],
                }
                for item in evidence
            ],
            "evidence": evidence,
            "evidence_reviews": [
                {
                    "url": item["url"],
                    "fitness": "FINAL",
                    "reason": "종료 조건을 확인했습니다.",
                }
                for item in evidence
            ],
            "self_review": {
                "criteria_clear": True,
                "result_period_complete": True,
                "findings_match_sources": True,
                "duplicate_publishers_checked": True,
                "contradiction_search_complete": True,
            },
        }
    )


def _공식_증거_번들(direction: Direction) -> EvidenceBundle:
    return _번들(
        [
            _증거(
                direction,
                url="https://example.com/official",
                authority="official",
                publisher="공식 기관",
            )
        ]
    )


def _결과(
    bundle: EvidenceBundle,
    decision: Literal["YES", "NO", "ESCALATED"],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """판정 Agent가 native structured output으로 반환할 결과 DTO payload."""
    return {
        "prediction_id": "prediction-1",
        "decision": decision,
        "summary": f"조사 결과는 {decision}입니다.",
        "evidence": [dict(item) for item in bundle.model_dump(mode="json")["evidence"]],
        "escalation_reason": reason,
    }


def _판정_실행(
    investigation: InvestigationInput,
    bundle: EvidenceBundle,
    outputs: list[dict[str, Any]],
) -> tuple[InvestigationResult, list[str], int]:
    instructions: list[str] = []
    calls = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        instructions.append(info.instructions or "")
        output = outputs[min(calls, len(outputs) - 1)]
        calls += 1
        return ModelResponse(parts=[TextPart(json.dumps(output))])

    with _agent.override(
        model=FunctionModel(model_function, profile=NATIVE_OUTPUT_PROFILE)
    ):
        result = asyncio.run(
            _agent.run(
                "판정을 제출하세요.",
                deps=JudgeDeps(investigation=investigation, bundle=bundle),
            )
        )
    return result.output, instructions, calls


class Test판정컨텍스트:
    def test_판정_컨텍스트에_입력과_번들을_넣고_resolve_after는_제외한다(self):
        bundle = _공식_증거_번들("YES")
        _, instructions, _ = _판정_실행(_입력(), bundle, [_결과(bundle, "YES")])

        assert "prediction-1" in instructions[0]
        assert "사건이 기준일까지 발생한다" in instructions[0]
        assert "조사 Agent의 증거 번들" in instructions[0]
        assert "resolve_after" not in instructions[0]
        assert "2025-01-02" not in instructions[0]

    def test_종합_검토와_이관_우선_지침이_프롬프트에_포함된다(self):
        bundle = _공식_증거_번들("YES")
        _, instructions, _ = _판정_실행(_입력(), bundle, [_결과(bundle, "YES")])

        assert "종합적으로 검토" in instructions[0]
        assert "잘못된 자동 판정보다\nESCALATED를 우선한다" in instructions[0]
        assert "그대로 복사" in instructions[0]


class Test판정출력검증:
    def test_모델이_결과_dto_양식으로_제출하면_그대로_반환한다(self):
        bundle = _공식_증거_번들("YES")
        result, _, calls = _판정_실행(_입력(), bundle, [_결과(bundle, "YES")])

        assert result.decision == "YES"
        assert result.evidence == bundle.evidence
        assert result.escalation_reason is None
        assert calls == 1

    def test_이관_결과는_이관_사유와_함께_반환한다(self):
        bundle = _공식_증거_번들("YES")
        result, _, calls = _판정_실행(
            _입력(),
            bundle,
            [_결과(bundle, "ESCALATED", reason="증거가 부족합니다.")],
        )

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == "증거가 부족합니다."
        assert calls == 1

    def test_결론과_다른_방향의_증거만_있으면_재시도시킨다(self):
        bundle = _공식_증거_번들("NO")
        result, _, calls = _판정_실행(
            _입력(),
            bundle,
            [
                _결과(bundle, "YES"),
                _결과(bundle, "ESCALATED", reason="YES 방향 증거가 없습니다."),
            ],
        )

        assert result.decision == "ESCALATED"
        assert calls == 2

    def test_사유_없는_이관은_재시도시킨다(self):
        bundle = _공식_증거_번들("YES")
        result, _, calls = _판정_실행(
            _입력(),
            bundle,
            [
                _결과(bundle, "ESCALATED"),
                _결과(bundle, "ESCALATED", reason="잠정 집계만 확인했습니다."),
            ],
        )

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == "잠정 집계만 확인했습니다."
        assert calls == 2


class TestResolve실행:
    def test_판정_가능_시점_전이면_조사와_판정을_호출하지_않고_이관한다(
        self,
        monkeypatch,
    ):
        def 백엔드를_만들면_실패한다(*_args, **_kwargs):
            raise AssertionError("검색 backend를 만들면 안 됩니다")

        monkeypatch.setattr(
            resolver, "create_search_backend", 백엔드를_만들면_실패한다
        )

        result = asyncio.run(
            resolver.resolve(
                _입력(resolve_after=datetime.now(UTC) + timedelta(days=1)),
            )
        )

        assert result.decision == "ESCALATED"
        assert result.evidence == []
        assert result.prediction_id == "prediction-1"

    def test_조사_gate_실패면_판정_없이_실패_사유로_이관한다(self, monkeypatch):
        bundle = _공식_증거_번들("YES")
        bundle.gate_failure = "제출한 검색어를 실제로 모두 실행해야 합니다."

        async def 가짜조사(_investigation, _backend):
            return bundle

        monkeypatch.setattr(resolver, "investigate", 가짜조사)
        monkeypatch.setattr(
            resolver, "create_search_backend", lambda: SimpleNamespace(name="fake")
        )

        def model_function(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            raise AssertionError("판정 Agent를 호출하면 안 됩니다")

        with _agent.override(
            model=FunctionModel(model_function, profile=NATIVE_OUTPUT_PROFILE)
        ):
            result = asyncio.run(resolver.resolve(_입력()))

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == bundle.gate_failure
        assert result.summary == bundle.summary

    def test_정상_번들이면_판정_agent의_결과를_반환한다(self, monkeypatch, caplog):
        bundle = _공식_증거_번들("YES")
        captured_backends = []

        async def 가짜조사(_investigation, backend):
            captured_backends.append(backend)
            return bundle

        sentinel = SimpleNamespace(name="sentinel")
        monkeypatch.setattr(resolver, "investigate", 가짜조사)
        monkeypatch.setattr(resolver, "create_search_backend", lambda: sentinel)

        def model_function(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(json.dumps(_결과(bundle, "YES")))])

        caplog.set_level(logging.INFO, logger=resolver.__name__)
        with _agent.override(
            model=FunctionModel(model_function, profile=NATIVE_OUTPUT_PROFILE)
        ):
            result = asyncio.run(resolver.resolve(_입력()))

        assert result.decision == "YES"
        assert result.evidence == bundle.evidence
        assert captured_backends == [sentinel]
        assert "agent usage" in caplog.text

    def test_모델이_prediction_id를_잘못_적어도_입력값으로_바로잡는다(
        self, monkeypatch
    ):
        bundle = _공식_증거_번들("YES")

        async def 가짜조사(_investigation, _backend):
            return bundle

        monkeypatch.setattr(resolver, "investigate", 가짜조사)
        monkeypatch.setattr(
            resolver, "create_search_backend", lambda: SimpleNamespace(name="fake")
        )
        wrong = _결과(bundle, "YES") | {"prediction_id": "prediction-999"}

        def model_function(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(json.dumps(wrong))])

        with _agent.override(
            model=FunctionModel(model_function, profile=NATIVE_OUTPUT_PROFILE)
        ):
            result = asyncio.run(resolver.resolve(_입력()))

        assert result.prediction_id == "prediction-1"

    def test_인자로_받은_검색_backend를_기본_선택보다_우선한다(self, monkeypatch):
        bundle = _공식_증거_번들("YES")
        captured_backends = []

        async def 가짜조사(_investigation, backend):
            captured_backends.append(backend)
            return bundle

        def 백엔드를_만들면_실패한다(*_args, **_kwargs):
            raise AssertionError("기본 backend를 만들면 안 됩니다")

        monkeypatch.setattr(resolver, "investigate", 가짜조사)
        monkeypatch.setattr(
            resolver, "create_search_backend", 백엔드를_만들면_실패한다
        )
        injected = SimpleNamespace(name="injected")

        def model_function(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(json.dumps(_결과(bundle, "YES")))])

        with _agent.override(
            model=FunctionModel(model_function, profile=NATIVE_OUTPUT_PROFILE)
        ):
            result = asyncio.run(
                resolver.resolve(_입력(), search_backend=injected)
            )

        assert result.decision == "YES"
        assert captured_backends == [injected]
