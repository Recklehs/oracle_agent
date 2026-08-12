import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle_agent.agents import resolver
from oracle_agent.agents.resolver import JudgeDeps, _agent
from oracle_agent.agents.searcher import EvidenceBundle, EvidenceFitness
from oracle_agent.models import InvestigationInput, InvestigationResult


models.ALLOW_MODEL_REQUESTS = False

Direction = Literal["YES", "NO"]


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
    original_publisher: str | None = None,
) -> dict[str, str]:
    return {
        "url": url,
        "title": f"{publisher} 발표",
        "publisher": publisher,
        "original_publisher": original_publisher or publisher,
        "authority": authority,
        "supports": direction,
        "finding": f"{direction}를 지지하는 사실을 확인했습니다.",
    }


def _번들(
    evidence: list[dict[str, str]],
    *,
    fitness: str = "FINAL",
    self_review: dict[str, Any] | None = None,
) -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "summary": "조사 결과를 요약했습니다.",
            "search_queries": [
                {"category": "OFFICIAL", "query": "사건 공식 결과"},
                {"category": "CURRENT", "query": "사건 기준일 최신 상태"},
                {"category": "SUPPORTS_YES", "query": "사건 발생 확인"},
                {"category": "SUPPORTS_NO", "query": "사건 미발생 반증"},
            ],
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
                    "fitness": fitness,
                    "reason": "종료 조건을 확인했습니다.",
                }
                for item in evidence
            ],
            "self_review": self_review
            or {
                "criteria_clear": True,
                "result_period_complete": True,
                "findings_match_sources": True,
                "duplicate_publishers_checked": True,
                "contradiction_search_complete": True,
            },
        }
    )


def _공식_증거_번들(direction: Direction, **kwargs: Any) -> EvidenceBundle:
    return _번들(
        [
            _증거(
                direction,
                url="https://example.com/official",
                authority="official",
                publisher="공식 기관",
            )
        ],
        **kwargs,
    )


def _고신뢰_증거_번들(direction: Direction, publishers: list[str]) -> EvidenceBundle:
    return _번들(
        [
            _증거(
                direction,
                url=f"https://source-{index}.example/report",
                authority="high_trust",
                publisher=f"보도사 {index}",
                original_publisher=publisher,
            )
            for index, publisher in enumerate(publishers, start=1)
        ]
    )


def _충돌_증거_번들() -> EvidenceBundle:
    return _번들(
        [
            _증거(
                "YES",
                url="https://yes.example/result",
                authority="official",
                publisher="YES 공식 기관",
            ),
            _증거(
                "NO",
                url="https://no.example/result",
                authority="high_trust",
                publisher="NO 고신뢰 기관",
            ),
        ]
    )


def _판정(
    decision: Literal["YES", "NO", "ESCALATED"],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "summary": f"조사 결과는 {decision}입니다.",
        "escalation_reason": reason,
    }


def _판정_실행(
    investigation: InvestigationInput,
    bundle: EvidenceBundle,
    outputs: list[dict[str, Any]],
) -> tuple[InvestigationResult, list[dict[str, Any]], int]:
    schemas: list[dict[str, Any]] = []
    calls = 0

    def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        schemas.append(info.output_tools[0].parameters_json_schema)
        output = outputs[min(calls, len(outputs) - 1)]
        calls += 1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, output)])

    with _agent.override(model=FunctionModel(model_function)):
        result = asyncio.run(
            _agent.run(
                "판정을 제출하세요.",
                deps=JudgeDeps(investigation=investigation, bundle=bundle),
            )
        )
    return result.output, schemas, calls


class Test판정출력경계:
    def test_모델은_증거를_제출할_수_없고_결과에는_번들_증거를_붙인다(self):
        bundle = _공식_증거_번들("YES")
        result, schemas, _ = _판정_실행(_입력(), bundle, [_판정("YES")])

        assert "evidence" not in schemas[0]["properties"]
        assert "prediction_id" not in schemas[0]["properties"]
        assert result.prediction_id == "prediction-1"
        assert result.evidence == bundle.evidence


class Test자동판정승인:
    def test_공식_출처가_yes를_명시하면_yes를_승인한다(self):
        result, _, calls = _판정_실행(_입력(), _공식_증거_번들("YES"), [_판정("YES")])

        assert result.decision == "YES"
        assert calls == 1

    def test_독립적인_고신뢰_원출처_두개가_no를_지지하면_no를_승인한다(self):
        result, _, calls = _판정_실행(
            _입력(official_sources=[]),
            _고신뢰_증거_번들("NO", ["기관 A", "기관 B"]),
            [_판정("NO")],
        )

        assert result.decision == "NO"
        assert calls == 1


class Test자기검토:
    @pytest.mark.parametrize(
        "field",
        [
            "criteria_clear",
            "result_period_complete",
            "findings_match_sources",
            "duplicate_publishers_checked",
            "contradiction_search_complete",
        ],
        ids=[
            "기준_명확성",
            "결과_기간_완료",
            "근거_일치",
            "원출처_중복",
            "반증_검색",
        ],
    )
    def test_자기검토_필수항목이_거짓이면_yes_no를_승인하지_않는다(self, field: str):
        bundle = _공식_증거_번들("YES")
        setattr(bundle.self_review, field, False)

        result, _, calls = _판정_실행(_입력(), bundle, [_판정("YES")] * 4)

        assert result.decision == "ESCALATED"
        assert "자기검토" in (result.escalation_reason or "")
        assert calls == 4

    def test_자기검토에_남은_조사가_있으면_yes_no를_승인하지_않는다(self):
        bundle = _공식_증거_번들("YES")
        bundle.self_review.missing_research = ["기준일 확인"]

        result, _, calls = _판정_실행(_입력(), bundle, [_판정("YES")] * 4)

        assert result.decision == "ESCALATED"
        assert "자기검토" in (result.escalation_reason or "")
        assert calls == 4

    def test_이관은_구체적_사유가_있으면_불완전한_자기검토를_허용한다(self):
        bundle = _공식_증거_번들("YES")
        bundle.self_review.criteria_clear = False

        result, _, calls = _판정_실행(
            _입력(),
            bundle,
            [_판정("ESCALATED", reason="시장 기준일 해석이 모호합니다.")],
        )

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == "시장 기준일 해석이 모호합니다."
        assert calls == 1


class Test증거적합성검토:
    def test_공식_출처라도_시점이_불명확하면_자동_승인하지_않는다(self):
        bundle = _공식_증거_번들("YES", fitness="STALE_OR_UNDATED")

        result, _, calls = _판정_실행(_입력(), bundle, [_판정("YES")] * 4)

        assert result.decision == "ESCALATED"
        assert "FINAL" in (result.escalation_reason or "")
        assert calls == 4

    @pytest.mark.parametrize(
        "fitness",
        ["PRELIMINARY", "FORECAST"],
        ids=["PRELIMINARY", "FORECAST"],
    )
    def test_확정되지_않은_자료이면_자동_승인하지_않는다(self, fitness: str):
        bundle = _공식_증거_번들("YES", fitness=fitness)

        result, _, _ = _판정_실행(_입력(), bundle, [_판정("YES")] * 4)

        assert result.decision == "ESCALATED"


class Test판정보완:
    def test_증거가_부족하면_보완_후_증거와_함께_이관한다(self):
        bundle = _고신뢰_증거_번들("YES", ["기관 A"])

        result, _, calls = _판정_실행(
            _입력(official_sources=[]),
            bundle,
            [_판정("YES")] * 4,
        )

        assert result.decision == "ESCALATED"
        assert result.evidence
        assert "현재 1개" in (result.escalation_reason or "")
        assert "2개 필요" in (result.escalation_reason or "")
        assert calls == 4

    def test_같은_원출처의_재게시_두개는_독립_증거_하나로_계산한다(self):
        bundle = _고신뢰_증거_번들("YES", ["원기관", "원기관"])

        result, _, calls = _판정_실행(
            _입력(official_sources=[]),
            bundle,
            [_판정("YES")] * 4,
        )

        assert result.decision == "ESCALATED"
        assert calls == 4

    def test_사유_없는_이관은_보완_후_구체적_사유를_받아들인다(self):
        result, _, calls = _판정_실행(
            _입력(),
            _공식_증거_번들("YES", fitness="PRELIMINARY"),
            [
                _판정("ESCALATED"),
                _판정("ESCALATED", reason="잠정 집계만 확인했습니다."),
            ],
        )

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == "잠정 집계만 확인했습니다."
        assert calls == 2

    def test_사유_없는_이관을_세번_보완하지_못하면_기본_사유로_이관한다(self):
        result, _, calls = _판정_실행(
            _입력(),
            _공식_증거_번들("YES", fitness="PRELIMINARY"),
            [_판정("ESCALATED")] * 4,
        )

        assert result.decision == "ESCALATED"
        assert "이관 사유" in (result.escalation_reason or "")
        assert calls == 4


class Test사람검토이관:
    def test_권위_있는_출처가_yes와_no로_충돌하면_즉시_이관한다(self):
        result, _, calls = _판정_실행(
            _입력(official_sources=[]),
            _충돌_증거_번들(),
            [_판정("YES")],
        )

        assert result.decision == "ESCALATED"
        assert "충돌" in (result.escalation_reason or "")
        assert calls == 1

    @pytest.mark.parametrize(
        "fitness",
        ["STALE_OR_UNDATED", "INCONCLUSIVE", "FORECAST"],
        ids=["낡은_자료", "확인_실패", "예상치"],
    )
    def test_확정되지_않은_적합성의_반대증거는_충돌로_보지_않는다(self, fitness: str):
        bundle = _번들(
            [
                _증거(
                    "NO",
                    url="https://no.example/result",
                    authority="official",
                    publisher="확정 공식 기관",
                ),
                _증거(
                    "YES",
                    url="https://yes.example/stale-profile",
                    authority="official",
                    publisher="미확정 공식 자료",
                ),
            ]
        )
        bundle.evidence_reviews[1].fitness = EvidenceFitness(fitness)

        result, _, calls = _판정_실행(
            _입력(official_sources=[]), bundle, [_판정("NO")]
        )

        assert result.decision == "NO"
        assert calls == 1

    def test_preliminary_적합성의_반대증거와_충돌하면_이관한다(self):
        bundle = _충돌_증거_번들()
        bundle.evidence_reviews[1].fitness = EvidenceFitness.PRELIMINARY

        result, _, calls = _판정_실행(
            _입력(official_sources=[]), bundle, [_판정("YES")]
        )

        assert result.decision == "ESCALATED"
        assert "충돌" in (result.escalation_reason or "")
        assert calls == 1


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

        with _agent.override(model=FunctionModel(model_function)):
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

        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, _판정("YES"))]
            )

        caplog.set_level(logging.INFO, logger=resolver.__name__)
        with _agent.override(model=FunctionModel(model_function)):
            result = asyncio.run(resolver.resolve(_입력()))

        assert result.decision == "YES"
        assert result.evidence == bundle.evidence
        assert captured_backends == [sentinel]
        assert "agent usage" in caplog.text

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

        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, _판정("YES"))]
            )

        with _agent.override(model=FunctionModel(model_function)):
            result = asyncio.run(
                resolver.resolve(_입력(), search_backend=injected)
            )

        assert result.decision == "YES"
        assert captured_backends == [injected]
