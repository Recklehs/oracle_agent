import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle_agent.agents import resolver
from oracle_agent.agents.resolver import _agent
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
    direction: Direction,
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


def _출력(
    decision: Literal["YES", "NO", "ESCALATED"],
    evidence: list[dict[str, str]],
    *,
    escalation_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "summary": f"조사 결과는 {decision}입니다.",
        "evidence": evidence,
        "escalation_reason": escalation_reason,
    }


def _공식_증거_출력(direction: Direction) -> dict[str, Any]:
    return _출력(
        direction,
        [
            _증거(
                direction,
                url="https://example.com/official",
                authority="official",
                publisher="공식 기관",
            )
        ],
    )


def _고신뢰_증거_출력(direction: Direction, publishers: list[str]) -> dict[str, Any]:
    return _출력(
        direction,
        [
            _증거(
                direction,
                url=f"https://source-{index}.example/report",
                authority="high_trust",
                publisher=f"보도사 {index}",
                original_publisher=publisher,
            )
            for index, publisher in enumerate(publishers, start=1)
        ],
    )


def _충돌_증거_출력() -> dict[str, Any]:
    return _출력(
        "YES",
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
        ],
    )


def _run_outputs(
    investigation: InvestigationInput,
    outputs: list[dict[str, Any]],
) -> tuple[InvestigationResult, list[dict[str, Any]], int]:
    schemas: list[dict[str, Any]] = []
    calls = 0

    def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        schemas.append(info.output_tools[0].parameters_json_schema)
        output = outputs[min(calls, len(outputs) - 1)]
        calls += 1
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, output)],
        )

    with _agent.override(model=FunctionModel(model_function), native_tools=[]):
        result = asyncio.run(_agent.run("조사 결과를 제출하세요.", deps=investigation))
    return result.output, schemas, calls


class Test최종출력경계:
    def test_모델_output_schema에는_prediction_id가_없고_결과에는_입력값을_붙인다(self):
        result, schemas, _ = _run_outputs(_입력(), [_공식_증거_출력("YES")])

        assert "prediction_id" not in schemas[0]["properties"]
        assert result.prediction_id == "prediction-1"


class Test자동판정승인:
    def test_공식_출처가_yes를_명시하면_yes를_승인한다(self):
        result, _, calls = _run_outputs(_입력(), [_공식_증거_출력("YES")])

        assert result.decision == "YES"
        assert calls == 1

    def test_독립적인_고신뢰_원출처_두개가_no를_지지하면_no를_승인한다(self):
        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [_고신뢰_증거_출력("NO", ["기관 A", "기관 B"])],
        )

        assert result.decision == "NO"
        assert calls == 1


class Test추가조사:
    def test_지정_공식_url을_누락하면_보완_조사한다(self):
        result, _, calls = _run_outputs(
            _입력(),
            [
                _고신뢰_증거_출력("YES", ["기관 A", "기관 B"]),
                _공식_증거_출력("YES"),
            ],
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_세번_보완해도_증거가_부족하면_증거와_함께_이관한다(self):
        insufficient = _고신뢰_증거_출력("YES", ["기관 A"])

        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [insufficient] * 4,
        )

        assert result.decision == "ESCALATED"
        assert result.evidence
        assert calls == 4

    def test_같은_원출처의_재게시_두개는_독립_증거_하나로_계산한다(self):
        duplicate = _고신뢰_증거_출력("YES", ["원기관", "원기관"])

        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [duplicate] * 4,
        )

        assert result.decision == "ESCALATED"
        assert calls == 4


class Test사람검토이관:
    def test_권위_있는_출처가_yes와_no로_충돌하면_즉시_이관한다(self):
        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [_충돌_증거_출력()],
        )

        assert result.decision == "ESCALATED"
        assert "충돌" in (result.escalation_reason or "")
        assert calls == 1


class TestResolver실행:
    def test_판정_가능_시점_전이면_모델을_호출하지_않고_이관한다(self, monkeypatch):
        def 모델을_만들면_실패한다():
            raise AssertionError("모델을 호출하면 안 됩니다")

        monkeypatch.setattr(resolver, "_production_model", 모델을_만들면_실패한다)

        result = asyncio.run(
            resolver.resolve(
                _입력(resolve_after=datetime.now(UTC) + timedelta(days=1)),
            )
        )

        assert result.decision == "ESCALATED"
        assert result.evidence == []
        assert result.prediction_id == "prediction-1"


def _http_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        return error
    raise AssertionError("HTTP 오류 응답이 필요합니다")


class TestProviderHttp재시도:
    def test_timeout과_지정된_http_상태만_재시도한다(self):
        timeout = httpx.ReadTimeout("timeout")
        rate_limit = httpx.Response(
            429,
            request=httpx.Request("GET", "https://api.openai.com"),
        )
        bad_request = httpx.Response(
            400,
            request=httpx.Request("GET", "https://api.openai.com"),
        )

        assert resolver._is_retryable_http_error(timeout)
        assert resolver._is_retryable_http_error(_http_status_error(rate_limit))
        assert not resolver._is_retryable_http_error(_http_status_error(bad_request))
