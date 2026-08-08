import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
import pytest
from pydantic_ai import ModelRetry, UnexpectedModelBehavior, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    ToolCallPart,
    ToolReturnPart,
)
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
    fitness: str = "FINAL",
    escalation_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "summary": f"조사 결과는 {decision}입니다.",
        "evidence": evidence,
        "search_queries": _검색_계획(),
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
        "evidence_reviews": [
            {
                "url": item["url"],
                "fitness": fitness,
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
        "escalation_reason": escalation_reason,
    }


def _검색_계획() -> list[dict[str, object]]:
    return [
        {"category": "OFFICIAL", "query": "사건 공식 결과", "target_domains": ["example.com"]},
        {"category": "CURRENT", "query": "사건 기준일 최신 상태", "target_domains": []},
        {"category": "SUPPORTS_YES", "query": "사건 발생 확인", "target_domains": []},
        {"category": "SUPPORTS_NO", "query": "사건 미발생 반증", "target_domains": []},
    ]


def _검색_후보들(
    count: int,
    *,
    discovered_by: str = "OFFICIAL",
) -> list[dict[str, str]]:
    return [
        {
            "url": (
                "https://example.com/official"
                if index == 1
                else f"https://candidate-{index}.example/result"
            ),
            "title": f"후보 {index}",
            "source_domain": "example.com",
            "discovered_by": discovered_by,
            "preliminary_authority": "official",
        }
        for index in range(1, count + 1)
    ]


def _공식_증거_출력(direction: Direction, *, fitness: str = "FINAL") -> dict[str, Any]:
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
        fitness=fitness,
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
    *,
    message_history: list[Any] | None = None,
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
        result = asyncio.run(
            _agent.run(
                "조사 결과를 제출하세요.",
                deps=investigation,
                message_history=message_history or _출력_조사_기록(outputs[-1]),
            )
        )
    return result.output, schemas, calls


def _조사_기록(
    *,
    query: str,
    source_url: str,
    fetch_error: bool = False,
    fetched_url: str | None = None,
    fetch: bool = True,
    search_return_id: str | None = None,
) -> list[Any]:
    search_id = f"search-{query}"
    fetch_id = f"fetch-{fetched_url or source_url}"
    fetch_url = fetched_url or source_url
    response_parts: list[Any] = [
        NativeToolCallPart(
            tool_name="web_search",
            tool_call_id=search_id,
            args={"query": query},
        ),
        NativeToolReturnPart(
            tool_name="web_search",
            tool_call_id=search_return_id or search_id,
            content={"status": "completed", "sources": [{"url": source_url}]},
        ),
    ]
    if fetch:
        response_parts.append(
            ToolCallPart(
                tool_name="web_fetch",
                tool_call_id=fetch_id,
                args={"url": fetch_url},
            )
        )
    fetch_content = (
        {"url": fetch_url, "error": "조회 실패"}
        if fetch_error
        else {"url": fetch_url, "content": "확정 결과"}
    )
    request_parts = (
        [
            ToolReturnPart(
                tool_name="web_fetch",
                tool_call_id=fetch_id,
                content=fetch_content,
            )
        ]
        if fetch
        else []
    )
    return [ModelResponse(parts=response_parts), ModelRequest(parts=request_parts)]


def _출력_조사_기록(output: dict[str, Any]) -> list[Any]:
    candidates = output["search_candidates"]
    return [
        message
        for index, query in enumerate(output["search_queries"])
        for message in _조사_기록(
            query=query["query"],
            source_url=candidates[index % len(candidates)]["url"],
        )
    ]


def _모든_후보_조사_기록(output: dict[str, Any]) -> list[Any]:
    return _출력_조사_기록(output) + [
        message
        for candidate in output["search_candidates"][4:]
        for message in _조사_기록(
            query=_검색_계획()[0]["query"],
            source_url=candidate["url"],
        )
    ]


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


class Test검색계획검토:
    def test_필수_검색_범주가_빠지면_보완_조사한다(self):
        output = _공식_증거_출력("YES")
        output["search_queries"] = output["search_queries"][:-1]

        result, _, calls = _run_outputs(_입력(), [output, _공식_증거_출력("YES")])

        assert result.decision == "YES"
        assert calls == 2

    def test_같은_검색어를_네_범주에_복사하면_보완_조사한다(self):
        invalid = _공식_증거_출력("YES")
        for query in invalid["search_queries"]:
            query["query"] = "사건 결과"

        result, _, calls = _run_outputs(_입력(), [invalid, _공식_증거_출력("YES")])

        assert result.decision == "YES"
        assert calls == 2


class Test조사지시문:
    @pytest.mark.parametrize(
        "required",
        ["OFFICIAL", "CURRENT", "SUPPORTS_YES", "SUPPORTS_NO", "원문", "최대 5개", "FINAL"],
        ids=["OFFICIAL", "CURRENT", "SUPPORTS_YES", "SUPPORTS_NO", "원문", "최대_5개", "FINAL"],
    )
    def test_조사지시문에_필수_검색과_원문검증_규칙이_있다(self, required: str):
        assert required in resolver.INVESTIGATION_INSTRUCTIONS


class Test후보조회한도:
    def test_검색으로_발견한_후보가_다섯개를_넘으면_보완_조사한다(self):
        output = _공식_증거_출력("YES")
        output["search_candidates"] = _검색_후보들(6)

        result, _, calls = _run_outputs(
            _입력(),
            [output, _공식_증거_출력("YES")],
            message_history=_모든_후보_조사_기록(output),
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_시장_지정_공식_후보는_다섯개_한도에서_제외한다(self):
        candidates = _검색_후보들(
            6,
            discovered_by="MARKET_OFFICIAL_SOURCE",
        )
        output = _출력(
            "YES",
            [
                _증거(
                    "YES",
                    url=candidate["url"],
                    authority="official",
                    publisher=f"공식 기관 {index}",
                )
                for index, candidate in enumerate(candidates, start=1)
            ],
        )
        output["search_candidates"] = candidates

        result, _, calls = _run_outputs(
            _입력(official_sources=[candidate["url"] for candidate in candidates]),
            [output],
            message_history=_모든_후보_조사_기록(output),
        )

        assert result.decision == "YES"
        assert calls == 1

    def test_시장_지정되지_않은_후보는_공식_표시여도_한도에_포함한다(self):
        output = _공식_증거_출력("YES")
        output["search_candidates"] = _검색_후보들(
            7,
            discovered_by="MARKET_OFFICIAL_SOURCE",
        )

        result, _, calls = _run_outputs(
            _입력(),
            [output, _공식_증거_출력("YES")],
            message_history=_모든_후보_조사_기록(output),
        )

        assert result.decision == "YES"
        assert calls == 2


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
    def test_yes_no_자기검토_필수항목이_거짓이면_보완_조사한다(self, field: str):
        incomplete = _공식_증거_출력("YES")
        incomplete["self_review"][field] = False

        result, _, calls = _run_outputs(
            _입력(),
            [incomplete, _공식_증거_출력("YES")],
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_yes_no_자기검토에_남은_조사가_있으면_보완_조사한다(self):
        incomplete = _공식_증거_출력("YES")
        incomplete["self_review"]["missing_research"] = ["기준일 확인"]

        result, _, calls = _run_outputs(
            _입력(),
            [incomplete, _공식_증거_출력("YES")],
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_이관은_구체적_사유가_있으면_불완전한_자기검토를_허용한다(self):
        output = _공식_증거_출력("YES")
        output["decision"] = "ESCALATED"
        output["self_review"]["criteria_clear"] = False
        output["escalation_reason"] = "시장 기준일 해석이 모호합니다."

        result, _, calls = _run_outputs(_입력(), [output])

        assert result.decision == "ESCALATED"
        assert result.escalation_reason == "시장 기준일 해석이 모호합니다."
        assert calls == 1


class Test증거적합성검토:
    def test_공식_출처라도_시점이_불명확하면_자동_승인하지_않는다(self):
        stale = _공식_증거_출력("YES", fitness="STALE_OR_UNDATED")

        result, _, calls = _run_outputs(_입력(), [stale] * 4)

        assert result.decision == "ESCALATED"
        assert "FINAL" in (result.escalation_reason or "")
        assert calls == 4

    @pytest.mark.parametrize(
        "fitness",
        ["PRELIMINARY", "FORECAST"],
        ids=["PRELIMINARY", "FORECAST"],
    )
    def test_확정되지_않은_자료이면_자동_승인하지_않는다(self, fitness: str):
        output = _공식_증거_출력("YES", fitness=fitness)

        result, _, _ = _run_outputs(_입력(), [output] * 4)

        assert result.decision == "ESCALATED"


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
        assert "현재 1개" in (result.escalation_reason or "")
        assert "2개 필요" in (result.escalation_reason or "")
        assert calls == 4

    def test_같은_원출처의_재게시_두개는_독립_증거_하나로_계산한다(self):
        duplicate = _고신뢰_증거_출력("YES", ["원기관", "원기관"])

        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [duplicate] * 4,
        )

        assert result.decision == "ESCALATED"
        assert calls == 4

    def test_공식_url을_세번_보완해도_확인하지_못하면_이관한다(self):
        missing_official = _고신뢰_증거_출력("YES", ["기관 A", "기관 B"])

        result, _, calls = _run_outputs(_입력(), [missing_official] * 4)

        assert result.decision == "ESCALATED"
        assert "공식 URL" in (result.escalation_reason or "")
        assert calls == 4

    def test_잘못된_output을_세번_수정하지_못하면_예외를_전달한다(self):
        calls = 0

        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {})],
            )

        with _agent.override(model=FunctionModel(model_function), native_tools=[]):
            with pytest.raises(UnexpectedModelBehavior):
                asyncio.run(_agent.run("조사 결과를 제출하세요.", deps=_입력()))

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


class Test조사실행기록:
    def test_실행하지_않은_검색어를_제출하면_보완_조사한다(self):
        unexecuted = _공식_증거_출력("YES")
        unexecuted["search_queries"][0]["query"] = "실행하지 않은 검색어"

        result, _, calls = _run_outputs(
            _입력(),
            [unexecuted, _공식_증거_출력("YES")],
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_검색결과에_없는_후보_url이면_보완_조사한다(self):
        undiscovered = _공식_증거_출력("YES")
        undiscovered["search_candidates"][0]["url"] = "https://unseen.example/result"

        result, _, calls = _run_outputs(
            _입력(),
            [undiscovered, _공식_증거_출력("YES")],
        )

        assert result.decision == "YES"
        assert calls == 2

    def test_원문을_조회하지_않은_evidence이면_보완_조사한다(self):
        output = _공식_증거_출력("YES")
        history = [
            message
            for query in _검색_계획()
            for message in _조사_기록(
                query=query["query"],
                source_url="https://example.com/official",
                fetch=False,
            )
        ]

        result, _, calls = _run_outputs(
            _입력(),
            [output] * 4,
            message_history=history,
        )

        assert result.decision == "ESCALATED"
        assert "원문" in (result.escalation_reason or "")
        assert calls == 4

    def test_원문_조회에_실패한_url을_결론_근거로_쓰면_보완_조사한다(self):
        output = _공식_증거_출력("YES")
        history = [
            message
            for query in _검색_계획()
            for message in _조사_기록(
                query=query["query"],
                source_url="https://example.com/official",
                fetch_error=True,
            )
        ]

        result, _, calls = _run_outputs(
            _입력(),
            [output] * 4,
            message_history=history,
        )

        assert result.decision == "ESCALATED"
        assert "실패" in (result.escalation_reason or "")
        assert calls == 4

    def test_시장_지정_공식_url은_검색결과에_없어도_후보로_인정한다(self):
        output = _공식_증거_출력("YES")
        history = [
            message
            for query in _검색_계획()
            for message in _조사_기록(
                query=query["query"],
                source_url="https://search.example/result",
                fetched_url="https://example.com/official",
            )
        ]

        result, _, calls = _run_outputs(_입력(), [output], message_history=history)

        assert result.decision == "YES"
        assert calls == 1

    def test_조회에_실패한_inconclusive_증거와_성공한_공식_증거가_있으면_yes를_승인한다(
        self,
    ):
        failed_url = "https://failed.example/result"
        confirmed_url = "https://confirmed.example/result"
        output = _출력(
            "YES",
            [
                _증거(
                    "INCONCLUSIVE",
                    url=failed_url,
                    authority="official",
                    publisher="실패한 공식 기관",
                ),
                _증거(
                    "YES",
                    url=confirmed_url,
                    authority="official",
                    publisher="확정한 공식 기관",
                ),
            ],
        )
        output["evidence_reviews"][0]["fitness"] = "INCONCLUSIVE"
        history = [
            message
            for index, query in enumerate(_검색_계획())
            for message in _조사_기록(
                query=query["query"],
                source_url=failed_url if index == 0 else confirmed_url,
                fetch_error=index == 0,
            )
        ]

        result, _, calls = _run_outputs(
            _입력(official_sources=[failed_url, confirmed_url]),
            [output],
            message_history=history,
        )

        assert result.decision == "YES"
        assert calls == 1

    def test_연결되지_않은_native_검색반환의_후보는_보완_조사한다(self):
        output = _고신뢰_증거_출력("YES", ["기관 A", "기관 B"])
        history = [
            message
            for index, query in enumerate(_검색_계획())
            for message in _조사_기록(
                query=query["query"],
                source_url=output["search_candidates"][index % 2]["url"],
                search_return_id=f"unmatched-{index}",
            )
        ]

        result, _, calls = _run_outputs(
            _입력(official_sources=[]),
            [output] * 4,
            message_history=history,
        )

        assert result.decision == "ESCALATED"
        assert "검색 결과" in (result.escalation_reason or "")
        assert calls == 4


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

    def test_agent_override가_조사기록없이_출력하면_자동판정하지_않는다(self):
        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        _공식_증거_출력("YES"),
                    )
                ],
            )

        with _agent.override(model=FunctionModel(model_function), native_tools=[]):
            result = asyncio.run(resolver.resolve(_입력()))

        assert result.decision == "ESCALATED"
        assert "검색어" in (result.escalation_reason or "")


class Test도구재시도:
    def test_도구가_계속_실패하면_최초_포함_세번_시도하고_예외를_전달한다(self):
        calls = 0

        def always_fails() -> str:
            nonlocal calls
            calls += 1
            raise ModelRetry("일시적인 도구 실패")

        def model_function(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("always_fails", {})])

        with _agent.override(
            model=FunctionModel(model_function),
            tools=[always_fails],
            native_tools=[],
        ):
            with pytest.raises(UnexpectedModelBehavior):
                asyncio.run(_agent.run("도구를 호출하세요.", deps=_입력()))

        assert calls == 3


class Test웹조회실패:
    def test_페이지를_계속_조회하지_못하면_세번_시도하고_실패_정보를_반환한다(
        self,
        monkeypatch,
    ):
        attempts = 0

        async def always_fails(_url: str):
            nonlocal attempts
            attempts += 1
            raise ModelRetry("페이지에 접근할 수 없습니다")

        monkeypatch.setattr(resolver._raw_web_fetch_tool, "function", always_fails)

        result = asyncio.run(resolver._fetch_web_page("https://unavailable.example"))

        assert result["url"] == "https://unavailable.example"
        assert "접근할 수 없습니다" in result["error"]
        assert attempts == 3


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

    def test_재시도_가능한_http_오류는_최초_포함_세번_시도한다(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503, request=request)

        async def no_sleep(_seconds: float) -> None:
            return None

        async def request() -> None:
            transport = resolver._retrying_transport(httpx.MockTransport(handler))
            transport.config["sleep"] = no_sleep
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(httpx.HTTPStatusError):
                    await client.get("https://api.openai.com/test")

        asyncio.run(request())

        assert attempts == 3
