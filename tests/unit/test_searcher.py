import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic_ai import BinaryContent, ModelRetry, UnexpectedModelBehavior, models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from oracle_agent.agents import searcher
from oracle_agent.agents.search_backends import SearchResult
from oracle_agent.agents.searcher import (
    EvidenceBundle,
    SearchDeps,
    SearchTrace,
    normalize_url,
)
from oracle_agent.models import InvestigationInput


models.ALLOW_MODEL_REQUESTS = False

Direction = Literal["YES", "NO"]


@pytest.fixture(autouse=True)
def exa_contents_차단(monkeypatch) -> list[str]:
    """단위 테스트가 실제 exa contents API를 호출하지 않게 기본 fallback을 무효화한다."""
    calls: list[str] = []

    async def fake_fetch(url: str, **_kwargs) -> None:
        calls.append(url)
        return None

    monkeypatch.setattr(searcher, "fetch_exa_contents", fake_fetch)
    return calls


class _가짜백엔드:
    name = "fake"

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def search(self, query: str, target_domains=()) -> list[SearchResult]:
        self.calls.append((query, tuple(target_domains)))
        return self.results


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


def _검색_계획() -> list[dict[str, object]]:
    return [
        {"query": "사건 공식 결과", "target_domains": ["example.com"]},
    ]


def _검색_후보들(
    count: int,
    *,
    discovered_by: str = "SEARCH",
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


def _조사출력(
    evidence: list[dict[str, str]],
    *,
    fitness: str = "FINAL",
) -> dict[str, Any]:
    return {
        "summary": "조사 결과를 요약했습니다.",
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
        "evidence": evidence,
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
    }


def _공식_증거_조사출력(direction: Direction, *, fitness: str = "FINAL") -> dict[str, Any]:
    return _조사출력(
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


def _고신뢰_증거_조사출력(direction: Direction, publishers: list[str]) -> dict[str, Any]:
    return _조사출력(
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


def _충돌_증거_조사출력() -> dict[str, Any]:
    return _조사출력(
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


def _기록(
    output: dict[str, Any],
    *,
    official_sources: list[str] = (),
    failed_urls: list[str] = (),
    extra_source_urls: list[str] = (),
    extra_fetched_urls: list[str] = (),
) -> SearchTrace:
    """제출된 출력과 일치하는 검색·조회 실행 기록을 만든다."""
    candidate_urls = {
        normalize_url(candidate["url"]) for candidate in output["search_candidates"]
    }
    official = {normalize_url(url) for url in official_sources}
    failed = {normalize_url(url) for url in failed_urls}
    fetched = (
        candidate_urls
        | official
        | {normalize_url(url) for url in extra_fetched_urls}
    ) - failed
    return SearchTrace(
        queries={query["query"].casefold() for query in output["search_queries"]},
        source_urls=candidate_urls | {normalize_url(url) for url in extra_source_urls},
        attempted_fetch_urls=fetched | failed,
        fetched_urls=fetched,
        failed_fetch_urls=failed,
    )


def _조사_실행(
    investigation: InvestigationInput,
    outputs: list[dict[str, Any]],
    trace: SearchTrace | None = None,
) -> tuple[EvidenceBundle, list[dict[str, Any]], int]:
    schemas: list[dict[str, Any]] = []
    calls = 0

    def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        schemas.append(info.output_tools[0].parameters_json_schema)
        output = outputs[min(calls, len(outputs) - 1)]
        calls += 1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, output)])

    deps = SearchDeps(investigation=investigation, backend=_가짜백엔드())
    deps.trace = trace or _기록(
        outputs[-1],
        official_sources=[str(url) for url in investigation.official_sources],
    )
    with searcher._agent.override(model=FunctionModel(model_function)):
        result = asyncio.run(searcher._agent.run("조사 결과를 제출하세요.", deps=deps))
    return result.output, schemas, calls


class Test조사출력경계:
    def test_모델_output_schema에는_gate_failure가_없다(self):
        bundle, schemas, _ = _조사_실행(_입력(), [_공식_증거_조사출력("YES")])

        assert "summary" in schemas[0]["properties"]
        assert "gate_failure" not in schemas[0]["properties"]
        assert bundle.gate_failure is None


class Test검색계획검토:
    def test_검색어를_하나도_제출하지_않으면_보완_조사한다(self):
        output = _공식_증거_조사출력("YES")
        output["search_queries"] = []

        bundle, _, calls = _조사_실행(_입력(), [output, _공식_증거_조사출력("YES")])

        assert bundle.gate_failure is None
        assert calls == 2

    def test_같은_검색어를_중복_제출하면_보완_조사한다(self):
        invalid = _공식_증거_조사출력("YES")
        invalid["search_queries"] = invalid["search_queries"] * 2

        bundle, _, calls = _조사_실행(_입력(), [invalid, _공식_증거_조사출력("YES")])

        assert bundle.gate_failure is None
        assert calls == 2

    def test_실행한_검색어를_기록에서_빠뜨리면_보완_조사한다(self):
        incomplete = _공식_증거_조사출력("YES")
        complete = _공식_증거_조사출력("YES")
        complete["search_queries"].append({"query": "사건 재검색", "target_domains": []})
        trace = _기록(complete, official_sources=["https://example.com/official/"])

        bundle, _, calls = _조사_실행(_입력(), [incomplete, complete], trace=trace)

        assert bundle.gate_failure is None
        assert calls == 2


class Test조사지시문:
    @pytest.mark.parametrize(
        "required",
        [
            "검색어 1개",
            "재검색",
            "글자 그대로",
            "원문",
            "최대 5개",
            "FINAL",
            "thin_content",
            "query",
            "render",
            "supports",
        ],
        ids=[
            "검색어_1개",
            "재검색",
            "글자_그대로",
            "원문",
            "최대_5개",
            "FINAL",
            "thin_content",
            "query_변형",
            "render",
            "supports_방향",
        ],
    )
    def test_조사지시문에_필수_검색과_원문검증_규칙이_있다(self, required: str):
        assert required in searcher.SEARCH_INSTRUCTIONS


class Test후보조회한도:
    def test_같은_검색_후보_url을_여섯번_제출해도_하나로_계산한다(self):
        output = _고신뢰_증거_조사출력("YES", ["기관 A", "기관 B"])
        first, second = output["search_candidates"]
        output["search_candidates"] = [first.copy() for _ in range(6)] + [second]

        bundle, _, calls = _조사_실행(_입력(official_sources=[]), [output])

        assert bundle.gate_failure is None
        assert calls == 1

    def test_검색_후보_원문을_다섯개_넘게_조회했으면_보완_없이_즉시_실패한다(self):
        output = _공식_증거_조사출력("YES")
        output["search_candidates"] = _검색_후보들(7)

        bundle, _, calls = _조사_실행(
            _입력(),
            [output],
            trace=_기록(output, official_sources=["https://example.com/official/"]),
        )

        assert bundle.gate_failure is not None
        assert "넘게" in bundle.gate_failure
        assert calls == 1

    def test_제출하지_않은_검색_후보_원문을_추가로_조회하면_보완_없이_실패한다(self):
        output = _공식_증거_조사출력("YES")
        trace = _기록(
            output,
            official_sources=["https://example.com/official/"],
            extra_fetched_urls=[
                f"https://hidden-{index}.example/result" for index in range(6)
            ],
        )

        bundle, _, calls = _조사_실행(_입력(), [output], trace=trace)

        assert bundle.gate_failure is not None
        assert "넘게" in bundle.gate_failure
        assert calls == 1

    def test_시장_지정_공식_후보는_다섯개_한도에서_제외한다(self):
        candidates = _검색_후보들(6, discovered_by="MARKET_OFFICIAL_SOURCE")
        output = _조사출력(
            [
                _증거(
                    "YES",
                    url=candidate["url"],
                    authority="official",
                    publisher=f"공식 기관 {index}",
                )
                for index, candidate in enumerate(candidates, start=1)
            ]
        )
        output["search_candidates"] = candidates

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[candidate["url"] for candidate in candidates]),
            [output],
        )

        assert bundle.gate_failure is None
        assert calls == 1

    def test_시장_지정되지_않은_후보는_공식_표시여도_한도에_포함한다(self):
        output = _공식_증거_조사출력("YES")
        output["search_candidates"] = _검색_후보들(
            7,
            discovered_by="MARKET_OFFICIAL_SOURCE",
        )

        bundle, _, calls = _조사_실행(
            _입력(),
            [output],
            trace=_기록(output, official_sources=["https://example.com/official/"]),
        )

        assert bundle.gate_failure is not None
        assert "넘게" in bundle.gate_failure
        assert calls == 1


class Test충돌우선:
    def test_확인된_충돌이면_불완전한_검색계획도_보완_없이_번들을_넘긴다(self):
        conflict = _충돌_증거_조사출력()
        conflict["search_queries"] = conflict["search_queries"][:-1]
        conflict["self_review"]["contradiction_search_complete"] = False

        bundle, _, calls = _조사_실행(_입력(official_sources=[]), [conflict])

        assert bundle.gate_failure is None
        assert {item["supports"] for item in bundle.evidence} == {"YES", "NO"}
        assert calls == 1

    def test_원문조회로_확인되지_않은_충돌이면_보완조사한다(self):
        conflict = _충돌_증거_조사출력()
        recovered = _고신뢰_증거_조사출력("YES", ["기관 A", "기관 B"])
        recovered["search_candidates"].append(conflict["search_candidates"][0])
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={
                normalize_url(candidate["url"])
                for candidate in conflict["search_candidates"]
                + recovered["search_candidates"]
            },
            attempted_fetch_urls={
                normalize_url(candidate["url"])
                for candidate in recovered["search_candidates"]
            },
            fetched_urls={
                normalize_url(candidate["url"])
                for candidate in recovered["search_candidates"]
            },
        )

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[]),
            [conflict, recovered],
            trace=trace,
        )

        assert bundle.gate_failure is None
        assert {item["supports"] for item in bundle.evidence} == {"YES"}
        assert calls == 2


class Test조사실행기록:
    def test_시장_지정_공식_url을_실제로_조회하지_않으면_실패한다(self):
        confirmed_url = "https://example.com/official"
        unobserved_url = "https://example.com/unobserved"
        output = _조사출력(
            [
                _증거(
                    "YES",
                    url=confirmed_url,
                    authority="official",
                    publisher="확정한 공식 기관",
                ),
                _증거(
                    "INCONCLUSIVE",
                    url=unobserved_url,
                    authority="official",
                    publisher="조회하지 않은 공식 기관",
                ),
            ]
        )
        output["evidence_reviews"][1]["fitness"] = "INCONCLUSIVE"
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={normalize_url(confirmed_url), normalize_url(unobserved_url)},
            attempted_fetch_urls={normalize_url(confirmed_url)},
            fetched_urls={normalize_url(confirmed_url)},
        )

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[confirmed_url, unobserved_url]),
            [output] * 4,
            trace=trace,
        )

        assert bundle.gate_failure is not None
        assert "공식 URL" in bundle.gate_failure
        assert calls == 4

    def test_실행하지_않은_검색어를_제출하면_보완_조사한다(self):
        unexecuted = _공식_증거_조사출력("YES")
        unexecuted["search_queries"][0]["query"] = "실행하지 않은 검색어"

        bundle, _, calls = _조사_실행(
            _입력(),
            [unexecuted, _공식_증거_조사출력("YES")],
        )

        assert bundle.gate_failure is None
        assert calls == 2

    def test_검색결과에_없는_후보_url이면_보완_조사한다(self):
        undiscovered = _공식_증거_조사출력("YES")
        undiscovered["search_candidates"][0]["url"] = "https://unseen.example/result"

        bundle, _, calls = _조사_실행(
            _입력(),
            [undiscovered, _공식_증거_조사출력("YES")],
        )

        assert bundle.gate_failure is None
        assert calls == 2

    def test_검색_결과와_query만_다른_후보는_gate를_통과한다(self):
        variant_url = "https://portal.example/data?stn=159&date=2026-08-08"
        output = _조사출력(
            [
                _증거(
                    "YES",
                    url=variant_url,
                    authority="official",
                    publisher="공식 기관",
                )
            ]
        )
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={normalize_url("https://portal.example/data")},
            attempted_fetch_urls={normalize_url(variant_url)},
            fetched_urls={normalize_url(variant_url)},
        )

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[]), [output], trace=trace
        )

        assert bundle.gate_failure is None
        assert calls == 1

    def test_같은_페이지의_변형을_여럿_조회하고_대표만_제출해도_통과한다(self):
        variant_url = "https://portal.example/data?stn=159&date=2026-08-08"
        output = _조사출력(
            [
                _증거(
                    "YES",
                    url=variant_url,
                    authority="official",
                    publisher="공식 기관",
                )
            ]
        )
        fetched = {
            normalize_url("https://portal.example/data"),
            normalize_url("https://portal.example/data?stn=108"),
            normalize_url(variant_url),
        }
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={normalize_url("https://portal.example/data")},
            attempted_fetch_urls=set(fetched),
            fetched_urls=set(fetched),
        )

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[]), [output], trace=trace
        )

        assert bundle.gate_failure is None
        assert calls == 1

    def test_조회한_페이지를_후보로_기록하지_않으면_보완_조사한다(self):
        output = _공식_증거_조사출력("YES")
        trace = _기록(
            output,
            official_sources=["https://example.com/official/"],
            extra_fetched_urls=["https://hidden.example/page"],
        )

        bundle, _, calls = _조사_실행(_입력(), [output] * 4, trace=trace)

        assert bundle.gate_failure is not None
        assert "후보" in bundle.gate_failure
        assert calls == 4

    def test_검색_결과와_path가_다른_후보는_보완_조사한다(self):
        undiscovered = _공식_증거_조사출력("YES")
        undiscovered["search_candidates"][0]["url"] = "https://example.com/other?x=1"

        bundle, _, calls = _조사_실행(
            _입력(),
            [undiscovered, _공식_증거_조사출력("YES")],
        )

        assert bundle.gate_failure is None
        assert calls == 2

    def test_원문을_조회하지_않은_evidence이면_실패한다(self):
        output = _공식_증거_조사출력("YES")
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={normalize_url("https://example.com/official")},
        )

        bundle, _, calls = _조사_실행(_입력(), [output] * 4, trace=trace)

        assert bundle.gate_failure is not None
        assert "원문" in bundle.gate_failure
        assert calls == 4

    def test_원문_조회에_실패한_url을_결론_근거로_쓰면_보완_조사한다(self):
        output = _공식_증거_조사출력("YES")
        trace = _기록(
            output,
            official_sources=["https://example.com/official/"],
            failed_urls=["https://example.com/official"],
        )

        bundle, _, calls = _조사_실행(_입력(), [output] * 4, trace=trace)

        assert bundle.gate_failure is not None
        assert "실패" in bundle.gate_failure
        assert calls == 4

    def test_시장_지정_공식_url은_검색결과에_없어도_후보로_인정한다(self):
        output = _공식_증거_조사출력("YES")
        trace = SearchTrace(
            queries={query["query"].casefold() for query in _검색_계획()},
            source_urls={normalize_url("https://search.example/result")},
            attempted_fetch_urls={normalize_url("https://example.com/official")},
            fetched_urls={normalize_url("https://example.com/official")},
        )

        bundle, _, calls = _조사_실행(_입력(), [output], trace=trace)

        assert bundle.gate_failure is None
        assert calls == 1

    def test_조회에_실패한_inconclusive_증거와_성공한_공식_증거는_통과한다(self):
        failed_url = "https://failed.example/result"
        confirmed_url = "https://confirmed.example/result"
        output = _조사출력(
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
            ]
        )
        output["evidence_reviews"][0]["fitness"] = "INCONCLUSIVE"
        trace = _기록(
            output,
            official_sources=[failed_url, confirmed_url],
            failed_urls=[failed_url],
        )

        bundle, _, calls = _조사_실행(
            _입력(official_sources=[failed_url, confirmed_url]),
            [output],
            trace=trace,
        )

        assert bundle.gate_failure is None
        assert calls == 1

    def test_과거_조회실패_후_성공한_url은_결론_근거로_회복한다(self):
        output = _공식_증거_조사출력("YES")
        trace = _기록(output, official_sources=["https://example.com/official/"])
        trace.failed_fetch_urls.add(normalize_url("https://example.com/official"))

        bundle, _, calls = _조사_실행(_입력(), [output], trace=trace)

        assert bundle.gate_failure is None
        assert calls == 1


class Test조사보완한도:
    def test_잘못된_output을_세번_수정하지_못하면_예외를_전달한다(self):
        calls = 0

        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {})])

        deps = SearchDeps(investigation=_입력(), backend=_가짜백엔드())
        with searcher._agent.override(model=FunctionModel(model_function)):
            with pytest.raises(UnexpectedModelBehavior):
                asyncio.run(searcher._agent.run("조사 결과를 제출하세요.", deps=deps))

        assert calls == 4

    def test_세번_보완해도_gate를_통과하지_못하면_실패_사유와_함께_번들을_반환한다(self):
        trace = _기록(
            _공식_증거_조사출력("YES"),
            official_sources=["https://example.com/official/"],
        )
        unexecuted = _공식_증거_조사출력("YES")
        unexecuted["search_queries"][0]["query"] = "실행하지 않은 검색어"

        bundle, _, calls = _조사_실행(_입력(), [unexecuted] * 4, trace=trace)

        assert bundle.gate_failure is not None
        assert "검색어" in bundle.gate_failure
        assert bundle.evidence
        assert calls == 4


class Test검색도구:
    def test_web_search가_검색어와_결과_url을_기록한다(self):
        backend = _가짜백엔드(
            [SearchResult(url="https://found.example/result", title="발견한 결과")]
        )
        deps = SearchDeps(investigation=_입력(), backend=backend)
        ctx = SimpleNamespace(deps=deps)

        results = asyncio.run(
            searcher._search_web(
                ctx,
                "사건 공식 결과",
                ["example.com"],
            )
        )

        assert [str(result.url) for result in results] == ["https://found.example/result"]
        assert backend.calls == [("사건 공식 결과", ("example.com",))]
        assert "사건 공식 결과" in deps.trace.queries
        assert normalize_url("https://found.example/result") in deps.trace.source_urls


class Test동적페이지fallback:
    def _조회(self, official_url: str) -> tuple[SearchDeps, Any]:
        deps = SearchDeps(
            investigation=_입력(official_sources=[official_url]),
            backend=_가짜백엔드(),
        )
        return deps, SimpleNamespace(deps=deps)

    def test_직접_조회_본문이_짧으면_exa_contents_본문으로_대체한다(self, monkeypatch):
        url = "https://spa.example/page"

        async def thin_fetch(_url: str):
            return {"url": url, "title": "제목", "content": 'window["fp"]=false'}

        async def rendered(fetch_url: str, **_kwargs):
            return {"url": fetch_url, "title": "렌더링 제목", "content": "본" * 600}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", thin_fetch)
        monkeypatch.setattr(searcher, "fetch_exa_contents", rendered)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert result["fetched_via"] == "exa_contents"
        assert result["content"] == "본" * 600
        assert normalize_url(url) in deps.trace.fetched_urls

    def test_본문이_충분하면_exa_contents를_호출하지_않는다(
        self, monkeypatch, exa_contents_차단
    ):
        url = "https://static.example/page"

        async def full_fetch(_url: str):
            return {"url": url, "title": "제목", "content": "가" * 600}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", full_fetch)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert "fetched_via" not in result
        assert "thin_content" not in result
        assert exa_contents_차단 == []
        assert normalize_url(url) in deps.trace.fetched_urls

    def test_직접_조회가_세번_실패해도_exa_contents가_성공하면_본문으로_회복한다(
        self, monkeypatch
    ):
        url = "https://blocked.example/page"

        async def always_fails(_url: str):
            raise ModelRetry("페이지에 접근할 수 없습니다")

        async def rendered(fetch_url: str, **_kwargs):
            return {"url": fetch_url, "title": "렌더링 제목", "content": "본" * 600}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", always_fails)
        monkeypatch.setattr(searcher, "fetch_exa_contents", rendered)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert result["fetched_via"] == "exa_contents"
        assert normalize_url(url) in deps.trace.fetched_urls
        assert deps.trace.failed_fetch_urls == set()

    def test_exa_contents도_본문이_짧으면_thin_content를_표시해_반환한다(
        self, monkeypatch
    ):
        url = "https://spa.example/page"

        async def thin_fetch(_url: str):
            return {"url": url, "title": "제목", "content": 'window["fp"]=false'}

        async def thin_rendered(fetch_url: str, **_kwargs):
            return {"url": fetch_url, "title": "렌더링 제목", "content": "짧음"}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", thin_fetch)
        monkeypatch.setattr(searcher, "fetch_exa_contents", thin_rendered)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert result["content"] == 'window["fp"]=false'
        assert "미만" in result["thin_content"]
        assert normalize_url(url) in deps.trace.fetched_urls

    def test_exa_key가_없으면_fallback_없이_thin_content만_표시한다(self, monkeypatch):
        url = "https://spa.example/page"

        async def thin_fetch(_url: str):
            return {"url": url, "title": "제목", "content": 'window["fp"]=false'}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", thin_fetch)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert "fetched_via" not in result
        assert "미만" in result["thin_content"]
        assert normalize_url(url) in deps.trace.fetched_urls

    def test_직접_조회와_fallback이_모두_실패하면_error를_반환한다(self, monkeypatch):
        url = "https://unavailable.example/page"

        async def always_fails(_url: str):
            raise ModelRetry("페이지에 접근할 수 없습니다")

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", always_fails)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert "접근할 수 없습니다" in result["error"]
        assert normalize_url(url) in deps.trace.failed_fetch_urls

    def test_render를_지정하면_직접_조회_없이_exa_contents_본문으로_조회한다(
        self, monkeypatch
    ):
        url = "https://spa.example/page"
        raw_calls: list[str] = []

        async def raw_fetch(fetch_url: str):
            raw_calls.append(fetch_url)
            return {"url": url, "title": "제목", "content": "가" * 600}

        async def rendered(fetch_url: str, **_kwargs):
            return {"url": fetch_url, "title": "렌더링 제목", "content": "본" * 600}

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", raw_fetch)
        monkeypatch.setattr(searcher, "fetch_exa_contents", rendered)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url, render=True))

        assert result["fetched_via"] == "exa_contents"
        assert raw_calls == []
        assert normalize_url(url) in deps.trace.fetched_urls

    def test_render_조회가_실패하면_error를_반환한다(self):
        url = "https://spa.example/page"
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url, render=True))

        assert "렌더링" in result["error"]
        assert normalize_url(url) in deps.trace.failed_fetch_urls

    def test_pdf_같은_binary_응답은_thin_판정_없이_그대로_반환한다(
        self, monkeypatch, exa_contents_차단
    ):
        url = "https://files.example/report.pdf"
        binary = BinaryContent(data=b"%PDF-1.7", media_type="application/pdf")

        async def binary_fetch(_url: str):
            return binary

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", binary_fetch)
        deps, ctx = self._조회(url)

        result = asyncio.run(searcher._fetch_web_page(ctx, url))

        assert result is binary
        assert exa_contents_차단 == []
        assert normalize_url(url) in deps.trace.fetched_urls


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

        monkeypatch.setattr(searcher._raw_web_fetch_tool, "function", always_fails)
        deps = SearchDeps(
            investigation=_입력(official_sources=["https://unavailable.example"]),
            backend=_가짜백엔드(),
        )
        ctx = SimpleNamespace(deps=deps)

        result = asyncio.run(
            searcher._fetch_web_page(ctx, "https://unavailable.example")
        )

        assert result["url"] == "https://unavailable.example"
        assert "접근할 수 없습니다" in result["error"]
        assert attempts == 3
        assert normalize_url("https://unavailable.example") in deps.trace.failed_fetch_urls

    def test_조회_규칙을_위반한_fetch는_조회_없이_skipped를_반환한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        ctx = SimpleNamespace(deps=deps)

        result = asyncio.run(
            searcher._fetch_web_page(ctx, "https://unknown.example/page")
        )

        assert result["url"] == "https://unknown.example/page"
        assert "skipped" in result
        assert deps.trace.attempted_fetch_urls == set()


class Test검색후보조회차단:
    def test_검색_결과에_없는_주소_조회를_거절한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        deps.trace.source_urls.add(normalize_url("https://a.example/result"))

        refusal = searcher._search_fetch_refusal(deps, "https://b.example/other")

        assert refusal is not None
        assert "검색 결과" in refusal

    def test_시장_지정_공식_url은_검색_결과에_없어도_조회를_허용한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=["https://official.example/page"]),
            backend=_가짜백엔드(),
        )

        refusal = searcher._search_fetch_refusal(deps, "https://official.example/page/")

        assert refusal is None

    def test_여섯번째_검색_후보_조회를_거절한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        for index in range(6):
            deps.trace.source_urls.add(
                normalize_url(f"https://candidate-{index}.example/result")
            )
        for index in range(5):
            deps.trace.attempted_fetch_urls.add(
                normalize_url(f"https://candidate-{index}.example/result")
            )

        refusal = searcher._search_fetch_refusal(
            deps, "https://candidate-5.example/result"
        )

        assert refusal is not None
        assert "최대 5개" in refusal

    def test_검색_결과_url과_query만_다른_주소는_조회를_허용한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        deps.trace.source_urls.add(normalize_url("https://portal.example/data"))

        refusal = searcher._search_fetch_refusal(
            deps, "https://portal.example/data?stn=159&date=2026-08-08"
        )

        assert refusal is None

    def test_공식_url의_query_변형도_새_페이지면_예산에_포함한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=["https://official.example/data"]),
            backend=_가짜백엔드(),
        )
        for index in range(5):
            deps.trace.attempted_fetch_urls.add(
                normalize_url(f"https://candidate-{index}.example/result")
            )

        refusal = searcher._search_fetch_refusal(
            deps, "https://official.example/data?stn=1"
        )

        assert refusal is not None
        assert "최대 5개" in refusal

    def test_path가_다른_주소는_같은_host여도_거절한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        deps.trace.source_urls.add(normalize_url("https://portal.example/data"))

        refusal = searcher._search_fetch_refusal(
            deps, "https://portal.example/other?stn=159"
        )

        assert refusal is not None
        assert "검색 결과" in refusal

    def test_query_변형은_같은_페이지라서_예산을_추가로_쓰지_않는다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        deps.trace.source_urls.add(normalize_url("https://portal.example/data"))
        deps.trace.attempted_fetch_urls.add(normalize_url("https://portal.example/data"))
        for index in range(4):
            url = f"https://candidate-{index}.example/result"
            deps.trace.source_urls.add(normalize_url(url))
            deps.trace.attempted_fetch_urls.add(normalize_url(url))

        refusal = searcher._search_fetch_refusal(
            deps, "https://portal.example/data?stn=159"
        )

        assert refusal is None

    def test_이미_시도한_검색_후보는_응답_전에도_다시_조회를_허용한다(self):
        deps = SearchDeps(
            investigation=_입력(official_sources=[]),
            backend=_가짜백엔드(),
        )
        for index in range(5):
            url = f"https://candidate-{index}.example/result"
            deps.trace.source_urls.add(normalize_url(url))
            deps.trace.attempted_fetch_urls.add(normalize_url(url))

        refusal = searcher._search_fetch_refusal(
            deps, "https://candidate-0.example/result"
        )

        assert refusal is None


class Test조사실행:
    def test_중복을_제외한_공식_url이_여섯개이면_도구호출한도를_스물둘로_늘린다(
        self,
        monkeypatch,
    ):
        captured_limits = None

        async def fake_run(_agent, *_args, **kwargs):
            nonlocal captured_limits
            captured_limits = kwargs["usage_limits"]
            return SimpleNamespace(
                output=EvidenceBundle.model_validate(_조사출력([])),
                usage=SimpleNamespace(
                    requests=0,
                    tool_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                ),
            )

        monkeypatch.setattr(type(searcher._agent), "run", fake_run)
        sources = [f"https://official-{index}.example/result" for index in range(6)]
        sources.append("https://official-0.example/result/")

        asyncio.run(
            searcher.investigate(_입력(official_sources=sources), _가짜백엔드())
        )

        assert captured_limits is not None
        assert captured_limits.tool_calls_limit == 22

    def test_조사기록없이_출력하면_통과시키지_않고_사용량을_기록한다(self, caplog):
        calls = 0

        def model_function(_messages: list[Any], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, _공식_증거_조사출력("YES"))]
            )

        caplog.set_level(logging.INFO, logger=searcher.__name__)
        with searcher._agent.override(model=FunctionModel(model_function)):
            bundle = asyncio.run(
                searcher.investigate(_입력(), _가짜백엔드())
            )

        assert bundle.gate_failure is not None
        assert calls == 4
        assert "agent usage" in caplog.text
