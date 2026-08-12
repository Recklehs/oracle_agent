"""검색 backend만 실제 API로 따로 검증하는 수동 시나리오.

조사·판정 Agent를 거치지 않고 각 backend의 `search()`만 호출해 후보 품질을
비교한다. 실행이 끝나면 `report/search-backend-*.md`에 시나리오별 비교
보고서를 남기고, `.env`에 `LOGFIRE_TOKEN`이 있으면 Logfire에도 검색 단위
span과 raw HTTP 요청·응답이 기록된다.

1. Git이 무시하는 `.env`에 사용할 backend의 API 키를 넣는다.
   - openai: `OPENAI_API_KEY`, gemini: `GEMINI_API_KEY`, exa: `EXA_API_KEY`,
     tavily: `TAVILY_API_KEY`, brave: `BRAVE_API_KEY`
   - 키가 없는 backend는 실패가 아니라 skip으로 표시된다.
2. 다음 스크립트로 이 파일만 명시적으로 실행한다.

./tests/live/run_search_backend_tests.sh

특정 backend나 시나리오만 실행하려면 pytest 인자를 그대로 넘긴다.

./tests/live/run_search_backend_tests.sh -k exa
./tests/live/run_search_backend_tests.sh -k 도메인
"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

import httpx
import logfire
import pytest
from openai import AsyncOpenAI


if os.getenv("RUN_ORACLE_LIVE_TESTS") != "1":
    pytest.skip("RUN_ORACLE_LIVE_TESTS=1일 때만 실행합니다", allow_module_level=True)

from oracle_agent.agents.search_backends import (
    BACKEND_FACTORIES,
    SearchBackend,
    SearchResult,
    create_search_backend,
)


BACKEND_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "exa": "EXA_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_API_KEY",
}
# Brave 무료 요금제가 초당 1회 제한이므로 검색 사이에 짧게 대기한다.
SECONDS_BETWEEN_SEARCHES = 1.5
REPORT_DIR = Path(__file__).parent / "report"


class _Scenario(NamedTuple):
    label: str
    query: str
    target_domains: tuple[str, ...] = ()


GENERAL_ENGLISH = _Scenario(
    label="영어 일반 검색",
    query="Who won the 2024 United States presidential election?",
)
GENERAL_KOREAN = _Scenario(
    label="한국어 일반 검색",
    query="2024 파리 올림픽 남자 100m 결승 금메달리스트",
)
OFFICIAL_DOMAIN = _Scenario(
    label="공식 도메인 제한 검색",
    query="December 2024 FOMC statement federal funds rate target range",
    target_domains=("federalreserve.gov",),
)

_records: list[dict[str, Any]] = []
_pause_before_next_search = False


@pytest.fixture(params=sorted(BACKEND_FACTORIES), ids=str)
def backend(request: pytest.FixtureRequest) -> SearchBackend:
    key_env = BACKEND_KEY_ENV[request.param]
    if not os.environ.get(key_env):
        pytest.skip(f".env의 {key_env}가 필요합니다")
    return create_search_backend(request.param)


async def _search_and_close(
    backend: SearchBackend, scenario: _Scenario
) -> list[SearchResult]:
    try:
        return await backend.search(scenario.query, scenario.target_domains)
    finally:
        # backend는 요청 스코프에서 쓰고 버리므로 내부 client를 직접 닫는다.
        client = getattr(backend, "_client", None)
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()
        elif isinstance(client, AsyncOpenAI):
            await client.close()


def _run_search(backend: SearchBackend, scenario: _Scenario) -> list[SearchResult]:
    global _pause_before_next_search
    if _pause_before_next_search:
        time.sleep(SECONDS_BETWEEN_SEARCHES)
    _pause_before_next_search = True

    results: list[SearchResult] = []
    error: str | None = None
    started = time.monotonic()
    try:
        with logfire.span(
            "검색 backend 라이브 {backend}/{scenario}",
            backend=backend.name,
            scenario=scenario.label,
            query=scenario.query,
            target_domains=list(scenario.target_domains),
            _tags=["search-backend"],
        ):
            results = asyncio.run(_search_and_close(backend, scenario))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _records.append(
            {
                "scenario": scenario,
                "backend": backend.name,
                "elapsed": time.monotonic() - started,
                "results": results,
                "error": error,
            }
        )
    print(f"\n[{backend.name}] {scenario.label} → {len(results)}건")
    for result in results:
        print(f"  - {result.url} | {result.title}")
    return results


def _hostname(result: SearchResult) -> str:
    return urlsplit(str(result.url)).hostname or ""


def _matches_domain(result: SearchResult, domains: tuple[str, ...]) -> bool:
    hostname = _hostname(result)
    return any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in domains
    )


class Test검색backend라이브검증:
    def test_영어_검색어를_검색하면_후보_url을_반환한다(self, backend: SearchBackend):
        results = _run_search(backend, GENERAL_ENGLISH)
        assert results

    def test_한국어_검색어를_검색하면_후보_url을_반환한다(self, backend: SearchBackend):
        results = _run_search(backend, GENERAL_KOREAN)
        assert results

    def test_공식_도메인을_제한하면_해당_도메인_후보가_포함된다(
        self, backend: SearchBackend
    ):
        results = _run_search(backend, OFFICIAL_DOMAIN)
        assert any(
            _matches_domain(result, OFFICIAL_DOMAIN.target_domains)
            for result in results
        )


def _write_report() -> None:
    if not _records:
        return
    REPORT_DIR.mkdir(exist_ok=True)
    now = datetime.now().astimezone()
    path = REPORT_DIR / f"search-backend-{now:%Y%m%d-%H%M%S}.md"
    missing = sorted(
        name for name, key in BACKEND_KEY_ENV.items() if not os.environ.get(key)
    )
    lines = [
        "# 검색 backend 라이브 비교 보고서",
        "",
        f"- 실행 시각: {now:%Y-%m-%d %H:%M:%S %Z}",
        f"- 키가 없어 건너뛴 backend: {', '.join(missing) if missing else '없음'}",
    ]
    for scenario in (GENERAL_ENGLISH, GENERAL_KOREAN, OFFICIAL_DOMAIN):
        rows = [record for record in _records if record["scenario"] == scenario]
        if not rows:
            continue
        lines += [
            "",
            f"## {scenario.label}",
            "",
            f"- 검색어: {scenario.query}",
            f"- 도메인 제한: {', '.join(scenario.target_domains) or '없음'}",
        ]
        for record in rows:
            results = record["results"]
            summary = f"### {record['backend']} — {len(results)}건, {record['elapsed']:.1f}초"
            if scenario.target_domains:
                matched = sum(
                    _matches_domain(result, scenario.target_domains)
                    for result in results
                )
                summary += f", 제한 도메인 일치 {matched}건"
            lines += ["", summary, ""]
            if record["error"]:
                lines.append(f"- 오류: {record['error']}")
            for result in results:
                suffix = f" ({result.published_at})" if result.published_at else ""
                lines.append(f"1. [{result.title}]({result.url}){suffix}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n검색 backend 비교 보고서: {path}")


@pytest.fixture(scope="module", autouse=True)
def _search_backend_report():
    yield
    _write_report()
