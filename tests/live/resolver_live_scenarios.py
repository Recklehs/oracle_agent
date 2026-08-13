"""
실제 Luna와 인터넷을 사용하는 수동 시나리오.

1. Git이 무시하는 `.env`의 `OPENAI_API_KEY=` 뒤에 직접 키를 넣는다.
2. 다음 스크립트로 이 파일만 명시적으로 실행한다.

./tests/live/run_live_tests.sh

키를 테스트 코드나 추적되는 파일에 기록하지 않는다.
"""

import asyncio
import os
from datetime import UTC, datetime

import pytest


if os.getenv("RUN_ORACLE_LIVE_TESTS") != "1":
    pytest.skip("RUN_ORACLE_LIVE_TESTS=1일 때만 실행합니다", allow_module_level=True)
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(".env의 OPENAI_API_KEY가 필요합니다", allow_module_level=True)

from oracle_agent.agents.provider import aclose_cached_models
from oracle_agent.agents.resolver import resolve
from oracle_agent.models import InvestigationInput


def _python_313_case() -> InvestigationInput:
    return InvestigationInput(
        prediction_id="live-python-313-release",
        prediction="Python 3.13.0 stable이 2024년 10월 8일까지 출시된다.",
        resolution_criteria=(
            "Python.org의 Python 3.13.0 공식 릴리스 날짜가 2024-10-08 23:59:59 UTC "
            "이전이면 YES, 이후이거나 그때까지 출시되지 않았으면 NO다."
        ),
        resolve_after=datetime(2024, 10, 9, tzinfo=UTC),
        official_sources=["https://www.python.org/downloads/release/python-3130/"],
    )


def _python_314_case() -> InvestigationInput:
    return InvestigationInput(
        prediction_id="live-python-314-before-2025",
        prediction="Python 3.14.0 stable이 2025년 1월 1일 전에 출시된다.",
        resolution_criteria=(
            "Python.org의 Python 3.14.0 공식 릴리스 날짜가 2025-01-01 00:00:00 UTC "
            "이전이면 YES, 해당 시각 이후이면 NO다."
        ),
        resolve_after=datetime(2025, 10, 8, tzinfo=UTC),
        official_sources=["https://www.python.org/downloads/release/python-3140/"],
    )


def _ambiguous_popularity_case() -> InvestigationInput:
    return InvestigationInput(
        prediction_id="live-python-popularity-2025",
        prediction="Python은 2025년에 인기 있는 프로그래밍 언어였다.",
        resolution_criteria=(
            "Python이 인기 있었다면 YES, 아니면 NO다. 인기의 측정 지표, 임계값, 지역, "
            "조사 대상은 지정하지 않는다."
        ),
        resolve_after=datetime(2026, 1, 1, tzinfo=UTC),
        official_sources=[],
    )


async def _resolve_and_close(investigation: InvestigationInput):
    try:
        return await resolve(investigation)
    finally:
        await aclose_cached_models()


@pytest.mark.parametrize(
    ("investigation", "expected"),
    [
        pytest.param(_python_313_case(), "YES", id="공식_릴리스_이전_yes"),
        pytest.param(_python_314_case(), "NO", id="기준일_이후_릴리스_no"),
        pytest.param(
            _ambiguous_popularity_case(),
            "ESCALATED",
            id="객관적_기준_없는_인기_이관",
        ),
    ],
)
def test_실제_luna가_과거_사건을_조사해_기대_결론을_낸다(
    investigation: InvestigationInput,
    expected: str,
):
    result = asyncio.run(_resolve_and_close(investigation))

    print(result.model_dump_json(indent=2))
    assert result.decision == expected
    if expected in {"YES", "NO"}:
        assert any(item["supports"] == expected for item in result.evidence)
    else:
        assert result.escalation_reason
