"""사용자가 지정한 실제 사건을 Luna와 인터넷으로 조사하는 수동 시나리오.

Git이 무시하는 `.env`에 `OPENAI_API_KEY`를 넣고 다음을 실행한다.

./tests/live/run_live_tests.sh
"""

import asyncio
import os
from datetime import UTC, datetime

import pytest


if os.getenv("RUN_ORACLE_LIVE_TESTS") != "1":
    pytest.skip("RUN_ORACLE_LIVE_TESTS=1일 때만 실행합니다", allow_module_level=True)
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(".env의 OPENAI_API_KEY가 필요합니다", allow_module_level=True)

from oracle_agent.agents.resolver import _production_model, resolve
from oracle_agent.models import InvestigationInput


def _scenario(
    prediction_id: str,
    prediction: str,
    resolve_after: datetime,
) -> InvestigationInput:
    return InvestigationInput(
        prediction_id=prediction_id,
        prediction=prediction,
        resolution_criteria=(
            f'"{prediction}"라는 예측이 사실이면 YES, 사실이 아니면 NO다.'
        ),
        resolve_after=resolve_after,
        official_sources=[],
    )


async def _resolve_and_close(investigation: InvestigationInput):
    try:
        return await resolve(investigation)
    finally:
        await _production_model().client.close()


@pytest.mark.parametrize(
    ("investigation", "expected"),
    [
        pytest.param(
            _scenario(
                "manual-world-cup-germany-2026",
                "2026년 북중미 월드컵의 우승팀은 독일이다",
                datetime(2026, 7, 20, tzinfo=UTC),
            ),
            "NO",
            id="북중미_월드컵_독일_no",
        ),
        pytest.param(
            _scenario(
                "manual-world-cup-spain-2026",
                "2026년 북중미 월드컵의 우승팀은 스페인이다",
                datetime(2026, 7, 20, tzinfo=UTC),
            ),
            "YES",
            id="북중미_월드컵_스페인_yes",
        ),
        pytest.param(
            _scenario(
                "manual-busan-high-temperature-2026-08-08",
                "2026년 8월 8일 부산의 최고기온은 40도를 넘었다",
                datetime(2026, 8, 8, 14, tzinfo=UTC),
            ),
            "NO",
            id="부산_최고기온_40도_초과_no",
        ),
        pytest.param(
            _scenario(
                "manual-democratic-party-incheon-2026-08-08",
                "2026년 8월 8일에 발표된 더불어민주당 당대표 인천 지역 결과에서 김민석 후보가 1위를 했다",
                datetime(2026, 8, 8, 13, tzinfo=UTC),
            ),
            "YES",
            id="더불어민주당_인천_김민석_1위_yes",
        ),
        pytest.param(
            _scenario(
                "manual-jeff-dean-google-2026-08-08",
                "2026년 8월 8일 기준으로 Jeff Dean은 여전히 구글의 직원이다",
                datetime(2026, 8, 8, tzinfo=UTC),
            ),
            "NO",
            id="jeff_dean_구글_직원_no",
        ),
        pytest.param(
            _scenario(
                "manual-president-approval-at-least-50-2026-08-06",
                "2026년 8월 6일 기준으로 이재명 대통령의 지지율은 50% 이상이다",
                datetime(2026, 8, 7, tzinfo=UTC),
            ),
            "NO",
            id="이재명_지지율_50퍼센트_이상_no",
        ),
        pytest.param(
            _scenario(
                "manual-president-approval-at-least-40-2026-08-06",
                "2026년 8월 6일 기준으로 이재명 대통령의 지지율은 40% 이상이다",
                datetime(2026, 8, 7, tzinfo=UTC),
            ),
            "YES",
            id="이재명_지지율_40퍼센트_이상_yes",
        ),
        pytest.param(
            _scenario(
                "manual-president-approval-at-least-90-2026-08-06",
                "2026년 8월 6일 기준으로 이재명 대통령의 지지율은 90% 이상이다",
                datetime(2026, 8, 7, tzinfo=UTC),
            ),
            "NO",
            id="이재명_지지율_90퍼센트_이상_no",
        ),
        pytest.param(
            _scenario(
                "manual-president-approval-at-most-90-2026-08-06",
                "2026년 8월 6일 기준으로 이재명 대통령의 지지율은 90% 이하이다",
                datetime(2026, 8, 7, tzinfo=UTC),
            ),
            "YES",
            id="이재명_지지율_90퍼센트_이하_yes",
        ),
    ],
)
def test_실제_luna가_수동_시나리오를_조사해_기대_결론을_낸다(
    investigation: InvestigationInput,
    expected: str,
):
    try:
        result = asyncio.run(_resolve_and_close(investigation))
    finally:
        _production_model.cache_clear()

    print(result.model_dump_json(indent=2))
    assert result.decision == expected
    assert any(item["supports"] == expected for item in result.evidence)
