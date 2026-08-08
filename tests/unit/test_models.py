from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import oracle_agent.models as models
from oracle_agent.models import JobStatus, Resolution, ResolutionJob


class Test판정작업상태:
    def test_완료_상태인데_판정값이_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="완료된 작업에는 판정값이 필요합니다"):
            ResolutionJob(status=JobStatus.RESOLVED)

    def test_이관_상태인데_사유가_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="이관된 작업에는 이관 사유가 필요합니다"):
            ResolutionJob(status=JobStatus.ESCALATED)

    def test_대기_상태에_판정값이_있으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="완료되지 않은 작업에는 판정값을 둘 수 없습니다"):
            ResolutionJob(status=JobStatus.QUEUED, resolution=Resolution.YES)


def _공식_증거(supports: str = "YES") -> dict[str, object]:
    return {
        "url": "https://example.com/final-result",
        "title": "공식 경기 결과",
        "publisher": "대한체육회",
        "original_publisher": "대한체육회",
        "authority": "official",
        "supports": supports,
        "finding": "대한민국이 결승전에서 승리했다.",
        "excerpt": "대한민국 우승",
        "published_at": datetime(2026, 8, 8, 19, 0, tzinfo=timezone.utc),
    }


class TestAgent입력DTO:
    def test_유효한_예측이면_입력_dto를_생성한다(self):
        investigation_input = models.InvestigationInput(
            prediction_id=" prediction-123 ",
            prediction=" 대한민국이 결승전에서 승리한다. ",
            resolution_criteria=" 공식 결과가 승리이면 YES다. ",
            resolve_after=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
        )

        assert investigation_input.prediction_id == "prediction-123"
        assert investigation_input.prediction == "대한민국이 결승전에서 승리한다."
        assert investigation_input.official_sources == []

    def test_prediction_id가_공백이면_검증에_실패한다(self):
        with pytest.raises(ValidationError):
            models.InvestigationInput(
                prediction_id="   ",
                prediction="대한민국이 결승전에서 승리한다.",
                resolution_criteria="공식 결과가 승리이면 YES다.",
                resolve_after=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
            )

    def test_판정_가능_시각에_타임존이_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError):
            models.InvestigationInput(
                prediction_id="prediction-123",
                prediction="대한민국이 결승전에서 승리한다.",
                resolution_criteria="공식 결과가 승리이면 YES다.",
                resolve_after=datetime(2026, 8, 8, 18, 0),
            )

    def test_공식_출처가_열개이면_입력_dto를_생성한다(self):
        investigation_input = models.InvestigationInput(
            prediction_id="prediction-123",
            prediction="대한민국이 결승전에서 승리한다.",
            resolution_criteria="공식 결과가 승리이면 YES다.",
            resolve_after=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
            official_sources=[
                f"https://official-{index}.example/result" for index in range(10)
            ],
        )

        assert len(investigation_input.official_sources) == 10

    def test_공식_출처가_열한개이면_검증에_실패한다(self):
        with pytest.raises(ValidationError):
            models.InvestigationInput(
                prediction_id="prediction-123",
                prediction="대한민국이 결승전에서 승리한다.",
                resolution_criteria="공식 결과가 승리이면 YES다.",
                resolve_after=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
                official_sources=[
                    f"https://official-{index}.example/result" for index in range(11)
                ],
            )


class TestAgent반환DTO:
    def test_같은_방향의_증거가_있는_yes_결과이면_반환_dto를_생성한다(self):
        result = models.InvestigationResult(
            prediction_id="prediction-123",
            decision="YES",
            summary="공식 결과에서 대한민국의 승리를 확인했다.",
            evidence=[_공식_증거()],
        )

        assert result.prediction_id == "prediction-123"
        assert result.decision == "YES"
        assert result.escalation_reason is None

    def test_escalated_결과인데_사유가_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="판정 불가 결과에는 이관 사유가 필요합니다"):
            models.InvestigationResult(
                prediction_id="prediction-123",
                decision="ESCALATED",
                summary="공식 결과를 확인할 수 없다.",
                evidence=[],
            )

    def test_결론과_같은_방향의_증거가_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="결론과 같은 방향의 증거가 필요합니다"):
            models.InvestigationResult(
                prediction_id="prediction-123",
                decision="NO",
                summary="공식 결과에서 패배를 확인했다.",
                evidence=[_공식_증거("YES")],
            )
