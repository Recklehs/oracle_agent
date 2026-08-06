import pytest
from pydantic import ValidationError

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
