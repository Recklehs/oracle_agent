from enum import StrEnum

from pydantic import BaseModel, model_validator


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


class Resolution(StrEnum):
    YES = "YES"
    NO = "NO"


class ResolutionJob(BaseModel):
    status: JobStatus = JobStatus.QUEUED
    resolution: Resolution | None = None
    escalation_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "ResolutionJob":
        if self.status is JobStatus.RESOLVED and self.resolution is None:
            raise ValueError("완료된 작업에는 판정값이 필요합니다")
        if self.status is JobStatus.ESCALATED and not self.escalation_reason:
            raise ValueError("이관된 작업에는 이관 사유가 필요합니다")
        if self.status is not JobStatus.RESOLVED and self.resolution is not None:
            raise ValueError("완료되지 않은 작업에는 판정값을 둘 수 없습니다")
        return self
