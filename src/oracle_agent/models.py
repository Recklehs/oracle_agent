from enum import StrEnum
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import AwareDatetime, AnyHttpUrl, BaseModel, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


class Resolution(StrEnum):
    YES = "YES"
    NO = "NO"


class InvestigationInput(BaseModel):
    # 서버에서 prediction_id 타입이 확정되면 확인 후 변경한다.
    prediction_id: NonEmptyText
    prediction: NonEmptyText
    resolution_criteria: NonEmptyText
    resolve_after: AwareDatetime
    official_sources: list[AnyHttpUrl] = Field(default_factory=list)


Evidence = TypedDict(
    "Evidence",
    {
        "url": AnyHttpUrl,
        "title": NonEmptyText,
        "publisher": NonEmptyText,
        "original_publisher": NonEmptyText,
        "authority": Literal["official", "high_trust", "other"],
        "supports": Literal["YES", "NO", "INCONCLUSIVE"],
        "finding": NonEmptyText,
        "excerpt": NotRequired[NonEmptyText | None],
        "published_at": NotRequired[AwareDatetime | None],
    },
)


class InvestigationResult(BaseModel):
    # 서버에서 prediction_id 타입이 확정되면 확인 후 변경한다.
    prediction_id: NonEmptyText
    decision: Literal["YES", "NO", "ESCALATED"]
    summary: NonEmptyText
    evidence: list[Evidence]
    escalation_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "InvestigationResult":
        if self.decision == "ESCALATED" and self.escalation_reason is None:
            raise ValueError("판정 불가 결과에는 이관 사유가 필요합니다")
        if self.decision != "ESCALATED" and self.escalation_reason is not None:
            raise ValueError("YES 또는 NO 결과에는 이관 사유를 둘 수 없습니다")
        if self.decision != "ESCALATED" and not any(
            item["supports"] == self.decision for item in self.evidence
        ):
            raise ValueError("결론과 같은 방향의 증거가 필요합니다")
        return self


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
