# Oracle Agent 조사 DTO 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 서버와 Oracle Agent 사이의 조사 입력 DTO와 `YES/NO/ESCALATED` 최종 조사 결과 DTO를 구현한다.

**Architecture:** 기존 `models.py`에 Pydantic 모델과 enum을 추가한다. 입력 검증과 결과 불변 조건은 DTO가 직접 보장하고, 의미 기반 출처 검토와 재조사 loop는 이후 Agent 구현에 남긴다.

**Tech Stack:** Python 3.12, Pydantic, pytest

## Global Constraints

- Python 요구 버전은 `>=3.12`다.
- 새 의존성을 추가하지 않는다.
- 구현 파일은 `src/oracle_agent/models.py`, 테스트 파일은 `tests/unit/test_models.py`만 변경한다.
- 테스트 식별자는 `test_<조건>이면_<기대결과>한다` 형태의 한국어 시나리오 이름을 사용한다.
- 실제 LLM이나 인터넷을 호출하지 않는다.

---

### Task 1: Agent 조사 입력 DTO

**Files:**
- Modify: `src/oracle_agent/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Consumes: Pydantic `AwareDatetime`, `AnyHttpUrl`
- Produces: `InvestigationInput(question: str, resolution_criteria: str, resolve_after: AwareDatetime, official_sources: list[AnyHttpUrl] = [])`

- [ ] **Step 1: 실패하는 입력 DTO 테스트 작성**

`tests/unit/test_models.py`에 다음 import와 테스트를 추가한다.

```python
from datetime import datetime, timezone

import oracle_agent.models as models


class TestAgent조사입력:
    def test_판정_가능_시각에_타임존이_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError):
            models.InvestigationInput(
                question="대한민국이 결승전에서 승리했는가?",
                resolution_criteria="공식 경기 결과가 승리이면 YES다.",
                resolve_after=datetime(2026, 8, 8, 18, 0),
            )

    @pytest.mark.parametrize(
        "field",
        ["question", "resolution_criteria"],
        ids=["예측문장", "종료조건"],
    )
    def test_필수_문자열이_공백이면_검증에_실패한다(self, field):
        values = {
            "question": "대한민국이 결승전에서 승리했는가?",
            "resolution_criteria": "공식 경기 결과가 승리이면 YES다.",
            "resolve_after": datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
        }
        values[field] = "   "

        with pytest.raises(ValidationError):
            models.InvestigationInput(**values)

    def test_유효한_입력이면_공식_출처_없이_생성한다(self):
        investigation_input = models.InvestigationInput(
            question=" 대한민국이 결승전에서 승리했는가? ",
            resolution_criteria=" 공식 경기 결과가 승리이면 YES다. ",
            resolve_after=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
        )

        assert investigation_input.question == "대한민국이 결승전에서 승리했는가?"
        assert investigation_input.resolution_criteria == "공식 경기 결과가 승리이면 YES다."
        assert investigation_input.official_sources == []
```

- [ ] **Step 2: 입력 DTO가 없어서 실패하는지 확인**

Run: `uv run pytest tests/unit/test_models.py::TestAgent조사입력 -v`

Expected: FAIL with `AttributeError: module 'oracle_agent.models' has no attribute 'InvestigationInput'`.

- [ ] **Step 3: 최소 입력 DTO 구현**

`src/oracle_agent/models.py`의 import와 모델에 다음을 추가한다.

```python
from typing import Annotated

from pydantic import AwareDatetime, AnyHttpUrl, BaseModel, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class InvestigationInput(BaseModel):
    question: NonEmptyText
    resolution_criteria: NonEmptyText
    resolve_after: AwareDatetime
    official_sources: list[AnyHttpUrl] = Field(default_factory=list)
```

- [ ] **Step 4: 입력 DTO 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_models.py::TestAgent조사입력 -v`

Expected: 4 tests PASS.

- [ ] **Step 5: 입력 DTO 커밋**

```bash
git add src/oracle_agent/models.py tests/unit/test_models.py
git commit -m "feat: Agent 조사 입력 DTO 추가"
```

### Task 2: 구조화된 증거 DTO

**Files:**
- Modify: `src/oracle_agent/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Consumes: Task 1의 `NonEmptyText`
- Produces: `SourceAuthority`, `EvidenceSupport`, `Evidence`

- [ ] **Step 1: 실패하는 증거 DTO 테스트 작성**

`tests/unit/test_models.py`에 다음 테스트를 추가한다.

```python
class Test조사증거:
    @pytest.mark.parametrize(
        "field",
        ["title", "publisher", "original_publisher", "summary"],
        ids=["제목", "게시자", "원게시자", "요약"],
    )
    def test_필수_문자열이_공백이면_검증에_실패한다(self, field):
        values = {
            "url": "https://example.com/final-result",
            "title": "공식 경기 결과",
            "publisher": "대한체육회",
            "original_publisher": "대한체육회",
            "authority": "official",
            "supports": "YES",
            "summary": "대한민국이 결승전에서 승리했다.",
        }
        values[field] = "   "

        with pytest.raises(ValidationError):
            models.Evidence(**values)

    def test_유효한_증거이면_구조화해_생성한다(self):
        evidence = models.Evidence(
            url="https://example.com/final-result",
            title="공식 경기 결과",
            publisher="대한체육회",
            original_publisher="대한체육회",
            authority=models.SourceAuthority.OFFICIAL,
            supports=models.EvidenceSupport.YES,
            summary="대한민국이 결승전에서 승리했다.",
            published_at=datetime(2026, 8, 8, 19, 0, tzinfo=timezone.utc),
        )

        assert evidence.authority is models.SourceAuthority.OFFICIAL
        assert evidence.supports is models.EvidenceSupport.YES
```

- [ ] **Step 2: 증거 DTO가 없어서 실패하는지 확인**

Run: `uv run pytest tests/unit/test_models.py::Test조사증거 -v`

Expected: FAIL with `AttributeError: module 'oracle_agent.models' has no attribute 'Evidence'`.

- [ ] **Step 3: 최소 enum과 증거 DTO 구현**

`src/oracle_agent/models.py`에 다음을 추가한다.

```python
class SourceAuthority(StrEnum):
    OFFICIAL = "official"
    HIGH_TRUST = "high_trust"
    OTHER = "other"


class EvidenceSupport(StrEnum):
    YES = "YES"
    NO = "NO"
    INCONCLUSIVE = "INCONCLUSIVE"


class Evidence(BaseModel):
    url: AnyHttpUrl
    title: NonEmptyText
    publisher: NonEmptyText
    original_publisher: NonEmptyText
    authority: SourceAuthority
    supports: EvidenceSupport
    summary: NonEmptyText
    published_at: AwareDatetime | None = None
```

- [ ] **Step 4: 증거 DTO 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_models.py::Test조사증거 -v`

Expected: 5 tests PASS.

- [ ] **Step 5: 증거 DTO 커밋**

```bash
git add src/oracle_agent/models.py tests/unit/test_models.py
git commit -m "feat: Agent 조사 증거 DTO 추가"
```

### Task 3: 최종 조사 결과 DTO

**Files:**
- Modify: `src/oracle_agent/models.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Consumes: Task 2의 `Evidence`, `EvidenceSupport`
- Produces: `InvestigationDecision`, `InvestigationResult(decision, evidence, summary, escalation_reason)`

- [ ] **Step 1: 실패하는 최종 결과 DTO 테스트 작성**

`tests/unit/test_models.py`에 다음 helper와 테스트를 추가한다.

```python
def _증거(supports):
    return models.Evidence(
        url="https://example.com/final-result",
        title="공식 경기 결과",
        publisher="대한체육회",
        original_publisher="대한체육회",
        authority=models.SourceAuthority.OFFICIAL,
        supports=supports,
        summary="공식 경기 결과를 확인했다.",
    )


class Test최종조사결과:
    def test_이관_결과인데_사유가_없으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="판정 불가 결과에는 이관 사유가 필요합니다"):
            models.InvestigationResult(
                decision="ESCALATED",
                evidence=[],
                summary="결과를 확인할 수 없다.",
            )

    @pytest.mark.parametrize(
        ("decision", "supports"),
        [("YES", models.EvidenceSupport.NO), ("NO", models.EvidenceSupport.YES)],
        ids=["YES에_NO근거", "NO에_YES근거"],
    )
    def test_판정과_같은_방향의_증거가_없으면_검증에_실패한다(self, decision, supports):
        with pytest.raises(ValidationError, match="판정과 같은 방향의 증거가 필요합니다"):
            models.InvestigationResult(
                decision=decision,
                evidence=[_증거(supports)],
                summary="공식 경기 결과를 확인했다.",
            )

    def test_확정_결과에_이관_사유가_있으면_검증에_실패한다(self):
        with pytest.raises(ValidationError, match="YES 또는 NO 결과에는 이관 사유를 둘 수 없습니다"):
            models.InvestigationResult(
                decision="YES",
                evidence=[_증거(models.EvidenceSupport.YES)],
                summary="대한민국이 결승전에서 승리했다.",
                escalation_reason="판정할 수 없다.",
            )

    def test_같은_방향의_증거가_있는_yes_결과이면_생성한다(self):
        result = models.InvestigationResult(
            decision=models.InvestigationDecision.YES,
            evidence=[_증거(models.EvidenceSupport.YES)],
            summary="대한민국이 결승전에서 승리했다.",
        )

        assert result.decision is models.InvestigationDecision.YES
        assert result.escalation_reason is None

    def test_사유가_있는_이관_결과이면_증거_없이_생성한다(self):
        result = models.InvestigationResult(
            decision=models.InvestigationDecision.ESCALATED,
            evidence=[],
            summary="판정 가능 시점이 아직 지나지 않았다.",
            escalation_reason="판정 가능 시점 전이다.",
        )

        assert result.decision is models.InvestigationDecision.ESCALATED
```

- [ ] **Step 2: 최종 결과 DTO가 없어서 실패하는지 확인**

Run: `uv run pytest tests/unit/test_models.py::Test최종조사결과 -v`

Expected: FAIL with `AttributeError: module 'oracle_agent.models' has no attribute 'InvestigationResult'`.

- [ ] **Step 3: 최소 최종 결과 DTO 구현**

`src/oracle_agent/models.py`에 다음을 추가한다.

```python
class InvestigationDecision(StrEnum):
    YES = "YES"
    NO = "NO"
    ESCALATED = "ESCALATED"


class InvestigationResult(BaseModel):
    decision: InvestigationDecision
    evidence: list[Evidence]
    summary: NonEmptyText
    escalation_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "InvestigationResult":
        if self.decision is InvestigationDecision.ESCALATED and self.escalation_reason is None:
            raise ValueError("판정 불가 결과에는 이관 사유가 필요합니다")
        if self.decision is not InvestigationDecision.ESCALATED and self.escalation_reason is not None:
            raise ValueError("YES 또는 NO 결과에는 이관 사유를 둘 수 없습니다")
        if self.decision is not InvestigationDecision.ESCALATED and not any(
            item.supports.value == self.decision.value for item in self.evidence
        ):
            raise ValueError("판정과 같은 방향의 증거가 필요합니다")
        return self
```

- [ ] **Step 4: 최종 결과 DTO 테스트 통과 확인**

Run: `uv run pytest tests/unit/test_models.py::Test최종조사결과 -v`

Expected: 6 tests PASS.

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `uv run pytest`

Expected: 기존 테스트를 포함해 모두 PASS.

- [ ] **Step 6: 최종 조사 결과 DTO 커밋**

```bash
git add src/oracle_agent/models.py tests/unit/test_models.py
git commit -m "feat: Agent 최종 조사 결과 DTO 추가"
```
