# 저장소 Bootstrap 및 main 보호 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 `develop`을 기준으로 GitHub의 보호된 `main`을 만들고, 최소 Python 구조·단위 테스트·통합 테스트·CI를 `develop → main` PR로 제안한다.

**Architecture:** GitHub 저장소는 안전한 기본값인 private로 생성한다. `main`은 현재 문서 커밋에서 시작하고 직접 변경을 막으며, 애플리케이션 기반은 `develop`에 TDD로 추가한다. CI가 성공하면 `test` 체크를 `main`의 필수 상태 검사로 등록한다.

**Tech Stack:** Git, GitHub CLI, GitHub Actions, Python 3.12+, uv, FastAPI, Pydantic, Pydantic AI, pytest

## Global Constraints

- 모든 커밋 메시지와 push 전후 설명은 한국어로 작성한다.
- 테스트 클래스와 함수의 실제 Python 식별자는 한국어 시나리오형 이름으로 작성한다.
- 단위 테스트와 통합 테스트를 기본 `pytest`에서 함께 실행한다.
- 하나의 Python 패키지 안에서 파일 단위로 책임을 나눈다.
- 구현이 없는 미래용 파일이나 interface는 만들지 않는다.
- GitHub 저장소는 `Recklehs/oracle_agent` private로 생성한다.
- `main`에는 PR 1개와 승인 리뷰 1개를 필수로 요구한다.
- `test` CI가 실제로 성공한 뒤에만 필수 상태 검사로 등록한다.

---

### Task 1: main 원격 Bootstrap과 초기 보호

**Files:**
- No file changes

**Interfaces:**
- Consumes: 로컬 `develop`의 현재 HEAD
- Produces: 로컬·원격 `main`, `origin`, 초기 branch protection

- [ ] **Step 1: develop 기준 main 생성**

```bash
git branch main develop
```

Expected: `main`과 `develop`이 같은 커밋을 가리킨다.

- [ ] **Step 2: private GitHub 저장소와 origin 생성**

```bash
gh repo create Recklehs/oracle_agent --private --source=. --remote=origin
```

Expected: `origin`이 `https://github.com/Recklehs/oracle_agent.git`을 가리킨다.

- [ ] **Step 3: main 최초 push와 기본 브랜치 설정**

```bash
git push -u origin main
gh repo edit Recklehs/oracle_agent --default-branch main
```

Expected: 원격 기본 브랜치가 `main`이다.

- [ ] **Step 4: PR과 리뷰를 필수로 설정**

GitHub branch protection API에 다음 payload를 `PUT /repos/Recklehs/oracle_agent/branches/main/protection`으로 보낸다.

```json
{
  "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
```

Expected: PR과 승인 리뷰 1개가 필요하며 아직 상태 검사는 요구하지 않는다. private 저장소 요금제에서 protection API가 거절되면 저장소 공개 전환을 임의로 하지 않고 사용자에게 선택을 요청한다.

### Task 2: Python 기반과 동작 가능한 최소 구조

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `src/oracle_agent/__init__.py`
- Create: `src/oracle_agent/models.py`
- Create: `src/oracle_agent/app.py`
- Create: `src/oracle_agent/agents/__init__.py`
- Create: `tests/unit/test_models.py`
- Create: `tests/integration/test_api.py`
- Modify: `docs/superpowers/plans/2026-08-06-repository-bootstrap.md`

**Interfaces:**
- Produces: `JobStatus`, `Resolution`, `ResolutionJob`, `GET /health`, import 가능한 `oracle_agent.agents` 패키지

- [ ] **Step 1: 프로젝트 설정 작성**

`pyproject.toml`은 Python 3.12 이상, `src` import 경로, pytest 상세 출력, runtime과 dev 의존성을 정의한다.

```toml
[project]
name = "oracle-agent"
version = "0.1.0"
description = "Prediction market resolution agent"
requires-python = ">=3.12"
dependencies = [
    "fastapi",
    "pydantic-ai",
    "uvicorn",
]

[dependency-groups]
dev = [
    "httpx",
    "pytest",
]

[tool.pytest.ini_options]
addopts = "-ra -v"
testpaths = ["tests"]
pythonpath = ["src"]
```

`.python-version`은 `3.12`, `.gitignore`는 `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `*.db`, `*.sqlite3`를 제외한다.

- [ ] **Step 2: 실패하는 단위 테스트 작성**

```python
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
```

각 테스트는 잘못된 상태 분기를 제거하거나 뒤집으면 실패해야 한다.

- [ ] **Step 3: 실패하는 통합 테스트 작성**

```python
from fastapi.testclient import TestClient

from oracle_agent.app import app


class Test헬스체크:
    def test_헬스체크를_호출하면_정상_상태를_반환한다(self):
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
```

이 테스트는 `/health` 라우트가 제거되거나 잘못된 상태를 반환하면 실패해야 한다.

- [ ] **Step 4: 의존성 잠금과 RED 확인**

```bash
uv lock
uv sync --dev
uv run pytest
```

Expected: `oracle_agent.models` 또는 `oracle_agent.app`이 없어 collection이 실패한다.

- [ ] **Step 5: 최소 구현 작성**

`src/oracle_agent/models.py`:

```python
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
```

`src/oracle_agent/app.py`:

```python
from fastapi import FastAPI


app = FastAPI(title="Oracle Agent")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

`src/oracle_agent/__init__.py`와 `src/oracle_agent/agents/__init__.py`에는 패키지 목적을 설명하는 한 줄 docstring만 둔다.

- [ ] **Step 6: GREEN 확인**

```bash
uv run pytest
```

Expected: 단위 테스트 3개와 통합 테스트 1개가 모두 통과한다.

- [ ] **Step 7: 한국어 메시지로 커밋**

```bash
git add .gitignore .python-version pyproject.toml uv.lock src tests docs/superpowers/plans/2026-08-06-repository-bootstrap.md
git commit -m "구성: Python 프로젝트 기반과 테스트 추가"
```

### Task 3: GitHub Actions 테스트 워크플로

**Files:**
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: `uv.lock`, 기본 `pytest`
- Produces: GitHub check context `test`

- [ ] **Step 1: 최소 CI 작성**

```yaml
name: 테스트

on:
  pull_request:
    branches: [main]
  push:
    branches: [develop]

permissions:
  contents: read

concurrency:
  group: test-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0
        with:
          python-version: "3.12"
          enable-cache: true
      - run: uv sync --locked --dev
      - run: uv run --frozen pytest
```

- [ ] **Step 2: 로컬 회귀 검사**

```bash
uv sync --locked --dev
uv run --frozen pytest
git diff --check
```

Expected: 4개 테스트가 통과하고 공백 오류가 없다.

- [ ] **Step 3: 한국어 메시지로 커밋**

```bash
git add .github/workflows/test.yml
git commit -m "CI: pull request 테스트 워크플로 추가"
```

### Task 4: develop push와 main 대상 PR 생성

**Files:**
- No additional file changes

**Interfaces:**
- Consumes: 로컬 `develop`의 기반·테스트·CI 커밋
- Produces: 원격 `develop`, ready-for-review PR

- [ ] **Step 1: 최종 로컬 검사와 push**

```bash
uv run --frozen pytest
git status --short --branch
git push -u origin develop
```

Expected: 원격 `develop`이 생성되고 작업 트리가 깨끗하다.

- [ ] **Step 2: develop → main PR 생성**

PR 제목은 `구성: Python 프로젝트 기반과 CI 추가`로 작성한다. 본문에는 변경 이유, 단위·통합 테스트, `uv run --frozen pytest` 결과를 한국어로 기록하고 draft가 아닌 review-ready PR로 생성한다.

### Task 5: CI 성공 확인과 test 필수 규칙 추가

**Files:**
- No file changes

**Interfaces:**
- Consumes: PR의 `test` check 성공 상태
- Produces: 리뷰 1개와 `test` 성공 없이는 병합할 수 없는 `main`

- [ ] **Step 1: PR CI 완료 대기**

```bash
gh pr checks --watch --interval 10
```

Expected: `test` check가 성공한다. 실패하면 필수 규칙을 추가하지 않고 로그를 조사해 수정한다.

- [ ] **Step 2: 성공한 test를 branch protection에 추가**

Task 1의 protection payload에서 `required_status_checks`만 다음 값으로 변경해 다시 적용한다.

```json
{
  "strict": true,
  "contexts": ["test"]
}
```

Expected: `main`은 PR, 승인 리뷰 1개, 최신 `test` 성공을 모두 요구한다.

- [ ] **Step 3: 보호 상태와 PR 병합 가능 상태 확인**

```bash
gh api repos/Recklehs/oracle_agent/branches/main/protection
gh pr view --json url,state,isDraft,reviewDecision,statusCheckRollup,mergeStateStatus
```

Expected: CI는 성공하고 PR은 다른 사용자의 승인 리뷰를 기다리는 상태다. 작성자 본인의 승인을 우회하지 않는다.
