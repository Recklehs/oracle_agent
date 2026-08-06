# AGENTS.md 작성 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 승인된 Oracle Agent 아키텍처 설계를 이후 작업자가 바로 따를 수 있는 루트 `AGENTS.md`로 정리한다.

**Architecture:** 루트 문서 하나에 프로젝트 목적, MVP 범위, 파일 책임, Pydantic AI 우선 사용 원칙, 판정 안전 규칙, 테스트 규칙, Git 메시지 규칙을 모은다. 별도 규칙 파일이나 중복 문서는 만들지 않는다.

**Tech Stack:** Markdown, Git

## Global Constraints

- 하나의 Python 패키지 안에서 파일 단위로 책임을 나눈다.
- Agent 관련 코드만 `src/oracle_agent/agents/`에 둔다.
- Pydantic AI의 공개 기능이 책임에 직접 맞으면 자체 구현보다 우선한다.
- 자동 판정 정확도를 자동 처리율보다 우선한다.
- 단위 테스트와 통합 테스트를 모두 작성한다.
- 테스트 클래스와 함수의 실제 Python 식별자는 한국어 시나리오형 이름으로 작성한다.
- 커밋 메시지와 푸시 작업을 설명하는 메시지는 한국어로 작성한다.

---

### Task 1: 루트 AGENTS.md 작성

**Files:**
- Create: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-08-06-oracle-agent-architecture-design.md`

**Interfaces:**
- Consumes: 승인된 아키텍처 설계의 목적, 범위, 구조, 정책, 테스트 규칙
- Produces: 저장소 전체에 적용되는 루트 `AGENTS.md`

- [ ] **Step 1: AGENTS.md 작성**

다음 섹션을 순서대로 작성한다.

```text
# Oracle Agent 작업 지침
## 프로젝트 목적
## MVP 범위
## 기술과 의존성 원칙
## 패키지 구조와 파일 책임
## 판정 작업 흐름
## 자동 판정 안전 규칙
## 오류 처리와 결과 전달
## 테스트 규칙
## Git 규칙
## 단순성 원칙
```

`Git 규칙`에는 커밋 제목과 본문, push 전후 상태 보고를 한국어로 작성하도록 명시한다. Conventional Commits 접두사를 사용하는 경우 접두사는 유지하되 설명은 한국어로 작성한다.

- [ ] **Step 2: 필수 규칙 포함 여부 확인**

Run:

```bash
rg -n '프로젝트 목적|Pydantic AI|WebSearch|WebFetch|자동 판정|ESCALATED|단위 테스트|통합 테스트|한국어|커밋|푸시' AGENTS.md
```

Expected: 모든 검색어가 관련 섹션에서 한 번 이상 확인된다.

- [ ] **Step 3: Markdown과 Git diff 확인**

Run:

```bash
git diff --check
git diff -- AGENTS.md
```

Expected: `git diff --check`가 출력 없이 성공하고, diff에는 `AGENTS.md` 한 파일만 새 작업 지침으로 표시된다.

- [ ] **Step 4: 한국어 커밋 메시지로 커밋**

```bash
git add AGENTS.md docs/superpowers/plans/2026-08-06-agents-guidance.md
git commit -m "문서: Oracle Agent 작업 지침 추가"
```
