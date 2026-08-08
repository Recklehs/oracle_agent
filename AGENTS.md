# Oracle Agent 작업 지침

이 문서는 저장소 전체에 적용된다. 구현 전에 이 문서와 관련 설계 문서를 읽고, 기존 결정과 충돌하는 변경은 먼저 사용자와 합의한다.

## 프로젝트 목적

Oracle Agent는 prediction market의 YES/NO 이진 시장을 종료할 때 현실 세계의 결과를 조사하고 판정하는 독립 API 서비스다.

시장이 지정한 공식 출처를 우선 확인하고, 부족하면 신뢰할 수 있는 추가 출처를 조사한다. 결과가 명확하면 `YES` 또는 `NO`로 자동 판정하고, 증거 부족·출처 충돌·조건 모호성이 있으면 `ESCALATED`로 사람에게 이관한다.

MVP의 최우선 기준은 자동 처리율이 아니라 잘못된 자동 판정을 최소화하는 것이다.

전체 서비스 설계는 `docs/superpowers/specs/2026-08-06-oracle-agent-architecture-design.md`를 참고한다. Agent의 조사·검토 흐름과 DTO는 `docs/superpowers/specs/2026-08-08-agent-investigation-dtos-design.md`를 우선한다.

## MVP 범위

포함한다.

- YES/NO 이진 시장 판정
- 비동기 작업 생성과 상태 조회
- 공식 출처 우선 조사와 추가 고신뢰 출처 조사
- Agent 내부 조사·검토·최대 3회 추가 조사와 사람 검토 이관
- SQLite 기반 작업·판정·증거 저장
- FastAPI 프로세스 내부 작업 처리
- 상태 조회 API와 완료 Webhook
- 서버 재시작 후 미완료 작업 재개

다음은 실제 요구가 생길 때까지 추가하지 않는다.

- 다중 선택 또는 수치형 시장
- 별도 워커 프로세스와 외부 작업 큐
- PostgreSQL
- 다중 Agent 합의 구조
- 관리용 웹 대시보드
- 구현이 하나뿐인 interface, adapter, factory

## 기술과 의존성 원칙

기본 기술은 Python, FastAPI, Pydantic AI, SQLite, pytest다.

Pydantic AI가 책임에 직접 맞는 공개 기능을 제공하면 자체 구현보다 먼저 사용한다. private API나 목적이 다른 기능을 억지로 재사용하지 않는다. 새 의존성을 추가하기 전에는 표준 라이브러리와 이미 설치된 의존성으로 해결할 수 있는지 확인한다.

Pydantic AI 사용 기준은 다음과 같다.

- 구조화된 최종 조사 결과: `output_type`과 Pydantic 모델
- 코드가 소유한 필드를 붙이는 최종 출력: `RunContext`를 받는 output function
- 출력 검증과 자기 수정: output function 또는 `output_validator`, `ModelRetry`
- 웹 검색과 페이지 조회: `pydantic_ai.capabilities.WebSearch`, `WebFetch`
- 실행 의존성: `deps_type`, `RunContext`
- 동시 실행 제한: `Agent(max_concurrency=...)` 또는 `ConcurrencyLimit`
- 요청·도구 호출·비용 제한: `UsageLimits`
- 모델 제공자 HTTP 재시도: 지원되는 경우 `pydantic_ai.retries.AsyncTenacityTransport`
- 실행 관찰: 실제 필요가 있을 때 `pydantic_ai.capabilities.Hooks`
- Agent 테스트: `TestModel`, `FunctionModel`, `Agent.override`, `ALLOW_MODEL_REQUESTS=False`
- 품질 평가: 실제 시장 사례가 쌓이면 Pydantic Evals

다음 책임은 Pydantic AI가 직접 제공하지 않으므로 애플리케이션에서 구현한다.

- 외부 HTTP 요청 접수: FastAPI
- 작업과 판정 결과 저장: 표준 `sqlite3`
- Webhook 전달: 프로젝트의 HTTP client

Pydantic AI의 durable execution은 Temporal, DBOS, Prefect, Restate 같은 별도 실행 인프라가 필요하므로 MVP에서는 사용하지 않는다. Deferred tools도 사람 검토 결과를 Agent 실행에 다시 주입해야 할 때만 도입한다.

## 패키지 구조와 파일 책임

하나의 Python 패키지 안에서 파일 단위로 책임을 나눈다. Agent 관련 코드만 함께 변경되므로 하위 디렉터리에 둔다.

```text
src/oracle_agent/
├── __init__.py
├── app.py
├── models.py
├── db.py
└── agents/
    ├── __init__.py
    ├── resolver.py
    ├── tools.py
    └── hooks.py       # 실제 Hook이 생길 때만 생성

tests/
├── unit/
│   ├── test_resolver.py
│   └── test_models.py
└── integration/
    └── test_api.py

tests/live/
└── resolver_live_scenarios.py  # 명시적 opt-in으로만 실행
```

- `app.py`: FastAPI 라우트, lifespan, 인프로세스 작업 루프, Agent 결과 저장과 Webhook 전달
- `models.py`: 요청, 작업, 증거, 조사 결과, 판정 결과의 Pydantic 모델과 enum
- `db.py`: SQLite 연결, 스키마 초기화, 작업 저장·선점·조회·상태 변경
- `agents/resolver.py`: 재사용 가능한 Pydantic AI Agent, 조사·검토·추가 조사 흐름, 지시문, 코드 소유 필드를 붙이는 output function
- `agents/tools.py`: 웹 검색과 페이지 조회 등 Agent 도구
- `agents/hooks.py`: 로깅·측정 등 실제 Hook이 필요할 때만 생성

프롬프트는 우선 `resolver.py`에 둔다. 여러 Agent가 공유하거나 파일 가독성을 해칠 만큼 길어질 때만 분리한다.

## 판정 작업 흐름

1. `POST /v1/resolution-jobs`가 요청을 검증한다.
2. 작업을 SQLite에 `queued`로 저장하고 `202 Accepted`, `job_id`, `status_url`을 반환한다.
3. FastAPI lifespan에서 시작된 비동기 작업 루프가 작업을 원자적으로 선점해 `running`으로 바꾼다.
4. Pydantic AI Agent가 공식 출처를 우선 조사하고 필요하면 추가 출처를 찾아 구조화된 조사 결과 초안을 만든다.
5. Agent가 초안의 근거와 결론을 검토한다.
6. 검토에서 문제가 발견되면 검토 의견을 반영해 최대 3회 추가 조사한다.
7. output function이 입력의 `prediction_id`를 붙이고 최종 `YES`, `NO`, `ESCALATED`와 구조화된 증거를 반환한다.
8. 결과를 `resolved`, `escalated`, `failed` 중 하나로 저장한다.
9. 저장 후 완료 또는 이관 Webhook을 전달한다.

상태 전이는 다음만 허용한다.

```text
queued → running → resolved | escalated | failed
```

`resolved`일 때만 `resolution`은 `YES` 또는 `NO`다. `escalated`에는 이관 사유가 있어야 한다.

MVP는 단일 Uvicorn 프로세스로 실행한다. 서버 시작 시 이전 프로세스가 남긴 모든 `running` 작업을 `queued`로 되돌리고 다시 처리한다. Agent 동시 실행 수는 Pydantic AI 기능으로 제한한다.

## Agent 최종 결과 안전 규칙

검토 과정은 Agent의 confidence 숫자만으로 `YES` 또는 `NO`를 승인하지 않는다. 다음 중 하나를 만족할 때만 `YES` 또는 `NO`를 최종 결과로 반환한다.

1. 시장이 지정한 공식 출처가 종료 조건과 결과를 명확히 확인한다.
2. 서로 독립적인 복수의 고신뢰 출처가 같은 결과를 지지한다.

공식 출처는 시장에 지정된 출처 또는 사건 결과를 발표할 권한이 있는 기관의 1차 자료다. 고신뢰 출처는 사건 당사 기관의 1차 자료나 편집 책임이 명확한 주요 보도 기관이다. 같은 보도자료의 재게시와 기사 전재는 독립 출처로 중복 계산하지 않는다.

다음은 실패가 아니라 `ESCALATED`다.

- 공식 출처와 다른 고신뢰 출처가 충돌한다.
- 독립적인 출처 수가 부족하다.
- 시장 종료 조건의 해석이 모호하다.
- 판정 시점이 아직 되지 않았다.
- 증거가 YES와 NO 중 하나를 명확히 지지하지 않는다.

## 오류 처리와 결과 전달

- 잘못된 입력은 작업을 만들지 않고 `4xx`로 거절한다.
- 모델 제공자 요청 재시도에는 가능한 경우 Pydantic AI retry transport를 사용한다.
- 검색 또는 LLM의 일시 오류는 최초 시도 이후 최대 2회 재시도한다.
- 예상하지 못한 오류는 작업을 `failed`로 저장한다.
- idempotency key로 중복 작업 생성을 막는다.
- 판정 결과를 SQLite에 먼저 저장한 뒤 Webhook을 보낸다.
- Webhook 실패는 판정 상태를 바꾸지 않으며 최초 시도를 포함해 최대 3회 시도한다.
- Webhook에 고유 `event_id`와 HMAC 서명을 포함한다.
- callback은 사전 등록하거나 허용한 호스트에만 전송한다.
- 최종 진실 공급원은 `GET /v1/resolution-jobs/{job_id}`다.

## 테스트 규칙

단위 테스트와 통합 테스트를 모두 중요하게 작성하며 기본 `pytest`에서 함께 실행한다.

- 단위 테스트: 모델 불변 조건, 상태 전이, 공식 출처 검토 승인, 복수 출처 검토 승인, 추가 조사, 이관
- 통합 테스트: FastAPI 요청, 임시 SQLite, 인프로세스 작업 루프, Agent 조사·검토 결과 처리, 상태 조회, Webhook, 재시작 복구
- 통합 테스트는 `tmp_path`로 테스트별 SQLite를 격리한다.
- Agent 모델은 `TestModel` 또는 `FunctionModel`과 `Agent.override`로 교체한다.
- `ALLOW_MODEL_REQUESTS=False`로 기본 테스트의 실제 모델 호출을 차단한다.
- 외부 검색과 Webhook만 `monkeypatch`로 대체한다.
- 실제 LLM과 인터넷을 호출하는 검증은 기본 테스트에 넣지 않는다.
- 실제 LLM과 인터넷을 호출하는 시나리오는 `tests/live/resolver_live_scenarios.py`에 둔다. 파일명이 pytest 기본 수집 패턴과 다르므로 전체 테스트에 포함되지 않으며, `RUN_ORACLE_LIVE_TESTS=1`과 파일 경로를 함께 지정한 수동 명령으로만 실행한다.

테스트 실패 node ID만 보고도 행동과 기대 결과를 한국어로 이해할 수 있어야 한다.

- 테스트 클래스와 함수의 실제 Python 식별자를 한국어 시나리오형으로 작성한다.
- 함수명은 `test_<조건>이면_<기대결과>한다` 형식을 우선한다.
- 주석이나 docstring만으로 테스트 목적을 설명하지 않는다.
- 하나의 테스트는 하나의 행동만 검증한다.
- 파라미터 테스트에는 각 사례를 구분하는 명시적 `id`를 넣는다.

```python
class Test최종결과검토:
    def test_공식_출처가_yes를_명시하면_yes를_승인한다(self):
        ...


class Test사람검토이관:
    def test_고신뢰_출처가_충돌하면_사람_검토로_이관한다(self):
        ...
```

`pyproject.toml`의 pytest 기본 설정은 다음을 사용한다.

```toml
[tool.pytest.ini_options]
addopts = "-ra -v"
testpaths = ["tests"]
```

## Git 규칙

- 커밋 제목과 본문은 한국어로 작성한다.
- Conventional Commits 접두사를 사용하는 경우 `feat:`, `fix:`, `docs:` 같은 접두사는 유지할 수 있지만 뒤의 설명은 한국어로 작성한다.
- 예: `docs: Oracle Agent 작업 지침 추가`
- push 작업 전후의 설명과 결과 보고도 한국어로 작성한다.
- 브랜치 이름, 원격 이름, 명령어처럼 기술적으로 정해진 식별자는 원문을 유지할 수 있다.
- push는 사용자가 요청했거나 명시된 작업 범위에 포함된 경우에만 수행한다.

## 단순성 원칙

- 구현이 하나뿐인 추상화와 미래를 위한 빈 파일을 만들지 않는다.
- FastAPI 라우트에 조사·검토 로직을 직접 넣지 않는다.
- Agent는 조사 결과를 검토하고 필요하면 추가 조사한 뒤 최종 `YES`, `NO`, `ESCALATED`를 반환한다.
- 파일은 책임을 이해하기 어려워졌을 때만 분리한다.
- 외부 큐, 별도 워커, PostgreSQL, 다중 Agent는 실제 필요가 확인될 때 도입한다.
- 입력 검증, 데이터 유실 방지, 보안, 금전 정산 테스트는 단순화를 이유로 생략하지 않는다.
