# Oracle Agent 아키텍처 설계

## 1. 프로젝트 목적

Oracle Agent는 prediction market의 YES/NO 이진 시장을 종료할 때 현실 세계의 결과를 조사하고 판정하는 독립 API 서비스다.

에이전트는 시장에 지정된 공식 출처를 우선 확인하고, 정보가 부족하면 신뢰할 수 있는 추가 출처를 탐색한다. 공식 출처에서 결과가 명확하거나 서로 독립적인 복수의 고신뢰 출처가 일치할 때만 `YES` 또는 `NO`로 자동 판정한다. 증거가 부족하거나 출처가 충돌하거나 종료 조건이 모호하면 `ESCALATED`로 사람에게 이관한다.

MVP의 최우선 기준은 자동 처리율이 아니라 잘못된 자동 판정을 최소화하는 것이다.

## 2. MVP 범위

포함한다.

- YES/NO 이진 시장 판정
- 비동기 판정 작업 생성과 상태 조회
- 지정 공식 출처 우선 조사
- 필요한 경우 추가 고신뢰 출처 조사
- 결정론적 자동 판정 정책
- 사람 검토 이관
- SQLite 기반 작업 및 증거 저장
- 완료 Webhook과 상태 조회 API
- 서버 재시작 후 미완료 작업 재개

초기에는 포함하지 않는다.

- 다중 선택 또는 수치형 시장
- 별도 워커 프로세스나 외부 작업 큐
- PostgreSQL
- 다중 Agent 합의 구조
- 관리용 웹 대시보드
- 실제 필요가 생기기 전의 저장소 interface, adapter, factory

## 3. 기술 선택

- Python
- FastAPI
- Pydantic AI
- SQLite
- pytest

Pydantic AI는 LLM 실행, 도구 호출, 구조화된 조사 결과 생성을 담당한다. HTTP 요청 처리, 작업 저장, 자동 판정 허용 정책, Webhook 전달은 애플리케이션 코드가 담당한다.

### Pydantic AI 기능 우선 원칙

Pydantic AI가 책임에 직접 맞는 공개 기능을 제공하면 자체 구현보다 해당 기능을 우선한다. 내부 API나 목적이 다른 기능을 억지로 재사용하지 않는다.

| 책임 | 적용 방식 |
| --- | --- |
| 외부 HTTP 요청 접수 | Pydantic AI 제공 기능이 아니므로 FastAPI로 구현한다. |
| 작업과 판정 결과 저장 | Pydantic AI의 SQLite 작업 저장 기능은 없으므로 표준 `sqlite3`로 구현한다. |
| 구조화된 조사 결과 | Pydantic AI `output_type`과 Pydantic 모델을 사용한다. |
| Agent 출력 검증과 자기 수정 | `output_validator`, `ModelRetry`, Pydantic 검증을 우선 사용한다. |
| 자동 정산 허용 여부 | 금전 정산 규칙은 모델 출력 검증과 분리하여 `resolution.py`의 결정론적 코드로 구현한다. |
| 웹 검색과 페이지 조회 | `pydantic_ai.capabilities.WebSearch`와 `WebFetch`를 우선 사용한다. 선택한 모델이 native 기능을 지원하지 않으면 공식 local fallback을 사용한다. |
| Agent 의존성 전달 | `deps_type`과 `RunContext`를 사용한다. |
| Agent 동시 실행 제한 | `Agent(max_concurrency=...)` 또는 `ConcurrencyLimit`을 사용한다. |
| 모델 요청·도구 호출·비용 제한 | `UsageLimits`를 사용한다. |
| 모델 제공자 HTTP 재시도 | 제공자가 custom HTTP client를 지원하면 `pydantic_ai.retries.AsyncTenacityTransport`를 사용한다. |
| Agent lifecycle 관찰 | 실제 로깅이나 측정 요구가 생기면 `pydantic_ai.capabilities.Hooks`를 사용한다. |
| Webhook 전송 | Pydantic AI 제공 기능이 아니므로 HTTP client로 직접 구현한다. |
| 단위 테스트용 모델 대체 | `TestModel`, `FunctionModel`, `Agent.override`, `ALLOW_MODEL_REQUESTS=False`를 사용한다. |
| Agent 품질 평가 | 실제 시장 사례가 쌓이면 Pydantic Evals를 사용한다. |

Pydantic AI의 durable execution은 Temporal, DBOS, Prefect, Restate 같은 별도 실행 시스템과의 통합이다. 현재 합의한 SQLite와 단일 FastAPI 프로세스 구조를 대체하기에는 운영 요소가 늘어나므로 MVP에서는 사용하지 않는다. 별도 실행 인프라가 실제로 필요해질 때 다시 평가한다.

Deferred tools는 사람 승인을 받은 뒤 같은 Agent 대화를 이어가는 기능이다. MVP의 `ESCALATED`는 조사 작업을 종료하고 외부 운영자에게 판정을 넘기므로 사용하지 않는다. 사람 검토 결과를 Agent 실행에 다시 주입하는 요구가 생기면 도입한다.

## 4. 패키지 구조

하나의 Python 패키지 안에서 파일 단위로 책임을 나눈다. 계층별 하위 패키지는 만들지 않는다. Agent 관련 코드만 응집도가 높고 함께 변경되므로 별도 디렉터리에 둔다.

```text
src/oracle_agent/
├── __init__.py
├── app.py
├── models.py
├── db.py
├── resolution.py
└── agents/
    ├── __init__.py
    ├── resolver.py
    ├── tools.py
    └── hooks.py       # 실제 Hook이 생길 때만 생성

tests/
├── unit/
│   ├── test_resolution.py
│   └── test_models.py
└── integration/
    └── test_api.py
```

### 파일별 책임

- `app.py`: FastAPI 라우트, 애플리케이션 lifespan, 인프로세스 작업 루프를 관리한다.
- `models.py`: 요청, 작업, 증거, 조사 결과, 판정 결과의 Pydantic 모델과 enum을 정의한다.
- `db.py`: SQLite 연결, 스키마 초기화, 작업 저장·선점·조회·상태 변경을 담당한다.
- `resolution.py`: Agent를 실행하고 결정론적 정책을 적용한 후 결과 저장과 Webhook 전달을 조율한다.
- `agents/resolver.py`: 재사용 가능한 Pydantic AI Agent, 지시문, 구조화된 출력 타입을 정의한다.
- `agents/tools.py`: 웹 검색과 페이지 조회 등 Agent가 사용하는 도구를 정의한다. 최종 정산 판정은 하지 않는다.
- `agents/hooks.py`: 실행 추적, 사용량 관찰, 도구 호출 관찰처럼 실제 Hook이 필요해질 때 추가한다. 정산 정책은 포함하지 않는다.

프롬프트는 `resolver.py`에 둔다. 여러 Agent가 공유하거나 파일을 읽기 어려울 만큼 길어질 때만 분리한다.

## 5. 판정 작업 흐름

1. 요청 서버가 `POST /v1/resolution-jobs`를 호출한다.
2. API는 입력을 검증하고 작업을 SQLite에 `queued`로 저장한다.
3. API는 `202 Accepted`, `job_id`, `status_url`을 즉시 반환한다.
4. FastAPI 프로세스 내부의 비동기 작업 루프가 새 작업을 감지한다.
5. 작업 루프는 작업을 원자적으로 선점하고 `running`으로 변경한다.
6. Pydantic AI Agent가 공식 출처를 우선 조사하고 필요하면 추가 출처를 찾는다.
7. Agent는 판정 후보, 근거, 출처 유형, 출처가 지지하는 결과를 구조화해 반환한다.
8. `resolution.py`의 결정론적 정책이 자동 판정 요건을 검사한다.
9. 결과를 `resolved`, `escalated`, 또는 `failed`로 SQLite에 저장한다.
10. 완료 또는 이관 이벤트를 Webhook으로 전달한다.

작업 상태는 다음과 같다.

```text
queued → running → resolved | escalated | failed
```

`resolved`일 때만 `resolution` 값이 `YES` 또는 `NO`다. `escalated`에는 이관 사유가, `failed`에는 내부 오류 정보가 저장된다.

## 6. 인프로세스 작업 처리

별도 워커 프로세스를 두지 않는다. FastAPI lifespan에서 `asyncio` 기반 작업 루프를 시작한다.

- 새 작업이 저장되면 루프를 즉시 깨운다.
- 서버 시작 시 남아 있는 `queued` 작업을 조회한다.
- 서버 시작 시 남아 있는 모든 `running` 작업은 이전 프로세스에서 중단된 것으로 보고 `queued`로 되돌린다.
- Pydantic AI의 `Agent(max_concurrency=...)`로 동시에 실행할 Agent 작업 수를 제한한다.
- MVP는 단일 Uvicorn 프로세스로 실행한다.
- 서버가 꺼져 있는 동안에는 처리하지 않지만 재시작하면 이어서 처리한다.

다중 서버 운영이나 처리량 증가가 실제로 필요할 때만 별도 워커와 외부 큐를 도입한다.

## 7. 자동 판정 정책

Agent의 신뢰도 숫자만으로 자동 판정하지 않는다. 다음 중 하나를 만족해야 한다.

1. 시장에 지정된 공식 출처가 종료 조건과 결과를 명확히 확인한다.
2. 서로 독립적인 복수의 고신뢰 출처가 같은 결과를 지지한다.

공식 출처는 시장에 지정된 출처 또는 사건 결과를 발표할 권한이 있는 기관의 1차 자료다. 고신뢰 출처는 사건 당사 기관의 1차 자료나 편집 책임이 명확한 주요 보도 기관이다. 같은 보도자료를 재게시하거나 기사를 전재한 출처들은 하나의 출처로 계산한다.

다음 상황은 `ESCALATED`다.

- 공식 출처와 다른 고신뢰 출처가 충돌한다.
- 독립적인 출처 수가 부족하다.
- 시장 종료 조건의 해석이 모호하다.
- 판정 시점이 아직 되지 않았다.
- 확보한 증거가 YES와 NO 중 하나를 명확히 지지하지 않는다.

## 8. 결과 반환

최종 진실 공급원은 상태 조회 API다.

```text
GET /v1/resolution-jobs/{job_id}
```

완료 Webhook은 빠른 알림을 위한 보조 수단이다.

- 판정 결과를 먼저 SQLite에 저장한 후 Webhook을 전송한다.
- Webhook 실패는 판정 상태를 실패로 바꾸지 않는다.
- 전송 실패는 최초 시도를 포함해 최대 3회까지 지수 백오프로 시도한다.
- 모든 이벤트에 고유한 `event_id`를 포함한다.
- 요청 서버는 `event_id`로 중복 전달을 제거한다.
- HMAC 서명으로 본문 위조를 방지한다.
- 운영 환경에서는 사전 등록하거나 허용한 callback 호스트만 호출한다.

Webhook 이벤트 유형은 `resolution.completed`와 `resolution.escalated`를 지원한다.

## 9. 오류 처리

- 잘못된 입력은 작업을 만들지 않고 `4xx`로 거절한다.
- 검색 또는 LLM의 일시 오류는 최초 시도 이후 최대 2회 재시도한다.
- 증거 부족, 출처 충돌, 조건 모호성은 시스템 오류가 아니라 `escalated`로 처리한다.
- 예상하지 못한 오류는 작업을 `failed`로 저장한다.
- 요청 서버가 보낸 idempotency key로 중복 작업 생성을 막는다.
- Webhook 전달 상태는 판정 상태와 별도로 저장한다.

## 10. 테스트 전략

단위 테스트와 통합 테스트를 모두 중요하게 다룬다.

### 단위 테스트

- Pydantic 모델의 불변 조건
- 공식 출처 기반 자동 판정
- 복수의 독립적인 고신뢰 출처 기반 자동 판정
- 출처 충돌과 증거 부족의 이관 처리
- 상태 전이 규칙

### 통합 테스트

- FastAPI 요청부터 SQLite 저장까지의 흐름
- 인프로세스 작업 루프의 작업 선점과 실행
- 가짜 Agent 결과를 사용한 판정 완료
- 상태 조회 API 결과
- Webhook 전송과 중복 방지
- 서버 시작 시 미완료 작업 복구

통합 테스트에서는 pytest의 `tmp_path`로 테스트별 임시 SQLite를 사용한다. Agent 모델은 Pydantic AI의 `TestModel` 또는 `FunctionModel`과 `Agent.override`로 대체하고, `ALLOW_MODEL_REQUESTS=False`로 실수로 실제 모델을 호출하지 못하게 한다. 외부 검색과 Webhook 전송만 `monkeypatch`로 대체하여 네트워크와 비용 없이 전체 내부 흐름을 검증한다.

실제 LLM과 인터넷을 호출하는 테스트는 기본 `pytest` 실행에 포함하지 않는다. 실제 시장 사례가 축적된 뒤 별도 live 테스트나 Pydantic Evals 데이터셋으로 추가한다.

### 한국어 테스트 이름 규칙

테스트가 실패했을 때 node ID만 보고도 검증하는 행동과 기대 결과를 한국어로 이해할 수 있어야 한다.

- 테스트 클래스와 함수의 실제 Python 식별자를 한국어 시나리오형 이름으로 작성한다.
- 함수명은 `test_<조건>이면_<기대결과>한다` 형식을 우선한다.
- 주석이나 docstring만으로 테스트 목적을 설명하지 않는다. 실패 요약에 나타나는 이름 자체가 설명이어야 한다.
- 하나의 테스트는 하나의 행동만 검증한다.
- 파라미터 테스트는 각 사례에 의미 있는 `id`를 지정한다. 한국어 함수명이 전체 행동을 설명하고, `id`는 개별 입력 사례를 구분한다.

예시:

```python
class Test자동판정:
    def test_공식_출처가_yes를_명시하면_yes로_자동_판정한다(self):
        ...


class Test사람검토이관:
    def test_고신뢰_출처가_충돌하면_사람_검토로_이관한다(self):
        ...
```

실패 요약은 다음처럼 읽혀야 한다.

```text
FAILED tests/unit/test_resolution.py::Test사람검토이관::test_고신뢰_출처가_충돌하면_사람_검토로_이관한다
```

`pyproject.toml`에서 pytest의 상세 출력을 기본값으로 설정한다.

```toml
[tool.pytest.ini_options]
addopts = "-ra -v"
testpaths = ["tests"]
```

기본 `pytest`는 단위 테스트와 통합 테스트를 모두 실행한다.

## 11. 단순성 원칙

- 실제 구현이 하나뿐인 추상화는 만들지 않는다.
- FastAPI 라우트에서 판정 로직을 직접 구현하지 않는다.
- Agent는 조사 결과를 만들고, 결정론적 코드는 자동 정산 허용 여부를 결정한다.
- 외부 큐, 별도 워커, PostgreSQL, 다중 Agent는 실제 요구가 생길 때 추가한다.
- 파일이 책임을 이해하기 어려울 만큼 커졌을 때만 분리한다.
- 금전 정산에 영향을 주는 입력 검증, 오류 처리, 보안, 테스트는 단순화를 이유로 생략하지 않는다.
