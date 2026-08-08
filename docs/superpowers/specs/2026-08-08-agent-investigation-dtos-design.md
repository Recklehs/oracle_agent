# Oracle Agent 예측 조사 DTO 설계

## 목적

서버가 특정 예측의 사실 여부를 확인해 달라고 요청하면 Oracle Agent가 웹에서 관련 사건의 결과를 조사하고, 조사 결과를 검토하고, 필요하면 추가 조사한 뒤 최종 결과 DTO를 서버에 반환한다.

최종 결과에는 예측이 맞으면 `YES`, 틀리면 `NO`, 신뢰할 수 있게 판정할 수 없으면 `ESCALATED`가 포함된다. 서버가 별도의 후단 판정 workflow를 실행하지 않으며, Oracle Agent가 조사와 검토를 마친 최종 결과를 만든다.

## 범위

이번 변경에는 다음을 포함한다.

- 서버가 Oracle Agent에 전달하는 조사 입력 DTO
- Oracle Agent가 서버에 반환하는 최종 조사 결과 DTO
- 출처 권위, 증거 방향, 최종 결과 enum
- 개별 증거 DTO
- DTO의 안전 불변 조건을 검증하는 단위 테스트
- DTO를 사용하는 조사·검토·재조사 process의 계약

다음은 이번 변경에서 구현하지 않는다.

- Pydantic AI Agent와 웹 도구의 실제 실행
- 조사 결과 reviewer와 재조사 loop
- FastAPI HTTP endpoint
- 공공데이터 OpenAPI 등의 RuleBased 검증
- SQLite 저장과 Webhook 전달

## Oracle Agent process

```text
서버
  → InvestigationInput
  → 조사 process
      → 지정 공식 출처 우선 확인
      → 필요 시 WebSearch와 WebFetch로 추가 조사
      → InvestigationResult 초안
  → 검토 process
      → 문제 있음: 검토 의견과 함께 조사 process로 반환
      → 문제 없음: 최종 InvestigationResult 승인
  → YES | NO | ESCALATED와 조사 근거를 서버에 반환
```

최초 조사 결과가 검토를 통과하지 못하면 검토 의견을 다음 조사에 전달한다. 추가 조사는 최대 2회 수행한다. 마지막 검토에서도 신뢰할 수 있는 `YES` 또는 `NO` 결론을 만들지 못하면 지금까지 확보한 증거와 구체적인 사유를 담아 `ESCALATED`로 반환한다.

### 조사 process

조사 process는 다음 순서를 따른다.

1. 예측 문장과 종료 조건을 함께 해석한다.
2. `resolve_after`가 지나지 않았으면 아직 판정할 수 없으므로 추가 웹 조사 없이 `ESCALATED` 초안을 만든다.
3. 지정된 `official_sources`가 있으면 먼저 확인한다.
4. 공식 출처만으로 결과가 명확하지 않으면 Pydantic AI의 `WebSearch`와 `WebFetch`로 추가 출처를 조사한다.
5. 각 출처가 예측의 `YES`, `NO` 중 어느 방향을 지지하는지 구조화한다.
6. 최종 결과와 근거를 `InvestigationResult` 초안으로 만든다.

### 검토 process

검토 process는 다음을 확인한다.

- 예측 문장과 종료 조건을 올바르게 해석했는가
- 출처의 실제 내용이 증거 요약과 결론을 뒷받침하는가
- 지정 공식 출처를 우선 확인했는가
- 복수 출처가 동일한 원 보도나 자료를 재게시한 것은 아닌가
- 공식 출처와 다른 고신뢰 출처가 충돌하지 않는가
- `YES`, `NO`, `ESCALATED`가 확보한 증거와 일치하는가

문제가 있지만 추가 조사로 보완할 수 있으면 부족한 근거와 확인할 항목을 조사 process에 돌려보낸다. 조건이 모호하거나 출처가 충돌하거나 추가 조사로도 증거가 부족하면 `ESCALATED`를 승인한다.

## DTO 모델

모든 모델과 enum은 기존 책임에 맞게 `src/oracle_agent/models.py`에 둔다. DTO만을 위한 새 파일이나 구현이 하나뿐인 interface는 만들지 않는다.

### `InvestigationInput`

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `question` | `str` | 확인할 예측 문장이며, 공백을 제거한 뒤 비어 있으면 안 된다. |
| `resolution_criteria` | `str` | `YES`와 `NO`를 구분하는 종료 조건이며, 공백을 제거한 뒤 비어 있으면 안 된다. |
| `resolve_after` | Pydantic `AwareDatetime` | 타임존이 포함된 판정 가능 시각이어야 한다. |
| `official_sources` | `list[AnyHttpUrl]` | 시장이 지정한 공식 출처 URL 목록이며, 없으면 빈 목록이다. |

`official_sources`는 이름이나 기관명 객체로 감싸지 않는다. 조사 process가 실제 페이지에서 게시 주체를 확인하므로 현재 interface에는 URL만 필요하다.

### `InvestigationDecision`

Oracle Agent가 서버에 반환하는 최종 결과를 다음 값으로 제한한다.

- `YES`: 예측이 맞다.
- `NO`: 예측이 틀리다.
- `ESCALATED`: 확보한 근거로 신뢰할 수 있게 판정할 수 없다.

### `SourceAuthority`

출처의 권위를 다음 값으로 제한한다.

- `OFFICIAL = "official"`: 시장 지정 출처 또는 사건 결과를 발표할 권한이 있는 기관의 1차 자료
- `HIGH_TRUST = "high_trust"`: 편집 책임이 명확한 주요 보도 기관 등 고신뢰 출처
- `OTHER = "other"`: 단독으로 신뢰할 수 있는 결론을 만들기 어려운 기타 출처

### `EvidenceSupport`

개별 증거가 예측을 지지하는 방향을 다음 값으로 제한한다.

- `YES`: 예측이 맞다는 방향을 지지한다.
- `NO`: 예측이 틀리다는 방향을 지지한다.
- `INCONCLUSIVE`: 어느 방향도 명확히 지지하지 않는다.

### `Evidence`

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `url` | `AnyHttpUrl` | 증거를 직접 확인할 수 있는 페이지 URL이다. |
| `title` | `str` | 공백을 제거한 뒤 비어 있으면 안 된다. |
| `publisher` | `str` | 현재 페이지를 게시한 주체이며 비어 있으면 안 된다. |
| `original_publisher` | `str` | 원 보도나 원 자료의 게시 주체이며 비어 있으면 안 된다. |
| `authority` | `SourceAuthority` | 출처 권위 분류다. |
| `supports` | `EvidenceSupport` | 해당 증거가 예측을 지지하는 방향이다. |
| `summary` | `str` | 종료 조건과 직접 관련된 사실 요약이며 비어 있으면 안 된다. |
| `published_at` | `AwareDatetime | None` | 페이지에서 확인할 수 있을 때만 기록한다. |

재게시나 기사 전재는 `original_publisher`를 동일하게 기록한다. 예를 들어 다른 사이트에 전재된 Reuters 기사라면 `publisher`는 현재 사이트, `original_publisher`는 `Reuters`다. 검토 process는 이 값을 사용해 동일한 원 출처를 독립 출처로 중복 계산하지 않는다.

### `InvestigationResult`

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `decision` | `InvestigationDecision` | 검토를 마친 Oracle Agent의 최종 결과다. |
| `evidence` | `list[Evidence]` | 조사한 증거 목록이며, 증거를 찾지 못한 `ESCALATED` 결과에서는 빈 목록일 수 있다. |
| `summary` | `str` | 예측, 확인한 실제 사건 결과, 결론의 관계를 설명하며 비어 있으면 안 된다. |
| `escalation_reason` | `str | None` | `ESCALATED`일 때 필수이며, `YES`와 `NO`에서는 없어야 한다. |

다음 불변 조건을 Pydantic 검증으로 강제한다.

- `decision`이 `YES` 또는 `NO`이면 같은 방향의 증거가 하나 이상 있어야 한다.
- `decision`이 `ESCALATED`이면 공백이 아닌 `escalation_reason`이 있어야 한다.
- `decision`이 `YES` 또는 `NO`이면 `escalation_reason`은 없어야 한다.

공식 출처 여부, 독립 출처 수, 출처 간 충돌처럼 의미 판단이 필요한 항목은 DTO 생성 검증만으로 승인하지 않고 검토 process에서 확인한다.

## 오류 처리

- 판정 시점이 아직 되지 않았거나 증거가 부족하거나 출처가 충돌하면 `ESCALATED` 결과를 반환한다.
- 조사와 검토를 최초 시도 이후 최대 2회 반복해도 결론을 승인할 수 없으면 `ESCALATED` 결과를 반환한다.
- 예상하지 못한 프로그래밍 오류나 저장 오류를 사실 판정 불가로 위장하지 않는다. 이런 오류는 서버의 작업 실패 처리로 전달한다.

## 테스트

`tests/unit/test_models.py`에 다음 행동을 각각 하나의 테스트로 추가한다.

- 입력의 `resolve_after`에 타임존이 없으면 검증에 실패한다.
- 공백뿐인 입력 문자열은 검증에 실패한다.
- `ESCALATED` 결과에 이관 사유가 없으면 검증에 실패한다.
- `YES` 또는 `NO` 결과에 같은 방향의 증거가 없으면 검증에 실패한다.
- `YES` 또는 `NO` 결과에 이관 사유가 있으면 검증에 실패한다.
- 유효한 입력과 증거를 가진 최종 조사 결과는 생성된다.

테스트 함수는 저장소 규칙에 따라 한국어 시나리오형 이름을 사용한다. 실제 모델 또는 인터넷 호출은 하지 않는다.
