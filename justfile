# Oracle Agent 데모용 CLI 인터페이스
# 사전 준비: .env 파일에 OPENAI_API_KEY 필요

# 레시피 목록 출력
default:
    @just --list

# 입력 형식을 안내하고 값을 직접 입력받아 판정 실행
demo:
    uv run --env-file .env python -m oracle_agent.cli interactive

# 입력 DTO(InvestigationInput) 형식의 JSON 템플릿 생성
template file="demo_input.json":
    uv run python -m oracle_agent.cli template {{file}}

# 입력 JSON을 검증하고 판정 실행, 결과(InvestigationResult) JSON 출력
resolve file="demo_input.json":
    uv run --env-file .env python -m oracle_agent.cli resolve {{file}}

# 입력 DTO의 JSON schema 출력
schema:
    uv run python -c "import json; from oracle_agent.models import InvestigationInput; print(json.dumps(InvestigationInput.model_json_schema(), ensure_ascii=False, indent=2))"
