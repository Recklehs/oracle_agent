"""InvestigationInput JSON을 받아 판정 결과 JSON을 출력하는 CLI 진입점.

사용법 (justfile 참고):
    uv run --env-file .env python -m oracle_agent.cli interactive
    uv run --env-file .env python -m oracle_agent.cli template [파일]
    uv run --env-file .env python -m oracle_agent.cli resolve <파일 | ->
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# input()의 줄 편집을 readline에 맡겨 한글 백스페이스가 글자 단위로 지워지게 한다.
import readline  # noqa: F401

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from oracle_agent.models import InvestigationInput, InvestigationResult


TEMPLATE = {
    "prediction_id": "demo-0001",
    "prediction": "2026년 북중미 월드컵의 우승팀은 스페인이다",
    "resolution_criteria": (
        '"2026년 북중미 월드컵의 우승팀은 스페인이다"라는 예측이 사실이면 YES, '
        "사실이 아니면 NO다."
    ),
    "resolve_after": datetime(2026, 7, 20, tzinfo=UTC).isoformat(),
    "official_sources": [],
}


INTERACTIVE_GUIDE = """\
=== Oracle Agent 판정 요청 ===

입력 형식 (InvestigationInput):
  prediction_id        예측 식별자 (시간 기반 자동 생성 또는 직접 입력)
  prediction           판정할 예측 문장
  resolution_criteria  YES/NO 종료 조건 (방향 선택)
  resolve_after        판정 가능 시점 (지금 또는 직접 입력)
  official_sources     공식 출처 URL 목록 (쉼표로 구분, 생략 가능)

각 항목을 입력하세요. [ ] 안의 기본값은 Enter로 그대로 사용합니다.
"""

_datetime_adapter = TypeAdapter(AwareDatetime)


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("  값을 입력하세요.", file=sys.stderr)


def _prompt_prediction_id() -> str:
    while True:
        choice = _prompt("prediction_id — 1) 시간 기반 자동 생성  2) 직접 입력", "1")
        if choice == "1":
            generated = datetime.now(UTC).strftime("demo-%Y%m%d-%H%M%S")
            print(f"  prediction_id = {generated}", file=sys.stderr)
            return generated
        if choice == "2":
            return _prompt("prediction_id")
        print("  1 또는 2를 입력하세요.", file=sys.stderr)


def _prompt_resolution_criteria(prediction: str) -> str:
    while True:
        choice = _prompt(
            "resolution_criteria — 1) 사실이면 YES  2) 사실이면 NO", "1"
        )
        if choice == "1":
            criteria = f'"{prediction}"라는 예측이 사실이면 YES, 사실이 아니면 NO다.'
        elif choice == "2":
            criteria = f'"{prediction}"라는 예측이 사실이면 NO, 사실이 아니면 YES다.'
        else:
            print("  1 또는 2를 입력하세요.", file=sys.stderr)
            continue
        print(f"  resolution_criteria = {criteria}", file=sys.stderr)
        return criteria


def _prompt_resolve_after() -> datetime:
    while True:
        choice = _prompt("resolve_after — 1) 지금  2) 직접 입력", "1")
        if choice == "1":
            now = datetime.now(UTC).replace(microsecond=0)
            print(f"  resolve_after = {now.isoformat()}", file=sys.stderr)
            return now
        if choice == "2":
            break
        print("  1 또는 2를 입력하세요.", file=sys.stderr)

    while True:
        raw = _prompt("resolve_after (YYYY-MM-DD 또는 ISO datetime)")
        if len(raw) == len("YYYY-MM-DD"):
            raw += "T00:00:00+00:00"
        try:
            return _datetime_adapter.validate_python(raw)
        except ValidationError:
            print(
                "  날짜 형식이 올바르지 않습니다. 예: 2026-07-20 또는 "
                "2026-07-20T09:00:00+09:00",
                file=sys.stderr,
            )


def _prompt_investigation() -> InvestigationInput:
    print(INTERACTIVE_GUIDE)
    prediction_id = _prompt_prediction_id()
    prediction = _prompt("prediction")
    resolution_criteria = _prompt_resolution_criteria(prediction)
    resolve_after = _prompt_resolve_after()
    sources_raw = _prompt("official_sources", "")
    while True:
        try:
            investigation = InvestigationInput(
                prediction_id=prediction_id,
                prediction=prediction,
                resolution_criteria=resolution_criteria,
                resolve_after=resolve_after,
                official_sources=[
                    url.strip() for url in sources_raw.split(",") if url.strip()
                ],
            )
            break
        except ValidationError as error:
            print(f"  URL 형식이 올바르지 않습니다:\n{error}", file=sys.stderr)
            sources_raw = _prompt("official_sources", "")

    print("\n입력 DTO:", file=sys.stderr)
    print(investigation.model_dump_json(indent=2), file=sys.stderr)
    return investigation


def _show_result(result: InvestigationResult) -> None:
    print("\n=== 판정 결과 ===")
    print(f"prediction_id: {result.prediction_id}")
    print(f"decision: {result.decision}")
    print(f"summary: {result.summary}")
    if result.escalation_reason:
        print(f"escalation_reason: {result.escalation_reason}")

    evidence = result.model_dump(mode="json")["evidence"]
    if not evidence:
        return
    print(f"\n수집 증거 {len(evidence)}건:")
    for index, item in enumerate(evidence, start=1):
        print(
            f"  [{index}] {item['supports']} · {item['authority']} · "
            f"{item['publisher']} — {item['title']}"
        )
    while True:
        try:
            raw = input("\n상세히 볼 증거 번호 (Enter면 종료): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not raw:
            return
        if raw.isdigit() and 1 <= int(raw) <= len(evidence):
            print(json.dumps(evidence[int(raw) - 1], ensure_ascii=False, indent=2))
        else:
            print(f"  1부터 {len(evidence)} 사이의 번호를 입력하세요.", file=sys.stderr)


def _read_input(source: str) -> InvestigationInput:
    raw = sys.stdin.read() if source == "-" else Path(source).read_text()
    return InvestigationInput.model_validate_json(raw)


async def _resolve(investigation: InvestigationInput) -> InvestigationResult:
    from oracle_agent.agents.provider import aclose_cached_models
    from oracle_agent.agents.resolver import resolve

    try:
        return await resolve(investigation)
    finally:
        await aclose_cached_models()


def main() -> int:
    parser = argparse.ArgumentParser(prog="oracle-agent", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template", help="입력 DTO 형식의 JSON 템플릿을 만든다"
    )
    template_parser.add_argument(
        "file", nargs="?", help="저장할 경로 (생략하면 stdout에 출력)"
    )

    resolve_parser = subparsers.add_parser(
        "resolve", help="입력 JSON을 검증하고 판정을 실행한다"
    )
    resolve_parser.add_argument("file", help="입력 JSON 경로 (-는 stdin)")

    subparsers.add_parser(
        "interactive", help="입력 형식을 안내하고 값을 직접 입력받아 판정한다"
    )

    args = parser.parse_args()

    if args.command == "template":
        rendered = json.dumps(TEMPLATE, ensure_ascii=False, indent=2)
        if args.file:
            Path(args.file).write_text(rendered + "\n")
            print(f"템플릿을 {args.file}에 저장했습니다. 수정 후 실행하세요:")
            print(f"  just resolve {args.file}")
        else:
            print(rendered)
        return 0

    if args.command == "interactive":
        try:
            investigation = _prompt_investigation()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되었습니다.", file=sys.stderr)
            return 1
    else:
        try:
            investigation = _read_input(args.file)
        except FileNotFoundError:
            print(f"입력 파일을 찾을 수 없습니다: {args.file}", file=sys.stderr)
            return 1
        except ValidationError as error:
            print("입력 DTO 검증 실패:", file=sys.stderr)
            print(error, file=sys.stderr)
            return 1

    print("입력 검증 완료. 조사를 시작합니다...", file=sys.stderr)
    result = asyncio.run(_resolve(investigation))
    if args.command == "interactive":
        _show_result(result)
    else:
        print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
