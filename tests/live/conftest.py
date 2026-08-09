"""라이브 테스트 실행마다 항목별 Agent 사용량 보고서를 report/에 남긴다."""

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


REPORT_DIR = Path(__file__).parent / "report"
RESOLVER_LOGGER = "oracle_agent.agents.resolver"
SECONDS_BETWEEN_TESTS = 30

_entries: list[dict[str, Any]] = []


def _scenario_label(name: str) -> str:
    """pytest가 이스케이프한 한글 시나리오 id를 사람이 읽을 수 있게 되돌린다."""
    decoded = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        name,
    )
    if "[" in decoded and decoded.endswith("]"):
        return decoded[decoded.index("[") + 1 : -1]
    return decoded


class _UsageCollector(logging.Handler):
    """resolver의 `agent usage` 로그에서 사용량 수치를 수집한다."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.usages: list[tuple[Any, ...]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, str) and record.msg.startswith("agent usage"):
            self.usages.append(tuple(record.args or ()))


_outcome_key = pytest.StashKey[str]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    if report.when == "call":
        item.stash[_outcome_key] = report.outcome
    return report


@pytest.fixture(autouse=True)
def _agent_usage_report(request: pytest.FixtureRequest):
    if _entries:
        time.sleep(SECONDS_BETWEEN_TESTS)
    logger = logging.getLogger(RESOLVER_LOGGER)
    collector = _UsageCollector()
    original_level = logger.level
    if not logger.isEnabledFor(logging.INFO):
        logger.setLevel(logging.INFO)
    logger.addHandler(collector)
    try:
        yield
    finally:
        logger.removeHandler(collector)
        logger.setLevel(original_level)
        _entries.append(
            {
                "test": _scenario_label(request.node.name),
                "outcome": request.node.stash.get(_outcome_key, "미실행"),
                "usages": collector.usages,
            }
        )


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _entries:
        return
    REPORT_DIR.mkdir(exist_ok=True)
    now = datetime.now().astimezone()
    path = REPORT_DIR / f"live-usage-{now:%Y%m%d-%H%M%S}.md"
    lines = [
        "# 라이브 테스트 Agent 사용량 보고서",
        "",
        f"- 실행 시각: {now:%Y-%m-%d %H:%M:%S %Z}",
        f"- pytest 종료 코드: {exitstatus}",
        "",
        "| 시나리오 | 결과 | requests | tool_calls | input_tokens | output_tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    totals = [0, 0, 0, 0]
    for entry in _entries:
        if entry["usages"]:
            usage = [
                sum(_as_int(values[index]) for values in entry["usages"] if len(values) > index)
                for index in range(4)
            ]
            totals = [total + value for total, value in zip(totals, usage)]
            cells = [str(value) for value in usage]
        else:
            cells = ["-"] * 4
        lines.append(f"| {entry['test']} | {entry['outcome']} | " + " | ".join(cells) + " |")
    lines.append("| 합계 | | " + " | ".join(str(total) for total in totals) + " |")
    lines += [
        "",
        "사용량이 `-`인 항목은 조사가 완주하지 못해 최종 사용량이 기록되지 않은 실행이다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
