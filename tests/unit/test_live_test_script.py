import os
import subprocess
from pathlib import Path


def test_어느_디렉터리에서_실행해도_env와_live_시나리오를_지정한다(tmp_path):
    저장소 = Path(__file__).resolve().parents[2]
    가짜_실행파일_폴더 = tmp_path / "bin"
    가짜_실행파일_폴더.mkdir()
    가짜_uv = 가짜_실행파일_폴더 / "uv"
    가짜_uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$RUN_ORACLE_LIVE_TESTS" > "$CAPTURE_ENV"\n'
        'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
    )
    가짜_uv.chmod(0o755)
    환경 = os.environ | {
        "PATH": f"{가짜_실행파일_폴더}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ENV": str(tmp_path / "env"),
        "CAPTURE_ARGS": str(tmp_path / "args"),
    }

    subprocess.run(
        [저장소 / "tests/live/run_live_tests.sh"],
        cwd=tmp_path,
        env=환경,
        check=True,
    )

    assert (tmp_path / "env").read_text().splitlines() == ["1"]
    assert (tmp_path / "args").read_text().splitlines() == [
        "run",
        "--env-file",
        str(저장소 / ".env"),
        "pytest",
        str(저장소 / "tests/live/resolver_live_scenarios.py"),
        str(저장소 / "tests/live/manual_test_scenario.py"),
        "-v",
        "-s",
    ]
