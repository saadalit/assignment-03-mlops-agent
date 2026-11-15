"""Run CI-equivalent checks and save dated, scrubbed command output and exit codes."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent.executor import redact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/review-2026-11-15"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = [
        ("lint", [sys.executable, "-m", "ruff", "check", "."]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        (
            "tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=agent",
                "--cov=api",
                "--cov=training",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ],
        ),
        ("configuration", [sys.executable, "-m", "scripts.validate_config"]),
        ("dvc-graph", [sys.executable, "-m", "dvc", "dag"]),
        ("repository-audit", [sys.executable, "-m", "scripts.audit_repository", "--history"]),
    ]
    results = []
    for name, command in checks:
        started = datetime.now(UTC).isoformat()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        output = redact(completed.stdout + completed.stderr)
        (args.output / f"{name}.txt").write_text(output, encoding="utf-8")
        results.append(
            {
                "check": name,
                "command": command,
                "started_at": started,
                "exit_code": completed.returncode,
                "output": f"{name}.txt",
            }
        )
        print(f"{name}: exit {completed.returncode}", flush=True)
    summary = {
        "source": "local CI-equivalent checks; not a hosted GitHub Actions run",
        "collected_at": datetime.now(UTC).isoformat(),
        "checks": results,
    }
    (args.output / "local-checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    main()
