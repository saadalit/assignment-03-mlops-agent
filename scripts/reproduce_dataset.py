"""Verify an immutable dataset reference in a disposable Git worktree."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("--remote", required=True, help="Existing local DVC remote path")
    args = parser.parse_args()
    if args.ref.startswith("-") or ".." in args.ref:
        raise ValueError("Invalid revision")
    repo = Path.cwd()
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{args.ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    env = {**os.environ, "PATH": str(repo / ".venv/bin") + os.pathsep + os.environ["PATH"]}
    with tempfile.TemporaryDirectory(prefix="assignment3-reproduce-") as directory:
        target = Path(directory) / "checkout"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(target), commit],
            check=True,
            capture_output=True,
        )
        try:
            for command in (
                [
                    "dvc",
                    "remote",
                    "add",
                    "--local",
                    "-f",
                    "-d",
                    "storage",
                    str(Path(args.remote).resolve()),
                ],
                ["dvc", "pull", "data/iris.csv.dvc"],
            ):
                subprocess.run(
                    command, cwd=target, env=env, check=True, capture_output=True, timeout=90
                )
            expected = yaml.safe_load((target / "data/iris.csv.dvc").read_text())["outs"][0]["md5"]
            actual = hashlib.md5((target / "data/iris.csv").read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError("Reproduced dataset checksum mismatch")
            print(
                json.dumps(
                    {
                        "ref": args.ref,
                        "git_commit": commit,
                        "expected_md5": expected,
                        "actual_md5": actual,
                        "match": True,
                        "method": "git worktree + real dvc pull from configured local remote",
                    },
                    indent=2,
                )
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(target)],
                check=True,
                capture_output=True,
            )


if __name__ == "__main__":
    main()
