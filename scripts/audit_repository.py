"""Audit submission files and optionally reachable Git blobs without printing secrets."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values

FORBIDDEN = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|\.local|\.venv|__pycache__|\.pytest_cache|\.ruff_cache|"
    r"mlruns|mlartifacts|models|artifacts)(?:/|$)|"
    r"(?:\.kubeconfig|\.pyc|\.db(?:-wal|-shm)?|\.log|\.coverage)$|"
    r"^\.dvc/(?:cache/|tmp/|config\.local$)"
)
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
]


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def submission_files():
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(set(p.decode() for p in paths.split(b"\0") if p))


def audit(history=False):
    files = submission_files()
    violations = []
    known = [
        v.encode()
        for k, v in dotenv_values(".env").items()
        if re.search(r"PASSWORD|TOKEN|SECRET|API_KEY", k) and v and len(v) >= 12
    ]

    def inspect(label, content):
        if any(p.search(content) for p in SECRET_PATTERNS) or any(s in content for s in known):
            violations.append({"file_or_object": label, "reason": "credential content detected"})

    for name in files:
        if name != ".env.example" and FORBIDDEN.search(name):
            violations.append({"file_or_object": name, "reason": "runtime or credential path"})
        path = Path(name)
        if path.is_file():
            inspect(name, path.read_bytes())
    blobs = 0
    if history:
        # Read object type and payload in one process; do not emit any object contents.
        objects = git("rev-list", "--objects", "--all").splitlines()
        ids = [line.split(b" ", 1)[0] for line in objects]
        batch = subprocess.run(
            ["git", "cat-file", "--batch"],
            input=b"\n".join(ids) + b"\n",
            check=True,
            capture_output=True,
        ).stdout
        position = 0
        for _ in ids:
            end = batch.index(b"\n", position)
            oid, kind, size = batch[position:end].split()
            position = end + 1
            length = int(size)
            if kind == b"blob":
                inspect("git:" + oid.decode(), batch[position : position + length])
                blobs += 1
            position += length + 1
    return {"files_checked": len(files), "history_blobs_checked": blobs, "violations": violations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    result = audit(args.history)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["violations"]))


if __name__ == "__main__":
    main()
