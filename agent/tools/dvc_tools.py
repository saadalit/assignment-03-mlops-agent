import hashlib
import json
import subprocess
import sys
from pathlib import PurePosixPath

import yaml

from agent.config import Settings
from agent.executor import ToolSpec
from agent.schemas import DatasetCompare, DatasetRef, Empty


class DVCTools:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _run(self, *args):
        if args[0] == "dvc":
            args = (sys.executable, "-m", "dvc", *args[1:])
        return subprocess.run(
            args,
            cwd=self.settings.repo,
            text=True,
            capture_output=True,
            check=True,
            timeout=self.settings.timeout,
        ).stdout

    def _commit(self, ref):
        if ref.startswith("-") or ".." in ref:
            raise ValueError("Invalid revision")
        return self._run("git", "rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def _pointer(self, ref, path):
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError("Dataset path must remain in repository")
        commit = self._commit(ref)
        pointer = yaml.safe_load(self._run("git", "show", f"{commit}:{path}"))
        return commit, pointer

    def list_datasets(self):
        paths = self._run("git", "ls-files", "--", "*.dvc").splitlines()
        return {
            "datasets": [{"pointer": p, **self.current_version("HEAD", p)} for p in paths],
            "pipeline": (self.settings.repo / "dvc.yaml").exists(),
        }

    def current_version(self, ref="HEAD", path="data/iris.csv.dvc"):
        commit, pointer = self._pointer(ref, path)
        outs = pointer["outs"]
        # md5 is DVC's content address; Git commit identifies the reproducible repository state.
        result = {"git_commit": commit, "requested_ref": ref, "pointer": path, "outputs": outs}
        if ref == "HEAD":
            output = (self.settings.repo / path).parent / outs[0]["path"]
            if output.is_file():
                result["workspace_md5"] = hashlib.md5(output.read_bytes()).hexdigest()
                result["matches_recorded_version"] = result["workspace_md5"] == outs[0].get("md5")
        return result

    def compare(self, ref, other_ref, path="data/iris.csv.dvc"):
        a, b = self.current_version(ref, path), self.current_version(other_ref, path)
        diff = json.loads(self._run("dvc", "diff", "--json", a["git_commit"], b["git_commit"]))
        return {"before": a, "after": b, "changed": a["outputs"] != b["outputs"], "diff": diff}

    def status(self):
        return {
            "changes": json.loads(self._run("dvc", "status", "--json")),
            "git_commit": self._commit("HEAD"),
        }

    def reproduction(self, ref="HEAD", path="data/iris.csv.dvc"):
        version = self.current_version(ref, path)
        commit = version["git_commit"]
        return {
            **version,
            "instructions": [
                f"git worktree add --detach ../dataset-reproduction {commit}",
                "cd ../dataset-reproduction",
                "uv sync --frozen",
                "uv run dvc remote add --local -d storage /absolute/path/to/dvc-remote",
                f"uv run dvc pull {path}",
                "Compare the downloaded file MD5 with outputs[0].md5 above.",
            ],
            "note": "Requires access to the DVC remote; no checkout was performed by this tool.",
        }

    def specs(self):
        return [
            ToolSpec(
                "dvc.list_datasets",
                "List Git-tracked DVC dataset pointers",
                Empty,
                self.list_datasets,
            ),
            ToolSpec(
                "dvc.current_version",
                "Read dataset MD5 and Git provenance at a revision",
                DatasetRef,
                self.current_version,
            ),
            ToolSpec(
                "dvc.compare",
                "Compare two recorded DVC dataset revisions",
                DatasetCompare,
                self.compare,
            ),
            ToolSpec(
                "dvc.status", "Detect changes in DVC-tracked data and pipeline", Empty, self.status
            ),
            ToolSpec(
                "dvc.reproduction",
                "Explain isolated reproduction of a recorded dataset",
                DatasetRef,
                self.reproduction,
            ),
        ]
