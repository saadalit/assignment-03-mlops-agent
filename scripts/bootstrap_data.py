"""Regenerate both demonstration DVC objects from the bundled sklearn dataset."""

import hashlib
import subprocess
from pathlib import Path

import yaml


def main():
    for version, ref in (("v1", "v1.0-data"), ("v2", "v2.0-data")):
        subprocess.run(["python", "-m", "training.prepare_data", "--version", version], check=True)
        pointer = subprocess.run(
            ["git", "show", f"{ref}:data/iris.csv.dvc"], capture_output=True, text=True, check=True
        ).stdout
        expected = yaml.safe_load(pointer)["outs"][0]["md5"]
        actual = hashlib.md5(Path("data/iris.csv").read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError("Regenerated dataset differs from committed reference")
        subprocess.run(["dvc", "add", "data/iris.csv"], check=True)
        subprocess.run(["dvc", "push", "data/iris.csv.dvc"], check=True)
    print("Both dataset versions populated in DVC cache/remote; workspace restored to v2")


if __name__ == "__main__":
    main()
