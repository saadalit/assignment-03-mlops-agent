"""Create local-only secrets and directories. Does not display credentials."""

import json
import secrets
from pathlib import Path

from dotenv import dotenv_values, set_key


def main():
    for path in (".local/airflow", ".local/mlflow", ".local/runs", ".local/dvc-remote"):
        Path(path).mkdir(parents=True, exist_ok=True)
    env = Path(".env")
    if not env.exists():
        env.write_text(Path(".env.example").read_text())
    values = dotenv_values(env)
    password = values.get("AIRFLOW_PASSWORD") or secrets.token_urlsafe(32)
    set_key(env, "AIRFLOW_PASSWORD", password)
    set_key(env, "AGENT_API_TOKEN", values.get("AGENT_API_TOKEN") or secrets.token_urlsafe(32))
    passwords = Path(".local/airflow/passwords.json")
    passwords.write_text(json.dumps({"agent": password}))
    # Airflow image UID 50000 must be able to write its development SQLite/log volume.
    Path(".local/airflow").chmod(0o777)
    passwords.chmod(0o644)
    env.chmod(0o600)
    print("Initialized ignored local configuration; no credentials printed")


if __name__ == "__main__":
    main()
