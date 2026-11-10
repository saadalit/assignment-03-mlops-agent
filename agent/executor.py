import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable

import httpx
from pydantic import BaseModel, ValidationError

from agent.permissions import PermissionDenied
from agent.schemas import ToolResult

logger = logging.getLogger("mlops.tools")


def redact(value: str) -> str:
    for key in ("AIRFLOW_PASSWORD", "AIRFLOW_TOKEN", "AGENT_API_TOKEN", "MLFLOW_TRACKING_TOKEN"):
        secret = os.getenv(key)
        if secret and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", value)
    value = re.sub(
        r"(?i)((?:password|token|api[_-]?key|secret)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]", value,
    )
    value = re.sub(r"https?://[^/@\s]+:[^/@\s]+@", "https://[REDACTED]@", value)
    value = re.sub(r"(?:gh[pousr]_[\w]+|github_pat_[\w]+|sk-[\w-]{10,})", "[REDACTED]", value)
    return value


def scrub(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"(?i)(password|token|secret|api_key)", k)
                else scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    return value


@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: type[BaseModel]
    handler: Callable
    confirmation: Callable[[dict], str] | None = None


class Executor:
    def __init__(self, specs: list[ToolSpec]):
        self.tools = {s.name: s for s in specs}

    def definitions(self):
        return [{"type": "function", "function": {
            "name": s.name.replace(".", "__"), "description": s.description,
            "parameters": s.inputs.model_json_schema(),
        }} for s in self.tools.values()]

    def execute(self, name: str, arguments: dict, request_id: str,
                confirmations: set[str] | None = None) -> ToolResult:
        start = time.monotonic()
        status, data, error = "ok", {}, None
        try:
            if name not in self.tools:
                raise ValueError("Unknown tool")
            spec = self.tools[name]
            args = spec.inputs.model_validate(arguments).model_dump()
            if spec.confirmation:
                from agent.permissions import require_confirmation
                require_confirmation(spec.confirmation(args), confirmations or set())
            data = spec.handler(**args)
            # Tool evidence may include arbitrary service logs; scrub before LLM/API/storage.
            data = scrub(json.loads(json.dumps(data, default=str)))
        except PermissionDenied as exc:
            status, error = "denied", str(exc)
        except (ValidationError, ValueError):
            status, error = "invalid", "Invalid tool name or arguments"
        except (TimeoutError, subprocess.TimeoutExpired, httpx.TimeoutException):
            status, error = "timeout", "Tool timed out; operational outcome is unknown"
        except Exception as exc:
            # Never disclose exception strings: SDK exceptions can embed credentials or bodies.
            status, error = "error", f"Service operation failed ({type(exc).__name__}); outcome unknown"
        duration = round((time.monotonic() - start) * 1000, 3)
        result = ToolResult(tool=name, status=status, data=data, error=error,
                            duration_ms=duration, request_id=request_id)
        logger.info(json.dumps({"tool": name if name in self.tools else "unknown",
                               "status": status, "duration_ms": duration,
                               "request_id": request_id}))
        return result
