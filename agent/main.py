import argparse
import hmac
import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException

from agent.config import Settings
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest, AgentResponse

app = FastAPI(title="Evidence-driven MLOps Agent", version="1.0.0")


def authorize(authorization: str | None = Header(default=None)):
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        raise HTTPException(503, "Set AGENT_API_TOKEN before using the agent HTTP API")
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(401, "Authentication required")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentResponse, dependencies=[Depends(authorize)])
def chat(request: AgentRequest):
    return Orchestrator(Settings.from_env()).run(request)


def main():
    parser = argparse.ArgumentParser(description="MLOps agent: evidence, uncertainty, recommendations")
    parser.add_argument("message")
    parser.add_argument("--confirm", action="append", default=[], help="Exact action being confirmed")
    parser.add_argument("--planner", choices=["rules", "ollama"], default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings.from_env()
    if args.planner:
        settings.planner = args.planner
    response = Orchestrator(settings).run(AgentRequest(message=args.message,
                                                     confirmations=args.confirm))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
