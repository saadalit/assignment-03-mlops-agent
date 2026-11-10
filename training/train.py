import argparse
import json
from pathlib import Path

from training import stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".local/runs/manual")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["validate", "preprocess", "train", "evaluate", "gate", "all"],
        default="all",
    )
    args = parser.parse_args()
    work = Path(args.workspace)
    work.mkdir(parents=True, exist_ok=True)
    info = {}
    if args.stage not in {"all", "validate"}:
        info = json.loads((work / "validation.json").read_text())
        info["max_depth"] = args.max_depth
    if args.stage == "gate":
        info = json.loads((work / "evaluation.json").read_text())
    for name in ["validate", "preprocess", "train", "evaluate", "gate"]:
        if args.stage not in {name, "all"}:
            continue
        if name == "validate":
            info = stages.validate(workspace=args.workspace)
        elif name == "preprocess":
            info = stages.preprocess(info)
        elif name == "train":
            info = stages.train(info, args.max_depth)
        elif name == "evaluate":
            info = stages.evaluate(info, args.min_f1)
        else:
            info = stages.quality_gate(info)
        if name == "gate":
            (work / "gate.json").write_text(json.dumps(info, indent=2))
        print(json.dumps({"stage": name, "status": "success"}))
    if args.register:
        info = stages.register_candidate(info)
    if args.export:
        stages.export_model(info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
