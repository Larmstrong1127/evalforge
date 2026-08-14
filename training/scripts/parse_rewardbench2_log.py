"""Turn a RewardBench 2 harness log into a results record.

`scripts/run_v2.py` prints its per-domain results and, with `--do_not_save`,
never writes an aggregate file. The numbers quoted in the model card and README
are therefore parsed straight out of the run log rather than transcribed by
hand, so a reader can diff the published table against the raw log.

The overall score is the **unweighted mean of the six domains**, matching the
RewardBench 2 leaderboard's "Average" column — deliberately not a row-weighted
mean, which would let the two largest domains dominate.

Usage:
    python training/scripts/parse_rewardbench2_log.py run.log \
        --model ... --params 184M --encoding text_pair --device cpu \
        --batch-size 32 --max-length 512 --wall-clock-seconds 7200 \
        --merge-into training/results/rewardbench2.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# "Factuality: 190.0/475 (0.4)" — and, for the Ties domain, which uses the
# harness's margin-aware scoring: "ties: Overall score 0.31".
SUBSET_LINE = re.compile(r"^([A-Za-z ]+): (\d+(?:\.\d+)?)/(\d+) \(([\d.]+)\)\s*$", re.M)
TIES_LINE = re.compile(r"^(\w+): Overall score ([\d.]+)\s*$", re.M)

DOMAINS = ("Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties")


def parse_log(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, _correct, _total, acc in SUBSET_LINE.findall(text):
        scores[name.strip()] = float(acc)
    for name, acc in TIES_LINE.findall(text):
        scores[name.strip().title()] = float(acc)
    if not scores:
        raise SystemExit("no per-domain result lines found in log")
    return scores


def overall(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--wall-clock-seconds", type=float, required=True)
    parser.add_argument("--merge-into", type=Path, default=None)
    args = parser.parse_args()

    scores = parse_log(args.log.read_text(encoding="utf-8", errors="replace"))
    record = {
        "model": args.model,
        "params": args.params,
        "encoding": args.encoding,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "wall_clock_seconds": round(args.wall_clock_seconds, 1),
        "per_domain": {k: round(v, 4) for k, v in sorted(scores.items())},
        "overall": round(overall(scores), 4),
    }
    print(json.dumps(record, indent=2))

    if args.merge_into:
        path = args.merge_into
        blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
        key = (args.model, args.encoding)
        blob["runs"] = [r for r in blob["runs"] if (r["model"], r["encoding"]) != key]
        blob["runs"].append(record)
        blob["runs"].sort(key=lambda r: (r["model"], r["encoding"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
        print(f"merged into {path}")


if __name__ == "__main__":
    main()
