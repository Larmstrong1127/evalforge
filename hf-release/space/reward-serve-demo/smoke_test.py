"""End-to-end Space smoke test: launch the real Gradio app, POST one real
request over HTTP to the exposed `/score` endpoint, and assert the model
actually scored the pair.

This is deliberately NOT a stub test (see test_scoring.py for those). It boots
the same `app.py` a Space boots, loads real 184M weights, and proves the
handler answers over the wire. Writes `smoke_proof.json` as the artifact.

Run (CPU only -- the GPU may be held by other work):

    CUDA_VISIBLE_DEVICES=-1 python smoke_test.py
    CUDA_VISIBLE_DEVICES=-1 REWARD_MODEL_REPO=../../../training/checkpoints/reward-lr2e5 \
        python smoke_test.py --port 7861
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "What causes seasons?"
RESPONSE_A = (
    "Earth's axis is tilted about 23.5 degrees relative to its orbital plane, so each "
    "hemisphere receives sunlight at a steeper angle for part of the year."
)
RESPONSE_B = "Because the Earth gets closer to the Sun in summer."


def post(url: str, payload: dict, timeout: float = 180.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {"status": response.status, "body": response.read().decode("utf-8")}


def get(url: str, timeout: float = 180.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return {"status": response.status, "body": response.read().decode("utf-8")}


def wait_for_server(base: str, attempts: int = 60) -> None:
    for _ in range(attempts):
        try:
            get(base + "/", timeout=5)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise SystemExit(f"server at {base} never came up")


def call_score(base: str) -> tuple[str, float]:
    """Gradio's two-step HTTP API: POST the args, then stream the event result."""
    started = time.perf_counter()
    posted = post(base + "/gradio_api/call/score", {"data": [PROMPT, RESPONSE_A, RESPONSE_B]})
    event_id = json.loads(posted["body"])["event_id"]
    stream = get(f"{base}/gradio_api/call/score/{event_id}")["body"]
    elapsed = time.perf_counter() - started

    for line in stream.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            if isinstance(payload, list) and payload:
                return payload[0], elapsed
    raise SystemExit(f"no data frame in stream:\n{stream}")


def redact_repo(repo: str) -> str:
    """Never write a local filesystem path into the committed proof artifact.

    The smoke test is normally pointed at a checkpoint directory on whatever
    machine is running it; that path is machine-identifying noise and does not
    belong in a file that ships to a public Hub repo.
    """
    if Path(repo).exists():
        return f"<local checkpoint directory: {Path(repo).name}>"
    return repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("smoke_proof.json"))
    args = parser.parse_args()

    import app

    demo = app.demo
    demo.queue(max_size=app.QUEUE_SIZE, default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        prevent_thread_lock=True,
        share=False,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        wait_for_server(base)
        cold_start = app.SCORER.loaded
        result, elapsed = call_score(base)
        second, elapsed_warm = call_score(base)
    finally:
        demo.close()

    ok = "Response A is preferred" in result and "Margin" in result
    proof = {
        "endpoint": "POST /gradio_api/call/score",
        "model_repo": redact_repo(app.MODEL_REPO),
        "model_loaded_before_first_request": cold_start,
        "model_loaded_after_first_request": app.SCORER.loaded,
        "prompt": PROMPT,
        "response_a": RESPONSE_A,
        "response_b": RESPONSE_B,
        "first_request_seconds_including_cold_model_load": round(elapsed, 2),
        "second_request_seconds_warm": round(elapsed_warm, 2),
        "responses_identical_across_calls": result == second,
        "result_markdown": result,
        "passed": ok,
    }
    args.out.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))
    print(f"\n{'SMOKE OK' if ok else 'SMOKE FAILED'} -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
