# EvalForge API

FastAPI backend for [EvalForge](../../README.md): async evaluation runner
with pluggable provider adapters (Claude, OpenAI, Gemini, Ollama) and
pluggable judges (exact-match, LLM-as-judge, and a fine-tuned local
DeBERTa hallucination judge), exposed over a REST API and a Typer CLI.

## Development

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"
pytest tests -q
ruff check evalforge tests
mypy evalforge
uvicorn evalforge.main:app --reload    # http://localhost:8000/docs
```

Provider API keys are read from `EVALFORGE_`-prefixed env vars or a local
`.env` (see `evalforge/config.py`). The local DeBERTa judge needs the
optional extra: `pip install -e ".[deberta]"`.

See the [root README](../../README.md) for the Docker Compose quickstart
and [`../../training`](../../training) for how the local judge was trained.
