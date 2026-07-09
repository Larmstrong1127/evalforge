# Contributing

Thanks for your interest in EvalForge. It's a small project with a simple
workflow.

## Setup

See the per-package READMEs: [`platform/api`](platform/api/README.md),
[`platform/dashboard`](platform/dashboard/README.md), and
[`training`](training/README.md). The short version:

```bash
# API
cd platform/api && pip install -e ".[dev]"
# Dashboard
cd platform/dashboard && npm install
```

## Before opening a PR

Everything CI checks, you can run locally:

```bash
# API
cd platform/api
ruff check . && mypy --strict evalforge && pytest -q

# Dashboard
cd platform/dashboard
npm run lint && npx tsc --noEmit && npm test && npm run build

# Training (only if you touched training/)
cd training
ruff check . && pytest -q
```

PRs that touch `platform/api/**` also run the [eval gate](.github/workflows/eval-gate.yml):
a fixed suite against a pinned local Ollama model, failing on score regression.

## Conventions

- **Conventional commits** (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, …).
- **Tests first** where practical — most of the codebase was built TDD and
  reviewers will ask where the test is.
- Architectural decisions get an ADR in [`docs/adr`](docs/adr) when they're
  non-obvious or reverse a previous decision.
- New providers and judges implement the existing protocols defined in
  `evalforge/providers/__init__.py` and `evalforge/judges/__init__.py`, and
  register in the same file's `get_*` registry — one file, one protocol,
  no framework.

## Extending

The two intended extension points:

- **Provider:** implement `generate()` returning text + token counts, add to
  the registry, add a pricing entry if it's a paid API.
- **Judge:** implement `async score(...) -> Judgment | None` (`None` =
  "cannot judge this item" — never fake a zero). Heavy dependencies go
  behind an optional extra like the existing `deberta` extra (see
  [ADR-004](docs/adr/ADR-004-judge-plugin-optional-extras.md)).
