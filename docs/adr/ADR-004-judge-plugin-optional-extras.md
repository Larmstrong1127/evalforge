# ADR-004: Judge plugin interface with heavy judges behind optional extras

## Status
Accepted (2026-07)

## Context
Judges range from trivial (`exact_match`, string comparison) to heavy (the
fine-tuned DeBERTa hallucination judge, which drags in torch + transformers —
multiple GB of dependencies most platform users will never need). All judges
must be interchangeable to the runner, and "this judge cannot score this
item" must be distinguishable from "scored it zero."

## Decision
- A single `Judge` protocol: `name` plus
  `async score(prompt, expected, output) -> Judgment | None`.
- Returning `None` means *cannot judge* (e.g. `exact_match` with no expected
  answer, or the DeBERTa judge with no context to check groundedness
  against); the runner records nothing rather than a fake zero.
- torch/transformers ship as an optional `deberta` extra
  (`pip install evalforge[deberta]`). `get_judge()` imports the DeBERTa
  module lazily, only when that judge is requested by name, so the base
  install never pays the import cost — and never breaks if the extra is
  absent but a different judge is requested.
- Even with the extra installed, model *weights* load lazily on the first
  real `score()` call, not at import or construction.

## Rationale
The registry is the extension point of the whole platform: adding a provider
or judge is one file implementing one protocol. Keeping the heavy judge out
of the base dependency set keeps `pip install` fast and the platform honest
about its footprint; the lazy import keeps the registry uniform (callers ask
by name, period). The `None` convention came out of Phase 1: averaging in
fake zeros for unjudgeable items silently corrupts scores.

## Consequence observed during implementation
Module-level (eager) torch imports inside `deberta_judge.py` — rather than
function-local lazy imports — were required for test mockability
(`@patch("...deberta_judge.AutoTokenizer")` needs the attribute to exist on
the module). This is fine precisely because the *module* itself is only
imported on request.

## Revisit when
A third heavy judge appears (consider a generic entry-point-based plugin
registry), or judge execution moves off the API process.
