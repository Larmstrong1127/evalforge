# ADR-002: Direct httpx calls, not vendor SDKs

## Status
Accepted (2026-07)

## Context
Each provider (Anthropic, OpenAI, Google, Ollama) ships a Python SDK.

## Decision
All four adapters call the HTTP APIs directly with httpx.

## Rationale
The adapters need exactly one operation (generate a completion) behind a
shared `Provider` protocol. Four SDKs means four dependency trees, four retry
behaviors, and four exception hierarchies to normalize anyway. Direct HTTP
keeps the adapters symmetric (~50 lines each), makes error taxonomy explicit
(retryable vs not, per status code), and tests mock uniformly with respx.

## Consequence observed during implementation
Symmetry across the four adapters made the retryable-status-code sets easy to
audit side by side (Anthropic includes 529 "overloaded," which the others
don't have; Gemini omits 502, which OpenAI includes) — a real per-vendor
difference that would have been hidden behind SDK-internal retry logic if we
had used the official clients instead.

## Revisit when
We need streaming, tool use, or other provider features where SDK ergonomics
start paying for their weight.
