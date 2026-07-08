# EvalForge Dashboard

Next.js frontend for [EvalForge](../../README.md): create prompt suites,
launch evaluation runs, watch them complete, browse results and costs,
cast blind A/B preference votes, and diff two runs side by side.

## Development

```bash
npm install
npm run dev      # http://localhost:3000 (expects the API on :8000)
npm run test     # vitest
npm run lint     # eslint
npm run build    # production build + type check
```

The API base URL is read from `NEXT_PUBLIC_API_BASE_URL`
(`.env.local`, defaults to `http://localhost:8000`).

See the [root README](../../README.md) for the full-stack quickstart via
Docker Compose.
