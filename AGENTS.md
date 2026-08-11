# AGENTS.md

DIALX course repo: 13 independent Python tasks (`t1_llm_api` … `t13_final_task`) that build AI apps directly on vendor APIs (OpenAI/Anthropic/Gemini). It is a learning repo, not a shipped product — there are **no tests, no linter config, no CI**. Each task ships as a README + code with `TODO` markers; the README for the task you're working on is the spec and the source of truth.

## Running code

- Setup (from repo root): `python -m venv .venv && .venv/bin/pip install -r requirements.txt`. Requires Python 3.11+ (system Python on macOS is too old).
- **Always run scripts from the repo root**, not from inside a task dir — code imports the `commons` package and sibling tasks via absolute module paths (e.g. `from t2_llms_output_tuning._clients._base_client import AIClient`).
- No test runner. Verification = executing the app. Many apps are interactive `input()` REPL loops (e.g. `t2/_main.py`); they can't run headless. Some print raw request/response JSON — expected.
- `t12_skills` warns Anthropic/OpenAI skills consume large token volumes; be careful when testing.

## Environment

- API keys come from env vars, read at import time in `commons/constants.py`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`. Most tasks require real keys and make paid calls.
- Vendor keys are also used as Postman env variables (collections in task dirs).

## Docker services

Start infra with `docker compose up -d` from the specific task dir (each task has its own compose file). Services are shared across tasks, so only start what a task's README asks for — port collisions are real (see t11 README which tells you to stop the old user-service container first).

- **User Service** (mock user backend): image `khshanovskyi/mockuserservice`, `localhost:8041`, endpoints `/v1/users*` + `/health` + Swagger at `/docs`. Used by t6, t8, t9, t10, t11, t13. `commons/user_service/client.py` already wraps it.
- **t5 (RAG advanced)**: Postgres+pgvector on **port 5433** (deliberately not 5432), user/password `postgres`.
- **t11 (MCP auth)**: also runs Keycloak on `localhost:8089` (admin/admin, realm pre-imported from `keycloak/`); MCP servers on ports 8007 (API key) and 8008 (OAuth).
- **t12 (skills)**: MCP Python code interpreter on `localhost:8050/mcp`.
- **t13 (final task)**: UMS MCP server `8005`, Redis `6379`, Redis Insight `6380`, agent app on `localhost:8011`.

## Conventions

- Tasks are **intentionally repetitive** (re-implementing clients/agents per provider) — duplicate code across tasks is by design.
- Internal/private modules are prefixed with `_`: `_main.py`, `_clients/`, `_openai_client.py`. Tasks commonly ship both an SDK-based client and a raw-HTTP `custom_client.py` side by side.
- Each task's README lists the exact TODO files in order; "switch client X for Y" steps usually mean editing `main()`/`app.py` in that task.
- Git branches: `main` (this), `main-detailed` (more detailed instructions), `completed` (solutions; does not exist for t13).
