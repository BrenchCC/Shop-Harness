# Repository Guidelines

## Project Structure & Module Organization

ShopHarness is a Python customer-service agent harness for cloud APIs and optionally locally served Qwen models.

- `shopharness/core/`: agent loop, context, permissions, retrieval, memory, and handoff.
- `shopharness/llm/`, `tools/`, and `flows/`: model clients, business tools, and resumable after-sales workflows.
- `shopharness/data/`: SQLite seed data; `shopharness/cli.py`: CLI and dependency assembly.
- `skills/<skill-name>/SKILL.md`: customer-service instructions and tool allowlists.
- `tests/`: pytest tests; `eval/`: scripted conversation and trajectory checks.
- `evolve/`: proposal gates, releases, training, and SFT/DPO exports.
- `scripts/`: model download and serving helpers. Consult `DESIGN.md` for architecture and `README.md` for workflows.

## Build, Test, and Development Commands

Use Python 3.10 or newer for the core package. The optional vLLM server has its own Python, platform, and GPU requirements. Run commands from the repository root so relative skill and data paths resolve correctly.

```bash
pip install -e '.[dev]'                 # Install package and test dependencies
python -m shopharness.cli --mock        # Interactive demo without a model server
python -m pytest                       # Run the test suite
python eval/run_eval.py --gate         # Fail on trajectory regressions
pip wheel . --no-deps -w dist           # Build the package wheel using Hatchling
```

For local inference, install `.[vllm]`, review paths in `scripts/serve_vllm.sh`, and run `bash scripts/serve_vllm.sh`. Connect with `python -m shopharness.cli --endpoint http://localhost:8000/v1`.

## Coding Style & Naming Conventions

Follow surrounding Python code: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and descriptive type hints. Keep changes focused. Name skill directories with hyphens, such as `return-sop`. No formatter or linter is configured in `pyproject.toml`.

## Testing Guidelines

Use `tests/test_<feature>.py` and `test_<behavior>` names. Reuse `tests/conftest.py` fixtures for temporary SQLite databases, isolated traces, and `MockLLM`. Keep default tests independent of GPU inference. Add regression checks for changed tool permissions, confirmation flows, handoffs, and persisted state. Extend `eval/scenarios.py` for conversation behavior changes. No numerical coverage threshold is configured.

## Commit & Pull Request Guidelines

The available history contains one `feat:` commit; follow that concise, imperative style, using an appropriate prefix such as `fix:` or `docs:`. PRs should explain the behavior change, link relevant issues, and report test and evaluation results. Include a sample conversation when responses or tool sequences change.

## Configuration & Runtime Data

Configure runtime paths and model settings in `shopharness/config.py`. Keep downloaded weights, generated databases, traces, credentials, and training exports out of commits. Preserve confirmation and audit checks when modifying write tools.
