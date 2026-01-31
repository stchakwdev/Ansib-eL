# Contributing to Ansib-eL

Thank you for your interest in contributing to Ansib-eL. This guide explains how to set up a development environment, follow project conventions, and submit changes.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git 2.30 or later
- A virtual environment manager (venv, virtualenv, or conda)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ansib-el

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify the installation
ai-git --version
python -c "from ansibel import GitWrapper; print('OK')"
```

### Running Tests

```bash
# Full test suite
make test

# With coverage report
make test-cov

# Single test
pytest tests/test_git_wrapper.py::TestGitWrapper::test_init_repo -v

# Async tests (tournament, trust lineage)
pytest tests/ -k "tournament or trust" -v
```

The project uses `pytest-asyncio` with `asyncio_mode = "auto"`, so async test functions are detected automatically.

### Integration Testing

`dogfood.py` runs Ansib-eL against its own repository as a self-hosted integration test:

```bash
python dogfood.py
```

`demo.py` provides a guided walkthrough of the full feature set:

```bash
python demo.py
```

## Project Layout

```
src/ansibel/
    __init__.py           # Public exports
    ansib_el.py           # AnsibElSystem unified API
    orchestrator.py       # Task breakdown, delegation, approval
    git_wrapper.py        # Git operations with AI metadata
    agent_system.py       # Agent lifecycle and isolation
    tournament.py         # Parallel execution, evaluation
    trust_lineage.py      # Trust scoring, decision lineage
    cli.py                # Click-based CLI (ai-git)
    exceptions.py         # Exception hierarchy
tests/
    test_git_wrapper.py   # Unit tests
docs/                     # Documentation
demo.py                  # Feature walkthrough
dogfood.py               # Self-hosted integration test
```

## Development Workflow

1. **Fork** the repository and clone your fork.
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feat/evaluation-strategy
   git checkout -b fix/trust-decay-overflow
   ```
3. **Make your changes.** Keep commits focused on a single logical change.
4. **Run checks** before committing:
   ```bash
   ruff format src/ tests/       # Auto-format
   ruff check src/ tests/        # Lint
   mypy src/                     # Type check
   make test                     # Tests
   ```
5. **Push** and open a pull request against `main`.

## Coding Standards

### Formatting and Linting

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- **Type checker**: `mypy` (Python 3.10 target)

### Conventions

- **Data models**: Use `dataclasses` for internal structures, Pydantic for validation at boundaries.
- **Async**: Tournament and trust modules use `asyncio` with `aiosqlite`. Test with `pytest-asyncio`.
- **Branch naming**: Agent branches follow `agent/{agent_id}/{purpose-slug}`.
- **Frozen dataclasses**: Immutable records (`DecisionRecord`, `ReasoningRecord`, `TrustScore`) use `@dataclass(frozen=True)`.
- **Protocols**: Interface contracts in `orchestrator.py` use `typing.Protocol` for structural subtyping.
- **Error handling**: Raise from the `ansibel.exceptions` hierarchy. Never use bare `except:`.

### Commit Messages

Write clear, imperative-mood commit messages:

```
Add composite evaluation strategy for tournaments

Combines complexity, test pass rate, and requirement match
scores with configurable weights.
```

## Pull Request Guidelines

- **CI must pass.** All checks (lint, type check, tests) run automatically.
- **One logical change per PR.** Split large changes into reviewable pieces.
- **Include tests** for new functionality or bug fixes.
- **Update documentation** if you change public APIs or CLI commands.
- **Fill out the PR template** with a summary, test plan, and related issues.

## Reporting Bugs and Feature Requests

- **Bugs**: Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
- **Features**: Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

Search existing issues before filing a new one.

## Areas of Interest

Looking for something to work on? These areas are particularly welcome:

- **Evaluation strategies**: New tournament scoring algorithms beyond the built-in set.
- **Trust algorithms**: Alternative scoring models (Bayesian, multi-factor).
- **IDE integrations**: VS Code or JetBrains plugins for `ai-git`.
- **CI/CD pipelines**: GitHub Actions, GitLab CI, or Jenkins integration recipes.
- **Agent templates**: Pre-configured agent profiles for common tasks.
- **Documentation**: Tutorials, examples, translations.
