# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ansib-eL is an AI-native version control system that extends Git for AI agent workflows. It tracks agent decisions (not just text changes), maintains full provenance and chain-of-thought reasoning, supports tournament-style parallel execution, and enforces human-in-the-loop approval for main branch merges.

**Status:** v1.0.0 Beta (Phase 1 - Git Extension)
**Language:** Python 3.8+

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install as editable package (registers `ai-git` CLI command)
pip install -e .

# Run tests
pytest test_git_wrapper.py -v

# Run a single test
pytest test_git_wrapper.py::TestGitWrapper::test_init_repo -v

# Type checking
mypy *.py

# Formatting
black *.py

# Linting
flake8 *.py

# Run the demo
python demo.py
```

## Architecture

The system is a 5-layer stack, all modules at the top level (flat structure, no subdirectories):

```
AnsibElSystem (ansib_el.py)           - Unified API entry point
  ├── Orchestrator (orchestrator.py)  - Task breakdown, delegation, approval gatekeeper
  ├── GitWrapper (git_wrapper.py)     - Git operations with AI metadata stored in .ai-git/
  ├── AgentManager (agent_system.py)  - Agent lifecycle, isolated workspace branches
  ├── TournamentOrchestrator (tournament.py) - Parallel agent execution via asyncio
  └── TrustLineageManager (trust_lineage.py) - Reputation scoring, decision lineage (SQLite)
```

**CLI** (`cli.py`): Click-based CLI exposed as `ai-git` command with subcommands: `init`, `status`, `commit`, `branch`, `review`, `merge`, `history`, `trust`.

### Data Flow

Human prompt -> Orchestrator (task breakdown) -> AgentManager (spawn N agents on isolated branches) -> TournamentOrchestrator (parallel execution, timeout) -> Human review queue -> TrustLineageManager (score update) -> Main branch merge

### Key Data Models

- **Agent** (`agent_system.py`): UUID identity, purpose, model version, status enum (IDLE/WORKING/COMPLETED/FAILED/TERMINATED), isolated workspace branch
- **Tournament** (`tournament.py`): SelectionMode (HUMAN_CHOICE, AUTO_BEST, THRESHOLD), SolutionStatus, async execution with configurable timeout
- **TrustScore** (`trust_lineage.py`): TrustTier enum (UNTRUSTED/LOW/MEDIUM/HIGH/VERIFIED), EMA-based scoring with 30-day decay half-life, SQLite persistence
- **DecisionRecord / ReasoningRecord** (`trust_lineage.py`): Immutable audit trail linking commits to decisions to chain-of-thought reasoning

### Storage

- `.ai-git/` directory within the repo stores metadata (`metadata.json`) and configuration (`config.yaml`)
- SQLite database for trust lineage (agent profiles, decision records, reasoning records, lineage records)

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANSIBEL_DEFAULT_MODEL` | Default model for agents | `gpt-5.2` |
| `ANSIBEL_TOURNAMENT_AGENTS` | Agents per tournament | `3` |
| `ANSIBEL_TOURNAMENT_TIMEOUT` | Tournament timeout (seconds) | `300` |
| `ANSIBEL_TRUST_THRESHOLD_HIGH` | High trust threshold | `0.8` |
| `ANSIBEL_TRUST_THRESHOLD_MEDIUM` | Medium trust threshold | `0.5` |
| `ANSIBEL_STORAGE_PATH` | Metadata storage directory | `.ai-git` |

## Key Patterns

- All modules use Python dataclasses extensively for data structures and Pydantic for validation
- Tournament system uses `asyncio` with `aiosqlite` for async parallel agent execution
- Agent branches follow the naming convention `agent/{agent_id}/{purpose-slug}`
- The `__init__.py` re-exports from `git_wrapper` and `cli` only; use direct imports for other modules (e.g., `from orchestrator import Orchestrator`)
- Tests use `pytest` with `setup_method`/`teardown_method` patterns and `tempfile` for isolated test directories
