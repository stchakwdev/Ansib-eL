# Ansib-eL: AI-Native Version Control System

> **Where autonomous agents write code, and humans retain sovereign control.**

<!-- Badges -->
[![CI](https://github.com/stchakwdev/Ansib-eL/actions/workflows/ci.yml/badge.svg)](https://github.com/stchakwdev/Ansib-eL/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Ansib-eL is a version control system designed for AI agent workflows. Unlike traditional Git which tracks text changes, Ansib-eL tracks **decisions made by agents**, maintaining complete provenance and accountability in AI-generated code.

---

## Core Philosophy

| Traditional Git | Ansib-eL |
|----------------|----------|
| Tracks text changes | Tracks agent decisions |
| Blame humans | Attribute to agent + model + prompt |
| Single execution | Tournament mode (best of N) |
| No trust model | Reputation scoring per agent |
| Opaque reasoning | Full lineage and chain-of-thought |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first repository, basic operations |
| [CLI Reference](docs/cli-reference.md) | All `ai-git` commands, options, and examples |
| [API Reference](docs/api-reference.md) | Python class and method documentation |
| [Architecture](docs/architecture.md) | System layers, data flow, storage layout |
| [Trust and Reputation](docs/concepts/trust-and-reputation.md) | Scoring algorithm, tiers, auto-approval |
| [Tournament Mode](docs/concepts/tournament-mode.md) | Parallel execution and evaluation |
| [Decision Lineage](docs/concepts/decision-lineage.md) | Audit trail and chain-of-thought |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HUMAN OPERATOR                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Repo Manager)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Task Breaker │  │  Delegator   │  │ Approval Gate    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            v               v               v
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Agent #1  │   │ Agent #2  │   │ Agent #3  │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          v
            ┌───────────────────────┐
            │    TOURNAMENT MODE    │
            │   (Best of N wins)    │
            └───────────┬───────────┘
                        v
            ┌───────────────────────┐
            │   HUMAN REVIEW QUEUE  │
            └───────────┬───────────┘
                        v
            ┌───────────────────────┐
            │   MAIN BRANCH MERGE   │
            └───────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ansib-el

# Install as editable package
pip install -e .
```

### CLI Usage

```bash
ai-git init
ai-git branch agent-001 --purpose "Implement authentication"
ai-git commit "Add OAuth flow" --agent-id agent-001 --model-version gpt-5.2
ai-git review
ai-git merge agent/agent-001/20240115
ai-git history
```

### Python API

```python
from ansibel.ansib_el import AnsibElSystem

system = AnsibElSystem("./my-project")
system.initialize()

result = system.process_prompt(
    "Add a login page with OAuth support",
    use_tournament=True,
    num_agents=3
)
```

See [Getting Started](docs/getting-started.md) for a full walkthrough.

---

## Key Features

### Agent Identity and Attribution

Every agent gets a UUID identity with model version, prompt hash, and confidence score tracked per commit. Full provenance from prompt to merge. [Learn more](docs/concepts/decision-lineage.md)

### Tournament Mode

Spawn N agents to solve the same task in parallel. Evaluate solutions with pluggable strategies (complexity, test pass rate, requirement match). Human selects the winner; losers are archived for training data. [Learn more](docs/concepts/tournament-mode.md)

### Trust and Reputation

EMA-based scoring with time decay. Trust tiers control auto-approval thresholds -- new agents always need review, proven agents can merge small changes autonomously. [Learn more](docs/concepts/trust-and-reputation.md)

### Human-in-the-Loop

Main branch requires human approval by default. All agent changes are queued for review. Branch protection levels from NONE to LOCKED. The orchestrator is the gatekeeper.

---

## Project Structure

```
ansib-el/
├── src/ansibel/
│   ├── __init__.py           # Package exports
│   ├── ansib_el.py           # Unified API (AnsibElSystem)
│   ├── orchestrator.py       # Task breakdown, delegation, approval
│   ├── git_wrapper.py        # Git operations with AI metadata
│   ├── agent_system.py       # Agent lifecycle management
│   ├── tournament.py         # Parallel execution system
│   ├── trust_lineage.py      # Trust scoring and lineage tracking
│   ├── cli.py                # CLI (ai-git command)
│   └── exceptions.py         # Exception hierarchy
├── tests/
│   └── test_git_wrapper.py   # Unit tests
├── docs/                     # Documentation
├── .github/                  # CI and templates
├── demo.py                   # Feature walkthrough
├── dogfood.py                # Self-hosted integration test
├── pyproject.toml            # Package configuration
├── Makefile                  # Development shortcuts
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANSIBEL_DEFAULT_MODEL` | Default model for agents | `gpt-5.2` |
| `ANSIBEL_TOURNAMENT_AGENTS` | Agents per tournament | `3` |
| `ANSIBEL_TOURNAMENT_TIMEOUT` | Tournament timeout (seconds) | `300` |
| `ANSIBEL_TRUST_THRESHOLD_HIGH` | High trust threshold | `0.8` |
| `ANSIBEL_TRUST_THRESHOLD_MEDIUM` | Medium trust threshold | `0.5` |
| `ANSIBEL_STORAGE_PATH` | Metadata storage directory | `.ai-git` |

### Repository Configuration

```yaml
# .ai-git/config.yaml
orchestrator:
  main_branch_protection: human_approval
  auto_merge_threshold: verified
tournament:
  default_agents: 3
  selection_mode: human_choice
trust:
  decay_half_life_days: 30
  recovery_rate: 0.1
```

---

## Roadmap

### Phase 1: Extension (Current)

Git wrapper with AI metadata, agent management, tournament system, trust scoring, CLI interface. Status: v1.0.0-beta.

### Phase 2: Standalone

Native storage engine to reduce Git dependency. Distributed agent protocols. Multi-model orchestration.

### Phase 3: Protocol Standard

Open Agent Commit format. Interoperability standards. Community plugin ecosystem.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and security architecture.

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

**Ansib-eL**: *Decisions tracked. Trust earned. Code evolved.*
