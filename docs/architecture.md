# Architecture

Ansib-eL is a five-layer system that wraps Git with AI-native capabilities. Each layer has a single responsibility and communicates through well-defined interfaces.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     HUMAN OPERATOR                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│              AnsibElSystem  (ansib_el.py)                    │
│                   Unified API entry point                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        v                   v                   v
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Orchestrator │  │  AgentManager    │  │ TrustLineage     │
│              │  │                  │  │ Manager          │
└──────┬───────┘  └────────┬─────────┘  └──────────────────┘
       │                   │
       v                   v
┌──────────────┐  ┌──────────────────┐
│  GitWrapper  │  │  Tournament      │
│              │  │  Orchestrator    │
└──────────────┘  └──────────────────┘
```

## Layer Descriptions

### AnsibElSystem (`ansib_el.py`)

The top-level facade that wires all subsystems together. It exposes a small public API: `initialize()`, `process_prompt()`, `review_and_approve()`, `get_status()`, and `get_agent_info()`. Callers never need to interact with lower layers directly unless they want fine-grained control.

**Key design choice:** Single entry point. External consumers (CLI, future web UI, SDK clients) depend only on `AnsibElSystem`, insulating them from internal refactors.

### Orchestrator (`orchestrator.py`)

The central coordinator responsible for three functions:

1. **Task breakdown** -- Parses human prompts into structured `Task` objects with priorities, dependencies, and acceptance criteria.
2. **Delegation** -- Routes tasks to agents (single or tournament mode) based on capabilities and availability.
3. **Approval gatekeeper** -- Enforces branch protection levels and manages the approval queue. No merge reaches `main` without passing through this layer.

**Key design choice:** Uses `typing.Protocol` for `GitWrapperInterface`, `AgentInterface`, `TournamentInterface`, and `HumanInterface`. This enables testing with mocks and future substitution of components without changing the orchestrator.

### GitWrapper (`git_wrapper.py`)

Handles all Git operations and AI metadata storage. Wraps `subprocess` calls to Git (with `shell=False` for security) and manages the `.ai-git/` directory structure.

**Key design choice:** Metadata lives outside Git's object model in `.ai-git/metadata/` as JSON files, indexed by commit hash. This avoids polluting Git's internal data while keeping metadata colocated with the repository.

### AgentManager (`agent_system.py`)

Manages agent lifecycle from spawn to termination. Each agent gets:

- A UUID identity
- An isolated workspace branch (`agent/{id}/{purpose-slug}`)
- An `AgentContext` providing environment isolation and workspace directories
- Pydantic validation at creation time

Also provides `AgentCommunicationBus` for inter-agent messaging (publish-subscribe with correlation IDs).

**Key design choice:** Agents are data objects, not running processes. The `AgentManager` tracks state; actual execution is delegated to the tournament or orchestrator layer. This keeps the agent model portable and serializable.

### TournamentOrchestrator (`tournament.py`)

Runs N agents in parallel against the same task using `asyncio` with semaphore-controlled concurrency. After execution, solutions are scored by pluggable `EvaluationStrategy` implementations and presented for review.

Built-in evaluation strategies:
- `ComplexityEvaluator` -- Penalizes high cyclomatic complexity
- `TestPassEvaluator` -- Scores test pass rate
- `RequirementMatchEvaluator` -- Keyword matching against requirements
- `CompositeEvaluator` -- Weighted combination of the above

**Key design choice:** Async throughout. Tournament execution, evaluation, and winner selection are all `async` methods. The `AsyncAgentAdapter` wraps the synchronous `AgentManager` to fit the async protocol, so tournament code stays non-blocking.

### TrustLineageManager (`trust_lineage.py`)

Two subsystems in one module:

1. **TrustScorer** -- Maintains per-agent trust scores using exponential moving average (EMA) with time decay. Records decisions, computes tiers, and determines auto-approval eligibility.
2. **LineageTracker** -- Records chain-of-thought reasoning, library choices, and parent-child commit relationships. Supports lineage queries and keyword search.

Both persist to SQLite via `aiosqlite`.

**Key design choice:** Frozen dataclasses for all records (`DecisionRecord`, `ReasoningRecord`, `TrustScore`). Once written, records are immutable, providing a tamper-evident audit trail.

## Data Flow

A human prompt flows through the system in this order:

```
1. Human prompt
   │
   v
2. AnsibElSystem.process_prompt()
   │
   v
3. Orchestrator.process_human_prompt()
   ├── Breaks prompt into Task objects
   ├── Determines execution strategy (single vs. parallel)
   │
   v
4. AgentManager.spawn_agent()  (x N for tournament)
   ├── Creates Agent with UUID
   ├── Creates isolated workspace branch
   │
   v
5. TournamentOrchestrator.run_tournament()  (if tournament mode)
   ├── Executes agents in parallel with asyncio
   ├── Applies evaluation strategies
   ├── Presents ReviewPresentation
   │
   v
6. Orchestrator.submit_for_approval()
   ├── Queues solution in approval list
   │
   v
7. Human review (CLI or API)
   ├── approve_solution() or reject_solution()
   │
   v
8. Orchestrator._execute_merge()
   ├── GitWrapper.merge_agent_branch()
   │
   v
9. TrustLineageManager.record_complete_decision()
   ├── Updates trust score (EMA)
   ├── Records reasoning and lineage
   │
   v
10. Main branch updated
```

## Storage Layout

```
my-project/
├── .git/                    # Standard Git repository
├── .ai-git/                 # Ansib-eL metadata root
│   ├── config.yaml          # Repository configuration
│   ├── metadata.json        # Global metadata index
│   ├── metadata/            # Per-commit AI metadata (JSON)
│   │   └── <commit-hash>.json
│   ├── agents/              # Agent records
│   │   └── <agent-id>.json
│   ├── trust_scores/        # Trust score snapshots
│   │   └── <agent-id>.json
│   ├── lineage/             # Decision lineage data
│   └── trust_lineage.db     # SQLite database (trust + lineage)
└── src/                     # Project source code
```

### Per-Commit Metadata Format

Each file in `.ai-git/metadata/` stores:

```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_version": "gpt-5.2",
  "prompt_hash": "sha256:a3f5c8...",
  "timestamp": "2024-01-15T10:30:00Z",
  "parent_task": "task-001",
  "confidence_score": 0.85,
  "reasoning": "Selected JWT for stateless auth...",
  "tool_calls": ["read_file", "write_file", "run_tests"]
}
```

## Key Design Decisions

### Flat Module Structure

All modules live at the top level of `src/ansibel/` rather than in nested packages. This keeps imports simple and avoids circular dependency issues during early development. As the project grows, modules may be grouped into sub-packages.

### Dataclasses + Pydantic

Internal data structures use `@dataclass` for simplicity and performance. Pydantic models are used at system boundaries (agent creation, API input) where validation is needed. The two coexist without conflict.

### Protocol-Based Interfaces

The orchestrator defines protocols (`GitWrapperInterface`, `AgentInterface`, etc.) using `typing.Protocol`. This provides structural subtyping without requiring inheritance, making it easy to substitute test doubles or alternative implementations.

### Asyncio for Tournaments

Tournament execution is fully async to support parallel agent runs without thread overhead. The `AsyncAgentAdapter` bridges the sync `AgentManager` into the async world. Non-tournament paths remain synchronous for simplicity.

### SQLite for Trust and Lineage

SQLite provides ACID guarantees, zero-configuration deployment, and good read performance for the trust and lineage data. WAL mode enables concurrent reads during tournament execution. The database schema includes indexes on `agent_id`, `commit_hash`, and `timestamp` for common query patterns.

### GitPython as Optional Dependency

While `GitWrapper` primarily uses `subprocess` for Git operations, `gitpython` is listed as a dependency for higher-level operations (repository inspection, branch listing). The subprocess path serves as a reliable fallback.
