# Decision Lineage

## Overview

Traditional version control records *what* changed and *who* changed it. Ansib-eL adds *why* through decision lineage -- an immutable audit trail that links every commit to the agent's reasoning, the task it was working on, the libraries it chose and why, and the chain of parent decisions that led to the current state.

This is critical for AI-generated code where understanding the rationale behind a change is as important as the change itself.

## Key Data Models

### DecisionRecord

Records the outcome of a human review of an agent's work.

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `str` | Unique identifier |
| `agent_id` | `str` | Agent that produced the work |
| `decision_type` | `DecisionType` | ACCEPTED, REJECTED, MODIFIED, AUTO_APPROVED, or REVIEWED |
| `commit_hash` | `str` | Associated commit |
| `timestamp` | `str` | ISO 8601 timestamp |
| `review_time_ms` | `int` | Time spent reviewing |
| `change_size` | `int` | Lines of code changed |
| `complexity` | `ChangeComplexity` | TRIVIAL, MINOR, MODERATE, MAJOR, or CRITICAL |
| `reviewer_notes` | `str` | Human reviewer's comments |
| `reviewer_id` | `str` | Who reviewed |

### ReasoningRecord

Captures the agent's chain-of-thought for a specific task.

| Field | Type | Description |
|-------|------|-------------|
| `reasoning_id` | `str` | Unique identifier |
| `agent_id` | `str` | Agent identity |
| `task_id` | `str` | Task being worked on |
| `commit_hash` | `str` | Resulting commit |
| `timestamp` | `str` | ISO 8601 timestamp |
| `reasoning` | `str` | Free-text chain-of-thought |
| `confidence` | `float` | Agent's self-reported confidence (0.0-1.0) |
| `supporting_evidence` | `List[str]` | References, benchmarks, or justifications |

### DecisionLineage

A composite view linking a commit to its full context.

| Field | Type | Description |
|-------|------|-------------|
| `commit_hash` | `str` | The commit |
| `agent_id` | `str` | Agent identity |
| `decision` | `DecisionRecord` | Review outcome |
| `reasoning` | `ReasoningRecord` | Agent's reasoning |
| `parent_commits` | `List[str]` | Commits this one depends on |
| `child_commits` | `List[str]` | Commits that depend on this one |
| `library_choices` | `Dict[str, str]` | Library name to rationale mapping |
| `full_chain` | `List[Dict]` | Complete ancestor chain |

## How It Works

When an agent completes a task and the work is reviewed, `record_complete_decision()` captures everything in a single call:

```python
decision_record, reasoning_record = manager.record_complete_decision(
    agent_id="agent-001",
    task_id="task-042",
    commit_hash="abc1234",
    decision=DecisionType.ACCEPTED,
    reasoning="Chose SQLAlchemy over raw SQL for type safety and migration support.",
    review_time_ms=12000,
    change_size=85,
    complexity=ChangeComplexity.MODERATE,
    confidence=0.88,
    supporting_evidence=["SQLAlchemy docs", "team familiarity survey"],
    parent_commits=["def5678"],
    library_choices={"sqlalchemy": "Type safety, migration support, team familiarity"}
)
```

This atomically:
1. Records the decision and updates the agent's trust score (via `TrustScorer`)
2. Stores the reasoning and lineage data (via `LineageTracker`)
3. Links parent commits to build the chain

## Querying Lineage

### Get Lineage for a Commit

```python
lineage = manager.get_decision_with_lineage("abc1234")

if lineage:
    print(f"Agent: {lineage.agent_id}")
    print(f"Decision: {lineage.decision.decision_type.name}")
    print(f"Reasoning: {lineage.reasoning.reasoning}")
    print(f"Parents: {lineage.parent_commits}")
    print(f"Libraries: {lineage.library_choices}")
```

### Trace the Decision Chain

Walk backward through the commit ancestry to see how decisions evolved:

```python
chain = manager.trust_lineage.lineage.trace_decision_chain(
    start_commit="abc1234",
    max_depth=10
)

for entry in chain:
    print(f"{entry.commit_hash[:8]} <- {entry.parent_commits}")
    if entry.reasoning:
        print(f"  Why: {entry.reasoning.reasoning[:100]}")
```

### Search Reasoning by Keyword

Find all reasoning records mentioning a specific term:

```python
results = manager.trust_lineage.lineage.search_reasoning(
    keyword="SQLAlchemy",
    limit=20
)

for record in results:
    print(f"Agent {record.agent_id}: {record.reasoning[:80]}...")
```

### Query Library Choices

Find out why a specific library was chosen for a given commit:

```python
rationale = manager.trust_lineage.lineage.get_reasoning_for_library_choice(
    commit_hash="abc1234",
    library="sqlalchemy"
)
print(rationale)  # "Type safety, migration support, team familiarity"
```

## Immutability

All lineage records use frozen dataclasses:

```python
@dataclass(frozen=True)
class DecisionRecord:
    record_id: str
    agent_id: str
    ...
```

Once created, records cannot be modified in memory. The SQLite storage layer enforces this by design -- records are inserted but never updated. This creates a tamper-evident log: any gap or inconsistency in the chain is detectable.

Prompt hashes (SHA-256) link each agent's work back to the exact prompt that produced it, preventing post-hoc attribution changes.

## Storage

Lineage data is stored in SQLite tables within `.ai-git/trust_lineage.db`:

| Table | Purpose |
|-------|---------|
| `decision_records` | Review outcomes indexed by agent_id and commit_hash |
| `reasoning_records` | Chain-of-thought indexed by agent_id and task_id |
| `lineage_chain` | Parent-child commit relationships |
| `library_choices` | Per-commit library selection rationale |

Indexes on `agent_id`, `commit_hash`, and `timestamp` support efficient queries across all tables.

## API Summary

### TrustLineageManager Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `record_complete_decision(...)` | `(DecisionRecord, ReasoningRecord)` | Record decision + reasoning atomically |
| `get_decision_with_lineage(commit_hash)` | `DecisionLineage` | Full lineage view for a commit |
| `get_agent_summary(agent_id)` | `Dict` | Trust score, tier, decision counts, recent history |

### LineageTracker Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `record_reasoning(...)` | `ReasoningRecord` | Store chain-of-thought |
| `get_lineage(commit_hash)` | `DecisionLineage` | Retrieve lineage for a commit |
| `trace_decision_chain(start_commit, max_depth)` | `List[DecisionLineage]` | Walk ancestor chain |
| `search_reasoning(keyword, agent_id, limit)` | `List[ReasoningRecord]` | Full-text search |
| `get_reasoning_for_library_choice(commit_hash, library)` | `str` | Why a library was chosen |
