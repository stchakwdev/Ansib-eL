# API Reference

This document covers the public Python API for Ansib-eL. All classes are importable from the `ansibel` package or its submodules.

## Package Structure

```
ansibel/
    __init__.py          # Re-exports: GitWrapper, AgentMetadata, MergeResult, main, exceptions
    ansib_el.py          # AnsibElSystem, SystemStatus
    orchestrator.py      # Orchestrator, Task, Solution, ApprovalResult, protocols
    git_wrapper.py       # GitWrapper, AgentMetadata, MergeResult
    agent_system.py      # AgentManager, Agent, AgentStatus, AgentContext
    tournament.py        # TournamentOrchestrator, Tournament, evaluation strategies
    trust_lineage.py     # TrustLineageManager, TrustScorer, LineageTracker, enums
    exceptions.py        # Exception hierarchy
```

---

## ansibel.ansib_el

### AnsibElSystem

The unified entry point for all Ansib-eL operations. Wires together all subsystems.

```python
class AnsibElSystem:
    def __init__(self, repo_path: str = ".") -> None
```

**Attributes:**
- `repo_path: Path` -- Absolute path to the repository
- `git: GitWrapper` -- Git operations
- `agents: AgentManager` -- Agent lifecycle
- `orchestrator: Orchestrator` -- Task coordination
- `tournament: TournamentOrchestrator` -- Parallel execution
- `trust_lineage: TrustLineageManager` -- Trust and lineage tracking

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Set up repository and all subsystems |
| `process_prompt(prompt, use_tournament=False, num_agents=3)` | `Dict[str, Any]` | Main entry point: break down prompt, execute, queue for review |
| `review_and_approve(approval_id, approve, comments="")` | `Dict[str, Any]` | Approve or reject a pending solution |
| `get_status()` | `SystemStatus` | Aggregate system status |
| `list_pending_approvals()` | `List[Dict[str, Any]]` | All solutions awaiting review |
| `get_agent_info(agent_id)` | `Dict[str, Any]` | Agent details including trust score |

### SystemStatus

```python
@dataclass
class SystemStatus:
    repo_initialized: bool
    active_agents: int
    pending_approvals: int
    total_commits: int
    trust_scores: Dict[str, float]
    recent_tournaments: List[str]
```

---

## ansibel.git_wrapper

### GitWrapper

Handles Git operations and `.ai-git/` metadata storage.

```python
class GitWrapper:
    AI_GIT_DIR = '.ai-git'

    def __init__(self, repo_path: str = '.') -> None
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `init_repo()` | `bool` | Initialize Git repo and `.ai-git/` structure |
| `is_initialized()` | `bool` | Check if repo has `.ai-git/` |
| `create_agent_branch(agent_id, purpose)` | `str` | Create and checkout `agent/{id}/{timestamp}` branch |
| `commit_with_metadata(message, metadata, files=None)` | `str` | Commit with AI metadata stored in `.ai-git/` |
| `commit_changes(message, files=None)` | `str` | Plain commit without AI metadata |
| `get_commit_metadata(commit_hash)` | `Optional[AgentMetadata]` | Retrieve metadata for a commit |
| `merge_agent_branch(branch_name, strategy="merge", target_branch=None)` | `MergeResult` | Merge with strategy (merge/squash/rebase) |
| `list_agent_branches(status=None)` | `List[Dict[str, Any]]` | List agent branches, optionally filtered |
| `get_diff(branch_a, branch_b)` | `str` | Unified diff between two branches |
| `get_status()` | `Dict[str, Any]` | Branch, dirty state, files, active agents |
| `get_ai_enhanced_history(limit=20)` | `List[Dict[str, Any]]` | Commit log with metadata annotations |
| `resolve_conflict(file_path, resolution)` | `bool` | Apply conflict resolution |
| `update_agent_trust_score(agent_id, score_delta)` | `float` | Adjust trust score, returns new value |
| `get_agent_trust_score(agent_id)` | `float` | Current trust score |

**Static Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `_validate_git_ref(ref_name)` | `str` | Validate ref name against injection; raises on invalid |
| `_atomic_write_json(path, data)` | `None` | Write JSON atomically via tempfile + rename |

### AgentMetadata

```python
@dataclass
class AgentMetadata:
    agent_id: str
    model_version: str
    prompt_hash: str
    timestamp: str
    parent_task: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[List[str]] = None
```

Has a `to_dict()` method for JSON serialization.

### MergeResult

```python
class MergeResult(NamedTuple):
    success: bool
    message: str
    conflicts: Optional[List[str]]
    merged_commit: Optional[str]
```

---

## ansibel.agent_system

### AgentManager

Manages agent lifecycle, workspace isolation, and persistence.

```python
class AgentManager:
    def __init__(self, storage_path: str) -> None
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `spawn_agent(purpose, model_version, prompt, task_id=None, parent_task_id=None)` | `Agent` | Create agent with UUID, workspace branch, and context |
| `get_agent(agent_id: UUID)` | `Optional[Agent]` | Look up by UUID |
| `get_agent_by_string_id(agent_id_str)` | `Optional[Agent]` | Look up by string ID |
| `list_active_agents()` | `List[Agent]` | Agents with status IDLE or WORKING |
| `list_all_agents()` | `List[Agent]` | All agents regardless of status |
| `list_agents_by_status(status)` | `List[Agent]` | Filter by AgentStatus |
| `get_agents_by_task(task_id)` | `List[Agent]` | All agents assigned to a task |
| `update_agent_status(agent_id, status)` | `bool` | Change agent status |
| `terminate_agent(agent_id, cleanup=True)` | `bool` | Set TERMINATED, optionally clean workspace |
| `get_agent_context(agent_id)` | `Optional[AgentContext]` | Get isolated execution context |
| `get_statistics()` | `Dict[str, Any]` | Counts by status, total agents |
| `cleanup_terminated(max_age_hours=24)` | `int` | Remove old terminated agents, returns count |

### Agent

```python
@dataclass
class Agent:
    agent_id: UUID
    purpose: str
    model_version: str
    prompt_hash: str
    created_at: datetime
    status: AgentStatus
    workspace_branch: str
    parent_task_id: Optional[str] = None
    metadata: Optional[AgentMetadata] = None
```

### AgentStatus

```python
class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
```

### AgentContext

Provides environment isolation for agent execution.

```python
class AgentContext:
    def __init__(self, agent_id: UUID, base_workspace_path: Path) -> None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Create workspace directories |
| `set_env_var(key, value)` | `None` | Set isolated environment variable |
| `set_env_vars(vars_dict)` | `None` | Set multiple variables |
| `get_env_var(key, default=None)` | `Optional[str]` | Read variable |
| `activate()` | `None` | Apply environment isolation |
| `deactivate()` | `None` | Restore original environment |
| `get_subprocess_env()` | `Dict[str, str]` | Environment dict for subprocess calls |
| `get_workspace_file_path(filename, subdir=None)` | `Path` | Path within workspace |
| `cleanup()` | `None` | Remove workspace directory |

Supports context manager protocol (`with agent_context:`).

### AgentCommunicationBus

Publish-subscribe messaging between agents.

```python
class AgentCommunicationBus:
    def send_message(sender_id, recipient_id, message_type, payload, correlation_id=None) -> AgentMessage
    def subscribe(agent_id, callback) -> None
    def unsubscribe(agent_id, callback) -> None
    def get_messages_for_agent(agent_id, message_type=None) -> List[AgentMessage]
    def clear_messages(agent_id) -> None
```

---

## ansibel.orchestrator

### Orchestrator

Central coordinator for task breakdown, delegation, and approval.

```python
class Orchestrator:
    DEFAULT_MAIN_BRANCH = "main"
    DEFAULT_PROTECTION_LEVEL = BranchProtectionLevel.HUMAN_APPROVAL

    def __init__(self, repo_path, git_wrapper, human_interface=None, tournament_system=None) -> None
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `process_human_prompt(prompt)` | `TaskBreakdown` | Parse prompt into structured tasks |
| `delegate_task(task, agent_pool, use_tournament=False)` | `DelegationResult` | Route task to agent(s) |
| `validate_and_merge(solution, force=False, reviewer=None)` | `MergeResult` | Human-in-the-loop merge gateway |
| `get_repo_status()` | `RepoStatus` | Current state of orchestrator |
| `set_branch_protection(branch_name, level)` | `bool` | Set protection level |
| `get_branch_protection(branch_name)` | `BranchProtectionLevel` | Query protection level |
| `lock_branch(branch_name)` | `bool` | Set to LOCKED |
| `unlock_branch(branch_name)` | `bool` | Set to NONE |
| `submit_for_approval(solution, priority=TaskPriority.MEDIUM)` | `str` | Queue for review, returns approval ID |
| `get_pending_approvals()` | `List[ApprovalRequest]` | All pending reviews |
| `approve_solution(approval_id, reviewer="", comments="")` | `ApprovalResult` | Approve and merge |
| `reject_solution(approval_id, reviewer="", comments="")` | `ApprovalResult` | Reject with feedback |
| `get_statistics()` | `Dict[str, Any]` | Delegation counts, merge stats |
| `export_state(output_path)` | `str` | Serialize state to JSON file |

### Key Data Classes

**Task**
```python
@dataclass
class Task:
    description: str
    requirements: List[str]
    acceptance_criteria: List[str]
    priority: TaskPriority
    dependencies: List[str]
    status: TaskStatus
    assigned_agent: Optional[str] = None
```

**Solution**
```python
@dataclass
class Solution:
    task_id: str
    agent_id: str
    changes: List[CodeChange]
    explanation: str
    tests_included: bool
    workspace_branch: Optional[str] = None
```

**ApprovalResult**
```python
@dataclass
class ApprovalResult:
    success: bool
    solution: Optional[Solution]
    message: str
    merged_commit: Optional[str] = None
```

**TaskBreakdown**
```python
@dataclass
class TaskBreakdown:
    original_prompt: str
    tasks: List[Task]
    execution_strategy: str
    context: Dict[str, Any]
```

### Enums

| Enum | Values |
|------|--------|
| `TaskStatus` | PENDING, ASSIGNED, IN_PROGRESS, COMPLETED, FAILED, CANCELLED |
| `TaskPriority` | CRITICAL, HIGH, MEDIUM, LOW |
| `MergeStatus` | PENDING_APPROVAL, APPROVED, REJECTED, MERGED, CONFLICT, ERROR |
| `BranchProtectionLevel` | NONE, REVIEW_REQUIRED, HUMAN_APPROVAL, LOCKED |

### Protocols

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `GitWrapperInterface` | get_current_branch, create_branch, checkout_branch, commit_changes, merge_branch, get_diff | Git abstraction |
| `AgentInterface` | id, capabilities, can_handle, execute, get_status | Agent abstraction |
| `TournamentInterface` | register_agent, execute_parallel, select_winner | Tournament abstraction |
| `HumanInterface` | request_approval, provide_feedback, is_available | Human interaction |

---

## ansibel.tournament

### TournamentOrchestrator

Manages parallel agent execution and solution evaluation.

```python
class TournamentOrchestrator:
    def __init__(self, agent_manager, git_wrapper, default_evaluators=None, max_concurrent_agents=5) -> None
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `create_tournament(task, agent_configs, selection_mode)` | `Tournament` | Set up a new tournament |
| `run_tournament(tournament_id)` | `TournamentResult` | Execute all agents (async) |
| `present_for_review(tournament_id)` | `ReviewPresentation` | Generate comparison view (async) |
| `select_winner(tournament_id, winner_id)` | `Solution` | Pick winning solution (async) |
| `archive_losers(tournament_id)` | `List[ArchivedSolution]` | Preserve non-winners (async) |
| `cancel_tournament(tournament_id)` | `bool` | Cancel a running tournament (async) |
| `register_progress_callback(callback)` | `None` | Subscribe to progress updates |
| `unregister_progress_callback(callback)` | `None` | Unsubscribe |

### Key Data Classes

**Tournament**
```python
@dataclass
class Tournament:
    tournament_id: str
    task: Task
    agent_configs: List[AgentConfig]
    selection_mode: SelectionMode
    solutions: List[Solution]
    status: TournamentStatus
    winner_id: Optional[str]
    evaluation_scores: Dict[str, float]
```

**AgentConfig**
```python
@dataclass
class AgentConfig:
    agent_id: str
    agent_type: str
    model_config: Dict[str, Any]
    system_prompt: str
    timeout_seconds: int = 300
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None
```

**Solution** (tournament module)
```python
@dataclass
class Solution:
    solution_id: str
    agent_id: str
    task_id: str
    files_changed: List[str]
    diff: str
    explanation: str
    metrics: Dict[str, Any]
    status: SolutionStatus
    execution_time_ms: int
    test_results: Optional[Dict[str, Any]] = None
```

### Enums

| Enum | Values |
|------|--------|
| `SelectionMode` | HUMAN_CHOICE, AUTO_BEST, THRESHOLD |
| `SolutionStatus` | PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED |
| `TournamentStatus` | CREATED, RUNNING, COMPLETED, CANCELLED, ERROR |

### Evaluation Strategies

All implement the abstract `EvaluationStrategy`:

```python
class EvaluationStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, solution: Solution, task: Task) -> float: ...
```

| Class | Scoring Criteria |
|-------|-----------------|
| `ComplexityEvaluator` | Lower cyclomatic complexity scores higher |
| `TestPassEvaluator` | Fraction of tests passing |
| `RequirementMatchEvaluator` | Keyword overlap with task requirements |
| `CompositeEvaluator` | Weighted combination of other strategies |

---

## ansibel.trust_lineage

### TrustLineageManager

Unified interface to trust scoring and lineage tracking.

```python
class TrustLineageManager:
    def __init__(self, storage_path: str) -> None
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `record_decision(agent_id, decision, commit_hash, ...)` | `DecisionRecord` | Record review outcome, update trust |
| `get_trust_score(agent_id)` | `TrustScore` | Current score with confidence |
| `get_trust_tier(agent_id)` | `TrustTier` | Tier classification |
| `get_agent_history(agent_id, limit=50)` | `List[DecisionRecord]` | Decision history |
| `should_auto_approve(agent_id, change_size)` | `bool` | Check auto-approval eligibility |
| `get_agent_profile(agent_id)` | `AgentProfile` | Full profile with statistics |
| `get_all_agents()` | `List[AgentProfile]` | All known agent profiles |
| `record_complete_decision(...)` | `(DecisionRecord, ReasoningRecord)` | Atomic decision + reasoning |
| `get_decision_with_lineage(commit_hash)` | `Optional[DecisionLineage]` | Full lineage for a commit |
| `get_agent_summary(agent_id)` | `Dict[str, Any]` | Trust + history summary |

### TrustScorer

Low-level trust scoring engine. Used internally by `TrustLineageManager`.

**Constants:**
- `EMA_ALPHA = 0.3`
- `DECAY_HALF_LIFE_DAYS = 30`
- `RECOVERY_RATE = 0.05`
- `MIN_SAMPLES_FOR_CONFIDENCE = 10`
- `MAX_SAMPLES_FOR_FULL_CONFIDENCE = 100`

### LineageTracker

Low-level lineage storage. Used internally by `TrustLineageManager`.

| Method | Returns | Description |
|--------|---------|-------------|
| `record_reasoning(...)` | `ReasoningRecord` | Store chain-of-thought |
| `get_lineage(commit_hash)` | `Optional[DecisionLineage]` | Retrieve lineage |
| `trace_decision_chain(start_commit, max_depth=10)` | `List[DecisionLineage]` | Walk ancestors |
| `search_reasoning(keyword, agent_id=None, limit=50)` | `List[ReasoningRecord]` | Full-text search |
| `get_reasoning_for_library_choice(commit_hash, library)` | `Optional[str]` | Library rationale |

### Key Data Classes

**TrustScore** (frozen)
```python
@dataclass(frozen=True)
class TrustScore:
    score: float       # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sample_count: int
    last_updated: str  # ISO 8601
```

**AgentProfile**
```python
@dataclass
class AgentProfile:
    agent_id: str
    name: str
    created_at: str
    trust_score: float
    tier: TrustTier
    total_decisions: int
    accepted_count: int
    rejected_count: int
    modified_count: int
    avg_review_time_ms: float
    last_activity: str
```

### Enums

| Enum | Values |
|------|--------|
| `TrustTier` | UNTRUSTED, LOW, MEDIUM, HIGH, VERIFIED |
| `DecisionType` | ACCEPTED, REJECTED, MODIFIED, AUTO_APPROVED, REVIEWED |
| `ChangeComplexity` | TRIVIAL(1), MINOR(2), MODERATE(3), MAJOR(5), CRITICAL(8) |

---

## ansibel.exceptions

```
AnsibElError                    # Base exception
    GitWrapperError             # Git operation failures
    TournamentError             # Tournament execution errors
    TrustError                  # Trust scoring errors
    AgentError                  # Agent management errors
        AgentNotFoundError      # Agent ID not found
        AgentValidationError    # Invalid agent parameters
        AgentContextError       # Workspace/context failures
```

All exceptions inherit from `AnsibElError`, which inherits from `Exception`.
