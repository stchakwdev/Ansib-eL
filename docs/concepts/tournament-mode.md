# Tournament Mode

## Overview

Tournament mode spawns multiple agents to work on the same task in parallel. Each agent produces an independent solution on its own branch. After execution, solutions are evaluated, compared, and presented for review. A human (or automatic threshold) selects the winner, which is merged; losing solutions are archived for future training data.

This approach trades compute cost for solution quality and diversity.

## How It Works

1. **Create tournament** -- Define the task, configure agent profiles, and choose a selection mode.
2. **Execute in parallel** -- Agents run concurrently with `asyncio`, controlled by a semaphore that limits maximum concurrent agents.
3. **Collect solutions** -- Each agent produces a `Solution` with file changes, diffs, explanations, metrics, and test results.
4. **Evaluate** -- Pluggable `EvaluationStrategy` implementations score each solution.
5. **Present for review** -- A `ReviewPresentation` summarizes all solutions with side-by-side diffs and recommendations.
6. **Select winner** -- Based on the selection mode, a winner is chosen and losers are archived.

## Selection Modes

| Mode | Behavior |
|------|----------|
| `HUMAN_CHOICE` | Present all solutions; human picks the winner |
| `AUTO_BEST` | Automatically select the highest-scoring solution |
| `THRESHOLD` | Auto-select if the top score exceeds a configurable threshold; otherwise fall back to human choice |

## Evaluation Strategies

Evaluation strategies score solutions on a 0.0-1.0 scale. They implement the `EvaluationStrategy` abstract class with a single `evaluate(solution, task)` method.

### Built-in Strategies

**ComplexityEvaluator**

Analyzes the cyclomatic complexity of changed code. Lower complexity scores higher, encouraging simpler solutions.

**TestPassEvaluator**

Scores based on the fraction of tests that pass. A solution where all tests pass scores 1.0; partial failures reduce the score proportionally.

**RequirementMatchEvaluator**

Checks how well the solution addresses the task requirements by matching keywords from the task description against the solution's explanation and diff content.

**CompositeEvaluator**

Combines multiple strategies with configurable weights:

```python
from ansibel.tournament import CompositeEvaluator, TestPassEvaluator, ComplexityEvaluator

evaluator = CompositeEvaluator(strategies=[
    (TestPassEvaluator(), 0.5),
    (ComplexityEvaluator(), 0.3),
    (RequirementMatchEvaluator(), 0.2),
])
```

### Custom Strategies

Implement the `EvaluationStrategy` interface:

```python
from ansibel.tournament import EvaluationStrategy, Solution, Task

class SecurityEvaluator(EvaluationStrategy):
    @property
    def name(self) -> str:
        return "security"

    def evaluate(self, solution: Solution, task: Task) -> float:
        # Score based on security analysis of the diff
        ...
        return score
```

## Review Presentation

When a tournament completes, `present_for_review()` generates a `ReviewPresentation` containing:

- **Task description** -- What the agents were asked to do
- **Solution comparisons** -- Side-by-side metrics for each solution
- **Diffs** -- Unified or side-by-side diffs via `DiffPresenter`
- **Agent metadata** -- Model version, execution time, configuration
- **Recommendations** -- Auto-generated suggestions based on evaluation scores

The `DiffPresenter` utility supports:
- Unified diff format (`format_unified_diff`)
- Side-by-side comparison (`format_side_by_side`)
- Solution summary (`format_solution_summary`)
- Cross-solution difference highlighting (`highlight_differences`)

## Archival

After winner selection, `archive_losers()` preserves non-winning solutions as `ArchivedSolution` objects containing:

- The full solution data
- The tournament ID for traceability
- The rejection reason (outscored, threshold not met, etc.)
- A comparison against the winner
- Training metadata for future model improvement

Archived solutions are valuable for fine-tuning and understanding which approaches work better for different task types.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANSIBEL_TOURNAMENT_AGENTS` | Default number of agents per tournament | `3` |
| `ANSIBEL_TOURNAMENT_TIMEOUT` | Timeout per agent in seconds | `300` |

### Config File (`.ai-git/config.yaml`)

```yaml
tournament:
  default_agents: 3
  selection_mode: human_choice
  max_concurrent_agents: 5
  timeout_seconds: 300
```

## API Examples

### Running a Tournament via AnsibElSystem

```python
from ansibel.ansib_el import AnsibElSystem

system = AnsibElSystem("./my-project")
system.initialize()

result = system.process_prompt(
    "Optimize the search algorithm",
    use_tournament=True,
    num_agents=3
)
```

### Direct Tournament API

```python
import asyncio
from ansibel.tournament import (
    TournamentOrchestrator, Task, AgentConfig, SelectionMode
)

# Define the task
task = Task(
    task_id="task-001",
    description="Implement connection pooling",
    context_files=["src/db.py"],
    requirements=["Must support max 10 connections", "Must handle timeouts"],
    test_commands=["pytest tests/test_db.py -v"],
)

# Configure agents
configs = [
    AgentConfig(agent_id="a1", agent_type="gpt-4", model_config={}, system_prompt="..."),
    AgentConfig(agent_id="a2", agent_type="claude-3", model_config={}, system_prompt="..."),
    AgentConfig(agent_id="a3", agent_type="gpt-4", model_config={}, system_prompt="..."),
]

# Create and run
orchestrator = TournamentOrchestrator(agent_manager, git_wrapper)
tournament = orchestrator.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)

result = asyncio.run(orchestrator.run_tournament(tournament.tournament_id))

# Review
review = asyncio.run(orchestrator.present_for_review(tournament.tournament_id))

# Select winner
winner = asyncio.run(orchestrator.select_winner(tournament.tournament_id, "a2"))

# Archive losers
archived = asyncio.run(orchestrator.archive_losers(tournament.tournament_id))
```

### Progress Callbacks

Monitor tournament execution in real time:

```python
def on_progress(tournament_id: str, info: dict):
    print(f"[{tournament_id}] {info}")

orchestrator.register_progress_callback(on_progress)
```
