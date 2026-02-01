"""
Tournament/Parallel Execution System for Ansib-eL (AI-Native Version Control)

This module implements the Tournament Logic where multiple agents attempt the same
task and the best solution is selected through parallel execution and evaluation.

Key Components:
- TournamentOrchestrator: Main orchestrator for spawning and managing agents
- Parallel execution with asyncio for concurrent agent runs
- Diff presentation for human review
- Pluggable evaluation strategies
- Solution archival for training data
"""

import asyncio
import uuid
import time
import difflib
import json
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Protocol, Set, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from concurrent.futures import TimeoutError as FutureTimeoutError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class SelectionMode(Enum):
    """Mode for selecting the winning solution from a tournament."""
    HUMAN_CHOICE = auto()   # Human reviews and selects winner
    AUTO_BEST = auto()      # Automatically select highest scoring solution
    THRESHOLD = auto()      # Select first solution meeting threshold score


class SolutionStatus(Enum):
    """Status of a solution in the tournament."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TournamentStatus(Enum):
    """Status of a tournament."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for spawning an agent in a tournament.
    
    Attributes:
        agent_id: Unique identifier for this agent configuration
        agent_type: Type/class of agent to spawn (e.g., "gpt-5.2", "claude-opus-4.5", "custom")
        model_config: Model-specific configuration (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt override
        timeout_seconds: Maximum time allowed for this agent to complete
        priority: Execution priority (higher = executed earlier)
        metadata: Additional agent-specific metadata
    """
    agent_id: str
    agent_type: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    timeout_seconds: float = 300.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())


@dataclass
class Task:
    """Represents a task to be solved by agents in a tournament.
    
    Attributes:
        task_id: Unique identifier for this task
        description: Human-readable task description
        context_files: List of file paths providing context
        requirements: Specific requirements or constraints
        test_commands: Commands to validate the solution
        expected_output: Expected output patterns for validation
        metadata: Additional task metadata
    """
    task_id: str
    description: str
    context_files: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class Solution:
    """Represents a solution produced by an agent.
    
    Attributes:
        solution_id: Unique identifier for this solution
        agent_id: ID of the agent that produced this solution
        task_id: ID of the task this solution addresses
        files_changed: Dictionary mapping file paths to their new content
        diff: Unified diff representation of changes
        explanation: Agent's explanation of the solution
        metrics: Solution quality metrics
        status: Current status of the solution
        created_at: Timestamp when solution was created
        completed_at: Timestamp when solution was completed
        execution_time_ms: Time taken to generate solution
        test_results: Results from running test commands
    """
    solution_id: str
    agent_id: str
    task_id: str
    files_changed: Dict[str, str] = field(default_factory=dict)
    diff: str = ""
    explanation: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    status: SolutionStatus = SolutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.solution_id:
            self.solution_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary for serialization."""
        return {
            "solution_id": self.solution_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "files_changed": self.files_changed,
            "diff": self.diff,
            "explanation": self.explanation,
            "metrics": self.metrics,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "test_results": self.test_results,
        }


@dataclass
class Tournament:
    """Represents a tournament instance with multiple competing agents.
    
    Attributes:
        tournament_id: Unique identifier for this tournament
        task: The task being solved
        agent_configs: List of agent configurations
        selection_mode: How the winner will be selected
        solutions: Map of agent_id to their solution
        status: Current tournament status
        created_at: Timestamp when tournament was created
        started_at: Timestamp when tournament started
        completed_at: Timestamp when tournament completed
        winner_id: ID of the winning solution (if selected)
        evaluation_scores: Scores from evaluation strategies
    """
    tournament_id: str
    task: Task
    agent_configs: List[AgentConfig]
    selection_mode: SelectionMode
    solutions: Dict[str, Solution] = field(default_factory=dict)
    status: TournamentStatus = TournamentStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    winner_id: Optional[str] = None
    evaluation_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.tournament_id:
            self.tournament_id = str(uuid.uuid4())
    
    def get_completed_solutions(self) -> List[Solution]:
        """Get all successfully completed solutions."""
        return [
            s for s in self.solutions.values()
            if s.status == SolutionStatus.COMPLETED
        ]
    
    def get_failed_solutions(self) -> List[Solution]:
        """Get all failed/timed out solutions."""
        return [
            s for s in self.solutions.values()
            if s.status in (SolutionStatus.FAILED, SolutionStatus.TIMEOUT)
        ]
    
    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get solutions ranked by their composite score."""
        scored_solutions = []
        for agent_id, scores in self.evaluation_scores.items():
            if scores:
                composite = sum(scores.values()) / len(scores)
                scored_solutions.append((agent_id, composite))
        return sorted(scored_solutions, key=lambda x: x[1], reverse=True)


@dataclass
class TournamentResult:
    """Results from running a tournament.
    
    Attributes:
        tournament_id: ID of the tournament
        solutions: All solutions produced
        winner: The winning solution (if selected)
        execution_summary: Summary of execution statistics
        evaluation_summary: Summary of evaluation results
    """
    tournament_id: str
    solutions: List[Solution]
    winner: Optional[Solution] = None
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "tournament_id": self.tournament_id,
            "solutions": [s.to_dict() for s in self.solutions],
            "winner": self.winner.to_dict() if self.winner else None,
            "execution_summary": self.execution_summary,
            "evaluation_summary": self.evaluation_summary,
        }


@dataclass
class ReviewPresentation:
    """Formatted presentation of solutions for human review.
    
    Attributes:
        tournament_id: ID of the tournament
        task_description: Description of the task
        solution_comparisons: Side-by-side comparisons
        diffs: Formatted diffs for each solution
        agent_metadata: Metadata about each agent
        recommendations: Auto-generated recommendations
    """
    tournament_id: str
    task_description: str
    solution_comparisons: List[Dict[str, Any]]
    diffs: Dict[str, str]
    agent_metadata: Dict[str, Dict[str, Any]]
    recommendations: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate markdown representation for review."""
        lines = [
            f"# Tournament Review: {self.tournament_id}",
            "",
            f"## Task: {self.task_description}",
            "",
            "## Solutions Overview",
            "",
        ]
        
        for comp in self.solution_comparisons:
            lines.extend([
                f"### Solution: {comp.get('agent_id', 'Unknown')}",
                f"- **Status**: {comp.get('status', 'Unknown')}",
                f"- **Score**: {comp.get('score', 'N/A')}",
                f"- **Execution Time**: {comp.get('execution_time', 'N/A')}ms",
                "",
            ])
        
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        lines.extend([
            "## Detailed Diffs",
            "",
        ])
        for agent_id, diff in self.diffs.items():
            lines.extend([
                f"### {agent_id}",
                "```diff",
                diff,
                "```",
                "",
            ])
        
        return "\n".join(lines)


@dataclass
class ArchivedSolution:
    """Archived solution with metadata for training.
    
    Attributes:
        archive_id: Unique identifier for this archive entry
        solution: The solution being archived
        tournament_id: ID of the tournament
        rejection_reason: Why this solution was not selected
        winner_comparison: Comparison with winning solution
        training_metadata: Metadata for training use
        archived_at: Timestamp of archival
    """
    archive_id: str
    solution: Solution
    tournament_id: str
    rejection_reason: Optional[str] = None
    winner_comparison: Dict[str, Any] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    archived_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.archive_id:
            self.archive_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert archive to dictionary for storage."""
        return {
            "archive_id": self.archive_id,
            "solution": self.solution.to_dict(),
            "tournament_id": self.tournament_id,
            "rejection_reason": self.rejection_reason,
            "winner_comparison": self.winner_comparison,
            "training_metadata": self.training_metadata,
            "archived_at": self.archived_at.isoformat(),
        }


# =============================================================================
# Protocols and Abstract Classes
# =============================================================================

class AgentManager(Protocol):
    """Protocol for agent management - to be implemented by the system."""
    
    async def spawn_agent(self, config: AgentConfig) -> str:
        """Spawn an agent with the given configuration. Returns agent ID."""
        ...
    
    async def execute_task(self, agent_id: str, task: Task) -> Solution:
        """Execute a task with the given agent. Returns the solution."""
        ...
    
    async def terminate_agent(self, agent_id: str) -> None:
        """Terminate an agent."""
        ...


class GitWrapper(Protocol):
    """Protocol for Git operations - to be implemented by the system."""
    
    async def get_file_content(self, path: str) -> str:
        """Get content of a file at the given path."""
        ...
    
    async def apply_diff(self, diff: str) -> bool:
        """Apply a diff to the working directory."""
        ...
    
    async def get_diff(self, source: str, target: str) -> str:
        """Get diff between two states."""
        ...


class EvaluationStrategy(ABC):
    """Abstract base class for solution evaluation strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this evaluation strategy."""
        pass
    
    @abstractmethod
    async def evaluate(self, solution: Solution, task: Task) -> float:
        """Evaluate a solution and return a score (0.0 to 1.0)."""
        pass


# =============================================================================
# Evaluation Strategy Implementations
# =============================================================================

class ComplexityEvaluator(EvaluationStrategy):
    """Evaluates solution based on code complexity metrics."""
    
    @property
    def name(self) -> str:
        return "complexity"
    
    async def evaluate(self, solution: Solution, task: Task) -> float:
        """Score based on cyclomatic complexity and lines of code."""
        total_lines = 0
        complexity_score = 1.0
        
        for filepath, content in solution.files_changed.items():
            lines = content.split('\n')
            total_lines += len(lines)
            
            # Simple complexity heuristic: count control structures
            control_structures = sum(
                content.count(keyword) 
                for keyword in ['if ', 'for ', 'while ', 'switch', 'try:', 'except']
            )
            
            # Penalize high complexity
            if len(lines) > 0:
                complexity_ratio = control_structures / len(lines)
                complexity_score *= max(0.5, 1.0 - complexity_ratio)
        
        # Prefer moderate line counts (not too short, not too long)
        length_score = 1.0
        if total_lines < 5:
            length_score = 0.7  # Too short might be incomplete
        elif total_lines > 500:
            length_score = 0.8  # Very long solutions
        
        return complexity_score * length_score


class TestPassEvaluator(EvaluationStrategy):
    """Evaluates solution based on test pass rate."""
    
    @property
    def name(self) -> str:
        return "test_pass"
    
    async def evaluate(self, solution: Solution, task: Task) -> float:
        """Score based on test results."""
        if not solution.test_results:
            return 0.5  # Neutral if no tests run
        
        passed = solution.test_results.get('passed', 0)
        total = solution.test_results.get('total', 0)
        
        if total == 0:
            return 0.5
        
        pass_rate = passed / total
        
        # Bonus for 100% pass rate
        if pass_rate == 1.0:
            return 1.0
        
        return pass_rate * 0.9  # Slight penalty for not passing all


class RequirementMatchEvaluator(EvaluationStrategy):
    """Evaluates solution based on requirement fulfillment."""
    
    @property
    def name(self) -> str:
        return "requirement_match"
    
    async def evaluate(self, solution: Solution, task: Task) -> float:
        """Score based on how well requirements are met."""
        if not task.requirements:
            return 1.0  # No requirements means perfect match
        
        # Simple keyword matching in explanation and diff
        explanation_lower = solution.explanation.lower()
        diff_lower = solution.diff.lower()
        
        matches = 0
        for req in task.requirements:
            req_lower = req.lower()
            # Check if requirement keywords appear in solution
            req_keywords = set(req_lower.split()) - {'the', 'a', 'an', 'to', 'of', 'in', 'and'}
            if req_keywords:
                match_count = sum(
                    1 for kw in req_keywords 
                    if kw in explanation_lower or kw in diff_lower
                )
                matches += match_count / len(req_keywords)
        
        return min(1.0, matches / len(task.requirements))


class CompositeEvaluator(EvaluationStrategy):
    """Combines multiple evaluation strategies with weights."""
    
    def __init__(self, strategies: List[tuple[EvaluationStrategy, float]]):
        """
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
        self._name = "composite"
    
    @property
    def name(self) -> str:
        return self._name
    
    async def evaluate(self, solution: Solution, task: Task) -> float:
        """Combine scores from all strategies."""
        if not self.strategies:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for strategy, weight in self.strategies:
            try:
                score = await strategy.evaluate(solution, task)
                total_score += score * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"Evaluation failed for {strategy.name}: {e}")
        
        if total_weight == 0:
            return 0.5
        
        return total_score / total_weight


# =============================================================================
# Diff Presentation
# =============================================================================

class DiffPresenter:
    """Formats and presents diffs for human review.
    
    This class provides multiple diff visualization formats including
    unified diff, side-by-side comparison, and highlighted differences.
    """
    
    def __init__(self, context_lines: int = 3):
        self.context_lines = context_lines
    
    def format_unified_diff(
        self, 
        original: Dict[str, str], 
        modified: Dict[str, str],
        from_label: str = "original",
        to_label: str = "modified"
    ) -> str:
        """Generate unified diff between original and modified files.
        
        Args:
            original: Dictionary mapping file paths to original content
            modified: Dictionary mapping file paths to modified content
            from_label: Label for original state
            to_label: Label for modified state
            
        Returns:
            Unified diff string
        """
        all_files = set(original.keys()) | set(modified.keys())
        diffs = []
        
        for filepath in sorted(all_files):
            original_content = original.get(filepath, "")
            modified_content = modified.get(filepath, "")
            
            if original_content != modified_content:
                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    modified_content.splitlines(keepends=True),
                    fromfile=f"{from_label}/{filepath}",
                    tofile=f"{to_label}/{filepath}",
                    n=self.context_lines
                )
                diffs.append(''.join(diff))
        
        return '\n'.join(diffs) if diffs else "No changes"
    
    def format_side_by_side(
        self,
        original: Dict[str, str],
        modified: Dict[str, str],
        width: int = 80
    ) -> str:
        """Generate side-by-side comparison.
        
        Args:
            original: Dictionary mapping file paths to original content
            modified: Dictionary mapping file paths to modified content
            width: Width of each column
            
        Returns:
            Side-by-side comparison string
        """
        all_files = set(original.keys()) | set(modified.keys())
        output = []
        
        for filepath in sorted(all_files):
            output.append(f"\n{'=' * (width * 2 + 5)}")
            output.append(f"File: {filepath}")
            output.append(f"{'=' * (width * 2 + 5)}\n")
            
            orig_lines = original.get(filepath, "").splitlines()
            mod_lines = modified.get(filepath, "").splitlines()
            
            max_lines = max(len(orig_lines), len(mod_lines))
            
            for i in range(max_lines):
                orig = orig_lines[i] if i < len(orig_lines) else ""
                mod = mod_lines[i] if i < len(mod_lines) else ""
                
                # Truncate to width
                orig_display = orig[:width-1].ljust(width)
                mod_display = mod[:width-1].ljust(width)
                
                # Mark differences
                marker = " "
                if orig != mod:
                    marker = "*"
                
                output.append(f"{marker} {orig_display} | {mod_display}")
        
        return '\n'.join(output)
    
    def format_solution_summary(self, solution: Solution) -> str:
        """Generate a summary of a solution's changes.
        
        Args:
            solution: The solution to summarize
            
        Returns:
            Formatted summary string
        """
        lines = [
            f"Solution: {solution.solution_id}",
            f"Agent: {solution.agent_id}",
            f"Status: {solution.status.value}",
            f"Files Changed: {len(solution.files_changed)}",
            f"Execution Time: {solution.execution_time_ms:.2f}ms",
            "",
            "Metrics:",
        ]
        
        for metric, value in solution.metrics.items():
            lines.append(f"  {metric}: {value}")
        
        if solution.explanation:
            lines.extend([
                "",
                "Explanation:",
                solution.explanation[:500] + "..." if len(solution.explanation) > 500 else solution.explanation
            ])
        
        return '\n'.join(lines)
    
    def highlight_differences(
        self,
        solutions: List[Solution],
        focus_files: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Highlight key differences between solutions.
        
        Args:
            solutions: List of solutions to compare
            focus_files: Specific files to focus on (None = all files)
            
        Returns:
            Dictionary mapping file paths to difference analysis
        """
        if not solutions:
            return {}
        
        # Collect all files
        all_files = set()
        for sol in solutions:
            all_files.update(sol.files_changed.keys())
        
        if focus_files:
            all_files = all_files & set(focus_files)
        
        differences = {}
        
        for filepath in all_files:
            file_versions = []
            for sol in solutions:
                content = sol.files_changed.get(filepath, "")
                file_versions.append({
                    "solution_id": sol.solution_id,
                    "agent_id": sol.agent_id,
                    "content": content,
                    "content_hash": hash(content) & 0xFFFFFFFF
                })
            
            # Find unique versions
            content_groups = {}
            for fv in file_versions:
                h = fv["content_hash"]
                if h not in content_groups:
                    content_groups[h] = []
                content_groups[h].append(fv)
            
            differences[filepath] = {
                "num_unique_versions": len(content_groups),
                "versions": list(content_groups.values()),
                "agents_per_version": [len(v) for v in content_groups.values()]
            }
        
        return differences


# =============================================================================
# Tournament Orchestrator
# =============================================================================

class TournamentOrchestrator:
    """Main orchestrator for tournament-based parallel agent execution.
    
    This class manages the complete tournament lifecycle:
    1. Creating tournaments with multiple agent configurations
    2. Executing agents in parallel with timeout handling
    3. Evaluating solutions using pluggable strategies
    4. Presenting results for human review
    5. Archiving rejected solutions for training
    
    Example:
        orchestrator = TournamentOrchestrator(agent_manager, git_wrapper)
        
        tournament = orchestrator.create_tournament(
            task=task,
            agent_configs=[config1, config2, config3],
            selection_mode=SelectionMode.HUMAN_CHOICE
        )
        
        result = await orchestrator.run_tournament(tournament.tournament_id)
        review = await orchestrator.present_for_review(tournament.tournament_id)
        winner = await orchestrator.select_winner(tournament.tournament_id, winner_id)
    """
    
    def __init__(
        self, 
        agent_manager: AgentManager,
        git_wrapper: Optional[GitWrapper] = None,
        default_evaluators: Optional[List[EvaluationStrategy]] = None,
        max_concurrent_agents: int = 5
    ):
        """Initialize the tournament orchestrator.
        
        Args:
            agent_manager: Manager for spawning and controlling agents
            git_wrapper: Optional Git wrapper for file operations
            default_evaluators: Default evaluation strategies
            max_concurrent_agents: Maximum agents to run concurrently
        """
        self.agent_manager = agent_manager
        self.git_wrapper = git_wrapper
        self.max_concurrent_agents = max_concurrent_agents
        
        # Default evaluators
        if default_evaluators is None:
            self.default_evaluators = [
                ComplexityEvaluator(),
                TestPassEvaluator(),
                RequirementMatchEvaluator()
            ]
        else:
            self.default_evaluators = default_evaluators
        
        # Tournament storage
        self._tournaments: Dict[str, Tournament] = {}
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Cancellation tokens
        self._cancellation_tokens: Dict[str, asyncio.Event] = {}
        
        # Diff presenter
        self._diff_presenter = DiffPresenter()
    
    def create_tournament(
        self,
        task: Task,
        agent_configs: List[AgentConfig],
        selection_mode: SelectionMode = SelectionMode.HUMAN_CHOICE
    ) -> Tournament:
        """Create a new tournament with the given task and agent configurations.
        
        Args:
            task: The task to be solved
            agent_configs: List of agent configurations to compete
            selection_mode: How the winner will be selected
            
        Returns:
            The created Tournament instance
        """
        # Sort agent configs by priority (higher first)
        sorted_configs = sorted(agent_configs, key=lambda c: -c.priority)
        
        tournament = Tournament(
            tournament_id=str(uuid.uuid4()),
            task=task,
            agent_configs=sorted_configs,
            selection_mode=selection_mode
        )
        
        self._tournaments[tournament.tournament_id] = tournament
        self._cancellation_tokens[tournament.tournament_id] = asyncio.Event()
        
        logger.info(f"Created tournament {tournament.tournament_id} with {len(agent_configs)} agents")
        
        return tournament
    
    async def run_tournament(self, tournament_id: str) -> TournamentResult:
        """Execute all agents in the tournament and return results.
        
        This method runs all agent configurations in parallel (up to max_concurrent_agents),
        with individual timeout handling for each agent. Failures are isolated so one
        agent's failure doesn't affect others.
        
        Args:
            tournament_id: ID of the tournament to run
            
        Returns:
            TournamentResult with all solutions and execution summary
        """
        tournament = self._get_tournament(tournament_id)
        cancel_token = self._cancellation_tokens.get(tournament_id)
        
        if tournament is None:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        tournament.status = TournamentStatus.RUNNING
        tournament.started_at = datetime.utcnow()
        
        self._notify_progress(tournament_id, {
            "status": "started",
            "total_agents": len(tournament.agent_configs)
        })
        
        # Create semaphore for limiting concurrent execution
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        
        # Execute all agents
        tasks = []
        for config in tournament.agent_configs:
            task = self._execute_agent_with_semaphore(
                tournament_id, config, tournament.task, semaphore, cancel_token
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = 0
        failure_count = 0
        timeout_count = 0
        
        for i, result in enumerate(results):
            config = tournament.agent_configs[i]
            
            if isinstance(result, Exception):
                # Create failed solution
                solution = Solution(
                    solution_id=str(uuid.uuid4()),
                    agent_id=config.agent_id,
                    task_id=tournament.task.task_id,
                    status=SolutionStatus.FAILED,
                    explanation=f"Execution failed: {str(result)}"
                )
                failure_count += 1
                logger.error(f"Agent {config.agent_id} failed: {result}")
            else:
                solution = result
                if solution.status == SolutionStatus.COMPLETED:
                    success_count += 1
                elif solution.status == SolutionStatus.TIMEOUT:
                    timeout_count += 1
                    failure_count += 1
            
            tournament.solutions[config.agent_id] = solution
        
        # Evaluate solutions
        await self._evaluate_solutions(tournament)
        
        tournament.status = TournamentStatus.COMPLETED
        tournament.completed_at = datetime.utcnow()
        
        # Build execution summary
        execution_summary = {
            "total_agents": len(tournament.agent_configs),
            "successful": success_count,
            "failed": failure_count,
            "timeouts": timeout_count,
            "total_execution_time_ms": sum(
                s.execution_time_ms for s in tournament.solutions.values()
            )
        }
        
        self._notify_progress(tournament_id, {
            "status": "completed",
            "execution_summary": execution_summary
        })
        
        # Auto-select winner if mode is AUTO_BEST or THRESHOLD
        winner = None
        if tournament.selection_mode == SelectionMode.AUTO_BEST:
            winner = await self._auto_select_winner(tournament)
        elif tournament.selection_mode == SelectionMode.THRESHOLD:
            winner = await self._threshold_select_winner(tournament, threshold=0.8)
        
        return TournamentResult(
            tournament_id=tournament_id,
            solutions=list(tournament.solutions.values()),
            winner=winner,
            execution_summary=execution_summary,
            evaluation_summary=tournament.evaluation_scores
        )
    
    async def present_for_review(self, tournament_id: str) -> ReviewPresentation:
        """Format tournament results for human review.
        
        Creates a comprehensive review presentation including:
        - Side-by-side solution comparisons
        - Formatted diffs for each solution
        - Agent metadata and performance metrics
        - Auto-generated recommendations
        
        Args:
            tournament_id: ID of the tournament to present
            
        Returns:
            ReviewPresentation formatted for human review
        """
        tournament = self._get_tournament(tournament_id)
        if tournament is None:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        # Build solution comparisons
        comparisons = []
        for agent_id, solution in tournament.solutions.items():
            scores = tournament.evaluation_scores.get(agent_id, {})
            composite_score = sum(scores.values()) / len(scores) if scores else 0.0
            
            comparisons.append({
                "agent_id": agent_id,
                "solution_id": solution.solution_id,
                "status": solution.status.value,
                "score": round(composite_score, 3),
                "execution_time": solution.execution_time_ms,
                "files_changed": len(solution.files_changed),
                "metrics": solution.metrics
            })
        
        # Sort by score descending
        comparisons.sort(key=lambda x: x["score"], reverse=True)
        
        # Build diffs
        diffs = {}
        for agent_id, solution in tournament.solutions.items():
            if solution.diff:
                diffs[agent_id] = solution.diff
            else:
                diffs[agent_id] = self._diff_presenter.format_solution_summary(solution)
        
        # Build agent metadata
        agent_metadata = {}
        for config in tournament.agent_configs:
            solution = tournament.solutions.get(config.agent_id)
            agent_metadata[config.agent_id] = {
                "agent_type": config.agent_type,
                "timeout_seconds": config.timeout_seconds,
                "priority": config.priority,
                "model_config": config.model_config,
                "solution_status": solution.status.value if solution else "unknown",
                "execution_time_ms": solution.execution_time_ms if solution else 0
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(tournament)
        
        return ReviewPresentation(
            tournament_id=tournament_id,
            task_description=tournament.task.description,
            solution_comparisons=comparisons,
            diffs=diffs,
            agent_metadata=agent_metadata,
            recommendations=recommendations
        )
    
    async def select_winner(
        self, 
        tournament_id: str, 
        winner_id: Optional[str] = None
    ) -> Solution:
        """Select the winning solution from a tournament.
        
        If winner_id is provided, that solution is selected. Otherwise,
        the highest-scoring solution is selected.
        
        Args:
            tournament_id: ID of the tournament
            winner_id: Optional specific solution ID to select as winner
            
        Returns:
            The selected winning solution
        """
        tournament = self._get_tournament(tournament_id)
        if tournament is None:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        if winner_id:
            # Find solution by ID
            winner = None
            for solution in tournament.solutions.values():
                if solution.solution_id == winner_id:
                    winner = solution
                    break
            if winner is None:
                raise ValueError(f"Solution {winner_id} not found in tournament")
        else:
            # Auto-select best
            winner = await self._auto_select_winner(tournament)
        
        tournament.winner_id = winner.solution_id
        
        logger.info(f"Selected winner {winner.solution_id} from tournament {tournament_id}")
        
        return winner
    
    async def archive_losers(self, tournament_id: str) -> List[ArchivedSolution]:
        """Archive all non-winning solutions for training data.
        
        Creates archive entries for each rejected solution, including:
        - The solution itself
        - Comparison with the winning solution
        - Metadata for training use
        
        Args:
            tournament_id: ID of the tournament
            
        Returns:
            List of archived solutions
        """
        tournament = self._get_tournament(tournament_id)
        if tournament is None:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        if tournament.winner_id is None:
            raise ValueError("No winner selected for tournament yet")
        
        # Find winner
        winner = None
        for sol in tournament.solutions.values():
            if sol.solution_id == tournament.winner_id:
                winner = sol
                break
        
        archived = []
        
        for agent_id, solution in tournament.solutions.items():
            if solution.solution_id == tournament.winner_id:
                continue  # Skip winner
            
            # Determine rejection reason
            if solution.status != SolutionStatus.COMPLETED:
                rejection_reason = f"Solution status: {solution.status.value}"
            else:
                winner_score = sum(tournament.evaluation_scores.get(winner.agent_id, {}).values())
                solution_score = sum(tournament.evaluation_scores.get(agent_id, {}).values())
                score_diff = winner_score - solution_score
                rejection_reason = f"Lower score than winner (diff: {score_diff:.3f})"
            
            # Create comparison with winner
            winner_comparison = self._compare_solutions(winner, solution)
            
            # Build training metadata
            training_metadata = {
                "task_type": tournament.task.metadata.get("type", "unknown"),
                "task_complexity": len(tournament.task.requirements),
                "num_competitors": len(tournament.agent_configs),
                "winner_agent_type": winner.agent_id if winner else "unknown",
                "loser_agent_type": agent_id,
                "winner_metrics": winner.metrics if winner else {},
                "loser_metrics": solution.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            archive = ArchivedSolution(
                archive_id=str(uuid.uuid4()),
                solution=solution,
                tournament_id=tournament_id,
                rejection_reason=rejection_reason,
                winner_comparison=winner_comparison,
                training_metadata=training_metadata
            )
            
            archived.append(archive)
        
        logger.info(f"Archived {len(archived)} solutions from tournament {tournament_id}")
        
        return archived
    
    async def cancel_tournament(self, tournament_id: str) -> bool:
        """Cancel a running tournament.
        
        Args:
            tournament_id: ID of the tournament to cancel
            
        Returns:
            True if cancellation was initiated
        """
        tournament = self._get_tournament(tournament_id)
        if tournament is None:
            return False
        
        tournament.status = TournamentStatus.CANCELLED
        
        # Signal cancellation
        token = self._cancellation_tokens.get(tournament_id)
        if token:
            token.set()
        
        logger.info(f"Cancelled tournament {tournament_id}")
        
        return True
    
    def register_progress_callback(
        self, 
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Register a callback for progress updates.
        
        Args:
            callback: Function called with (tournament_id, progress_info)
        """
        self._progress_callbacks.append(callback)
    
    def unregister_progress_callback(
        self, 
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Unregister a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------
    
    def _get_tournament(self, tournament_id: str) -> Optional[Tournament]:
        """Get tournament by ID."""
        return self._tournaments.get(tournament_id)
    
    def _notify_progress(self, tournament_id: str, info: Dict[str, Any]) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(tournament_id, info)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def _execute_agent_with_semaphore(
        self,
        tournament_id: str,
        config: AgentConfig,
        task: Task,
        semaphore: asyncio.Semaphore,
        cancel_token: Optional[asyncio.Event]
    ) -> Solution:
        """Execute an agent with semaphore-controlled concurrency."""
        async with semaphore:
            return await self._execute_agent_safe(
                tournament_id, config, task, cancel_token
            )
    
    async def _execute_agent_safe(
        self,
        tournament_id: str,
        config: AgentConfig,
        task: Task,
        cancel_token: Optional[asyncio.Event]
    ) -> Solution:
        """Execute an agent with timeout and error handling."""
        start_time = time.time()
        
        self._notify_progress(tournament_id, {
            "status": "agent_started",
            "agent_id": config.agent_id
        })
        
        try:
            # Check for cancellation before starting
            if cancel_token and cancel_token.is_set():
                return Solution(
                    solution_id=str(uuid.uuid4()),
                    agent_id=config.agent_id,
                    task_id=task.task_id,
                    status=SolutionStatus.CANCELLED,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Execute with timeout
            solution = await asyncio.wait_for(
                self.agent_manager.execute_task(config.agent_id, task),
                timeout=config.timeout_seconds
            )
            
            solution.execution_time_ms = (time.time() - start_time) * 1000
            
            self._notify_progress(tournament_id, {
                "status": "agent_completed",
                "agent_id": config.agent_id,
                "solution_status": solution.status.value
            })
            
            return solution
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            logger.warning(f"Agent {config.agent_id} timed out after {config.timeout_seconds}s")
            
            self._notify_progress(tournament_id, {
                "status": "agent_timeout",
                "agent_id": config.agent_id
            })
            
            return Solution(
                solution_id=str(uuid.uuid4()),
                agent_id=config.agent_id,
                task_id=task.task_id,
                status=SolutionStatus.TIMEOUT,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Agent {config.agent_id} failed: {e}")
            
            self._notify_progress(tournament_id, {
                "status": "agent_failed",
                "agent_id": config.agent_id,
                "error": str(e)
            })
            
            return Solution(
                solution_id=str(uuid.uuid4()),
                agent_id=config.agent_id,
                task_id=task.task_id,
                status=SolutionStatus.FAILED,
                explanation=f"Execution failed: {str(e)}",
                execution_time_ms=execution_time
            )
    
    async def _evaluate_solutions(self, tournament: Tournament) -> None:
        """Evaluate all solutions in a tournament."""
        for agent_id, solution in tournament.solutions.items():
            if solution.status != SolutionStatus.COMPLETED:
                continue
            
            scores = {}
            for evaluator in self.default_evaluators:
                try:
                    score = await evaluator.evaluate(solution, tournament.task)
                    scores[evaluator.name] = score
                except Exception as e:
                    logger.warning(f"Evaluation failed for {evaluator.name}: {e}")
                    scores[evaluator.name] = 0.0
            
            tournament.evaluation_scores[agent_id] = scores
    
    async def _auto_select_winner(self, tournament: Tournament) -> Solution:
        """Automatically select the best solution based on evaluation scores."""
        leaderboard = tournament.get_leaderboard()
        
        if not leaderboard:
            # No scores, pick first completed solution
            for solution in tournament.solutions.values():
                if solution.status == SolutionStatus.COMPLETED:
                    return solution
            # No completed solutions, raise error
            raise ValueError("No completed solutions to select as winner")
        
        best_agent_id = leaderboard[0][0]
        return tournament.solutions[best_agent_id]
    
    async def _threshold_select_winner(
        self, 
        tournament: Tournament, 
        threshold: float
    ) -> Optional[Solution]:
        """Select first solution meeting the threshold score."""
        for agent_id, scores in tournament.evaluation_scores.items():
            if scores:
                composite = sum(scores.values()) / len(scores)
                if composite >= threshold:
                    return tournament.solutions[agent_id]
        
        # No solution met threshold, return best
        return await self._auto_select_winner(tournament)
    
    def _compare_solutions(
        self, 
        winner: Solution, 
        other: Solution
    ) -> Dict[str, Any]:
        """Compare two solutions and return comparison metrics."""
        return {
            "winner_files": len(winner.files_changed),
            "other_files": len(other.files_changed),
            "winner_execution_time": winner.execution_time_ms,
            "other_execution_time": other.execution_time_ms,
            "common_files": set(winner.files_changed.keys()) & set(other.files_changed.keys()),
            "winner_only_files": set(winner.files_changed.keys()) - set(other.files_changed.keys()),
            "other_only_files": set(other.files_changed.keys()) - set(winner.files_changed.keys()),
        }
    
    def _generate_recommendations(self, tournament: Tournament) -> List[str]:
        """Generate recommendations based on tournament results."""
        recommendations = []
        
        completed = tournament.get_completed_solutions()
        failed = tournament.get_failed_solutions()
        
        if not completed:
            recommendations.append("No solutions completed successfully. Consider simplifying the task or increasing timeouts.")
            return recommendations
        
        # Check for high failure rate
        failure_rate = len(failed) / len(tournament.agent_configs)
        if failure_rate > 0.5:
            recommendations.append(f"High failure rate ({failure_rate*100:.0f}%). Consider reviewing task complexity.")
        
        # Check for consensus
        leaderboard = tournament.get_leaderboard()
        if len(leaderboard) > 1:
            top_score = leaderboard[0][1]
            second_score = leaderboard[1][1]
            if top_score - second_score < 0.1:
                recommendations.append("Top solutions have similar scores. Human review recommended.")
        
        # Check execution times
        avg_time = sum(s.execution_time_ms for s in completed) / len(completed)
        if avg_time > 60000:  # > 60 seconds
            recommendations.append(f"Average execution time is high ({avg_time/1000:.1f}s). Consider optimization.")
        
        return recommendations


# =============================================================================
# Example Usage and Testing
# =============================================================================

class MockAgentManager:
    """Mock agent manager for testing."""
    
    def __init__(self, simulate_failures: bool = False):
        self.agents: Dict[str, AgentConfig] = {}
        self.simulate_failures = simulate_failures
    
    async def spawn_agent(self, config: AgentConfig) -> str:
        self.agents[config.agent_id] = config
        return config.agent_id
    
    async def execute_task(self, agent_id: str, task: Task) -> Solution:
        config = self.agents.get(agent_id)
        if config is None:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Simulate occasional failures
        if self.simulate_failures and agent_id.endswith("_fail"):
            raise RuntimeError("Simulated agent failure")
        
        # Generate a mock solution
        solution = Solution(
            solution_id=str(uuid.uuid4()),
            agent_id=agent_id,
            task_id=task.task_id,
            files_changed={
                "src/main.py": f"# Solution by {agent_id}\ndef main():\n    print('Hello from {agent_id}')\n    return 42\n",
                "tests/test_main.py": f"# Tests for {agent_id}\ndef test_main():\n    assert main() == 42\n"
            },
            diff=f"--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,4 @@\n+# Solution by {agent_id}\n def main():\n     return 42\n",
            explanation=f"This solution implements the requirements using approach {agent_id}",
            status=SolutionStatus.COMPLETED,
            test_results={"passed": 5, "total": 5, "failed": 0},
            metrics={
                "lines_added": 10 + hash(agent_id) % 20,
                "lines_removed": 2,
                "complexity_score": 0.7 + (hash(agent_id) % 30) / 100
            }
        )
        
        return solution
    
    async def terminate_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            del self.agents[agent_id]


async def main():
    """Example usage of the tournament system."""
    print("=" * 60)
    print("Ansib-eL Tournament System Demo")
    print("=" * 60)
    
    # Create mock agent manager
    agent_manager = MockAgentManager(simulate_failures=True)
    
    # Create orchestrator
    orchestrator = TournamentOrchestrator(
        agent_manager=agent_manager,
        max_concurrent_agents=3
    )
    
    # Register progress callback
    def on_progress(tournament_id: str, info: Dict[str, Any]):
        print(f"  [Progress] {info.get('status')} - {info}")
    
    orchestrator.register_progress_callback(on_progress)
    
    # Create a task
    task = Task(
        task_id="task-001",
        description="Implement a function that returns the answer to life, the universe, and everything",
        requirements=[
            "Function should be named 'main'",
            "Should return integer 42",
            "Include basic tests"
        ],
        test_commands=["python -m pytest tests/"],
        metadata={"type": "function_implementation", "complexity": "low"}
    )
    
    # Create agent configurations
    agent_configs = [
        AgentConfig(
            agent_id="gpt-5.2-agent",
            agent_type="gpt-5.2",
            model_config={"temperature": 0.7, "max_tokens": 2000},
            timeout_seconds=60,
            priority=10
        ),
        AgentConfig(
            agent_id="claude-agent",
            agent_type="claude-opus-4.5",
            model_config={"temperature": 0.5, "max_tokens": 2000},
            timeout_seconds=60,
            priority=9
        ),
        AgentConfig(
            agent_id="gemini-3-flash-agent",
            agent_type="gemini-3-flash",
            model_config={"temperature": 0.8, "max_tokens": 1500},
            timeout_seconds=45,
            priority=5
        ),
        AgentConfig(
            agent_id="fail-agent_fail",  # Will simulate failure
            agent_type="unstable",
            timeout_seconds=30,
            priority=1
        )
    ]
    
    # Create tournament
    print("\n1. Creating tournament...")
    tournament = orchestrator.create_tournament(
        task=task,
        agent_configs=agent_configs,
        selection_mode=SelectionMode.HUMAN_CHOICE
    )
    print(f"   Tournament ID: {tournament.tournament_id}")
    print(f"   Agents: {len(agent_configs)}")
    
    # Run tournament
    print("\n2. Running tournament (parallel execution)...")
    result = await orchestrator.run_tournament(tournament.tournament_id)
    
    print(f"\n   Results:")
    print(f"   - Total solutions: {len(result.solutions)}")
    print(f"   - Successful: {result.execution_summary.get('successful', 0)}")
    print(f"   - Failed: {result.execution_summary.get('failed', 0)}")
    print(f"   - Timeouts: {result.execution_summary.get('timeouts', 0)}")
    
    # Present for review
    print("\n3. Generating review presentation...")
    review = await orchestrator.present_for_review(tournament.tournament_id)
    
    print(f"\n   Solutions Overview:")
    for comp in review.solution_comparisons:
        print(f"   - {comp['agent_id']}: score={comp['score']}, status={comp['status']}, time={comp['execution_time']:.0f}ms")
    
    if review.recommendations:
        print(f"\n   Recommendations:")
        for rec in review.recommendations:
            print(f"   - {rec}")
    
    # Select winner
    print("\n4. Selecting winner...")
    winner = await orchestrator.select_winner(tournament.tournament_id)
    print(f"   Winner: {winner.solution_id} (agent: {winner.agent_id})")
    print(f"   Files changed: {len(winner.files_changed)}")
    
    # Archive losers
    print("\n5. Archiving rejected solutions...")
    archived = await orchestrator.archive_losers(tournament.tournament_id)
    print(f"   Archived: {len(archived)} solutions")
    
    for archive in archived:
        print(f"   - {archive.solution.agent_id}: {archive.rejection_reason}")
    
    # Show markdown review
    print("\n6. Review Markdown Preview:")
    print("-" * 40)
    print(review.to_markdown()[:1000] + "...")
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
