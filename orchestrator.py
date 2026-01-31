"""
Ansib-eL Core Orchestrator Module

This module implements the central "Repo Manager" for the AI-Native Version Control System.
The Orchestrator acts as a project manager that:
- Breaks down human prompts into executable tasks
- Delegates tasks to sub-agents
- Controls main branch access with human-in-the-loop validation
- Manages repository state and pending approvals

Author: AI Systems Architect
"""

from __future__ import annotations

import uuid
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)
from queue import Queue, Empty
from threading import Lock


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TaskStatus(Enum):
    """Status states for a task lifecycle."""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class MergeStatus(Enum):
    """Possible outcomes of a merge operation."""
    PENDING_APPROVAL = auto()
    APPROVED = auto()
    REJECTED = auto()
    MERGED = auto()
    CONFLICT = auto()
    ERROR = auto()


class BranchProtectionLevel(Enum):
    """Protection levels for branches."""
    NONE = auto()           # No protection
    REVIEW_REQUIRED = auto()  # Requires review but not necessarily human
    HUMAN_APPROVAL = auto()   # Requires explicit human approval
    LOCKED = auto()           # Completely locked


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class TaskId:
    """Unique identifier for tasks."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AgentId:
    """Unique identifier for agents."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass
class CodeChange:
    """Represents a code change in the repository."""
    file_path: str
    diff_content: str
    description: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "diff_content": self.diff_content,
            "description": self.description,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class Task:
    """
    Represents an individual task derived from a human prompt.
    
    Attributes:
        id: Unique task identifier
        description: Human-readable task description
        requirements: List of specific requirements to fulfill
        acceptance_criteria: Criteria for task completion
        priority: Task priority level
        estimated_effort: Estimated effort in story points or hours
        dependencies: List of task IDs that must complete before this task
        metadata: Additional task metadata
        status: Current task status
        assigned_agent: ID of agent assigned to this task
        created_at: Task creation timestamp
        completed_at: Task completion timestamp
    """
    description: str
    requirements: List[str]
    acceptance_criteria: List[str]
    id: TaskId = field(default_factory=TaskId)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_effort: float = 1.0
    dependencies: List[TaskId] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[AgentId] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def mark_completed(self) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        logger.info(f"Task {self.id} marked as completed")
    
    def mark_failed(self, reason: str) -> None:
        """Mark the task as failed with a reason."""
        self.status = TaskStatus.FAILED
        self.metadata["failure_reason"] = reason
        logger.error(f"Task {self.id} marked as failed: {reason}")
    
    def can_execute(self, completed_tasks: Set[TaskId]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": str(self.id),
            "description": self.description,
            "requirements": self.requirements,
            "acceptance_criteria": self.acceptance_criteria,
            "priority": self.priority.name,
            "estimated_effort": self.estimated_effort,
            "dependencies": [str(dep) for dep in self.dependencies],
            "status": self.status.name,
            "assigned_agent": str(self.assigned_agent) if self.assigned_agent else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class TaskBreakdown:
    """
    Represents the decomposition of a human prompt into executable tasks.
    
    Attributes:
        original_prompt: The original human prompt
        tasks: List of decomposed tasks
        execution_strategy: Recommended execution strategy (sequential/parallel/mixed)
        context: Additional context extracted from the prompt
    """
    original_prompt: str
    tasks: List[Task]
    execution_strategy: str = "sequential"
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_critical_tasks(self) -> List[Task]:
        """Return tasks with CRITICAL priority."""
        return [t for t in self.tasks if t.priority == TaskPriority.CRITICAL]
    
    def get_execution_order(self) -> List[Task]:
        """
        Return tasks in dependency-respecting execution order.
        Uses topological sort for dependency resolution.
        """
        # Simple topological sort
        completed: Set[TaskId] = set()
        ordered: List[Task] = []
        remaining = list(self.tasks)
        
        while remaining:
            progress = False
            for task in remaining[:]:
                if task.can_execute(completed):
                    ordered.append(task)
                    completed.add(task.id)
                    remaining.remove(task)
                    progress = True
            
            if not progress and remaining:
                # Circular dependency detected, break with error
                raise ValueError(f"Circular dependency detected in tasks: {remaining}")
        
        return ordered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert breakdown to dictionary representation."""
        return {
            "original_prompt": self.original_prompt,
            "tasks": [task.to_dict() for task in self.tasks],
            "execution_strategy": self.execution_strategy,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Solution:
    """
    Represents a solution produced by an agent.
    
    Attributes:
        task_id: ID of the task this solution addresses
        agent_id: ID of the agent that produced this solution
        changes: List of code changes
        explanation: Explanation of the solution approach
        tests_included: Whether tests are included
        documentation: Any documentation produced
    """
    task_id: TaskId
    agent_id: AgentId
    changes: List[CodeChange]
    explanation: str
    tests_included: bool = False
    documentation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_affected_files(self) -> Set[str]:
        """Return set of files affected by this solution."""
        return {change.file_path for change in self.changes}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary representation."""
        return {
            "task_id": str(self.task_id),
            "agent_id": str(self.agent_id),
            "changes": [change.to_dict() for change in self.changes],
            "explanation": self.explanation,
            "tests_included": self.tests_included,
            "documentation": self.documentation,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DelegationResult:
    """
    Result of delegating a task to an agent.
    
    Attributes:
        success: Whether delegation was successful
        task_id: ID of the delegated task
        assigned_agent: ID of the agent assigned
        message: Status message
        estimated_completion: Estimated completion time
    """
    success: bool
    task_id: TaskId
    assigned_agent: Optional[AgentId]
    message: str
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeResult:
    """
    Result of a merge operation.
    
    Attributes:
        status: Status of the merge operation
        solution: The solution being merged
        message: Human-readable result message
        commit_hash: Git commit hash if merge was successful
        conflicts: List of conflicts if any
        approved_by: ID of approver if human approval was required
    """
    status: MergeStatus
    solution: Solution
    message: str
    commit_hash: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)
    approved_by: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RepoStatus:
    """
    Snapshot of repository state.
    
    Attributes:
        current_branch: Current active branch
        pending_approvals: Number of solutions awaiting approval
        active_tasks: Number of tasks currently in progress
        completed_tasks: Number of completed tasks
        failed_tasks: Number of failed tasks
        branch_protection: Current branch protection level
        last_commit: Hash of last commit
        working_tree_clean: Whether working tree has uncommitted changes
    """
    current_branch: str
    pending_approvals: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    branch_protection: BranchProtectionLevel
    last_commit: Optional[str]
    working_tree_clean: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self) -> bool:
        """Check if repository is in a healthy state."""
        return self.failed_tasks == 0 and self.working_tree_clean


@dataclass
class ApprovalRequest:
    """
    Represents a pending approval request for human review.
    
    Attributes:
        id: Unique approval request ID
        solution: Solution awaiting approval
        submitted_at: When the request was submitted
        priority: Priority of the approval request
    """
    solution: Solution
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: datetime = field(default_factory=datetime.now)
    priority: TaskPriority = TaskPriority.MEDIUM
    reviewed_by: Optional[str] = None
    review_comments: Optional[str] = None
    
    def approve(self, reviewer: str, comments: Optional[str] = None) -> None:
        """Mark the approval request as approved."""
        self.reviewed_by = reviewer
        self.review_comments = comments
        logger.info(f"Approval request {self.id} approved by {reviewer}")
    
    def reject(self, reviewer: str, comments: str) -> None:
        """Mark the approval request as rejected."""
        self.reviewed_by = reviewer
        self.review_comments = comments
        logger.info(f"Approval request {self.id} rejected by {reviewer}: {comments}")


# =============================================================================
# Protocols (Interfaces)
# =============================================================================

class GitWrapperInterface(Protocol):
    """Protocol for Git wrapper integration."""
    
    def get_current_branch(self) -> str:
        """Get the name of the current branch."""
        ...
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        """Create a new branch."""
        ...
    
    def checkout_branch(self, branch_name: str) -> bool:
        """Checkout a branch."""
        ...
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> str:
        """Commit changes and return commit hash."""
        ...
    
    def merge_branch(self, branch_name: str, strategy: str = "recursive") -> MergeResult:
        """Merge a branch into current branch."""
        ...
    
    def get_last_commit(self) -> Optional[str]:
        """Get hash of last commit."""
        ...
    
    def is_working_tree_clean(self) -> bool:
        """Check if working tree is clean."""
        ...
    
    def get_branch_protection(self, branch_name: str) -> BranchProtectionLevel:
        """Get protection level for a branch."""
        ...
    
    def set_branch_protection(self, branch_name: str, level: BranchProtectionLevel) -> bool:
        """Set protection level for a branch."""
        ...


class AgentInterface(Protocol):
    """Protocol for agent integration."""
    
    @property
    def id(self) -> AgentId:
        """Unique agent identifier."""
        ...
    
    @property
    def capabilities(self) -> List[str]:
        """List of agent capabilities."""
        ...
    
    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle the given task."""
        ...
    
    def execute(self, task: Task) -> Solution:
        """Execute the task and return a solution."""
        ...
    
    def get_status(self) -> TaskStatus:
        """Get current agent status."""
        ...


class TournamentInterface(Protocol):
    """Protocol for tournament/parallel execution integration."""
    
    def register_agent(self, agent: AgentInterface) -> None:
        """Register an agent for tournament participation."""
        ...
    
    def execute_parallel(self, task: Task, agents: List[AgentInterface]) -> List[Solution]:
        """Execute task with multiple agents in parallel."""
        ...
    
    def select_winner(self, solutions: List[Solution]) -> Solution:
        """Select the best solution from candidates."""
        ...


class HumanInterface(Protocol):
    """Protocol for human-in-the-loop integration."""
    
    def request_approval(self, request: ApprovalRequest) -> bool:
        """Request human approval for a solution."""
        ...
    
    def provide_feedback(self, task_id: TaskId, feedback: str) -> None:
        """Provide feedback on a task or solution."""
        ...
    
    def is_available(self) -> bool:
        """Check if human reviewer is available."""
        ...


# =============================================================================
# Core Orchestrator Class
# =============================================================================

class Orchestrator:
    """
    Central orchestrator for the Ansib-eL AI-Native Version Control System.
    
    The Orchestrator acts as the "Repo Manager" - it doesn't write code but
    oversees repository state, manages task delegation, and controls access
    to protected branches through human-in-the-loop validation.
    
    Key Responsibilities:
    1. Break down human prompts into executable tasks
    2. Delegate tasks to appropriate sub-agents
    3. Control main branch access with approval gates
    4. Manage repository state and health
    
    Example:
        >>> orchestrator = Orchestrator("/path/to/repo")
        >>> breakdown = orchestrator.process_human_prompt("Add user authentication")
        >>> for task in breakdown.tasks:
        ...     result = orchestrator.delegate_task(task, agent_pool)
        >>> # Solutions await human approval before merging to main
    """
    
    # Default configuration
    DEFAULT_MAIN_BRANCH = "main"
    DEFAULT_PROTECTION_LEVEL = BranchProtectionLevel.HUMAN_APPROVAL
    
    def __init__(
        self,
        repo_path: str,
        git_wrapper: Optional[GitWrapperInterface] = None,
        human_interface: Optional[HumanInterface] = None,
        tournament_system: Optional[TournamentInterface] = None,
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            repo_path: Path to the Git repository
            git_wrapper: Optional Git wrapper instance (for testing/injection)
            human_interface: Optional human interface instance
            tournament_system: Optional tournament system for parallel execution
        """
        self.repo_path = Path(repo_path).resolve()
        self._git = git_wrapper  # Will be initialized lazily
        self._human = human_interface
        self._tournament = tournament_system
        
        # Task management
        self._tasks: Dict[TaskId, Task] = {}
        self._solutions: Dict[TaskId, Solution] = {}
        self._task_lock = Lock()
        
        # Approval queue
        self._approval_queue: Queue[ApprovalRequest] = Queue()
        self._approved_solutions: List[Solution] = []
        self._rejected_solutions: List[Solution] = []
        
        # Branch protection
        self._protected_branches: Dict[str, BranchProtectionLevel] = {
            self.DEFAULT_MAIN_BRANCH: self.DEFAULT_PROTECTION_LEVEL
        }
        
        # Statistics
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "merges_approved": 0,
            "merges_rejected": 0,
        }
        
        logger.info(f"Orchestrator initialized for repository: {self.repo_path}")
    
    # -------------------------------------------------------------------------
    # Core Public Methods
    # -------------------------------------------------------------------------
    
    def process_human_prompt(self, prompt: str) -> TaskBreakdown:
        """
        Break down a human prompt into executable tasks.
        
        This method analyzes the human prompt and decomposes it into
        a structured set of tasks with dependencies, priorities, and
        acceptance criteria.
        
        Args:
            prompt: The natural language prompt from the human
            
        Returns:
            TaskBreakdown containing decomposed tasks and execution strategy
            
        Example:
            >>> breakdown = orchestrator.process_human_prompt(
            ...     "Add user authentication with login and signup"
            ... )
            >>> print(f"Created {len(breakdown.tasks)} tasks")
        """
        logger.info(f"Processing human prompt: {prompt[:50]}...")
        
        # TODO: Integration Point - Connect to LLM for intelligent task breakdown
        # This is where an LLM would analyze the prompt and create tasks
        
        # For now, implement a simple rule-based breakdown
        tasks = self._breakdown_prompt(prompt)
        
        # Determine execution strategy based on task dependencies
        strategy = self._determine_execution_strategy(tasks)
        
        breakdown = TaskBreakdown(
            original_prompt=prompt,
            tasks=tasks,
            execution_strategy=strategy,
            context={"prompt_length": len(prompt), "word_count": len(prompt.split())}
        )
        
        # Register tasks
        with self._task_lock:
            for task in tasks:
                self._tasks[task.id] = task
                self._stats["tasks_created"] += 1
        
        logger.info(f"Prompt breakdown complete: {len(tasks)} tasks created")
        return breakdown
    
    def delegate_task(
        self,
        task: Task,
        agent_pool: List[AgentInterface],
        use_tournament: bool = False,
    ) -> DelegationResult:
        """
        Delegate a task to an appropriate agent from the pool.
        
        Args:
            task: The task to delegate
            agent_pool: Available agents to choose from
            use_tournament: Whether to use tournament mode (parallel execution)
            
        Returns:
            DelegationResult indicating success/failure and assignment details
            
        Raises:
            ValueError: If no suitable agent found in pool
            
        Example:
            >>> result = orchestrator.delegate_task(task, [agent1, agent2, agent3])
            >>> if result.success:
            ...     print(f"Task assigned to {result.assigned_agent}")
        """
        logger.info(f"Delegating task {task.id}: {task.description[:40]}...")
        
        if not agent_pool:
            return DelegationResult(
                success=False,
                task_id=task.id,
                assigned_agent=None,
                message="No agents available in pool"
            )
        
        # Find suitable agents
        suitable_agents = [a for a in agent_pool if a.can_handle(task)]
        
        if not suitable_agents:
            return DelegationResult(
                success=False,
                task_id=task.id,
                assigned_agent=None,
                message=f"No agent capable of handling task: {task.description}"
            )
        
        # Update task status
        task.status = TaskStatus.ASSIGNED
        
        if use_tournament and self._tournament and len(suitable_agents) > 1:
            # Tournament mode - execute with multiple agents and select best
            return self._delegate_tournament(task, suitable_agents)
        else:
            # Single agent mode - select best match
            return self._delegate_single(task, suitable_agents)
    
    def validate_and_merge(
        self,
        solution: Solution,
        force: bool = False,
        reviewer: Optional[str] = None,
    ) -> MergeResult:
        """
        Validate a solution and merge if approved.
        
        This is the human-in-the-loop gateway. Solutions cannot be merged
        to protected branches without explicit approval.
        
        Args:
            solution: The solution to validate and merge
            force: Whether to bypass protection (admin only)
            reviewer: ID of the human reviewer
            
        Returns:
            MergeResult indicating the outcome
            
        Example:
            >>> result = orchestrator.validate_and_merge(solution, reviewer="alice")
            >>> if result.status == MergeStatus.MERGED:
            ...     print(f"Successfully merged: {result.commit_hash}")
        """
        logger.info(f"Validating solution for task {solution.task_id}")
        
        current_branch = self._get_git_wrapper().get_current_branch()
        protection = self._get_branch_protection(current_branch)
        
        # Check if protection can be bypassed
        if force:
            logger.warning(f"Force merge requested for {solution.task_id}")
            protection = BranchProtectionLevel.NONE
        
        # Route based on protection level
        if protection == BranchProtectionLevel.HUMAN_APPROVAL:
            return self._merge_with_human_approval(solution, reviewer)
        elif protection == BranchProtectionLevel.REVIEW_REQUIRED:
            return self._merge_with_review(solution, reviewer)
        else:
            return self._merge_direct(solution)
    
    def get_repo_status(self) -> RepoStatus:
        """
        Get current repository status snapshot.
        
        Returns:
            RepoStatus containing comprehensive repository state
            
        Example:
            >>> status = orchestrator.get_repo_status()
            >>> print(f"Pending approvals: {status.pending_approvals}")
            >>> print(f"Repository healthy: {status.is_healthy()}")
        """
        git = self._get_git_wrapper()
        
        with self._task_lock:
            active = sum(1 for t in self._tasks.values() 
                        if t.status == TaskStatus.IN_PROGRESS)
            completed = sum(1 for t in self._tasks.values() 
                          if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() 
                        if t.status == TaskStatus.FAILED)
        
        current_branch = git.get_current_branch()
        
        return RepoStatus(
            current_branch=current_branch,
            pending_approvals=self._approval_queue.qsize(),
            active_tasks=active,
            completed_tasks=completed,
            failed_tasks=failed,
            branch_protection=self._get_branch_protection(current_branch),
            last_commit=git.get_last_commit(),
            working_tree_clean=git.is_working_tree_clean(),
        )
    
    # -------------------------------------------------------------------------
    # Branch Protection Methods
    # -------------------------------------------------------------------------
    
    def set_branch_protection(
        self,
        branch_name: str,
        level: BranchProtectionLevel,
    ) -> bool:
        """
        Set protection level for a branch.
        
        Args:
            branch_name: Name of the branch to protect
            level: Protection level to apply
            
        Returns:
            True if protection was set successfully
        """
        self._protected_branches[branch_name] = level
        logger.info(f"Set protection level for {branch_name}: {level.name}")
        
        # Also update via Git wrapper if available
        git = self._get_git_wrapper()
        if hasattr(git, 'set_branch_protection'):
            return git.set_branch_protection(branch_name, level)
        return True
    
    def get_branch_protection(self, branch_name: str) -> BranchProtectionLevel:
        """Get protection level for a branch."""
        return self._get_branch_protection(branch_name)
    
    def lock_branch(self, branch_name: str) -> bool:
        """Completely lock a branch (no commits allowed)."""
        return self.set_branch_protection(branch_name, BranchProtectionLevel.LOCKED)
    
    def unlock_branch(self, branch_name: str) -> bool:
        """Remove all protection from a branch."""
        return self.set_branch_protection(branch_name, BranchProtectionLevel.NONE)
    
    # -------------------------------------------------------------------------
    # Approval Queue Management
    # -------------------------------------------------------------------------
    
    def submit_for_approval(self, solution: Solution, priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """
        Submit a solution for human approval.
        
        Args:
            solution: Solution to be approved
            priority: Priority of the approval request
            
        Returns:
            Approval request ID
        """
        request = ApprovalRequest(solution=solution, priority=priority)
        self._approval_queue.put(request)
        logger.info(f"Solution submitted for approval: {request.id}")
        return request.id
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get list of all pending approval requests."""
        # Convert queue to list without consuming
        items = list(self._approval_queue.queue)
        return sorted(items, key=lambda x: x.priority.value)
    
    def approve_solution(
        self,
        approval_id: str,
        reviewer: str,
        comments: Optional[str] = None,
    ) -> bool:
        """
        Approve a pending solution.
        
        Args:
            approval_id: ID of the approval request
            reviewer: ID of the approving reviewer
            comments: Optional review comments
            
        Returns:
            True if approval was processed
        """
        # Find and remove from queue
        temp_queue = Queue()
        found = None
        
        while not self._approval_queue.empty():
            try:
                req = self._approval_queue.get_nowait()
                if req.id == approval_id:
                    found = req
                else:
                    temp_queue.put(req)
            except Empty:
                break
        
        # Restore queue
        while not temp_queue.empty():
            self._approval_queue.put(temp_queue.get())
        
        if found:
            found.approve(reviewer, comments)
            self._approved_solutions.append(found.solution)
            self._stats["merges_approved"] += 1
            logger.info(f"Solution {approval_id} approved by {reviewer}")
            return True
        
        logger.warning(f"Approval request {approval_id} not found")
        return False
    
    def reject_solution(
        self,
        approval_id: str,
        reviewer: str,
        comments: str,
    ) -> bool:
        """Reject a pending solution with feedback."""
        # Similar to approve but mark as rejected
        temp_queue = Queue()
        found = None
        
        while not self._approval_queue.empty():
            try:
                req = self._approval_queue.get_nowait()
                if req.id == approval_id:
                    found = req
                else:
                    temp_queue.put(req)
            except Empty:
                break
        
        while not temp_queue.empty():
            self._approval_queue.put(temp_queue.get())
        
        if found:
            found.reject(reviewer, comments)
            self._rejected_solutions.append(found.solution)
            self._stats["merges_rejected"] += 1
            logger.info(f"Solution {approval_id} rejected by {reviewer}")
            return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Statistics and Reporting
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "pending_approvals": self._approval_queue.qsize(),
            "approved_solutions": len(self._approved_solutions),
            "rejected_solutions": len(self._rejected_solutions),
            "total_tasks": len(self._tasks),
        }
    
    def export_state(self, output_path: Optional[str] = None) -> str:
        """
        Export orchestrator state to JSON.
        
        Args:
            output_path: Path to save state file (optional)
            
        Returns:
            Path to the saved state file
        """
        state = {
            "repo_path": str(self.repo_path),
            "statistics": self._stats,
            "protected_branches": {
                name: level.name for name, level in self._protected_branches.items()
            },
            "tasks": {str(tid): task.to_dict() for tid, task in self._tasks.items()},
            "pending_approvals": len(self._approval_queue.queue),
            "timestamp": datetime.now().isoformat(),
        }
        
        if output_path is None:
            output_path = self.repo_path / ".ansibel" / "orchestrator_state.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State exported to {output_path}")
        return str(output_path)
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _breakdown_prompt(self, prompt: str) -> List[Task]:
        """
        Break down a prompt into tasks.
        
        TODO: Integration Point - Replace with LLM-based intelligent breakdown
        """
        tasks = []
        prompt_lower = prompt.lower()
        
        # Simple keyword-based breakdown for demonstration
        if "authentication" in prompt_lower or "login" in prompt_lower:
            tasks.append(Task(
                description="Design authentication system architecture",
                requirements=["Define auth flow", "Choose auth mechanism"],
                acceptance_criteria=["Architecture document created"],
                priority=TaskPriority.HIGH,
            ))
            tasks.append(Task(
                description="Implement user login endpoint",
                requirements=["Create login API", "Validate credentials"],
                acceptance_criteria=["Login works with valid credentials"],
                priority=TaskPriority.HIGH,
                dependencies=[tasks[-1].id],
            ))
            tasks.append(Task(
                description="Implement user signup endpoint",
                requirements=["Create signup API", "Validate input"],
                acceptance_criteria=["Users can register successfully"],
                priority=TaskPriority.HIGH,
                dependencies=[tasks[-1].id],
            ))
        elif "test" in prompt_lower:
            tasks.append(Task(
                description="Write unit tests",
                requirements=["Cover core functionality", "Achieve 80% coverage"],
                acceptance_criteria=["All tests pass"],
                priority=TaskPriority.MEDIUM,
            ))
        else:
            # Generic single task
            tasks.append(Task(
                description=prompt,
                requirements=["Implement as specified"],
                acceptance_criteria=["Feature works as expected"],
                priority=TaskPriority.MEDIUM,
            ))
        
        return tasks
    
    def _determine_execution_strategy(self, tasks: List[Task]) -> str:
        """Determine the best execution strategy for tasks."""
        if len(tasks) <= 1:
            return "sequential"
        
        # Check for dependencies
        has_dependencies = any(task.dependencies for task in tasks)
        
        if has_dependencies:
            return "mixed"  # Some parallel, some sequential
        return "parallel"
    
    def _delegate_single(
        self,
        task: Task,
        agents: List[AgentInterface],
    ) -> DelegationResult:
        """Delegate to a single best-matching agent."""
        # Select agent with best capability match
        best_agent = agents[0]  # Simplified - would use scoring
        
        task.assigned_agent = best_agent.id
        task.status = TaskStatus.IN_PROGRESS
        
        # TODO: Integration Point - Actually execute via agent
        # For now, simulate async execution
        
        return DelegationResult(
            success=True,
            task_id=task.id,
            assigned_agent=best_agent.id,
            message=f"Task assigned to agent {best_agent.id}",
            estimated_completion=datetime.now(),  # Would be actual estimate
        )
    
    def _delegate_tournament(
        self,
        task: Task,
        agents: List[AgentInterface],
    ) -> DelegationResult:
        """Delegate using tournament mode (parallel execution)."""
        if self._tournament:
            # Use tournament system
            self._tournament.register_agents(agents)
            solutions = self._tournament.execute_parallel(task, agents)
            winner = self._tournament.select_winner(solutions)
            
            return DelegationResult(
                success=True,
                task_id=task.id,
                assigned_agent=winner.agent_id,
                message=f"Tournament completed, winner: {winner.agent_id}",
            )
        else:
            # Fallback to single agent
            logger.warning("Tournament system not available, using single agent")
            return self._delegate_single(task, agents)
    
    def _get_branch_protection(self, branch_name: str) -> BranchProtectionLevel:
        """Get protection level for a branch."""
        return self._protected_branches.get(
            branch_name,
            BranchProtectionLevel.NONE
        )
    
    def _merge_with_human_approval(
        self,
        solution: Solution,
        reviewer: Optional[str],
    ) -> MergeResult:
        """Merge requiring explicit human approval."""
        # Submit for approval if not already
        approval_id = self.submit_for_approval(solution)
        
        # If reviewer provided, try to process immediately
        if reviewer and self._human and self._human.is_available():
            request = next(
                (r for r in self._approval_queue.queue if r.id == approval_id),
                None
            )
            if request and self._human.request_approval(request):
                # Approved - proceed with merge
                return self._execute_merge(solution, reviewer)
            else:
                return MergeResult(
                    status=MergeStatus.REJECTED,
                    solution=solution,
                    message="Solution rejected by human reviewer",
                )
        
        # Queued for later approval
        return MergeResult(
            status=MergeStatus.PENDING_APPROVAL,
            solution=solution,
            message=f"Solution queued for approval (ID: {approval_id})",
        )
    
    def _merge_with_review(
        self,
        solution: Solution,
        reviewer: Optional[str],
    ) -> MergeResult:
        """Merge requiring review (can be automated or human)."""
        # TODO: Integration Point - Automated review checks
        # For now, require human
        return self._merge_with_human_approval(solution, reviewer)
    
    def _merge_direct(self, solution: Solution) -> MergeResult:
        """Merge without approval requirements."""
        return self._execute_merge(solution, None)
    
    def _execute_merge(
        self,
        solution: Solution,
        approver: Optional[str],
    ) -> MergeResult:
        """Execute the actual merge operation."""
        git = self._get_git_wrapper()
        
        try:
            # TODO: Integration Point - Apply changes via Git wrapper
            # commit_hash = git.commit_changes(
            #     message=f"Implement: {solution.explanation}",
            #     files=list(solution.get_affected_files())
            # )
            
            # Update task status
            if solution.task_id in self._tasks:
                self._tasks[solution.task_id].mark_completed()
                self._stats["tasks_completed"] += 1
            
            return MergeResult(
                status=MergeStatus.MERGED,
                solution=solution,
                message="Successfully merged",
                commit_hash="simulated_commit_hash",  # Would be actual hash
                approved_by=approver,
            )
        
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return MergeResult(
                status=MergeStatus.ERROR,
                solution=solution,
                message=f"Merge failed: {str(e)}",
            )
    
    def _get_git_wrapper(self) -> GitWrapperInterface:
        """Get or initialize Git wrapper."""
        if self._git is None:
            # TODO: Integration Point - Initialize actual Git wrapper
            # For now, return a mock for demonstration
            self._git = MockGitWrapper(self.repo_path)
        return self._git


# =============================================================================
# Mock Implementations for Testing/Demonstration
# =============================================================================

class MockGitWrapper:
    """Mock Git wrapper for demonstration purposes."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._branch = "main"
        self._commit = "abc123"
        self._clean = True
    
    def get_current_branch(self) -> str:
        return self._branch
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        return True
    
    def checkout_branch(self, branch_name: str) -> bool:
        self._branch = branch_name
        return True
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> str:
        self._commit = f"commit_{uuid.uuid4().hex[:8]}"
        return self._commit
    
    def merge_branch(self, branch_name: str, strategy: str = "recursive") -> MergeResult:
        return MergeResult(
            status=MergeStatus.MERGED,
            solution=Solution(
                task_id=TaskId(),
                agent_id=AgentId(),
                changes=[],
                explanation="Mock merge",
            ),
            message="Mock merge successful",
            commit_hash=self._commit,
        )
    
    def get_last_commit(self) -> Optional[str]:
        return self._commit
    
    def is_working_tree_clean(self) -> bool:
        return self._clean
    
    def get_branch_protection(self, branch_name: str) -> BranchProtectionLevel:
        return BranchProtectionLevel.HUMAN_APPROVAL if branch_name == "main" else BranchProtectionLevel.NONE
    
    def set_branch_protection(self, branch_name: str, level: BranchProtectionLevel) -> bool:
        return True


class MockAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, name: str, capabilities: List[str]):
        self._id = AgentId()
        self.name = name
        self._capabilities = capabilities
        self._status = TaskStatus.PENDING
    
    @property
    def id(self) -> AgentId:
        return self._id
    
    @property
    def capabilities(self) -> List[str]:
        return self._capabilities
    
    def can_handle(self, task: Task) -> bool:
        # Simple capability matching
        task_desc = task.description.lower()
        return any(cap in task_desc for cap in self._capabilities)
    
    def execute(self, task: Task) -> Solution:
        self._status = TaskStatus.IN_PROGRESS
        # Simulate execution
        self._status = TaskStatus.COMPLETED
        return Solution(
            task_id=task.id,
            agent_id=self._id,
            changes=[],
            explanation=f"Executed by {self.name}",
        )
    
    def get_status(self) -> TaskStatus:
        return self._status


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of the Orchestrator functionality.
    
    This example shows how to:
    1. Initialize the orchestrator
    2. Process a human prompt into tasks
    3. Delegate tasks to agents
    4. Submit solutions for approval
    5. Get repository status
    """
    
    print("=" * 60)
    print("Ansib-eL Orchestrator Demo")
    print("=" * 60)
    
    # 1. Initialize orchestrator
    orchestrator = Orchestrator("/tmp/demo_repo")
    print(f"\n1. Initialized orchestrator for: {orchestrator.repo_path}")
    
    # 2. Create some mock agents
    agents = [
        MockAgent("BackendAgent", ["authentication", "api", "endpoint"]),
        MockAgent("FrontendAgent", ["ui", "component", "interface"]),
        MockAgent("TestAgent", ["test", "testing", "coverage"]),
    ]
    print(f"\n2. Created {len(agents)} agents: {[a.name for a in agents]}")
    
    # 3. Process a human prompt
    prompt = "Add user authentication with login and signup endpoints"
    breakdown = orchestrator.process_human_prompt(prompt)
    
    print(f"\n3. Processed prompt: '{prompt}'")
    print(f"   Created {len(breakdown.tasks)} tasks:")
    for i, task in enumerate(breakdown.tasks, 1):
        print(f"   {i}. [{task.priority.name}] {task.description}")
    
    # 4. Delegate tasks
    print(f"\n4. Delegating tasks...")
    delegation_results = []
    for task in breakdown.get_execution_order():
        result = orchestrator.delegate_task(task, agents)
        delegation_results.append(result)
        status = "OK" if result.success else "FAIL"
        print(f"   [{status}] Task '{task.description[:30]}...' -> {result.assigned_agent}")
    
    # 5. Create a mock solution and submit for approval
    mock_solution = Solution(
        task_id=breakdown.tasks[0].id,
        agent_id=agents[0].id,
        changes=[
            CodeChange(
                file_path="auth.py",
                diff_content="+ def login(): pass",
                description="Added login function",
            )
        ],
        explanation="Implemented user authentication system",
        tests_included=True,
    )
    
    approval_id = orchestrator.submit_for_approval(mock_solution)
    print(f"\n5. Submitted solution for approval (ID: {approval_id})")
    
    # 6. Get repository status
    status = orchestrator.get_repo_status()
    print(f"\n6. Repository Status:")
    print(f"   Current branch: {status.current_branch}")
    print(f"   Pending approvals: {status.pending_approvals}")
    print(f"   Active tasks: {status.active_tasks}")
    print(f"   Completed tasks: {status.completed_tasks}")
    print(f"   Branch protection: {status.branch_protection.name}")
    print(f"   Repository healthy: {status.is_healthy()}")
    
    # 7. Get statistics
    stats = orchestrator.get_statistics()
    print(f"\n7. Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 8. Export state
    state_path = orchestrator.export_state()
    print(f"\n8. State exported to: {state_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
