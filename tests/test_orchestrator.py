"""
Comprehensive unit tests for the Ansib-eL Orchestrator class.

Tests cover:
- Prompt breakdown (_breakdown_prompt) for various prompt categories
- Task execution order via topological sort and circular dependency detection
- Task delegation (empty pool, single agent, tournament mode)
- Approval queue management (submit, approve, reject, non-existent)
- Branch protection (set, get, lock, unlock)
- State export and statistics
- Task and TaskBreakdown data structures
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ansibel.orchestrator import (
    AgentId,
    ApprovalResult,
    BranchProtectionLevel,
    CodeChange,
    MergeResult,
    MergeStatus,
    Orchestrator,
    Solution,
    Task,
    TaskBreakdown,
    TaskId,
    TaskPriority,
    TaskStatus,
)

# =============================================================================
# Mock / Fake Helpers
# =============================================================================


class FakeGitWrapper:
    """Fake GitWrapper that satisfies GitWrapperInterface."""

    def __init__(self, branch: str = "main", last_commit: str = "abc123", clean: bool = True):
        self._branch = branch
        self._last_commit = last_commit
        self._clean = clean
        self._protections: dict[str, BranchProtectionLevel] = {}

    def get_current_branch(self) -> str:
        return self._branch

    def create_branch(self, branch_name: str, from_branch: str | None = None) -> bool:
        return True

    def checkout_branch(self, branch_name: str) -> bool:
        self._branch = branch_name
        return True

    def commit_changes(self, message: str, files: list[str] | None = None) -> str:
        return "deadbeef"

    def merge_branch(self, branch_name: str, strategy: str = "recursive") -> MergeResult:
        return MergeResult(
            status=MergeStatus.MERGED,
            solution=MagicMock(),
            message="Merged",
        )

    def get_last_commit(self) -> str | None:
        return self._last_commit

    def is_working_tree_clean(self) -> bool:
        return self._clean

    def get_branch_protection(self, branch_name: str) -> BranchProtectionLevel:
        return self._protections.get(branch_name, BranchProtectionLevel.NONE)

    def set_branch_protection(self, branch_name: str, level: BranchProtectionLevel) -> bool:
        self._protections[branch_name] = level
        return True


class FakeAgent:
    """Fake agent that satisfies AgentInterface."""

    def __init__(
        self,
        agent_id: AgentId | None = None,
        capabilities: list[str] | None = None,
        can_handle_all: bool = True,
    ):
        self._id = agent_id or AgentId()
        self._capabilities = capabilities or ["coding"]
        self._can_handle_all = can_handle_all

    @property
    def id(self) -> AgentId:
        return self._id

    @property
    def capabilities(self) -> list[str]:
        return self._capabilities

    def can_handle(self, task: Task) -> bool:
        return self._can_handle_all

    def execute(self, task: Task) -> Solution:
        return Solution(
            task_id=task.id,
            agent_id=self._id,
            changes=[],
            explanation="Fake solution",
        )

    def get_status(self) -> TaskStatus:
        return TaskStatus.IDLE


def _make_solution(task_id: TaskId | None = None, agent_id: AgentId | None = None) -> Solution:
    """Helper to create a Solution for tests."""
    return Solution(
        task_id=task_id or TaskId(),
        agent_id=agent_id or AgentId(),
        changes=[
            CodeChange(
                file_path="auth.py",
                diff_content="+def login(): pass",
                description="Add login stub",
            )
        ],
        explanation="Stub implementation",
    )


# =============================================================================
# Tests
# =============================================================================


class TestBreakdownPrompt:
    """Tests for Orchestrator._breakdown_prompt via process_human_prompt."""

    def _make_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(
            repo_path=str(tmp_path),
            git_wrapper=FakeGitWrapper(),
        )

    def test_testing_prompt_produces_test_task(self, tmp_path: Path) -> None:
        """'Write unit tests for auth module' should produce a testing task."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Write unit tests for auth module")

        assert len(breakdown.tasks) >= 1
        descriptions = " ".join(t.description.lower() for t in breakdown.tasks)
        assert "test" in descriptions

    def test_fix_prompt_produces_fix_task(self, tmp_path: Path) -> None:
        """'Fix the database connection bug' should produce a fix/bug task."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Fix the database connection bug")

        assert len(breakdown.tasks) >= 1
        first_task = breakdown.tasks[0]
        assert first_task.description.lower().startswith("fix:")
        assert first_task.priority == TaskPriority.HIGH

    def test_auth_prompt_produces_feature_tasks(self, tmp_path: Path) -> None:
        """'Add user authentication' should produce authentication-related tasks."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Add user authentication")

        # Authentication prompts create multiple tasks (design + login + signup)
        assert len(breakdown.tasks) == 3
        descriptions = [t.description.lower() for t in breakdown.tasks]
        assert any("authentication" in d or "auth" in d for d in descriptions)
        assert any("login" in d for d in descriptions)
        assert any("signup" in d for d in descriptions)

    def test_refactor_prompt_produces_refactor_task(self, tmp_path: Path) -> None:
        """'Refactor the database layer' should produce a refactoring task."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Refactor the database layer")

        assert len(breakdown.tasks) >= 1
        first_task = breakdown.tasks[0]
        assert first_task.description.lower().startswith("refactor:")
        assert "Maintain existing behavior" in first_task.requirements

    def test_documentation_prompt_produces_doc_task(self, tmp_path: Path) -> None:
        """'Document the API endpoints' should produce a documentation task."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Document the API endpoints")

        assert len(breakdown.tasks) >= 1
        first_task = breakdown.tasks[0]
        assert first_task.priority == TaskPriority.LOW
        assert "Write documentation" in first_task.requirements

    def test_generic_prompt_produces_general_task(self, tmp_path: Path) -> None:
        """An unrecognized prompt falls through to the generic handler."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Do something unusual and novel")

        assert len(breakdown.tasks) == 1
        task = breakdown.tasks[0]
        assert task.description == "Do something unusual and novel"
        assert task.priority == TaskPriority.MEDIUM

    def test_delete_prompt(self, tmp_path: Path) -> None:
        """'Remove the legacy endpoint' should produce a removal task."""
        orch = self._make_orchestrator(tmp_path)
        breakdown = orch.process_human_prompt("Remove the legacy endpoint")

        assert len(breakdown.tasks) >= 1
        assert breakdown.tasks[0].description.lower().startswith("remove:")


class TestTaskExecutionOrder:
    """Tests for topological sort in TaskBreakdown.get_execution_order."""

    def test_simple_dependency_chain(self) -> None:
        """Tasks with linear dependencies should be ordered correctly."""
        task_a = Task(description="A", requirements=[], acceptance_criteria=[])
        task_b = Task(
            description="B",
            requirements=[],
            acceptance_criteria=[],
            dependencies=[task_a.id],
        )
        task_c = Task(
            description="C",
            requirements=[],
            acceptance_criteria=[],
            dependencies=[task_b.id],
        )

        breakdown = TaskBreakdown(
            original_prompt="test",
            tasks=[task_c, task_b, task_a],  # out of order
        )

        ordered = breakdown.get_execution_order()
        descriptions = [t.description for t in ordered]
        assert descriptions == ["A", "B", "C"]

    def test_independent_tasks_all_returned(self) -> None:
        """Tasks with no dependencies should all appear in the result."""
        tasks = [
            Task(description=f"Task {i}", requirements=[], acceptance_criteria=[]) for i in range(4)
        ]
        breakdown = TaskBreakdown(original_prompt="test", tasks=tasks)

        ordered = breakdown.get_execution_order()
        assert len(ordered) == 4

    def test_circular_dependency_raises(self) -> None:
        """Circular dependencies should raise ValueError."""
        task_a = Task(description="A", requirements=[], acceptance_criteria=[])
        task_b = Task(description="B", requirements=[], acceptance_criteria=[])

        # Inject circular dependency after creation
        task_a.dependencies = [task_b.id]
        task_b.dependencies = [task_a.id]

        breakdown = TaskBreakdown(original_prompt="test", tasks=[task_a, task_b])

        with pytest.raises(ValueError, match="Circular dependency"):
            breakdown.get_execution_order()


class TestDelegateTask:
    """Tests for Orchestrator.delegate_task."""

    def _make_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(
            repo_path=str(tmp_path),
            git_wrapper=FakeGitWrapper(),
        )

    def test_empty_agent_pool(self, tmp_path: Path) -> None:
        """Delegation with empty pool should fail gracefully."""
        orch = self._make_orchestrator(tmp_path)
        task = Task(description="Test", requirements=[], acceptance_criteria=[])

        result = orch.delegate_task(task, agent_pool=[])

        assert result.success is False
        assert result.assigned_agent is None
        assert "No agents available" in result.message

    def test_single_capable_agent(self, tmp_path: Path) -> None:
        """Delegation with one capable agent should succeed."""
        orch = self._make_orchestrator(tmp_path)
        task = Task(description="Test", requirements=[], acceptance_criteria=[])
        agent = FakeAgent()

        result = orch.delegate_task(task, agent_pool=[agent])

        assert result.success is True
        assert result.assigned_agent == agent.id
        assert task.status == TaskStatus.IN_PROGRESS

    def test_no_capable_agents(self, tmp_path: Path) -> None:
        """Delegation should fail when no agents can handle the task."""
        orch = self._make_orchestrator(tmp_path)
        task = Task(description="Test", requirements=[], acceptance_criteria=[])
        agent = FakeAgent(can_handle_all=False)

        result = orch.delegate_task(task, agent_pool=[agent])

        assert result.success is False
        assert "No agent capable" in result.message

    def test_tournament_mode_fallback_no_system(self, tmp_path: Path) -> None:
        """Tournament mode without tournament system falls back to single delegation."""
        orch = self._make_orchestrator(tmp_path)
        task = Task(description="Test", requirements=[], acceptance_criteria=[])
        agents = [FakeAgent(), FakeAgent()]

        result = orch.delegate_task(task, agent_pool=agents, use_tournament=True)

        # Falls back to single agent since no tournament system configured
        assert result.success is True


class TestApprovalQueue:
    """Tests for approval queue: submit, approve, reject."""

    def _make_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(
            repo_path=str(tmp_path),
            git_wrapper=FakeGitWrapper(),
        )

    def test_submit_for_approval(self, tmp_path: Path) -> None:
        """Submitting a solution should return an approval ID and add to queue."""
        orch = self._make_orchestrator(tmp_path)
        solution = _make_solution()

        approval_id = orch.submit_for_approval(solution)

        assert isinstance(approval_id, str)
        assert len(approval_id) > 0
        pending = orch.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0].solution is solution

    def test_approve_solution(self, tmp_path: Path) -> None:
        """Approving a solution should remove it from the queue and record it."""
        orch = self._make_orchestrator(tmp_path)
        solution = _make_solution()
        approval_id = orch.submit_for_approval(solution)

        result = orch.approve_solution(approval_id, reviewer="alice", comments="Looks good")

        assert isinstance(result, ApprovalResult)
        assert result.success is True
        assert result.solution is solution
        assert len(orch.get_pending_approvals()) == 0
        stats = orch.get_statistics()
        assert stats["merges_approved"] == 1
        assert stats["approved_solutions"] == 1

    def test_reject_solution(self, tmp_path: Path) -> None:
        """Rejecting a solution should remove from queue and track it."""
        orch = self._make_orchestrator(tmp_path)
        solution = _make_solution()
        approval_id = orch.submit_for_approval(solution)

        result = orch.reject_solution(approval_id, reviewer="bob", comments="Needs work")

        assert isinstance(result, ApprovalResult)
        assert result.success is True
        assert result.solution is solution
        assert result.merged_commit is None
        assert len(orch.get_pending_approvals()) == 0
        stats = orch.get_statistics()
        assert stats["merges_rejected"] == 1
        assert stats["rejected_solutions"] == 1

    def test_approve_nonexistent_id(self, tmp_path: Path) -> None:
        """Approving a non-existent approval ID should return ApprovalResult with success=False."""
        orch = self._make_orchestrator(tmp_path)

        result = orch.approve_solution("nonexistent-id", reviewer="alice")

        assert isinstance(result, ApprovalResult)
        assert result.success is False
        assert result.solution is None

    def test_reject_nonexistent_id(self, tmp_path: Path) -> None:
        """Rejecting a non-existent approval ID should return ApprovalResult with success=False."""
        orch = self._make_orchestrator(tmp_path)

        result = orch.reject_solution("nonexistent-id", reviewer="bob", comments="N/A")

        assert isinstance(result, ApprovalResult)
        assert result.success is False
        assert result.solution is None


class TestBranchProtection:
    """Tests for branch protection: set, get, lock, unlock."""

    def _make_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(
            repo_path=str(tmp_path),
            git_wrapper=FakeGitWrapper(),
        )

    def test_default_main_branch_protection(self, tmp_path: Path) -> None:
        """Main branch should default to HUMAN_APPROVAL protection."""
        orch = self._make_orchestrator(tmp_path)

        level = orch.get_branch_protection("main")

        assert level == BranchProtectionLevel.HUMAN_APPROVAL

    def test_unprotected_branch_returns_none(self, tmp_path: Path) -> None:
        """An unprotected branch should return NONE protection level."""
        orch = self._make_orchestrator(tmp_path)

        level = orch.get_branch_protection("feature/xyz")

        assert level == BranchProtectionLevel.NONE

    def test_set_branch_protection(self, tmp_path: Path) -> None:
        """Setting branch protection should be retrievable."""
        orch = self._make_orchestrator(tmp_path)

        orch.set_branch_protection("develop", BranchProtectionLevel.REVIEW_REQUIRED)

        assert orch.get_branch_protection("develop") == BranchProtectionLevel.REVIEW_REQUIRED

    def test_lock_branch(self, tmp_path: Path) -> None:
        """Locking a branch should set LOCKED protection level."""
        orch = self._make_orchestrator(tmp_path)

        orch.lock_branch("release")

        assert orch.get_branch_protection("release") == BranchProtectionLevel.LOCKED

    def test_unlock_branch(self, tmp_path: Path) -> None:
        """Unlocking a branch should set NONE protection level."""
        orch = self._make_orchestrator(tmp_path)
        orch.lock_branch("staging")

        orch.unlock_branch("staging")

        assert orch.get_branch_protection("staging") == BranchProtectionLevel.NONE


class TestStateExportAndStatistics:
    """Tests for export_state and get_statistics."""

    def _make_orchestrator(self, tmp_path: Path) -> Orchestrator:
        return Orchestrator(
            repo_path=str(tmp_path),
            git_wrapper=FakeGitWrapper(),
        )

    def test_initial_statistics(self, tmp_path: Path) -> None:
        """Fresh orchestrator should have zeroed statistics."""
        orch = self._make_orchestrator(tmp_path)

        stats = orch.get_statistics()

        assert stats["tasks_created"] == 0
        assert stats["tasks_completed"] == 0
        assert stats["tasks_failed"] == 0
        assert stats["merges_approved"] == 0
        assert stats["merges_rejected"] == 0
        assert stats["total_tasks"] == 0

    def test_statistics_after_prompt(self, tmp_path: Path) -> None:
        """Processing a prompt should increment tasks_created."""
        orch = self._make_orchestrator(tmp_path)
        orch.process_human_prompt("Fix the login bug")

        stats = orch.get_statistics()

        assert stats["tasks_created"] >= 1
        assert stats["total_tasks"] >= 1

    def test_export_state_creates_file(self, tmp_path: Path) -> None:
        """export_state should create a JSON file with orchestrator state."""
        orch = self._make_orchestrator(tmp_path)
        orch.process_human_prompt("Add new feature")

        output_file = tmp_path / "state.json"
        result_path = orch.export_state(str(output_file))

        assert Path(result_path).exists()
        with open(result_path) as f:
            state = json.load(f)
        assert "repo_path" in state
        assert "statistics" in state
        assert "protected_branches" in state
        assert "tasks" in state
        assert state["statistics"]["tasks_created"] >= 1

    def test_export_state_default_path(self, tmp_path: Path) -> None:
        """export_state without output_path should use default .ansibel directory."""
        orch = self._make_orchestrator(tmp_path)

        result_path = orch.export_state()

        assert Path(result_path).exists()
        assert ".ansibel" in result_path
        assert "orchestrator_state.json" in result_path


class TestTaskDataStructures:
    """Tests for Task and TaskBreakdown data model methods."""

    def test_task_mark_completed(self) -> None:
        """mark_completed should set status and completed_at."""
        task = Task(description="Test", requirements=[], acceptance_criteria=[])

        task.mark_completed()

        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_task_mark_failed(self) -> None:
        """mark_failed should set status and store failure reason in metadata."""
        task = Task(description="Test", requirements=[], acceptance_criteria=[])

        task.mark_failed("Out of memory")

        assert task.status == TaskStatus.FAILED
        assert task.metadata["failure_reason"] == "Out of memory"

    def test_task_can_execute_no_deps(self) -> None:
        """Task with no dependencies should always be executable."""
        task = Task(description="Test", requirements=[], acceptance_criteria=[])

        assert task.can_execute(completed_tasks=set()) is True

    def test_task_can_execute_with_deps(self) -> None:
        """Task should only execute when all dependency IDs are in the completed set."""
        dep_id = TaskId()
        task = Task(
            description="Test",
            requirements=[],
            acceptance_criteria=[],
            dependencies=[dep_id],
        )

        assert task.can_execute(completed_tasks=set()) is False
        assert task.can_execute(completed_tasks={dep_id}) is True

    def test_task_to_dict(self) -> None:
        """to_dict should include all relevant fields."""
        task = Task(
            description="Sample task",
            requirements=["req1"],
            acceptance_criteria=["ac1"],
            priority=TaskPriority.HIGH,
        )

        d = task.to_dict()

        assert d["description"] == "Sample task"
        assert d["requirements"] == ["req1"]
        assert d["acceptance_criteria"] == ["ac1"]
        assert d["priority"] == "HIGH"
        assert d["status"] == "PENDING"

    def test_task_breakdown_get_critical_tasks(self) -> None:
        """get_critical_tasks should return only CRITICAL priority tasks."""
        critical_task = Task(
            description="Critical",
            requirements=[],
            acceptance_criteria=[],
            priority=TaskPriority.CRITICAL,
        )
        medium_task = Task(
            description="Medium",
            requirements=[],
            acceptance_criteria=[],
            priority=TaskPriority.MEDIUM,
        )
        breakdown = TaskBreakdown(
            original_prompt="test",
            tasks=[critical_task, medium_task],
        )

        critical_tasks = breakdown.get_critical_tasks()

        assert len(critical_tasks) == 1
        assert critical_tasks[0].description == "Critical"

    def test_task_breakdown_to_dict(self) -> None:
        """TaskBreakdown.to_dict should serialize the full breakdown."""
        task = Task(description="T", requirements=[], acceptance_criteria=[])
        breakdown = TaskBreakdown(
            original_prompt="Build a feature",
            tasks=[task],
            execution_strategy="parallel",
            context={"key": "value"},
        )

        d = breakdown.to_dict()

        assert d["original_prompt"] == "Build a feature"
        assert d["execution_strategy"] == "parallel"
        assert d["context"] == {"key": "value"}
        assert len(d["tasks"]) == 1


class TestExecutionStrategy:
    """Tests for _determine_execution_strategy via process_human_prompt."""

    def test_single_task_sequential(self, tmp_path: Path) -> None:
        """A single task should produce sequential strategy."""
        orch = Orchestrator(repo_path=str(tmp_path), git_wrapper=FakeGitWrapper())
        breakdown = orch.process_human_prompt("Fix the database connection bug")

        assert breakdown.execution_strategy == "sequential"

    def test_authentication_prompt_mixed_strategy(self, tmp_path: Path) -> None:
        """Authentication prompt with dependencies should produce mixed strategy."""
        orch = Orchestrator(repo_path=str(tmp_path), git_wrapper=FakeGitWrapper())
        breakdown = orch.process_human_prompt("Add user authentication")

        # Multiple tasks with dependencies -> mixed
        assert breakdown.execution_strategy == "mixed"
