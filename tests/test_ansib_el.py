"""
Tests for ansibel.ansib_el -- AnsibElSystem integration class.

Tests cover initialisation, status retrieval, agent info, prompt
processing (mocked), and review/approval workflows.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from ansibel.ansib_el import AnsibElSystem, SystemStatus
from ansibel.orchestrator import (
    AgentId,
    CodeChange,
    Solution,
    Task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "test@ansibel.dev",
    "GIT_COMMITTER_NAME": "Test Author",
    "GIT_COMMITTER_EMAIL": "test@ansibel.dev",
}


def _init_git_repo(path: Path) -> None:
    """Create a bare git repository with an initial commit."""
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True, env=env)
    subprocess.run(
        ["git", "config", "user.email", "test@ansibel.dev"],
        cwd=str(path),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Author"],
        cwd=str(path),
        capture_output=True,
        check=True,
        env=env,
    )
    readme = path / "README.md"
    readme.write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(path),
        capture_output=True,
        check=True,
        env=env,
    )


def _make_solution(task: Task) -> Solution:
    """Create a minimal Solution for testing approval workflows."""
    return Solution(
        task_id=task.id,
        agent_id=AgentId(),
        changes=[
            CodeChange(
                file_path="test.py",
                diff_content="+# added",
                description="Test change",
            )
        ],
        explanation="Test solution explanation",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_path(tmp_path: Path) -> Path:
    """Return a temp directory with an initialised git repository."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    _init_git_repo(repo)
    return repo


@pytest.fixture()
def system(repo_path: Path) -> AnsibElSystem:
    """Return an AnsibElSystem instance (not yet initialised)."""
    return AnsibElSystem(str(repo_path))


@pytest.fixture()
def initialized_system(repo_path: Path) -> AnsibElSystem:
    """Return an AnsibElSystem that has been initialised."""
    sys = AnsibElSystem(str(repo_path))
    result = sys.initialize()
    assert result is True
    return sys


# ---------------------------------------------------------------------------
# 1. System initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_initialize_returns_true(self, system: AnsibElSystem) -> None:
        assert system.initialize() is True

    def test_initialized_flag_set(self, system: AnsibElSystem) -> None:
        system.initialize()
        assert system._initialized is True

    def test_orchestrator_created_after_init(self, system: AnsibElSystem) -> None:
        system.initialize()
        assert system.orchestrator is not None

    def test_tournament_created_after_init(self, system: AnsibElSystem) -> None:
        system.initialize()
        assert system.tournament is not None

    def test_ai_git_dir_created(self, system: AnsibElSystem) -> None:
        system.initialize()
        assert (system.repo_path / ".ai-git").is_dir()


# ---------------------------------------------------------------------------
# 2. Not-initialised guard
# ---------------------------------------------------------------------------


class TestNotInitialisedGuard:
    def test_process_prompt_raises_before_init(self, system: AnsibElSystem) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            system.process_prompt("Add login page")

    def test_review_and_approve_raises_before_init(self, system: AnsibElSystem) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            system.review_and_approve("fake-id", approve=True)


# ---------------------------------------------------------------------------
# 3. get_status() returns SystemStatus
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_before_init(self, system: AnsibElSystem) -> None:
        status = system.get_status()
        assert isinstance(status, SystemStatus)
        assert status.repo_initialized is False
        assert status.active_agents == 0
        assert status.pending_approvals == 0

    def test_status_after_init(self, initialized_system: AnsibElSystem) -> None:
        status = initialized_system.get_status()
        assert isinstance(status, SystemStatus)
        assert status.repo_initialized is True


# ---------------------------------------------------------------------------
# 4. list_pending_approvals()
# ---------------------------------------------------------------------------


class TestPendingApprovals:
    def test_empty_before_init(self, system: AnsibElSystem) -> None:
        approvals = system.list_pending_approvals()
        assert approvals == []

    def test_empty_after_init(self, initialized_system: AnsibElSystem) -> None:
        approvals = initialized_system.list_pending_approvals()
        assert approvals == []


# ---------------------------------------------------------------------------
# 5. get_agent_info() with invalid agent ID
# ---------------------------------------------------------------------------


class TestGetAgentInfo:
    def test_invalid_agent_returns_error(self, initialized_system: AnsibElSystem) -> None:
        fake_id = str(uuid4())
        info = initialized_system.get_agent_info(fake_id)
        assert isinstance(info, dict)
        assert "error" in info
        assert info["error"] == "Agent not found"

    def test_invalid_uuid_raises(self, initialized_system: AnsibElSystem) -> None:
        with pytest.raises((ValueError, Exception)):
            initialized_system.get_agent_info("not-a-valid-uuid")


# ---------------------------------------------------------------------------
# 6. process_prompt -- single agent mode (mocked)
# ---------------------------------------------------------------------------


class TestProcessPromptSingle:
    def test_single_agent_mode(self, initialized_system: AnsibElSystem) -> None:
        """process_prompt with use_tournament=False should delegate to _run_single_agent_task."""
        mock_result = {
            "mode": "single",
            "task_id": "mock-task",
            "status": "delegated",
            "message": "ok",
            "pending_approval": True,
        }

        with patch.object(
            initialized_system, "_run_single_agent_task", return_value=mock_result
        ) as mocked:
            result = initialized_system.process_prompt(
                "Add a search bar", use_tournament=False, num_agents=1
            )

        assert result["status"] == "success"
        assert result["tasks_processed"] >= 1
        assert mocked.called


# ---------------------------------------------------------------------------
# 7. process_prompt -- tournament mode (mocked)
# ---------------------------------------------------------------------------


class TestProcessPromptTournament:
    def test_tournament_mode(self, initialized_system: AnsibElSystem) -> None:
        """process_prompt with use_tournament=True should delegate to _run_tournament_task."""
        mock_result = {
            "mode": "tournament",
            "tournament_id": "t-001",
            "solutions_generated": 3,
            "status": "completed",
            "review_presentation": "mock markdown",
            "pending_approval": True,
        }

        with patch.object(
            initialized_system, "_run_tournament_task", return_value=mock_result
        ) as mocked:
            result = initialized_system.process_prompt(
                "Refactor user module", use_tournament=True, num_agents=3
            )

        assert result["status"] == "success"
        assert result["tasks_processed"] >= 1
        assert mocked.called


# ---------------------------------------------------------------------------
# 8. Review and approve workflow
# ---------------------------------------------------------------------------


class TestReviewApprove:
    def test_approve_workflow(self, initialized_system: AnsibElSystem) -> None:
        """Submit a solution for approval, then approve it."""
        # Create a task and solution to submit
        task = Task(
            description="Test task for approval",
            requirements=["req1"],
            acceptance_criteria=["works"],
        )
        solution = _make_solution(task)

        # Submit for approval
        approval_id = initialized_system.orchestrator.submit_for_approval(solution)

        # orchestrator.approve_solution now returns ApprovalResult natively.
        # We still mock record_decision since it hits the DB.
        with patch.object(initialized_system.trust_lineage, "record_decision"):
            result = initialized_system.review_and_approve(
                approval_id, approve=True, comments="Looks good"
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# 9. Review and reject workflow
# ---------------------------------------------------------------------------


class TestReviewReject:
    def test_reject_workflow(self, initialized_system: AnsibElSystem) -> None:
        """Submit a solution for approval, then reject it."""
        task = Task(
            description="Test task for rejection",
            requirements=["req1"],
            acceptance_criteria=["works"],
        )
        solution = _make_solution(task)

        approval_id = initialized_system.orchestrator.submit_for_approval(solution)

        # orchestrator.reject_solution now returns ApprovalResult natively.
        with patch.object(initialized_system.trust_lineage, "record_decision"):
            result = initialized_system.review_and_approve(
                approval_id, approve=False, comments="Needs rework"
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# 10. SystemStatus dataclass fields
# ---------------------------------------------------------------------------


class TestSystemStatusDataclass:
    def test_all_fields_present(self) -> None:
        status = SystemStatus(
            repo_initialized=True,
            active_agents=5,
            pending_approvals=2,
            total_commits=100,
            trust_scores={"agent-1": 0.85},
            recent_tournaments=["t-001"],
        )
        assert status.repo_initialized is True
        assert status.active_agents == 5
        assert status.pending_approvals == 2
        assert status.total_commits == 100
        assert "agent-1" in status.trust_scores
        assert status.recent_tournaments == ["t-001"]

    def test_default_uninitialised(self) -> None:
        status = SystemStatus(
            repo_initialized=False,
            active_agents=0,
            pending_approvals=0,
            total_commits=0,
            trust_scores={},
            recent_tournaments=[],
        )
        assert status.repo_initialized is False
        assert status.total_commits == 0


# ---------------------------------------------------------------------------
# 11. Multiple initialisations are safe
# ---------------------------------------------------------------------------


class TestMultipleInit:
    def test_double_init_succeeds(self, repo_path: Path) -> None:
        sys = AnsibElSystem(str(repo_path))
        assert sys.initialize() is True
        assert sys.initialize() is True
        assert sys._initialized is True

    def test_double_init_preserves_components(self, repo_path: Path) -> None:
        sys = AnsibElSystem(str(repo_path))
        sys.initialize()
        sys.initialize()
        # After second init, orchestrator is reassigned but system still works
        assert sys.orchestrator is not None


# ---------------------------------------------------------------------------
# 12. Constructor sets repo_path correctly
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_repo_path_resolved(self, repo_path: Path) -> None:
        sys = AnsibElSystem(str(repo_path))
        assert sys.repo_path == repo_path.resolve()

    def test_not_initialised_by_default(self, repo_path: Path) -> None:
        sys = AnsibElSystem(str(repo_path))
        assert sys._initialized is False
        assert sys.orchestrator is None
        assert sys.tournament is None


# ---------------------------------------------------------------------------
# 13. process_prompt returns correct structure
# ---------------------------------------------------------------------------


class TestProcessPromptStructure:
    def test_result_keys(self, initialized_system: AnsibElSystem) -> None:
        """Verify the result dict has expected keys."""
        with patch.object(
            initialized_system,
            "_run_single_agent_task",
            return_value={"mode": "single", "status": "delegated"},
        ):
            result = initialized_system.process_prompt(
                "Add logging", use_tournament=False, num_agents=1
            )

        assert "status" in result
        assert "tasks_processed" in result
        assert "results" in result
        assert "pending_approvals" in result
        assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# 14. Approval for non-existent ID
# ---------------------------------------------------------------------------


class TestApprovalNotFound:
    def test_approve_nonexistent_id(self, initialized_system: AnsibElSystem) -> None:
        """Approving a non-existent approval_id should return success=False."""
        # orchestrator.approve_solution now returns ApprovalResult natively
        result = initialized_system.review_and_approve(
            "nonexistent-approval-id", approve=True, comments="test"
        )

        assert result["success"] is False


# ---------------------------------------------------------------------------
# 15. get_status after process_prompt reflects changes
# ---------------------------------------------------------------------------


class TestStatusAfterProcessing:
    def test_status_reflects_pending_approvals(self, initialized_system: AnsibElSystem) -> None:
        """After submitting a solution, pending_approvals should increase."""
        task = Task(
            description="Test pending count",
            requirements=["req"],
            acceptance_criteria=["works"],
        )
        solution = _make_solution(task)

        initialized_system.orchestrator.submit_for_approval(solution)

        status = initialized_system.get_status()
        assert status.pending_approvals >= 1
