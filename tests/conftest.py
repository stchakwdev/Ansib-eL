"""
Shared pytest fixtures and relocated mock classes for Ansib-eL test suite.

This module provides:
- MockGitWrapper: In-memory mock mimicking the GitWrapper API
- MockAgent: Lightweight mock agent with UUID identity
- MockAgentManager: Async mock implementing tournament.py AgentManager Protocol
- Fixtures for temporary repos, real manager instances, and test data factories
"""

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from ansibel.agent_system import AgentManager, AgentStatus
from ansibel.git_wrapper import AgentMetadata as GitAgentMetadata
from ansibel.git_wrapper import GitWrapper
from ansibel.git_wrapper import MergeResult as GitMergeResult
from ansibel.orchestrator import Orchestrator, Task, TaskPriority
from ansibel.tournament import (
    AgentConfig,
    Solution,
    SolutionStatus,
)
from ansibel.tournament import (
    Task as TournamentTask,
)
from ansibel.trust_lineage import TrustLineageManager

# ============================================================================
# Relocated Mock Classes
# ============================================================================


class MockGitWrapper:
    """In-memory mock that mimics the GitWrapper API for testing.

    Tracks branches, commits, files, and metadata without touching the
    filesystem or requiring a real git repository.
    """

    def __init__(self, repo_path: str = "/tmp/mock-repo") -> None:
        self.repo_path = Path(repo_path)
        self._initialized = False
        self._current_branch = "main"
        self._branches: dict[str, list[str]] = {"main": []}
        self._commits: dict[str, dict[str, Any]] = {}
        self._metadata_store: dict[str, GitAgentMetadata] = {}
        self._files: dict[str, str] = {}
        self._dirty = False
        self._branch_protection: dict[str, Any] = {}
        self._commit_counter = 0

    # -- Initialisation ------------------------------------------------------

    def init_repo(self) -> bool:
        self._initialized = True
        return True

    def is_initialized(self) -> bool:
        return self._initialized

    # -- Branch operations ---------------------------------------------------

    def create_branch(self, branch_name: str, from_branch: str | None = None) -> bool:
        base = from_branch or self._current_branch
        self._branches[branch_name] = list(self._branches.get(base, []))
        return True

    def create_agent_branch(self, agent_id: str, purpose: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"agent/{agent_id}/{timestamp}"
        self.create_branch(branch_name)
        self.checkout_branch(branch_name)
        return branch_name

    def checkout_branch(self, branch_name: str) -> bool:
        if branch_name not in self._branches:
            return False
        self._current_branch = branch_name
        return True

    def get_current_branch(self) -> str:
        return self._current_branch

    # -- Commit operations ---------------------------------------------------

    def commit_changes(self, message: str, files: list[str] | None = None) -> str:
        self._commit_counter += 1
        commit_hash = hashlib.sha1(
            f"{message}-{self._commit_counter}-{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()
        self._commits[commit_hash] = {
            "message": message,
            "branch": self._current_branch,
            "files": files or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._branches[self._current_branch].append(commit_hash)
        self._dirty = False
        return commit_hash

    def commit_with_metadata(
        self,
        message: str,
        metadata: GitAgentMetadata,
        files: list[str] | None = None,
    ) -> str:
        commit_hash = self.commit_changes(message, files)
        self._metadata_store[commit_hash] = metadata
        return commit_hash

    def get_commit_metadata(self, commit_hash: str) -> GitAgentMetadata | None:
        return self._metadata_store.get(commit_hash)

    # -- Merge operations ----------------------------------------------------

    def merge_branch(
        self, branch_name: str, strategy: str = "recursive"
    ) -> GitMergeResult:
        if branch_name not in self._branches:
            return GitMergeResult(
                success=False,
                message=f"Branch {branch_name} not found",
                conflicts=[],
            )
        # Simulate a clean merge
        self._branches[self._current_branch].extend(self._branches[branch_name])
        merged_hash = self.commit_changes(f"Merge branch '{branch_name}'")
        return GitMergeResult(
            success=True,
            message=f"Successfully merged {branch_name}",
            conflicts=[],
            merged_commit=merged_hash,
        )

    def merge_agent_branch(
        self,
        branch_name: str,
        strategy: str = "merge",
        target_branch: str | None = None,
    ) -> GitMergeResult:
        original = self._current_branch
        if target_branch:
            self.checkout_branch(target_branch)
        result = self.merge_branch(branch_name, strategy)
        if target_branch:
            self.checkout_branch(original)
        return result

    # -- Diff / status -------------------------------------------------------

    def get_diff(self, branch_a: str, branch_b: str) -> str:
        commits_a = set(self._branches.get(branch_a, []))
        commits_b = set(self._branches.get(branch_b, []))
        diff_commits = commits_b - commits_a
        if not diff_commits:
            return ""
        return f"diff --mock a/{branch_a} b/{branch_b}\n+{len(diff_commits)} commit(s) differ"

    def get_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "branch": self._current_branch,
            "is_dirty": self._dirty,
            "untracked_files": [],
            "modified_files": [],
            "staged_files": [],
            "active_agents": [],
            "pending_merges": 0,
        }

    def is_working_tree_clean(self) -> bool:
        return not self._dirty

    def get_last_commit(self) -> str | None:
        branch_commits = self._branches.get(self._current_branch, [])
        return branch_commits[-1] if branch_commits else None

    # -- Branch protection ---------------------------------------------------

    def get_branch_protection(self, branch_name: str) -> Any:
        from ansibel.orchestrator import BranchProtectionLevel

        return self._branch_protection.get(branch_name, BranchProtectionLevel.NONE)

    def set_branch_protection(self, branch_name: str, level: Any) -> bool:
        self._branch_protection[branch_name] = level
        return True

    # -- Agent branch listing ------------------------------------------------

    def list_agent_branches(self, status: str | None = None) -> list[dict[str, Any]]:
        return [
            {"name": name, "status": "active"}
            for name in self._branches
            if name.startswith("agent/")
        ]

    # -- Trust score helpers (subset of GitWrapper API) ----------------------

    def update_agent_trust_score(self, agent_id: str, score_delta: float) -> float:
        # Simplified in-memory implementation
        return 0.5 + score_delta

    def get_agent_trust_score(self, agent_id: str) -> float:
        return 0.5

    # -- History -------------------------------------------------------------

    def get_ai_enhanced_history(self, limit: int = 20) -> list[dict[str, Any]]:
        history = []
        for commit_hash, info in list(self._commits.items())[-limit:]:
            metadata = self._metadata_store.get(commit_hash)
            history.append(
                {
                    "hash": commit_hash[:8],
                    "full_hash": commit_hash,
                    "message": info["message"],
                    "author": "mock-author",
                    "date": info["timestamp"],
                    "metadata": metadata.to_dict() if metadata else None,
                }
            )
        return history


class MockAgent:
    """Lightweight mock agent with UUID identity for testing."""

    def __init__(
        self,
        purpose: str = "test-task",
        model_version: str = "gpt-5.2-test",
        status: AgentStatus = AgentStatus.IDLE,
        workspace_branch: str | None = None,
        agent_id: UUID | None = None,
    ) -> None:
        self.agent_id: UUID = agent_id or uuid4()
        self.purpose = purpose
        self.model_version = model_version
        self.prompt_hash = hashlib.sha256(purpose.encode()).hexdigest()
        self.created_at = datetime.now(timezone.utc)
        self.status = status
        self.workspace_branch = (
            workspace_branch or f"agent/{str(self.agent_id)[:8]}/mock"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "purpose": self.purpose,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "workspace_branch": self.workspace_branch,
        }


class MockAgentManager:
    """Async mock implementing the tournament.py ``AgentManager`` Protocol.

    Provides ``spawn_agent``, ``execute_task``, and ``terminate_agent``
    as async methods suitable for use with ``TournamentOrchestrator``.
    """

    def __init__(self) -> None:
        self.spawned_agents: dict[str, AgentConfig] = {}
        self.executed_tasks: list[dict[str, Any]] = []
        self.terminated_agents: list[str] = []
        # Allow tests to inject custom solution generators
        self._solution_factory: Any | None = None

    def set_solution_factory(self, factory: Any) -> None:
        """Set a callable ``(agent_id, task) -> Solution`` for custom results."""
        self._solution_factory = factory

    async def spawn_agent(self, config: AgentConfig) -> str:
        agent_id = config.agent_id or str(uuid4())
        self.spawned_agents[agent_id] = config
        return agent_id

    async def execute_task(self, agent_id: str, task: TournamentTask) -> Solution:
        self.executed_tasks.append({"agent_id": agent_id, "task_id": task.task_id})
        if self._solution_factory is not None:
            return self._solution_factory(agent_id, task)
        return Solution(
            solution_id=str(uuid4()),
            agent_id=agent_id,
            task_id=task.task_id,
            files_changed={"test_file.py": "# mock solution\npass\n"},
            diff="+ # mock solution\n+ pass",
            explanation=f"Mock solution by agent {agent_id}",
            status=SolutionStatus.COMPLETED,
        )

    async def terminate_agent(self, agent_id: str) -> None:
        self.terminated_agents.append(agent_id)


# ============================================================================
# Fixtures: Temporary Git Repository
# ============================================================================


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary directory with an initialised git repo and an initial commit.

    Returns the ``Path`` to the repository root.
    """
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@ansibel.dev",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@ansibel.dev",
    }

    subprocess.run(
        ["git", "init"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
        env=env,
    )

    # Set local config so commits work regardless of global git config
    subprocess.run(
        ["git", "config", "user.email", "test@ansibel.dev"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Author"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
        env=env,
    )

    # Create an initial commit so HEAD exists
    readme = repo_dir / "README.md"
    readme.write_text("# Test Repository\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
        env=env,
    )

    return repo_dir


# ============================================================================
# Fixtures: Real Component Instances
# ============================================================================


@pytest.fixture
def git_wrapper(tmp_repo: Path) -> GitWrapper:
    """Return a ``GitWrapper`` instance bound to the temporary git repo."""
    wrapper = GitWrapper(str(tmp_repo))
    wrapper.init_repo()
    return wrapper


@pytest.fixture
def agent_manager(tmp_path: Path) -> AgentManager:
    """Return an ``AgentManager`` with a temporary JSON storage path."""
    storage_file = tmp_path / "agents" / "agents.json"
    return AgentManager(str(storage_file))


@pytest.fixture
def trust_manager(tmp_path: Path) -> TrustLineageManager:
    """Return a ``TrustLineageManager`` backed by a temporary SQLite database."""
    db_path = tmp_path / "trust" / "trust_lineage.db"
    return TrustLineageManager(str(db_path))


@pytest.fixture
def orchestrator(tmp_repo: Path, git_wrapper: GitWrapper) -> Orchestrator:
    """Return an ``Orchestrator`` wired to a real ``GitWrapper`` on a temp repo."""
    return Orchestrator(repo_path=str(tmp_repo), git_wrapper=git_wrapper)


# ============================================================================
# Fixtures: Mock Instances
# ============================================================================


@pytest.fixture
def mock_git_wrapper() -> MockGitWrapper:
    """Return a fresh ``MockGitWrapper`` instance (already initialised)."""
    mgw = MockGitWrapper()
    mgw.init_repo()
    return mgw


@pytest.fixture
def mock_agent_manager() -> MockAgentManager:
    """Return a fresh ``MockAgentManager`` instance."""
    return MockAgentManager()


# ============================================================================
# Test Data Factories
# ============================================================================


@pytest.fixture
def sample_task() -> Task:
    """Return a minimal ``orchestrator.Task`` for testing."""
    return Task(
        description="Implement user login endpoint",
        requirements=["Create login API", "Validate credentials", "Return JWT token"],
        acceptance_criteria=[
            "Login works with valid credentials",
            "Invalid credentials are rejected",
        ],
        priority=TaskPriority.HIGH,
    )


@pytest.fixture
def sample_agent_metadata() -> GitAgentMetadata:
    """Return a sample ``git_wrapper.AgentMetadata`` instance."""
    return GitAgentMetadata(
        agent_id=str(uuid4()),
        model_version="gpt-5.2",
        prompt_hash=hashlib.sha256(b"test prompt for fixture").hexdigest(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        parent_task=None,
        confidence_score=0.92,
        reasoning="Fixture-generated agent metadata for testing.",
        tool_calls=["read_file", "write_file"],
    )


@pytest.fixture
def sample_agent_id() -> UUID:
    """Return a deterministic UUID suitable for repeatable tests."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_tournament_task() -> TournamentTask:
    """Return a sample ``tournament.Task`` for tournament tests."""
    return TournamentTask(
        task_id=str(uuid4()),
        description="Refactor authentication module",
        context_files=["src/auth.py", "src/models.py"],
        requirements=["Maintain backward compatibility", "Add type hints"],
        test_commands=["pytest tests/test_auth.py -v"],
    )


@pytest.fixture
def sample_agent_configs() -> list[AgentConfig]:
    """Return a list of three ``AgentConfig`` objects for tournament tests."""
    return [
        AgentConfig(
            agent_id=str(uuid4()),
            agent_type="gpt-5.2",
            model_config={"temperature": 0.7},
            timeout_seconds=60.0,
            priority=2,
        ),
        AgentConfig(
            agent_id=str(uuid4()),
            agent_type="claude-opus-4.5",
            model_config={"temperature": 0.5},
            timeout_seconds=60.0,
            priority=1,
        ),
        AgentConfig(
            agent_id=str(uuid4()),
            agent_type="gpt-5.2",
            model_config={"temperature": 0.3},
            timeout_seconds=60.0,
            priority=0,
        ),
    ]
