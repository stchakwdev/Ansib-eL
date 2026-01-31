"""
Git-specific integration tests for the Ansib-eL system.

Tests cover:
1. Init -> create branch -> commit with metadata -> merge -> verify history
2. Conflict creation and resolution
3. Multiple agent branches simultaneously
4. Branch listing and cleanup
5. Metadata persistence (commit with metadata, read it back, verify)

All tests use real git operations via GitWrapper and temporary repositories.
"""

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ansibel.git_wrapper import AgentMetadata, GitWrapper, MergeResult

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(repo_path: Path) -> None:
    """Initialise a git repo with an initial commit at *repo_path*."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Integration Test",
        "GIT_AUTHOR_EMAIL": "test@ansibel.dev",
        "GIT_COMMITTER_NAME": "Integration Test",
        "GIT_COMMITTER_EMAIL": "test@ansibel.dev",
    }
    subprocess.run(
        ["git", "init"],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@ansibel.dev"],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.name", "Integration Test"],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
        env=env,
    )
    readme = repo_path / "README.md"
    readme.write_text("# Integration Test Repo\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
        env=env,
    )


def _make_metadata(agent_id: str = "agent-001") -> AgentMetadata:
    """Create a sample AgentMetadata instance."""
    return AgentMetadata(
        agent_id=agent_id,
        model_version="gpt-4-turbo",
        prompt_hash=hashlib.sha256(b"test prompt").hexdigest(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        parent_task="task-001",
        confidence_score=0.95,
        reasoning="Integration test commit reasoning.",
        tool_calls=["read_file", "write_file"],
    )


def _write_and_stage(repo_path: Path, filename: str, content: str) -> None:
    """Write a file and stage it in git."""
    filepath = repo_path / filename
    filepath.write_text(content)
    subprocess.run(
        ["git", "add", filename],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# 1. Full git workflow: init -> branch -> commit with metadata -> merge -> history
# ---------------------------------------------------------------------------


class TestFullGitWorkflow:
    """End-to-end git operations through GitWrapper."""

    def test_init_branch_commit_merge_history(self, tmp_path: Path) -> None:
        """Full lifecycle: init, create agent branch, commit with metadata,
        merge back, and verify the commit appears in history with metadata."""
        repo_dir = tmp_path / "workflow-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        assert gw.init_repo() is True
        assert gw.is_initialized() is True

        # Capture the main branch name before creating agent branch
        status_before = gw.get_status()
        main_branch = status_before["branch"]

        # Create an agent branch
        agent_id = "agent-workflow-001"
        branch_name = gw.create_agent_branch(agent_id, "implement feature X")
        assert branch_name.startswith("agent/")
        assert agent_id in branch_name

        # Write a file and commit with metadata on the agent branch
        _write_and_stage(repo_dir, "feature_x.py", "# Feature X\nprint('hello')\n")
        metadata = _make_metadata(agent_id)
        commit_hash = gw.commit_with_metadata(
            message="Add feature X implementation",
            metadata=metadata,
            files=["feature_x.py"],
        )
        assert commit_hash is not None
        assert len(commit_hash) >= 7

        # Switch back to main and merge
        subprocess.run(
            ["git", "checkout", main_branch],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        merge_result = gw.merge_agent_branch(branch_name, target_branch=main_branch)
        assert isinstance(merge_result, MergeResult)
        assert merge_result.success is True
        assert len(merge_result.conflicts) == 0

        # Verify the commit appears in history with metadata
        history = gw.get_ai_enhanced_history(limit=10)
        assert len(history) >= 1

        # Find our commit in the history
        our_commits = [
            h
            for h in history
            if h.get("metadata") is not None and h["metadata"].get("agent_id") == agent_id
        ]
        assert len(our_commits) >= 1, "The commit with agent metadata should appear in history"


# ---------------------------------------------------------------------------
# 2. Conflict creation and resolution
# ---------------------------------------------------------------------------


class TestConflictDetection:
    """Test conflict detection when merging branches with conflicting changes."""

    def test_conflicting_merge_detected(self, tmp_path: Path) -> None:
        """Conflicting changes on two branches should be detected on merge."""
        repo_dir = tmp_path / "conflict-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        status = gw.get_status()
        main_branch = status["branch"]

        # Create a shared file on main
        _write_and_stage(repo_dir, "shared.txt", "original content\n")
        subprocess.run(
            ["git", "commit", "-m", "Add shared file"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        # Create first agent branch and modify shared.txt
        branch_a = gw.create_agent_branch("agent-a", "modify shared file version A")
        (repo_dir / "shared.txt").write_text("version A content\n")
        subprocess.run(
            ["git", "add", "shared.txt"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent A modifies shared.txt"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        # Go back to main
        subprocess.run(
            ["git", "checkout", main_branch],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        # Create second agent branch and modify shared.txt differently
        branch_b = gw.create_agent_branch("agent-b", "modify shared file version B")
        (repo_dir / "shared.txt").write_text("version B content\n")
        subprocess.run(
            ["git", "add", "shared.txt"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent B modifies shared.txt"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )

        # Go back to main and merge branch A first (should succeed)
        subprocess.run(
            ["git", "checkout", main_branch],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        merge_a = gw.merge_agent_branch(branch_a, target_branch=main_branch)
        assert merge_a.success is True

        # Now try merging branch B -- should detect conflict
        merge_b = gw.merge_agent_branch(branch_b, target_branch=main_branch)
        # The merge should either fail or report conflicts
        assert merge_b.success is False or len(merge_b.conflicts) > 0, (
            "Merging conflicting branches should detect conflicts"
        )


# ---------------------------------------------------------------------------
# 3. Multiple agent branches simultaneously
# ---------------------------------------------------------------------------


class TestMultipleAgentBranches:
    """Test creating and committing on several agent branches in isolation."""

    def test_multiple_branches_isolation(self, tmp_path: Path) -> None:
        """Commits on different agent branches should be isolated from each other."""
        repo_dir = tmp_path / "multi-branch-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        status = gw.get_status()
        main_branch = status["branch"]

        agent_branches = []
        agent_ids = ["agent-alpha", "agent-beta", "agent-gamma"]

        for agent_id in agent_ids:
            # Return to main before creating each branch
            subprocess.run(
                ["git", "checkout", main_branch],
                cwd=str(repo_dir),
                capture_output=True,
                check=True,
            )

            branch_name = gw.create_agent_branch(agent_id, f"task for {agent_id}")
            agent_branches.append(branch_name)

            # Write a unique file on this branch
            filename = f"{agent_id}_work.py"
            _write_and_stage(repo_dir, filename, f"# Work by {agent_id}\n")

            metadata = _make_metadata(agent_id)
            gw.commit_with_metadata(
                message=f"Work by {agent_id}",
                metadata=metadata,
                files=[filename],
            )

        # Verify each branch has only its own file
        for i, branch_name in enumerate(agent_branches):
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=str(repo_dir),
                capture_output=True,
                check=True,
            )

            expected_file = repo_dir / f"{agent_ids[i]}_work.py"
            assert expected_file.exists(), (
                f"{expected_file.name} should exist on branch {branch_name}"
            )

            # Other agents' files should NOT exist on this branch
            for j, other_agent_id in enumerate(agent_ids):
                if i != j:
                    other_file = repo_dir / f"{other_agent_id}_work.py"
                    assert not other_file.exists(), (
                        f"{other_file.name} should NOT exist on branch {branch_name}"
                    )

    def test_multiple_branches_all_merge_cleanly(self, tmp_path: Path) -> None:
        """Non-overlapping changes on multiple branches should merge cleanly."""
        repo_dir = tmp_path / "multi-merge-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        status = gw.get_status()
        main_branch = status["branch"]

        branches = []
        for idx in range(3):
            subprocess.run(
                ["git", "checkout", main_branch],
                cwd=str(repo_dir),
                capture_output=True,
                check=True,
            )
            agent_id = f"agent-merge-{idx}"
            branch_name = gw.create_agent_branch(agent_id, f"non-overlapping task {idx}")
            branches.append(branch_name)

            filename = f"module_{idx}.py"
            _write_and_stage(repo_dir, filename, f"# Module {idx}\n")
            subprocess.run(
                ["git", "commit", "-m", f"Add module_{idx}.py"],
                cwd=str(repo_dir),
                capture_output=True,
                check=True,
            )

        # Merge all branches back to main
        subprocess.run(
            ["git", "checkout", main_branch],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
        )
        for branch_name in branches:
            result = gw.merge_agent_branch(branch_name, target_branch=main_branch)
            assert result.success is True, (
                f"Merging {branch_name} should succeed (non-overlapping changes)"
            )
            assert len(result.conflicts) == 0

        # All files should exist on main after merges
        for idx in range(3):
            assert (repo_dir / f"module_{idx}.py").exists()


# ---------------------------------------------------------------------------
# 4. Branch listing and naming conventions
# ---------------------------------------------------------------------------


class TestBranchListingAndCleanup:
    """Test listing agent branches and verifying naming conventions."""

    def test_list_agent_branches(self, tmp_path: Path) -> None:
        """Created agent branches should appear in listing with correct metadata."""
        repo_dir = tmp_path / "list-branches-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        status = gw.get_status()
        main_branch = status["branch"]

        # Create several agent branches
        created_branches = []
        for i in range(3):
            subprocess.run(
                ["git", "checkout", main_branch],
                cwd=str(repo_dir),
                capture_output=True,
                check=True,
            )
            agent_id = f"agent-list-{i}"
            branch_name = gw.create_agent_branch(agent_id, f"listing test task {i}")
            created_branches.append(branch_name)

        # List all agent branches
        branches = gw.list_agent_branches()
        assert len(branches) >= 3

        # Each branch should have the expected metadata keys
        for branch_info in branches:
            assert "name" in branch_info
            assert "agent_id" in branch_info
            assert "purpose" in branch_info
            assert "created_at" in branch_info
            assert "status" in branch_info

    def test_branch_naming_convention(self, tmp_path: Path) -> None:
        """Agent branches should follow the agent/{agent_id}/{timestamp} pattern."""
        repo_dir = tmp_path / "naming-convention-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        agent_id = "agent-naming-test"
        branch_name = gw.create_agent_branch(agent_id, "naming convention test")

        # Branch should start with "agent/"
        assert branch_name.startswith("agent/"), (
            f"Branch name '{branch_name}' should start with 'agent/'"
        )
        # Branch should contain the agent_id
        assert agent_id in branch_name, (
            f"Branch name '{branch_name}' should contain agent_id '{agent_id}'"
        )
        # Branch name should have 3 segments separated by "/"
        parts = branch_name.split("/")
        assert len(parts) == 3, (
            f"Branch name '{branch_name}' should have format agent/{{id}}/{{timestamp}}"
        )
        assert parts[0] == "agent"

    def test_list_branches_filter_by_status(self, tmp_path: Path) -> None:
        """Listing with status filter should return only matching branches."""
        repo_dir = tmp_path / "filter-branches-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        # Create an agent branch (status defaults to 'active')
        gw.create_agent_branch("agent-filter", "filter test")

        active_branches = gw.list_agent_branches(status="active")
        assert len(active_branches) >= 1

        # Filter by a status that should not match
        merged_branches = gw.list_agent_branches(status="merged")
        assert len(merged_branches) == 0


# ---------------------------------------------------------------------------
# 5. Metadata persistence
# ---------------------------------------------------------------------------


class TestMetadataPersistence:
    """Test that commit metadata survives round-trip storage and retrieval."""

    def test_commit_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Metadata written via commit_with_metadata should be fully retrievable."""
        repo_dir = tmp_path / "metadata-roundtrip-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        agent_id = "agent-meta-rt"
        original_metadata = AgentMetadata(
            agent_id=agent_id,
            model_version="gpt-4-turbo",
            prompt_hash=hashlib.sha256(b"roundtrip test prompt").hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_task="task-roundtrip-001",
            confidence_score=0.88,
            reasoning="This is a detailed reasoning for the metadata roundtrip test.",
            tool_calls=["search_code", "write_file", "run_tests"],
        )

        # Write a file and commit with metadata
        _write_and_stage(repo_dir, "roundtrip.py", "# Roundtrip test\n")
        commit_hash = gw.commit_with_metadata(
            message="Metadata roundtrip test commit",
            metadata=original_metadata,
            files=["roundtrip.py"],
        )

        # Retrieve metadata
        retrieved = gw.get_commit_metadata(commit_hash)
        assert retrieved is not None

        # Verify all fields match
        assert retrieved.agent_id == original_metadata.agent_id
        assert retrieved.model_version == original_metadata.model_version
        assert retrieved.prompt_hash == original_metadata.prompt_hash
        assert retrieved.timestamp == original_metadata.timestamp
        assert retrieved.parent_task == original_metadata.parent_task
        assert retrieved.confidence_score == pytest.approx(original_metadata.confidence_score)
        assert retrieved.reasoning == original_metadata.reasoning
        assert retrieved.tool_calls == original_metadata.tool_calls

    def test_metadata_survives_new_wrapper_instance(self, tmp_path: Path) -> None:
        """Metadata should be readable from a new GitWrapper instance on the same repo."""
        repo_dir = tmp_path / "metadata-persist-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        # First wrapper instance: write metadata
        gw1 = GitWrapper(str(repo_dir))
        gw1.init_repo()

        metadata = _make_metadata("agent-persist")
        _write_and_stage(repo_dir, "persist.py", "# Persist test\n")
        commit_hash = gw1.commit_with_metadata(
            message="Persistence test commit",
            metadata=metadata,
            files=["persist.py"],
        )

        # Second wrapper instance: read metadata
        gw2 = GitWrapper(str(repo_dir))
        retrieved = gw2.get_commit_metadata(commit_hash)

        assert retrieved is not None
        assert retrieved.agent_id == metadata.agent_id
        assert retrieved.model_version == metadata.model_version
        assert retrieved.reasoning == metadata.reasoning

    def test_metadata_for_nonexistent_commit(self, tmp_path: Path) -> None:
        """Querying metadata for a non-existent commit should return None."""
        repo_dir = tmp_path / "metadata-none-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        result = gw.get_commit_metadata("0000000000000000000000000000000000000000")
        assert result is None

    def test_multiple_commits_each_have_own_metadata(self, tmp_path: Path) -> None:
        """Each commit should maintain its own independent metadata."""
        repo_dir = tmp_path / "metadata-multi-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        commits = []
        for i in range(3):
            agent_id = f"agent-multi-meta-{i}"
            metadata = AgentMetadata(
                agent_id=agent_id,
                model_version=f"model-v{i}",
                prompt_hash=hashlib.sha256(f"prompt-{i}".encode()).hexdigest(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence_score=0.7 + (i * 0.1),
                reasoning=f"Reasoning for commit {i}",
            )

            filename = f"file_{i}.py"
            _write_and_stage(repo_dir, filename, f"# File {i}\n")
            commit_hash = gw.commit_with_metadata(
                message=f"Commit {i} with metadata",
                metadata=metadata,
                files=[filename],
            )
            commits.append((commit_hash, agent_id, f"model-v{i}"))

        # Verify each commit has its own distinct metadata
        for commit_hash, expected_agent_id, expected_model in commits:
            retrieved = gw.get_commit_metadata(commit_hash)
            assert retrieved is not None, f"Metadata for commit {commit_hash[:8]} should exist"
            assert retrieved.agent_id == expected_agent_id
            assert retrieved.model_version == expected_model

    def test_metadata_includes_optional_fields(self, tmp_path: Path) -> None:
        """Optional metadata fields (tool_calls, reasoning, etc.) should persist."""
        repo_dir = tmp_path / "metadata-optional-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        gw = GitWrapper(str(repo_dir))
        gw.init_repo()

        metadata_with_all = AgentMetadata(
            agent_id="agent-optional-fields",
            model_version="gpt-4",
            prompt_hash=hashlib.sha256(b"optional fields test").hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_task="parent-task-42",
            confidence_score=0.99,
            reasoning="Detailed reasoning about architectural choices.",
            tool_calls=["analyze_deps", "refactor_module", "run_linter"],
        )

        _write_and_stage(repo_dir, "optional_test.py", "# Optional fields\n")
        commit_hash = gw.commit_with_metadata(
            message="Commit with all optional fields",
            metadata=metadata_with_all,
            files=["optional_test.py"],
        )

        retrieved = gw.get_commit_metadata(commit_hash)
        assert retrieved is not None
        assert retrieved.parent_task == "parent-task-42"
        assert retrieved.confidence_score == pytest.approx(0.99)
        assert retrieved.reasoning == "Detailed reasoning about architectural choices."
        assert retrieved.tool_calls == ["analyze_deps", "refactor_module", "run_linter"]
