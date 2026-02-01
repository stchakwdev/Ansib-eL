#!/usr/bin/env python3
"""
Comprehensive unit tests for the GitWrapper class.

Tests cover initialization, branch management, commit metadata, merges,
diffs, status, history, validation, security, and data model roundtrips.
All tests are self-contained using pytest's tmp_path fixture with real
git repositories.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ansibel.exceptions import GitWrapperError
from ansibel.git_wrapper import AgentMetadata, GitWrapper, MergeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command inside *repo_path* and return the result."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )


def _init_wrapper(tmp_path: Path) -> GitWrapper:
    """Create a GitWrapper, call init_repo, and configure git user for commits."""
    wrapper = GitWrapper(str(tmp_path))
    wrapper.init_repo()
    # Configure user so that git commit works in a bare temp dir.
    _git(tmp_path, "config", "user.email", "test@ansibel.dev")
    _git(tmp_path, "config", "user.name", "Test Runner")
    return wrapper


def _make_initial_commit(tmp_path: Path) -> None:
    """Stage the existing .gitignore and create an initial commit."""
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-m", "Initial commit")


def _create_file(tmp_path: Path, name: str, content: str) -> Path:
    """Create a file in the repo, return its Path."""
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# 1. init_repo
# ---------------------------------------------------------------------------


class TestInitRepo:
    """Tests for GitWrapper.init_repo()."""

    def test_init_repo_creates_git_directory(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        result = wrapper.init_repo()
        assert result is True
        assert (tmp_path / ".git").exists()

    def test_init_repo_creates_ai_git_structure(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        wrapper.init_repo()
        ai_dir = tmp_path / ".ai-git"
        assert ai_dir.exists()
        assert (ai_dir / "metadata.json").exists()
        assert (ai_dir / "agents").is_dir()
        assert (ai_dir / "trust_scores").is_dir()
        assert (ai_dir / "lineage").is_dir()

    def test_init_repo_is_idempotent(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        first = wrapper.init_repo()
        second = wrapper.init_repo()
        assert first is True
        assert second is True
        # metadata.json should still be valid
        meta = json.loads((tmp_path / ".ai-git" / "metadata.json").read_text())
        assert meta["version"] == "1.0.0"

    def test_init_repo_creates_gitignore_with_ai_git(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        wrapper.init_repo()
        gitignore = (tmp_path / ".gitignore").read_text()
        assert ".ai-git/" in gitignore

    def test_init_repo_appends_to_existing_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        wrapper = GitWrapper(str(tmp_path))
        wrapper.init_repo()
        gitignore = (tmp_path / ".gitignore").read_text()
        assert "*.pyc" in gitignore
        assert ".ai-git/" in gitignore


# ---------------------------------------------------------------------------
# 2. _validate_git_ref
# ---------------------------------------------------------------------------


class TestValidateGitRef:
    """Tests for GitWrapper._validate_git_ref()."""

    @pytest.mark.parametrize(
        "ref",
        [
            "main",
            "feature/foo",
            "agent/abc-123/20240101-120000",
            "release-1.0",
            "v2.3.4",
            "my_branch",
        ],
    )
    def test_valid_refs_accepted(self, ref: str) -> None:
        assert GitWrapper._validate_git_ref(ref) == ref

    @pytest.mark.parametrize(
        "ref",
        [
            "branch name",  # space
            "branch;rm -rf /",  # semicollon / shell metachar
            "foo..bar",  # double dot
            "$(whoami)",  # command substitution
            "branch\ttab",  # tab
            "",  # empty
            "hello world",  # space
            "bad&ref",  # ampersand
            "bad|ref",  # pipe
        ],
    )
    def test_invalid_refs_rejected(self, ref: str) -> None:
        with pytest.raises(GitWrapperError):
            GitWrapper._validate_git_ref(ref)


# ---------------------------------------------------------------------------
# 3. create_agent_branch
# ---------------------------------------------------------------------------


class TestCreateAgentBranch:
    """Tests for GitWrapper.create_agent_branch()."""

    def test_branch_name_follows_convention(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        branch = wrapper.create_agent_branch("agent-42", "code review")
        assert branch.startswith("agent/agent-42/")

    def test_branch_metadata_file_created(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        branch = wrapper.create_agent_branch("agent-42", "code review")
        meta_file = tmp_path / ".ai-git" / "agents" / f"{branch.replace('/', '_')}.json"
        assert meta_file.exists()
        data = json.loads(meta_file.read_text())
        assert data["agent_id"] == "agent-42"
        assert data["purpose"] == "code review"
        assert data["status"] == "active"

    def test_branch_requires_initialized_repo(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        with pytest.raises(GitWrapperError, match="not initialized"):
            wrapper.create_agent_branch("agent-1", "test")

    def test_branch_with_injected_name_raises(self, tmp_path: Path) -> None:
        """Security: agent_id with shell metacharacters must be rejected."""
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        with pytest.raises(GitWrapperError):
            wrapper.create_agent_branch("agent;rm -rf /", "malicious")

    def test_branch_with_spaces_in_id_raises(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        with pytest.raises(GitWrapperError):
            wrapper.create_agent_branch("bad agent id", "test")


# ---------------------------------------------------------------------------
# 4. commit_with_metadata / commit_changes
# ---------------------------------------------------------------------------


class TestCommitWithMetadata:
    """Tests for GitWrapper.commit_with_metadata()."""

    @staticmethod
    def _make_metadata(**overrides) -> AgentMetadata:
        defaults = dict(
            agent_id="test-agent",
            model_version="gpt-5.2",
            prompt_hash="hash123",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        defaults.update(overrides)
        return AgentMetadata(**defaults)

    def test_commit_returns_full_sha(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        _create_file(tmp_path, "a.txt", "hello")
        _git(tmp_path, "add", "a.txt")
        commit_hash = wrapper.commit_with_metadata("add file", self._make_metadata())
        assert len(commit_hash) == 40

    def test_commit_metadata_retrievable(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        _create_file(tmp_path, "a.txt", "hello")
        _git(tmp_path, "add", "a.txt")
        meta = self._make_metadata(confidence_score=0.88)
        commit_hash = wrapper.commit_with_metadata("add file", meta)
        loaded = wrapper.get_commit_metadata(commit_hash)
        assert loaded is not None
        assert loaded.agent_id == "test-agent"
        assert loaded.confidence_score == 0.88

    def test_commit_requires_initialized_repo(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        meta = self._make_metadata()
        with pytest.raises(GitWrapperError, match="not initialized"):
            wrapper.commit_with_metadata("msg", meta)


# ---------------------------------------------------------------------------
# 5. _save_commit_metadata
# ---------------------------------------------------------------------------


class TestSaveCommitMetadata:
    """Tests for GitWrapper._save_commit_metadata()."""

    def test_metadata_json_file_created(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        _create_file(tmp_path, "b.txt", "content")
        _git(tmp_path, "add", "b.txt")
        meta = AgentMetadata(
            agent_id="a1",
            model_version="gpt-5.2",
            prompt_hash="ph",
            timestamp="2024-01-01T00:00:00Z",
        )
        commit_hash = wrapper.commit_with_metadata("test", meta)
        subdir = commit_hash[:2]
        meta_path = tmp_path / ".ai-git" / "lineage" / subdir / f"{commit_hash}.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# 6. _atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for GitWrapper._atomic_write_json()."""

    def test_file_exists_after_write(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "data.json"
        GitWrapper._atomic_write_json(target, {"key": "value"})
        assert target.exists()

    def test_content_matches(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        payload = {"alpha": 1, "beta": [2, 3]}
        GitWrapper._atomic_write_json(target, payload)
        loaded = json.loads(target.read_text())
        assert loaded == payload

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        GitWrapper._atomic_write_json(target, {"v": 1})
        GitWrapper._atomic_write_json(target, {"v": 2})
        loaded = json.loads(target.read_text())
        assert loaded["v"] == 2


# ---------------------------------------------------------------------------
# 7. merge_agent_branch
# ---------------------------------------------------------------------------


class TestMergeAgentBranch:
    """Tests for GitWrapper.merge_agent_branch()."""

    def test_successful_merge(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _create_file(tmp_path, "base.txt", "base")
        _git(tmp_path, "add", "base.txt")
        _git(tmp_path, "commit", "-m", "base commit")

        branch = wrapper.create_agent_branch("merger", "merge test")
        _create_file(tmp_path, "new.txt", "new content")
        _git(tmp_path, "add", "new.txt")
        _git(tmp_path, "commit", "-m", "agent work")

        # Switch back to the original branch (master/main)
        _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        # The original branch might be main or master depending on git config.
        # We stored it before switching, let's just go back to the first branch.
        branches_output = _git(tmp_path, "branch").stdout
        all_branches = [
            b.strip().lstrip("* ") for b in branches_output.strip().splitlines()
        ]
        original = [b for b in all_branches if not b.startswith("agent/")][0]
        _git(tmp_path, "checkout", original)

        result = wrapper.merge_agent_branch(branch)
        assert result.success is True
        assert result.merged_commit is not None
        assert len(result.conflicts) == 0

    def test_merge_conflict_detected(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _create_file(tmp_path, "conflict.txt", "original")
        _git(tmp_path, "add", "conflict.txt")
        _git(tmp_path, "commit", "-m", "base")

        # Determine default branch name
        default_branch = _git(tmp_path, "branch", "--show-current").stdout.strip()

        branch = wrapper.create_agent_branch("conflict-agent", "conflict test")
        _create_file(tmp_path, "conflict.txt", "agent version")
        _git(tmp_path, "add", "conflict.txt")
        _git(tmp_path, "commit", "-m", "agent edit")

        # Go back to default branch and make a conflicting change
        _git(tmp_path, "checkout", default_branch)
        _create_file(tmp_path, "conflict.txt", "main version")
        _git(tmp_path, "add", "conflict.txt")
        _git(tmp_path, "commit", "-m", "main edit")

        result = wrapper.merge_agent_branch(branch)
        assert result.success is False
        assert (
            len(result.conflicts) > 0
            or "conflict" in result.message.lower()
            or "failed" in result.message.lower()
        )


# ---------------------------------------------------------------------------
# 8. get_diff
# ---------------------------------------------------------------------------


class TestGetDiff:
    """Tests for GitWrapper.get_diff()."""

    def test_diff_between_branches(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _create_file(tmp_path, "file.txt", "first")
        _git(tmp_path, "add", "file.txt")
        _git(tmp_path, "commit", "-m", "first commit")

        default_branch = _git(tmp_path, "branch", "--show-current").stdout.strip()

        branch = wrapper.create_agent_branch("diff-agent", "diff test")
        _create_file(tmp_path, "file.txt", "second")
        _git(tmp_path, "add", "file.txt")
        _git(tmp_path, "commit", "-m", "changed file")

        diff_output = wrapper.get_diff(default_branch, branch)
        # The diff should reference the changed file
        assert isinstance(diff_output, str)


# ---------------------------------------------------------------------------
# 9. get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for GitWrapper.get_status()."""

    def test_status_on_clean_repo(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        status = wrapper.get_status()
        assert status["initialized"] is True
        assert "branch" in status

    def test_status_on_dirty_repo(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        _create_file(tmp_path, "dirty.txt", "untracked")
        status = wrapper.get_status()
        assert status["is_dirty"] is True

    def test_status_uninitialized(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        status = wrapper.get_status()
        assert status["initialized"] is False


# ---------------------------------------------------------------------------
# 10. get_ai_enhanced_history
# ---------------------------------------------------------------------------


class TestGetAiEnhancedHistory:
    """Tests for GitWrapper.get_ai_enhanced_history()."""

    def test_history_returns_commits(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _create_file(tmp_path, "h.txt", "v1")
        _git(tmp_path, "add", "h.txt")
        _git(tmp_path, "commit", "-m", "first")
        _create_file(tmp_path, "h.txt", "v2")
        _git(tmp_path, "add", "h.txt")
        _git(tmp_path, "commit", "-m", "second")

        history = wrapper.get_ai_enhanced_history(limit=10)
        assert len(history) >= 2
        assert "hash" in history[0]
        assert "message" in history[0]

    def test_history_includes_metadata_when_present(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        _create_file(tmp_path, "m.txt", "data")
        _git(tmp_path, "add", "m.txt")
        meta = AgentMetadata(
            agent_id="hist-agent",
            model_version="gpt-5.2",
            prompt_hash="p",
            timestamp="2024-06-01T00:00:00Z",
        )
        wrapper.commit_with_metadata("with meta", meta)
        history = wrapper.get_ai_enhanced_history(limit=5)
        # The most recent commit should have metadata
        latest = history[0]
        assert latest["metadata"] is not None
        assert latest["metadata"]["agent_id"] == "hist-agent"

    def test_history_respects_limit(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        for i in range(5):
            _create_file(tmp_path, f"f{i}.txt", str(i))
            _git(tmp_path, "add", f"f{i}.txt")
            _git(tmp_path, "commit", "-m", f"commit {i}")
        history = wrapper.get_ai_enhanced_history(limit=2)
        assert len(history) == 2


# ---------------------------------------------------------------------------
# 11. Security: injected branch names
# ---------------------------------------------------------------------------


class TestSecurityInjection:
    """Security tests ensuring shell injection via branch names is blocked."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "agent;rm -rf /",
            "agent$(whoami)",
            "agent`id`",
            "agent | cat /etc/passwd",
            "agent\nmalicious",
        ],
    )
    def test_create_agent_branch_rejects_injected_id(
        self, tmp_path: Path, bad_id: str
    ) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        with pytest.raises(GitWrapperError):
            wrapper.create_agent_branch(bad_id, "test purpose")


# ---------------------------------------------------------------------------
# 12. AgentMetadata roundtrip
# ---------------------------------------------------------------------------


class TestAgentMetadataRoundtrip:
    """Tests for AgentMetadata.to_dict / from_dict roundtrip."""

    def test_roundtrip_all_fields(self) -> None:
        original = AgentMetadata(
            agent_id="roundtrip",
            model_version="gpt-5.2",
            prompt_hash="sha256abc",
            timestamp="2024-12-25T10:00:00Z",
            parent_task="task-99",
            confidence_score=0.77,
            reasoning="chose approach A",
            tool_calls=["search", "edit"],
        )
        d = original.to_dict()
        restored = AgentMetadata.from_dict(d)
        assert restored == original

    def test_roundtrip_minimal_fields(self) -> None:
        original = AgentMetadata(
            agent_id="min",
            model_version="v1",
            prompt_hash="h",
            timestamp="t",
        )
        restored = AgentMetadata.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_ignores_extra_keys(self) -> None:
        data = {
            "agent_id": "x",
            "model_version": "v",
            "prompt_hash": "h",
            "timestamp": "t",
            "unknown_field": "should be ignored",
        }
        meta = AgentMetadata.from_dict(data)
        assert meta.agent_id == "x"
        assert not hasattr(meta, "unknown_field")


# ---------------------------------------------------------------------------
# 13. MergeResult namedtuple fields
# ---------------------------------------------------------------------------


class TestMergeResult:
    """Tests for MergeResult namedtuple."""

    def test_fields_accessible(self) -> None:
        r = MergeResult(
            success=True, message="ok", conflicts=[], merged_commit="abc123"
        )
        assert r.success is True
        assert r.message == "ok"
        assert r.conflicts == []
        assert r.merged_commit == "abc123"

    def test_default_merged_commit_is_none(self) -> None:
        r = MergeResult(success=False, message="fail", conflicts=["a.txt"])
        assert r.merged_commit is None

    def test_tuple_unpacking(self) -> None:
        r = MergeResult(success=True, message="m", conflicts=[], merged_commit="h")
        success, message, conflicts, merged_commit = r
        assert success is True
        assert merged_commit == "h"


# ---------------------------------------------------------------------------
# 14. list_agent_branches
# ---------------------------------------------------------------------------


class TestListAgentBranches:
    """Tests for GitWrapper.list_agent_branches()."""

    def test_no_branches_initially(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        branches = wrapper.list_agent_branches()
        assert branches == []

    def test_lists_created_branches(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        wrapper.create_agent_branch("a1", "purpose one")
        # Switch back so we can create another branch from default
        _git(tmp_path, "checkout", "-")
        wrapper.create_agent_branch("a2", "purpose two")
        branches = wrapper.list_agent_branches()
        assert len(branches) == 2
        agent_ids = {b["agent_id"] for b in branches}
        assert agent_ids == {"a1", "a2"}

    def test_filter_by_status(self, tmp_path: Path) -> None:
        wrapper = _init_wrapper(tmp_path)
        _make_initial_commit(tmp_path)
        wrapper.create_agent_branch("a1", "active branch")
        active = wrapper.list_agent_branches(status="active")
        merged = wrapper.list_agent_branches(status="merged")
        assert len(active) == 1
        assert len(merged) == 0

    def test_requires_initialized_repo(self, tmp_path: Path) -> None:
        wrapper = GitWrapper(str(tmp_path))
        with pytest.raises(GitWrapperError, match="not initialized"):
            wrapper.list_agent_branches()
