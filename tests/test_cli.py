"""
Tests for ansibel.cli -- Click-based CLI commands.

Uses Click's CliRunner to invoke CLI commands in isolated temporary
directories with pre-initialised git repositories.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from ansibel.cli import cli

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
    """Initialise a git repo with an initial commit at *path*."""
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
    readme.write_text("# Test\n")
    subprocess.run(
        ["git", "add", "README.md"], cwd=str(path), capture_output=True, check=True, env=env
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(path),
        capture_output=True,
        check=True,
        env=env,
    )


def _init_ai_git(path: Path) -> None:
    """Run ai-git init in *path* by invoking the CLI."""
    runner = CliRunner()
    saved = os.getcwd()
    try:
        os.chdir(str(path))
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0, f"init failed: {result.output}"
    finally:
        os.chdir(saved)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_dir(tmp_path: Path) -> Path:
    """Return a tmp directory with git init + initial commit + ai-git init."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    _init_ai_git(repo)
    return repo


# ---------------------------------------------------------------------------
# 1. init command -- initialise repository
# ---------------------------------------------------------------------------


class TestInitCommand:
    def test_init_creates_ai_git_dir(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo))
            result = runner.invoke(cli, ["init"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert (repo / ".ai-git").is_dir()
        assert "initialized" in result.output.lower() or "Successfully" in result.output

    def test_init_already_initialised_warns(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["init"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "already initialized" in result.output.lower() or "already" in result.output.lower()

    def test_init_force_reinitialises(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["init", "--force"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert (repo_dir / ".ai-git").is_dir()


# ---------------------------------------------------------------------------
# 2. status command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_shows_branch(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["status"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        # Should mention branch name somewhere
        assert "branch" in result.output.lower() or "Branch" in result.output

    def test_status_json_output(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["status", "--json-output"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "branch" in data
        assert "initialized" in data or "is_dirty" in data


# ---------------------------------------------------------------------------
# 3. branch command
# ---------------------------------------------------------------------------


class TestBranchCommand:
    def test_branch_creates_agent_branch(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["branch", "agent-001", "--purpose", "Fix login bug"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "agent-001" in result.output or "Created branch" in result.output


# ---------------------------------------------------------------------------
# 4. commit command
# ---------------------------------------------------------------------------


class TestCommitCommand:
    def test_commit_with_metadata(self, repo_dir: Path) -> None:
        # Create a file to commit
        test_file = repo_dir / "feature.py"
        test_file.write_text("# new feature\n")

        env = {**os.environ, **GIT_ENV}
        subprocess.run(
            ["git", "add", "feature.py"],
            cwd=str(repo_dir),
            capture_output=True,
            check=True,
            env=env,
        )

        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(
                cli,
                [
                    "commit",
                    "Add feature file",
                    "--agent-id",
                    "agent-001",
                    "--model-version",
                    "gpt-4",
                    "--confidence",
                    "0.95",
                ],
            )
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "commit" in result.output.lower() or "Created" in result.output


# ---------------------------------------------------------------------------
# 5. merge command
# ---------------------------------------------------------------------------


class TestMergeCommand:
    def test_merge_nonexistent_branch_fails(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["merge", "nonexistent-branch", "--yes"])
        finally:
            os.chdir(saved)

        # Should fail with non-zero exit or error message
        assert (
            result.exit_code != 0
            or "failed" in result.output.lower()
            or "error" in result.output.lower()
        )


# ---------------------------------------------------------------------------
# 6. history command
# ---------------------------------------------------------------------------


class TestHistoryCommand:
    def test_history_shows_commits(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["history"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        # Should display something (header at minimum)
        assert (
            "History" in result.output
            or "commit" in result.output.lower()
            or len(result.output) > 0
        )

    def test_history_json_output(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["history", "--json-output"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# 7. agents command
# ---------------------------------------------------------------------------


class TestAgentsCommand:
    def test_agents_list_empty(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["agents"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0

    def test_agents_json_output(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["agents", "--json-output"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# 8. trust command
# ---------------------------------------------------------------------------


class TestTrustCommand:
    def test_trust_update_score(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["trust", "agent-001", "0.1"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "trust" in result.output.lower() or "score" in result.output.lower()


# ---------------------------------------------------------------------------
# 9. diff command
# ---------------------------------------------------------------------------


class TestDiffCommand:
    def test_diff_single_branch_arg(self, repo_dir: Path) -> None:
        """diff with one argument compares that branch against current."""
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            # Get current branch name
            env = {**os.environ, **GIT_ENV}
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                env=env,
            )
            current = branch_result.stdout.strip()
            result = runner.invoke(cli, ["diff", current])
        finally:
            os.chdir(saved)

        # Should succeed (possibly empty diff)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 10. JSON output mode (status --json-output)
# ---------------------------------------------------------------------------


class TestJsonOutputMode:
    def test_status_json_is_valid(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["status", "-j"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert "branch" in parsed


# ---------------------------------------------------------------------------
# 11. Error on uninitialised repo
# ---------------------------------------------------------------------------


class TestUninitialised:
    def test_status_fails_without_init(self, tmp_path: Path) -> None:
        repo = tmp_path / "bare"
        repo.mkdir()
        _init_git_repo(repo)
        # Do NOT run ai-git init

        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo))
            result = runner.invoke(cli, ["status"])
        finally:
            os.chdir(saved)

        assert result.exit_code != 0
        assert "not" in result.output.lower() or "init" in result.output.lower()

    def test_history_fails_without_init(self, tmp_path: Path) -> None:
        repo = tmp_path / "bare"
        repo.mkdir()
        _init_git_repo(repo)

        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo))
            result = runner.invoke(cli, ["history"])
        finally:
            os.chdir(saved)

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 12. Help text output
# ---------------------------------------------------------------------------


class TestHelpText:
    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AI-Native" in result.output or "ai-git" in result.output.lower()

    def test_init_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output or "init" in result.output.lower()


# ---------------------------------------------------------------------------
# 13. Version output
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


# ---------------------------------------------------------------------------
# 14. Metadata command
# ---------------------------------------------------------------------------


class TestMetadataCommand:
    def test_metadata_no_metadata_warns(self, repo_dir: Path) -> None:
        """Requesting metadata for an unknown commit hash should warn."""
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["metadata", "0000000000000000000000000000000000000000"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "no metadata" in result.output.lower() or "No metadata" in result.output


# ---------------------------------------------------------------------------
# 15. Review command -- no pending branches
# ---------------------------------------------------------------------------


class TestReviewCommand:
    def test_review_no_pending(self, repo_dir: Path) -> None:
        runner = CliRunner()
        saved = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            result = runner.invoke(cli, ["review"])
        finally:
            os.chdir(saved)

        assert result.exit_code == 0
        assert "no pending" in result.output.lower() or "No pending" in result.output
