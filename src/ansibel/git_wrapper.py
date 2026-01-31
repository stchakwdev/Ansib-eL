#!/usr/bin/env python3
"""
Git Wrapper for Ansib-eL (AI-Native Version Control System)

This module extends standard Git functionality with AI-specific metadata
and workflows for tracking agent decisions and maintaining branch isolation.
"""

import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from ansibel.exceptions import GitWrapperError

# Try to import GitPython, fall back to subprocess
try:
    import git
    from git import Repo

    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False

# Regex for validating git ref names (branch names, tags, etc.)
_VALID_GIT_REF_RE = re.compile(r"^[a-zA-Z0-9._\-/]+$")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai-git")


@dataclass
class AgentMetadata:
    """Metadata for AI agent commits."""

    agent_id: str
    model_version: str
    prompt_hash: str
    timestamp: str
    parent_task: str | None = None
    confidence_score: float | None = None
    reasoning: str | None = None
    tool_calls: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMetadata":
        """Create metadata from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MergeResult(NamedTuple):
    """Result of a merge operation."""

    success: bool
    message: str
    conflicts: list[str]
    merged_commit: str | None = None


# Keep backward-compatible re-export
__all__ = ["GitWrapper", "AgentMetadata", "MergeResult", "GitWrapperError"]


class GitWrapper:
    """
    Wrapper class for Git operations with AI-native extensions.

    This class provides:
    - Standard git operations via GitPython or subprocess
    - AI metadata storage and retrieval
    - Agent branch management
    - Merge and conflict resolution helpers
    """

    AI_GIT_DIR = ".ai-git"
    METADATA_FILE = "metadata.json"
    AGENTS_DIR = "agents"
    TRUST_DIR = "trust_scores"
    LINEAGE_DIR = "lineage"

    @staticmethod
    def _validate_git_ref(ref_name: str) -> str:
        """Validate a git ref name to prevent command injection.

        Args:
            ref_name: The ref name to validate.

        Returns:
            The validated ref name.

        Raises:
            GitWrapperError: If the ref name is invalid.
        """
        if not ref_name or not _VALID_GIT_REF_RE.match(ref_name) or ".." in ref_name:
            raise GitWrapperError(
                f"Invalid git ref name: {ref_name!r}. "
                "Only alphanumeric characters, '.', '-', '_', and '/' are allowed. "
                "Consecutive dots ('..') are not permitted."
            )
        return ref_name

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        """Atomically write JSON data to a file using tempfile + os.replace().

        Args:
            path: Target file path.
            data: JSON-serializable data to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def __init__(self, repo_path: str = ".") -> None:
        """
        Initialize the Git wrapper.

        Args:
            repo_path: Path to the git repository

        Raises:
            GitWrapperError: If the path is not a valid git repository
        """
        self.repo_path = Path(repo_path).resolve()
        self.ai_git_dir = self.repo_path / self.AI_GIT_DIR

        # Initialize git repository interface
        self.repo: Repo | None = None
        if GITPYTHON_AVAILABLE:
            try:
                self.repo = Repo(self.repo_path)
                logger.info(f"Initialized GitPython for repo: {self.repo_path}")
            except git.InvalidGitRepositoryError:
                logger.warning(f"No git repository found at {self.repo_path}")
        else:
            logger.info("GitPython not available, using subprocess")

    def _run_git_command(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a git command via subprocess.

        Args:
            args: Git command arguments
            check: Whether to raise on non-zero exit

        Returns:
            CompletedProcess instance
        """
        cmd = ["git"] + args
        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, cwd=self.repo_path, capture_output=True, text=True, check=False
        )

        if check and result.returncode != 0:
            raise GitWrapperError(f"Git command failed: {result.stderr}")

        return result

    def _ensure_ai_git_structure(self) -> None:
        """Create the .ai-git directory structure if it doesn't exist."""
        directories = [
            self.ai_git_dir,
            self.ai_git_dir / self.AGENTS_DIR,
            self.ai_git_dir / self.TRUST_DIR,
            self.ai_git_dir / self.LINEAGE_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def _get_metadata_path(self, commit_hash: str) -> Path:
        """Get the path for storing commit metadata."""
        # Store metadata in subdirectories based on first 2 chars of hash
        subdir = commit_hash[:2]
        return self.ai_git_dir / self.LINEAGE_DIR / subdir / f"{commit_hash}.json"

    def _save_commit_metadata(self, commit_hash: str, metadata: AgentMetadata) -> None:
        """Save metadata for a specific commit."""
        metadata_path = self._get_metadata_path(commit_hash)
        self._atomic_write_json(metadata_path, metadata.to_dict())
        logger.info(f"Saved metadata for commit {commit_hash}")

    def _load_commit_metadata(self, commit_hash: str) -> AgentMetadata | None:
        """Load metadata for a specific commit."""
        metadata_path = self._get_metadata_path(commit_hash)

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return AgentMetadata.from_dict(data)

    def init_repo(self) -> bool:
        """
        Initialize an ai-git repository.

        This initializes both a standard git repository (if needed)
        and the ai-git metadata structure.

        Returns:
            True if initialization was successful
        """
        try:
            # Initialize git repo if needed
            if not (self.repo_path / ".git").exists():
                if GITPYTHON_AVAILABLE:
                    Repo.init(self.repo_path)
                else:
                    self._run_git_command(["init"])
                logger.info(f"Initialized git repository at {self.repo_path}")

            # Create ai-git structure
            self._ensure_ai_git_structure()

            # Create initial metadata file
            metadata_file = self.ai_git_dir / self.METADATA_FILE
            if not metadata_file.exists():
                initial_metadata = {
                    "initialized_at": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0.0",
                    "agents": {},
                    "branches": {},
                }
                with open(metadata_file, "w") as f:
                    json.dump(initial_metadata, f, indent=2)

            # Add .ai-git to .gitignore if not already present
            gitignore = self.repo_path / ".gitignore"
            gitignore_entry = f"{self.AI_GIT_DIR}/\n"

            if gitignore.exists():
                with open(gitignore) as f:
                    content = f.read()
                if self.AI_GIT_DIR not in content:
                    with open(gitignore, "a") as f:
                        f.write(gitignore_entry)
            else:
                with open(gitignore, "w") as f:
                    f.write(gitignore_entry)

            logger.info(f"Initialized ai-git repository at {self.repo_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if ai-git has been initialized."""
        return (self.ai_git_dir / self.METADATA_FILE).exists()

    def create_agent_branch(self, agent_id: str, purpose: str) -> str:
        """
        Create an isolated branch for an agent.

        Args:
            agent_id: Unique identifier for the agent
            purpose: Description of the agent's purpose

        Returns:
            Name of the created branch

        Raises:
            GitWrapperError: If branch creation fails
        """
        if not self.is_initialized():
            raise GitWrapperError(
                "Repository not initialized. Run 'ai-git init' first."
            )

        # Validate agent_id before using in branch name
        self._validate_git_ref(agent_id)

        # Generate unique branch name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"agent/{agent_id}/{timestamp}"

        try:
            if GITPYTHON_AVAILABLE and self.repo:
                # Create and checkout new branch
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
            else:
                self._run_git_command(["checkout", "-b", branch_name])

            # Record branch metadata
            branch_metadata = {
                "agent_id": agent_id,
                "purpose": purpose,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
            }

            branch_file = (
                self.ai_git_dir
                / self.AGENTS_DIR
                / f"{branch_name.replace('/', '_')}.json"
            )
            with open(branch_file, "w") as f:
                json.dump(branch_metadata, f, indent=2)

            logger.info(f"Created agent branch: {branch_name}")
            return branch_name

        except Exception as e:
            raise GitWrapperError(f"Failed to create agent branch: {e}") from e

    def commit_with_metadata(
        self, message: str, metadata: AgentMetadata, files: list[str] | None = None
    ) -> str:
        """
        Create a commit with AI metadata.

        Args:
            message: Commit message
            metadata: Agent metadata to associate with the commit
            files: Specific files to commit (None for all staged)

        Returns:
            Hash of the created commit

        Raises:
            GitWrapperError: If commit fails
        """
        if not self.is_initialized():
            raise GitWrapperError(
                "Repository not initialized. Run 'ai-git init' first."
            )

        try:
            # Stage files if specified
            if files:
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.index.add(files)
                else:
                    self._run_git_command(["add"] + files)

            # Create commit with metadata in message trailer
            metadata_trailer = (
                f"\n\n[ai-git-metadata]\n{json.dumps(metadata.to_dict())}"
            )
            full_message = message + metadata_trailer

            if GITPYTHON_AVAILABLE and self.repo:
                commit = self.repo.index.commit(full_message)
                commit_hash = commit.hexsha
            else:
                # Use environment variable to pass message
                self._run_git_command(["commit", "-m", full_message])
                # Get the latest commit hash
                hash_result = self._run_git_command(["rev-parse", "HEAD"])
                commit_hash = hash_result.stdout.strip()

            # Save metadata separately for easy retrieval
            self._save_commit_metadata(commit_hash, metadata)

            logger.info(f"Created commit {commit_hash[:8]} with metadata")
            return commit_hash

        except Exception as e:
            raise GitWrapperError(f"Failed to create commit: {e}") from e

    def commit_changes(self, message: str, files: list[str] | None = None) -> str:
        """
        Create a plain commit (no AI metadata).

        Args:
            message: Commit message
            files: Specific files to stage and commit (None for all staged)

        Returns:
            Hash of the created commit

        Raises:
            GitWrapperError: If commit fails
        """
        if not self.is_initialized():
            raise GitWrapperError(
                "Repository not initialized. Run 'ai-git init' first."
            )

        try:
            if files:
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.index.add(files)
                else:
                    self._run_git_command(["add"] + files)

            if GITPYTHON_AVAILABLE and self.repo:
                commit = self.repo.index.commit(message)
                commit_hash = commit.hexsha
            else:
                self._run_git_command(["commit", "-m", message])
                hash_result = self._run_git_command(["rev-parse", "HEAD"])
                commit_hash = hash_result.stdout.strip()

            logger.info(f"Created commit {commit_hash[:8]}")
            return commit_hash

        except Exception as e:
            raise GitWrapperError(f"Failed to create commit: {e}") from e

    def get_commit_metadata(self, commit_hash: str) -> AgentMetadata | None:
        """
        Retrieve metadata for a specific commit.

        Args:
            commit_hash: Hash of the commit

        Returns:
            AgentMetadata if found, None otherwise
        """
        # First try loading from our storage
        metadata = self._load_commit_metadata(commit_hash)
        if metadata:
            return metadata

        # Fallback: try parsing from commit message
        try:
            if GITPYTHON_AVAILABLE and self.repo:
                commit = self.repo.commit(commit_hash)
                message = str(commit.message)
            else:
                result = self._run_git_command(
                    ["log", "-1", "--format=%B", commit_hash]
                )
                message = str(result.stdout)

            # Look for metadata trailer
            if "[ai-git-metadata]" in message:
                json_start = message.find("[ai-git-metadata]") + len(
                    "[ai-git-metadata]"
                )
                json_str = message[json_start:].strip()
                data = json.loads(json_str)
                return AgentMetadata.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to parse metadata from commit: {e}")

        return None

    def merge_agent_branch(
        self,
        branch_name: str,
        strategy: str = "merge",
        target_branch: str | None = None,
    ) -> MergeResult:
        """
        Merge an agent branch with conflict resolution.

        Args:
            branch_name: Name of the agent branch to merge
            strategy: Merge strategy ('merge', 'squash', 'rebase')
            target_branch: Target branch (default: current branch)

        Returns:
            MergeResult with success status and details
        """
        if not self.is_initialized():
            raise GitWrapperError(
                "Repository not initialized. Run 'ai-git init' first."
            )

        original_branch = None
        conflicts = []

        try:
            # Save current branch
            if GITPYTHON_AVAILABLE and self.repo:
                original_branch = self.repo.active_branch.name
            else:
                result = self._run_git_command(["branch", "--show-current"])
                original_branch = result.stdout.strip()

            # Checkout target branch if specified
            if target_branch:
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.heads[target_branch].checkout()
                else:
                    self._run_git_command(["checkout", target_branch])

            # Perform merge based on strategy
            if strategy == "merge":
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.git.merge(branch_name, no_ff=True)
                else:
                    self._run_git_command(
                        ["merge", "--no-ff", branch_name], check=False
                    )

            elif strategy == "squash":
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.git.merge(branch_name, squash=True)
                else:
                    self._run_git_command(
                        ["merge", "--squash", branch_name], check=False
                    )

            elif strategy == "rebase":
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.git.rebase(branch_name)
                else:
                    self._run_git_command(["rebase", branch_name], check=False)

            else:
                return MergeResult(
                    success=False, message=f"Unknown strategy: {strategy}", conflicts=[]
                )

            # Check for conflicts
            if GITPYTHON_AVAILABLE and self.repo:
                if self.repo.index.unmerged_blobs():
                    conflicts = list(self.repo.index.unmerged_blobs().keys())
            else:
                result = self._run_git_command(
                    ["diff", "--name-only", "--diff-filter=U"], check=False
                )
                if result.stdout.strip():
                    conflicts = result.stdout.strip().split("\n")

            if conflicts:
                return MergeResult(
                    success=False,
                    message=f"Merge has conflicts in {len(conflicts)} file(s)",
                    conflicts=[str(c) for c in conflicts],
                )

            # Get merged commit hash
            if GITPYTHON_AVAILABLE and self.repo:
                merged_commit = self.repo.head.commit.hexsha
            else:
                result = self._run_git_command(["rev-parse", "HEAD"])
                merged_commit = result.stdout.strip()

            # Update branch metadata
            branch_file = (
                self.ai_git_dir
                / self.AGENTS_DIR
                / f"{branch_name.replace('/', '_')}.json"
            )
            if branch_file.exists():
                with open(branch_file) as f:
                    metadata = json.load(f)
                metadata["status"] = "merged"
                metadata["merged_at"] = datetime.now(timezone.utc).isoformat()
                metadata["merged_commit"] = merged_commit
                with open(branch_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            return MergeResult(
                success=True,
                message=f"Successfully merged {branch_name}",
                conflicts=[],
                merged_commit=merged_commit,
            )

        except Exception as e:
            # Attempt to abort merge/rebase on failure
            try:
                if GITPYTHON_AVAILABLE and self.repo:
                    self.repo.git.merge("--abort")
                else:
                    self._run_git_command(["merge", "--abort"], check=False)
            except (GitWrapperError, OSError, Exception):
                pass

            return MergeResult(
                success=False,
                message=f"Merge failed: {str(e)}",
                conflicts=[str(c) for c in conflicts],
            )

        finally:
            # Restore original branch
            if original_branch and target_branch:
                try:
                    if GITPYTHON_AVAILABLE and self.repo:
                        self.repo.heads[original_branch].checkout()
                    else:
                        self._run_git_command(
                            ["checkout", original_branch], check=False
                        )
                except (GitWrapperError, OSError, Exception):
                    pass

    def list_agent_branches(self, status: str | None = None) -> list[dict[str, Any]]:
        """
        List all agent branches with metadata.

        Args:
            status: Filter by status ('active', 'merged', 'closed')

        Returns:
            List of branch information dictionaries
        """
        if not self.is_initialized():
            raise GitWrapperError(
                "Repository not initialized. Run 'ai-git init' first."
            )

        branches: list[dict[str, Any]] = []
        agents_dir = self.ai_git_dir / self.AGENTS_DIR

        if not agents_dir.exists():
            return branches

        for branch_file in agents_dir.glob("*.json"):
            with open(branch_file) as f:
                metadata = json.load(f)

            branch_name = branch_file.stem.replace("_", "/")

            if status and metadata.get("status") != status:
                continue

            branches.append({"name": branch_name, **metadata})

        return sorted(branches, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_diff(self, branch_a: str, branch_b: str) -> str:
        """
        Get the diff between two branches.

        Args:
            branch_a: First branch name
            branch_b: Second branch name

        Returns:
            Diff output as string
        """
        try:
            if GITPYTHON_AVAILABLE and self.repo:
                # Get commits for each branch
                commit_a = self.repo.commit(branch_a)
                commit_b = self.repo.commit(branch_b)

                # Generate diff
                diff = commit_a.diff(commit_b, create_patch=True)
                return "\n".join(str(d) for d in diff)
            else:
                result = self._run_git_command(["diff", f"{branch_a}...{branch_b}"])
                return str(result.stdout)

        except Exception as e:
            raise GitWrapperError(f"Failed to get diff: {e}") from e

    def get_status(self) -> dict[str, Any]:
        """
        Get repository status with AI information.

        Returns:
            Dictionary with status information
        """
        if not self.is_initialized():
            return {"initialized": False}

        status = {
            "initialized": True,
            "ai_git_version": "1.0.0",
        }

        try:
            if GITPYTHON_AVAILABLE and self.repo:
                status["branch"] = self.repo.active_branch.name
                status["is_dirty"] = self.repo.is_dirty()
                status["untracked_files"] = self.repo.untracked_files
                status["modified_files"] = [
                    item.a_path for item in self.repo.index.diff(None)
                ]
                status["staged_files"] = [
                    item.a_path for item in self.repo.index.diff("HEAD")
                ]
            else:
                # Get branch
                result = self._run_git_command(["branch", "--show-current"])
                status["branch"] = result.stdout.strip()

                # Check for changes
                result = self._run_git_command(["status", "--porcelain"])
                lines = (
                    result.stdout.strip().split("\n") if result.stdout.strip() else []
                )

                status["untracked_files"] = [
                    line[3:] for line in lines if line.startswith("??")
                ]
                status["modified_files"] = [
                    line[3:]
                    for line in lines
                    if line.startswith(" M") or line.startswith("M ")
                ]
                status["staged_files"] = [
                    line[3:]
                    for line in lines
                    if line.startswith("A ") or line.startswith("M ")
                ]
                status["is_dirty"] = len(lines) > 0

            # Add agent information
            active_agents = self.list_agent_branches(status="active")
            status["active_agents"] = active_agents
            status["pending_merges"] = len(active_agents)

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            status["error"] = str(e)

        return status

    def get_ai_enhanced_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get commit history with AI metadata.

        Args:
            limit: Maximum number of commits to retrieve

        Returns:
            List of commit information with metadata
        """
        history = []

        try:
            if GITPYTHON_AVAILABLE and self.repo:
                for commit in self.repo.iter_commits(max_count=limit):
                    metadata = self.get_commit_metadata(commit.hexsha)

                    history.append(
                        {
                            "hash": commit.hexsha[:8],
                            "full_hash": commit.hexsha,
                            "message": str(commit.message).split("\n")[0],
                            "author": str(commit.author),
                            "date": commit.committed_datetime.isoformat(),
                            "metadata": metadata.to_dict() if metadata else None,
                        }
                    )
            else:
                # Use git log with format
                format_str = "%H|%an|%ad|%s"
                result = self._run_git_command(
                    ["log", f"-{limit}", f"--format={format_str}"]
                )

                for line in result.stdout.strip().split("\n"):
                    if "|" in line:
                        parts = line.split("|", 3)
                        if len(parts) >= 4:
                            commit_hash = parts[0]
                            metadata = self.get_commit_metadata(commit_hash)

                            history.append(
                                {
                                    "hash": commit_hash[:8],
                                    "full_hash": commit_hash,
                                    "author": parts[1],
                                    "date": parts[2],
                                    "message": parts[3],
                                    "metadata": (
                                        metadata.to_dict() if metadata else None
                                    ),
                                }
                            )

        except Exception as e:
            logger.error(f"Failed to get history: {e}")

        return history

    def resolve_conflict(self, file_path: str, resolution: str) -> bool:
        """
        Mark a conflicted file as resolved.

        Args:
            file_path: Path to the conflicted file
            resolution: Resolution content or 'ours'/'theirs'

        Returns:
            True if resolution was successful
        """
        try:
            if resolution in ("ours", "theirs"):
                if GITPYTHON_AVAILABLE and self.repo:
                    if resolution == "ours":
                        self.repo.git.checkout("--ours", file_path)
                    else:
                        self.repo.git.checkout("--theirs", file_path)
                else:
                    self._run_git_command(["checkout", f"--{resolution}", file_path])
            else:
                # Write resolution content
                full_path = self.repo_path / file_path
                with open(full_path, "w") as f:
                    f.write(resolution)

            # Stage the resolved file
            if GITPYTHON_AVAILABLE and self.repo:
                self.repo.index.add([file_path])
            else:
                self._run_git_command(["add", file_path])

            logger.info(f"Resolved conflict in {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            return False

    def update_agent_trust_score(self, agent_id: str, score_delta: float) -> float:
        """
        Update the trust score for an agent.

        Args:
            agent_id: Agent identifier
            score_delta: Change in trust score

        Returns:
            New trust score
        """
        trust_file = self.ai_git_dir / self.TRUST_DIR / f"{agent_id}.json"

        if trust_file.exists():
            with open(trust_file) as f:
                data = json.load(f)
        else:
            data = {
                "agent_id": agent_id,
                "score": 0.5,  # Default starting score
                "history": [],
            }

        # Update score (clamp between 0 and 1)
        new_score = max(0.0, min(1.0, data["score"] + score_delta))
        data["score"] = new_score
        data["history"].append(
            {"delta": score_delta, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

        with open(trust_file, "w") as f:
            json.dump(data, f, indent=2)

        return float(new_score)

    def get_agent_trust_score(self, agent_id: str) -> float:
        """
        Get the trust score for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Trust score (0.0 to 1.0)
        """
        trust_file = self.ai_git_dir / self.TRUST_DIR / f"{agent_id}.json"

        if trust_file.exists():
            with open(trust_file) as f:
                data = json.load(f)
            return float(data.get("score", 0.5))

        return 0.5  # Default score
