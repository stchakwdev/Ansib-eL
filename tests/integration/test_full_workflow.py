"""
End-to-end integration tests for the Ansib-eL system.

Tests cover:
1. Full workflow: init system -> verify status -> check trust system is ready
2. System init + status: Initialize AnsibElSystem, check SystemStatus fields
3. Trust lineage integration: Create TrustLineageManager, record decisions,
   verify scores update, check lineage retrieval
"""

import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from ansibel.ansib_el import AnsibElSystem, SystemStatus
from ansibel.trust_lineage import (
    ChangeComplexity,
    DecisionType,
    TrustLineageManager,
    TrustTier,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(repo_path: Path) -> None:
    """Initialise a bare git repo with an initial commit at *repo_path*."""
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


# ---------------------------------------------------------------------------
# 1. Full workflow: init -> status -> trust system ready
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """End-to-end: initialise the system, verify status, confirm trust readiness."""

    def test_init_status_trust_ready(self, tmp_path: Path) -> None:
        """System initialises, status reports correctly, and trust system is usable."""
        repo_dir = tmp_path / "full-workflow-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        # -- Step 1: Initialise system --
        system = AnsibElSystem(str(repo_dir))
        result = system.initialize()
        assert result is True, "System initialization should succeed"

        # -- Step 2: Verify status --
        status = system.get_status()
        assert isinstance(status, SystemStatus)
        assert status.repo_initialized is True
        assert status.active_agents == 0
        assert status.pending_approvals == 0
        assert isinstance(status.trust_scores, dict)
        assert isinstance(status.recent_tournaments, list)

        # -- Step 3: Trust system is ready --
        # The trust lineage manager should be usable immediately after init
        trust_mgr = system.trust_lineage
        assert trust_mgr is not None

        # Querying a non-existent agent should return the default score
        unknown_agent_id = uuid4()
        score = trust_mgr.trust.get_trust_score(unknown_agent_id)
        assert score.score == pytest.approx(0.5, abs=0.01)
        assert score.confidence == pytest.approx(0.0)
        assert score.sample_count == 0

    def test_init_creates_ai_git_directory(self, tmp_path: Path) -> None:
        """Initialization should create the .ai-git metadata directory."""
        repo_dir = tmp_path / "ai-git-dir-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        system = AnsibElSystem(str(repo_dir))
        system.initialize()

        ai_git_dir = repo_dir / ".ai-git"
        assert ai_git_dir.exists(), ".ai-git directory should be created"
        assert (ai_git_dir / "metadata.json").exists(), "metadata.json should exist"

    def test_init_is_idempotent(self, tmp_path: Path) -> None:
        """Calling initialize() twice should not fail or corrupt state."""
        repo_dir = tmp_path / "idempotent-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        system = AnsibElSystem(str(repo_dir))
        assert system.initialize() is True
        assert system.initialize() is True

        status = system.get_status()
        assert status.repo_initialized is True


# ---------------------------------------------------------------------------
# 2. System init + status field validation
# ---------------------------------------------------------------------------


class TestSystemInitAndStatus:
    """Validate that AnsibElSystem.get_status() populates all fields correctly."""

    def test_status_fields_after_init(self, tmp_path: Path) -> None:
        """All SystemStatus fields should be populated with correct types."""
        repo_dir = tmp_path / "status-fields-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        system = AnsibElSystem(str(repo_dir))
        system.initialize()

        status = system.get_status()

        # Verify each field type and sensible value
        assert isinstance(status.repo_initialized, bool)
        assert status.repo_initialized is True

        assert isinstance(status.active_agents, int)
        assert status.active_agents >= 0

        assert isinstance(status.pending_approvals, int)
        assert status.pending_approvals >= 0

        assert isinstance(status.total_commits, int)
        assert status.total_commits >= 0

        assert isinstance(status.trust_scores, dict)
        assert isinstance(status.recent_tournaments, list)

    def test_status_before_init(self, tmp_path: Path) -> None:
        """Status before initialize() should show repo_initialized=False."""
        repo_dir = tmp_path / "pre-init-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        system = AnsibElSystem(str(repo_dir))
        # Do NOT call system.initialize()

        status = system.get_status()
        assert status.repo_initialized is False
        assert status.active_agents == 0
        assert status.pending_approvals == 0
        assert status.total_commits == 0
        assert status.trust_scores == {}
        assert status.recent_tournaments == []

    def test_git_wrapper_status_after_init(self, tmp_path: Path) -> None:
        """The underlying GitWrapper should report initialized=True."""
        repo_dir = tmp_path / "git-wrapper-status-repo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        system = AnsibElSystem(str(repo_dir))
        system.initialize()

        git_status = system.git.get_status()
        assert git_status["initialized"] is True
        assert "branch" in git_status


# ---------------------------------------------------------------------------
# 3. Trust lineage integration
# ---------------------------------------------------------------------------


class TestTrustLineageIntegration:
    """Integration tests for TrustLineageManager: record, score, lineage."""

    def test_record_decision_updates_score(self, tmp_path: Path) -> None:
        """Recording a decision should update the agent's trust score."""
        db_path = str(tmp_path / "trust_test.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # Record an ACCEPTED decision
        record = manager.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash="abc123def456",
            review_time_ms=500,
            change_size=50,
            complexity=ChangeComplexity.MODERATE,
        )

        assert record is not None
        assert record.agent_id == agent_id
        assert record.decision_type == DecisionType.ACCEPTED

        # Score should now reflect the positive decision
        score = manager.trust.get_trust_score(agent_id)
        assert score.score > 0.0
        assert score.sample_count == 1

    def test_multiple_decisions_adjust_score(self, tmp_path: Path) -> None:
        """Multiple decisions should progressively adjust the trust score."""
        db_path = str(tmp_path / "trust_multi.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # Record several ACCEPTED decisions
        for i in range(5):
            manager.trust.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"commit_{i:04d}",
                review_time_ms=200,
                change_size=30,
                complexity=ChangeComplexity.MINOR,
            )

        score_after_accepts = manager.trust.get_trust_score(agent_id)
        assert score_after_accepts.sample_count == 5
        high_score = score_after_accepts.score

        # Now record a REJECTED decision -- score should decrease
        manager.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.REJECTED,
            commit_hash="commit_rejected",
            review_time_ms=1000,
            change_size=100,
            complexity=ChangeComplexity.MAJOR,
        )

        score_after_reject = manager.trust.get_trust_score(agent_id)
        assert score_after_reject.sample_count == 6
        assert score_after_reject.score < high_score, "Score should decrease after a rejection"

    def test_trust_tier_reflects_score(self, tmp_path: Path) -> None:
        """Trust tier should match the score range after decisions."""
        db_path = str(tmp_path / "trust_tier.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # A new agent with no decisions defaults to 0.5
        tier = manager.trust.get_trust_tier(agent_id)
        assert isinstance(tier, TrustTier)

        # Record many ACCEPTED decisions to push score higher
        for i in range(15):
            manager.trust.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"tier_commit_{i:04d}",
                review_time_ms=300,
                change_size=40,
                complexity=ChangeComplexity.MODERATE,
            )

        score = manager.trust.get_trust_score(agent_id)
        tier = manager.trust.get_trust_tier(agent_id)

        # The tier should be at least MEDIUM given many accepted decisions
        assert tier in (TrustTier.MEDIUM, TrustTier.HIGH, TrustTier.VERIFIED), (
            f"Expected at least MEDIUM tier after 15 accepts, got {tier} with score {score.score}"
        )

    def test_agent_history_retrieval(self, tmp_path: Path) -> None:
        """Agent decision history should be retrievable in reverse chronological order."""
        db_path = str(tmp_path / "trust_history.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # Record a mix of decisions
        decisions = [
            DecisionType.ACCEPTED,
            DecisionType.ACCEPTED,
            DecisionType.REJECTED,
            DecisionType.MODIFIED,
            DecisionType.ACCEPTED,
        ]
        for i, decision in enumerate(decisions):
            manager.trust.record_decision(
                agent_id=agent_id,
                decision=decision,
                commit_hash=f"hist_commit_{i:04d}",
                review_time_ms=100 * (i + 1),
                change_size=20,
                complexity=ChangeComplexity.MINOR,
            )

        history = manager.trust.get_agent_history(agent_id)
        assert len(history) == 5

        # Should be reverse chronological (most recent first)
        for i in range(len(history) - 1):
            assert history[i].timestamp >= history[i + 1].timestamp

    def test_lineage_retrieval_for_commit(self, tmp_path: Path) -> None:
        """Recording reasoning + decision should produce a retrievable lineage."""
        db_path = str(tmp_path / "trust_lineage.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()
        commit_hash = "lineage_abc123"

        # Record a complete decision (both trust and lineage)
        decision_rec, reasoning_rec = manager.record_complete_decision(
            agent_id=agent_id,
            task_id="task-001",
            commit_hash=commit_hash,
            decision=DecisionType.ACCEPTED,
            reasoning="Chose this approach because it minimizes coupling.",
            review_time_ms=750,
            change_size=80,
            complexity=ChangeComplexity.MODERATE,
            confidence=0.85,
            supporting_evidence=["design_doc.md", "benchmark_results.txt"],
            parent_commits=["parent_abc"],
            library_choices={"requests": "Well-maintained HTTP library"},
        )

        assert decision_rec is not None
        assert reasoning_rec is not None
        assert reasoning_rec.confidence == pytest.approx(0.85)

        # Retrieve lineage for the commit
        lineage = manager.get_decision_with_lineage(commit_hash)
        assert lineage is not None
        assert lineage.commit_hash == commit_hash
        assert lineage.agent_id == agent_id

        # Verify reasoning is retrievable
        assert lineage.reasoning is not None
        assert "minimizes coupling" in lineage.reasoning.reasoning

        # Verify decision is retrievable
        assert lineage.decision is not None
        assert lineage.decision.decision_type == DecisionType.ACCEPTED

        # Verify parent commits
        assert "parent_abc" in lineage.parent_commits

        # Verify library choices
        assert "requests" in lineage.library_choices

    def test_agent_profile_after_decisions(self, tmp_path: Path) -> None:
        """Agent profile should reflect cumulative decision statistics."""
        db_path = str(tmp_path / "trust_profile.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # Record 3 accepts and 1 reject
        for i in range(3):
            manager.trust.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"profile_commit_{i}",
                review_time_ms=200,
                change_size=25,
                complexity=ChangeComplexity.MINOR,
            )

        manager.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.REJECTED,
            commit_hash="profile_commit_rejected",
            review_time_ms=500,
            change_size=50,
            complexity=ChangeComplexity.MODERATE,
        )

        profile = manager.trust.get_agent_profile(agent_id)
        assert profile is not None
        assert profile.agent_id == agent_id
        assert profile.total_decisions == 4
        assert profile.accepted_count == 3
        assert profile.rejected_count == 1

    def test_agent_summary(self, tmp_path: Path) -> None:
        """get_agent_summary should return a comprehensive summary dict."""
        db_path = str(tmp_path / "trust_summary.db")
        manager = TrustLineageManager(db_path)

        agent_id = uuid4()

        # Record a few decisions
        for i in range(3):
            manager.trust.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"summary_commit_{i}",
                review_time_ms=300,
                change_size=40,
                complexity=ChangeComplexity.MODERATE,
            )

        summary = manager.get_agent_summary(agent_id)
        assert "profile" in summary
        assert "recent_decisions" in summary
        assert len(summary["recent_decisions"]) == 3
        assert "can_auto_approve_small_changes" in summary
        assert "can_auto_approve_medium_changes" in summary
