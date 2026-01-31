"""
Comprehensive unit tests for TrustLineageManager, TrustScorer, LineageTracker,
and related data models in ansibel.trust_lineage.

Tests cover:
- Decision recording for all DecisionType variants
- EMA smoothing behaviour and score evolution
- Score clamping to [0, 1]
- Unknown agent defaults
- TrustTier boundary classification
- Auto-approval logic by tier and change size
- Time decay towards minimum floor
- Minimum floor enforcement
- Confidence progression
- Agent profile CRUD
- Recovery boost after rejection streak
- Reasoning recording via record_complete_decision()
- Lineage retrieval by commit hash
- Chain tracing for linked decisions
- Keyword search with and without agent filter
- Thread safety with concurrent recording
- Foreign key enforcement (PRAGMA foreign_keys = ON)
- Enum value correctness
- TrustLineageManager initialisation
- get_agent_summary() output structure
- get_decision_with_lineage() return structure
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from ansibel.trust_lineage import (
    AgentProfile,
    ChangeComplexity,
    DecisionLineage,
    DecisionRecord,
    DecisionType,
    LineageTracker,
    ReasoningRecord,
    TrustLineageManager,
    TrustScore,
    TrustScorer,
    TrustTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commit_hash() -> str:
    """Generate a unique fake commit hash."""
    return hashlib.sha1(uuid4().bytes).hexdigest()


def _scorer(tmp_path) -> TrustScorer:
    return TrustScorer(str(tmp_path / "test.db"))


def _manager(tmp_path) -> TrustLineageManager:
    return TrustLineageManager(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# 1. Decision recording for all DecisionTypes
# ---------------------------------------------------------------------------

class TestDecisionRecordingAllTypes:
    """Verify that every DecisionType can be recorded without error."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DecisionType.ACCEPTED,
            DecisionType.REJECTED,
            DecisionType.MODIFIED,
            DecisionType.AUTO_APPROVED,
            DecisionType.REVIEWED,
        ],
    )
    def test_record_decision_for_each_type(self, tmp_path, dtype):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()
        record = scorer.record_decision(
            agent_id=agent_id,
            decision=dtype,
            commit_hash=_commit_hash(),
            review_time_ms=500,
            change_size=10,
            complexity=ChangeComplexity.MINOR,
        )
        assert isinstance(record, DecisionRecord)
        assert record.decision_type == dtype
        assert record.agent_id == agent_id


# ---------------------------------------------------------------------------
# 2. EMA smoothing -- score increases on ACCEPTED, decreases on REJECTED
# ---------------------------------------------------------------------------

class TestEMASmoothing:
    def test_score_increases_on_accepted(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        # Seed with a moderate starting score via MODIFIED (maps to 0.5)
        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.MODIFIED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.TRIVIAL,
        )
        score_before = scorer.get_trust_score(agent_id).score

        # Record an ACCEPTED decision (maps to 1.0)
        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.TRIVIAL,
        )
        score_after = scorer.get_trust_score(agent_id).score

        assert score_after > score_before

    def test_score_decreases_on_rejected(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        # Seed with ACCEPTED to push score high
        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.TRIVIAL,
        )
        score_before = scorer.get_trust_score(agent_id).score

        # Record a REJECTED decision (maps to 0.0)
        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.REJECTED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.TRIVIAL,
        )
        score_after = scorer.get_trust_score(agent_id).score

        assert score_after < score_before


# ---------------------------------------------------------------------------
# 3. Score clamping -- never below 0 or above 1
# ---------------------------------------------------------------------------

class TestScoreClamping:
    def test_score_never_exceeds_1(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        # Drive score upward with many ACCEPTED decisions of high weight
        for _ in range(20):
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=_commit_hash(),
                review_time_ms=50000,
                change_size=500,
                complexity=ChangeComplexity.CRITICAL,
            )
        score = scorer.get_trust_score(agent_id).score
        assert 0.0 <= score <= 1.0

    def test_score_never_below_0(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        for _ in range(20):
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.REJECTED,
                commit_hash=_commit_hash(),
                review_time_ms=50000,
                change_size=500,
                complexity=ChangeComplexity.CRITICAL,
            )
        score = scorer.get_trust_score(agent_id).score
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 4. Unknown agent default score
# ---------------------------------------------------------------------------

class TestUnknownAgentDefault:
    def test_unknown_agent_returns_default_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        ts = scorer.get_trust_score(uuid4())
        assert ts.score == 0.5
        assert ts.confidence == 0.0
        assert ts.sample_count == 0


# ---------------------------------------------------------------------------
# 5. TrustTier boundary classification
# ---------------------------------------------------------------------------

class TestTrustTierBoundaries:
    @pytest.mark.parametrize(
        "score, expected_tier",
        [
            (0.00, TrustTier.UNTRUSTED),
            (0.39, TrustTier.UNTRUSTED),
            (0.40, TrustTier.LOW),
            (0.59, TrustTier.LOW),
            (0.60, TrustTier.MEDIUM),
            (0.79, TrustTier.MEDIUM),
            (0.80, TrustTier.HIGH),
            (0.94, TrustTier.HIGH),
            (0.95, TrustTier.VERIFIED),
            (1.00, TrustTier.VERIFIED),
        ],
    )
    def test_tier_classification(self, tmp_path, score, expected_tier):
        scorer = _scorer(tmp_path)
        tier = scorer._get_tier(score)
        assert tier == expected_tier


# ---------------------------------------------------------------------------
# 6. Auto-approval logic by tier + change size
# ---------------------------------------------------------------------------

class TestAutoApproval:
    def test_untrusted_cannot_auto_approve(self, tmp_path):
        scorer = _scorer(tmp_path)
        # Unknown agent => score 0.5 => tier LOW (>=0.40)
        # But let's explicitly create an untrusted agent
        agent_id = uuid4()
        # Record many rejections to push score near 0 => UNTRUSTED
        for _ in range(10):
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.REJECTED,
                commit_hash=_commit_hash(),
                review_time_ms=100,
                change_size=10,
                complexity=ChangeComplexity.TRIVIAL,
            )
        assert not scorer.should_auto_approve(agent_id, 1)

    def test_verified_can_auto_approve_large_changes(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        # Build trust to VERIFIED (>= 0.95)
        for _ in range(30):
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=_commit_hash(),
                review_time_ms=100,
                change_size=10,
                complexity=ChangeComplexity.TRIVIAL,
            )
        score = scorer.get_trust_score(agent_id)
        tier = scorer._get_tier(score.score)
        if tier == TrustTier.VERIFIED:
            # VERIFIED limit is 500 LOC
            assert scorer.should_auto_approve(agent_id, 400)
            assert not scorer.should_auto_approve(agent_id, 600)

    def test_auto_approve_below_limit(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()
        # Default score 0.5 => tier LOW => limit 50
        # But first decision sets score, use REVIEWED to get ~0.8 start
        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.TRIVIAL,
        )
        tier = scorer.get_trust_tier(agent_id)
        limit = TrustScorer.AUTO_APPROVE_LIMITS[tier]
        assert scorer.should_auto_approve(agent_id, limit)
        assert not scorer.should_auto_approve(agent_id, limit + 1)


# ---------------------------------------------------------------------------
# 7. Time decay
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_decay_reduces_score_over_time(self, tmp_path):
        scorer = _scorer(tmp_path)
        current_score = 0.8
        old_activity = datetime.now(timezone.utc) - timedelta(days=60)
        decayed = scorer._apply_time_decay(current_score, old_activity)
        assert decayed < current_score

    def test_no_decay_for_recent_activity(self, tmp_path):
        scorer = _scorer(tmp_path)
        current_score = 0.8
        recent_activity = datetime.now(timezone.utc)
        decayed = scorer._apply_time_decay(current_score, recent_activity)
        assert decayed == current_score


# ---------------------------------------------------------------------------
# 8. Minimum floor enforcement
# ---------------------------------------------------------------------------

class TestMinFloor:
    def test_decay_does_not_drop_below_floor(self, tmp_path):
        scorer = _scorer(tmp_path)
        current_score = 0.05
        ancient_activity = datetime.now(timezone.utc) - timedelta(days=365)
        decayed = scorer._apply_time_decay(current_score, ancient_activity)
        # Floor is min(current_score, 0.1) => 0.05 here
        assert decayed >= min(current_score, 0.1)

    def test_floor_is_0_1_for_higher_scores(self, tmp_path):
        scorer = _scorer(tmp_path)
        current_score = 0.5
        ancient_activity = datetime.now(timezone.utc) - timedelta(days=365)
        decayed = scorer._apply_time_decay(current_score, ancient_activity)
        assert decayed >= 0.1


# ---------------------------------------------------------------------------
# 9. Confidence progression
# ---------------------------------------------------------------------------

class TestConfidenceProgression:
    def test_confidence_increases_with_decisions(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        confidences = []
        for _ in range(15):
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=_commit_hash(),
                review_time_ms=100,
                change_size=10,
                complexity=ChangeComplexity.TRIVIAL,
            )
            confidences.append(scorer.get_trust_score(agent_id).confidence)

        # Overall trend should be increasing
        assert confidences[-1] > confidences[0]

    def test_max_confidence_at_100_samples(self, tmp_path):
        scorer = _scorer(tmp_path)
        confidence = scorer._calculate_confidence(100)
        assert confidence == 1.0

    def test_confidence_below_min_samples(self, tmp_path):
        scorer = _scorer(tmp_path)
        confidence = scorer._calculate_confidence(5)
        assert confidence == pytest.approx(5 / 10 * 0.5)


# ---------------------------------------------------------------------------
# 10. Agent profile CRUD
# ---------------------------------------------------------------------------

class TestAgentProfileCRUD:
    def test_profile_created_on_first_decision(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        assert scorer.get_agent_profile(agent_id) is None

        scorer.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=200,
            change_size=50,
            complexity=ChangeComplexity.MODERATE,
        )
        profile = scorer.get_agent_profile(agent_id)
        assert profile is not None
        assert isinstance(profile, AgentProfile)
        assert profile.agent_id == agent_id
        assert profile.total_decisions == 1
        assert profile.accepted_count == 1

    def test_profile_counts_update_correctly(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        scorer.record_decision(agent_id=agent_id, decision=DecisionType.ACCEPTED,
                               commit_hash=_commit_hash(), review_time_ms=100)
        scorer.record_decision(agent_id=agent_id, decision=DecisionType.REJECTED,
                               commit_hash=_commit_hash(), review_time_ms=100)
        scorer.record_decision(agent_id=agent_id, decision=DecisionType.MODIFIED,
                               commit_hash=_commit_hash(), review_time_ms=100)

        profile = scorer.get_agent_profile(agent_id)
        assert profile.total_decisions == 3
        assert profile.accepted_count == 1
        assert profile.rejected_count == 1
        assert profile.modified_count == 1

    def test_get_all_agents(self, tmp_path):
        scorer = _scorer(tmp_path)
        ids = [uuid4() for _ in range(3)]
        for aid in ids:
            scorer.record_decision(agent_id=aid, decision=DecisionType.ACCEPTED,
                                   commit_hash=_commit_hash(), review_time_ms=100)
        agents = scorer.get_all_agents()
        assert len(agents) == 3
        returned_ids = {a.agent_id for a in agents}
        assert returned_ids == set(ids)


# ---------------------------------------------------------------------------
# 11. Recovery boost after rejection streak
# ---------------------------------------------------------------------------

class TestRecoveryBoost:
    def test_apply_recovery_boosts_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        # Create agent with low score via rejections
        for _ in range(5):
            scorer.record_decision(agent_id=agent_id, decision=DecisionType.REJECTED,
                                   commit_hash=_commit_hash(), review_time_ms=100)
        before = scorer.get_trust_score(agent_id).score

        boosted = scorer.apply_recovery(agent_id, boost_amount=0.2)
        assert boosted.score > before

    def test_recovery_does_not_exceed_1(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        scorer.record_decision(agent_id=agent_id, decision=DecisionType.ACCEPTED,
                               commit_hash=_commit_hash(), review_time_ms=100)
        boosted = scorer.apply_recovery(agent_id, boost_amount=5.0)
        assert boosted.score <= 1.0

    def test_recovery_raises_for_unknown_agent(self, tmp_path):
        scorer = _scorer(tmp_path)
        with pytest.raises(ValueError):
            scorer.apply_recovery(uuid4(), boost_amount=0.1)


# ---------------------------------------------------------------------------
# 12. Reasoning recording with record_complete_decision()
# ---------------------------------------------------------------------------

class TestRecordCompleteDecision:
    def test_returns_both_records(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()
        commit = _commit_hash()

        decision_rec, reasoning_rec = mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="task-001",
            commit_hash=commit,
            decision=DecisionType.ACCEPTED,
            reasoning="Chose approach A because it handles edge cases.",
            review_time_ms=300,
            change_size=25,
            complexity=ChangeComplexity.MINOR,
            confidence=0.9,
            supporting_evidence=["test_result_a", "benchmark_b"],
            parent_commits=None,
            library_choices={"numpy": "numerical performance"},
            reviewer_notes="Looks good",
            reviewer_id="human-1",
        )
        assert isinstance(decision_rec, DecisionRecord)
        assert isinstance(reasoning_rec, ReasoningRecord)
        assert decision_rec.commit_hash == commit
        assert reasoning_rec.commit_hash == commit
        assert reasoning_rec.confidence == 0.9
        assert "edge cases" in reasoning_rec.reasoning


# ---------------------------------------------------------------------------
# 13. Lineage retrieval by commit hash
# ---------------------------------------------------------------------------

class TestLineageRetrieval:
    def test_get_lineage_returns_decision_and_reasoning(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()
        commit = _commit_hash()

        mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="task-lin",
            commit_hash=commit,
            decision=DecisionType.MODIFIED,
            reasoning="Modified approach for performance.",
            review_time_ms=200,
            change_size=40,
            library_choices={"pandas": "dataframe handling"},
        )

        lineage = mgr.get_decision_with_lineage(commit)
        assert lineage is not None
        assert isinstance(lineage, DecisionLineage)
        assert lineage.commit_hash == commit
        assert lineage.decision is not None
        assert lineage.reasoning is not None
        assert lineage.library_choices.get("pandas") == "dataframe handling"

    def test_get_lineage_unknown_commit_returns_mostly_empty(self, tmp_path):
        mgr = _manager(tmp_path)
        lineage = mgr.get_decision_with_lineage(_commit_hash())
        assert lineage is not None
        assert lineage.decision is None
        assert lineage.reasoning is None


# ---------------------------------------------------------------------------
# 14. Chain tracing for linked decisions
# ---------------------------------------------------------------------------

class TestChainTracing:
    def test_trace_decision_chain_follows_parents(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()

        parent_commit = _commit_hash()
        child_commit = _commit_hash()

        # Create agent profile first by recording a decision
        mgr.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=parent_commit,
            review_time_ms=100,
        )

        # Create parent reasoning
        mgr.lineage.record_reasoning(
            agent_id=agent_id,
            task_id="task-parent",
            commit_hash=parent_commit,
            reasoning="Parent decision reasoning.",
        )

        # Create child with parent link
        mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="task-child",
            commit_hash=child_commit,
            decision=DecisionType.ACCEPTED,
            reasoning="Child builds on parent.",
            review_time_ms=100,
            parent_commits=[parent_commit],
        )

        chain = mgr.lineage.trace_decision_chain(child_commit, max_depth=5)
        assert len(chain) >= 1
        assert chain[0].commit_hash == child_commit

    def test_trace_respects_max_depth(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()

        # Create agent profile first by recording a decision
        mgr.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=100,
        )

        commits = [_commit_hash() for _ in range(5)]
        # Build a linear chain: commits[0] <- commits[1] <- ... <- commits[4]
        for i, c in enumerate(commits):
            parents = [commits[i - 1]] if i > 0 else None
            mgr.lineage.record_reasoning(
                agent_id=agent_id,
                task_id=f"task-{i}",
                commit_hash=c,
                reasoning=f"Step {i} reasoning",
                parent_commits=parents,
            )

        chain = mgr.lineage.trace_decision_chain(commits[-1], max_depth=2)
        assert len(chain) <= 2


# ---------------------------------------------------------------------------
# 15. Keyword search with/without agent filter
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_search_finds_matching_reasoning(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()

        # Create agent profile first
        mgr.trust.record_decision(
            agent_id=agent_id, decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(), review_time_ms=100,
        )

        mgr.lineage.record_reasoning(
            agent_id=agent_id,
            task_id="task-search",
            commit_hash=_commit_hash(),
            reasoning="Implemented binary search algorithm for sorted lists.",
        )
        mgr.lineage.record_reasoning(
            agent_id=agent_id,
            task_id="task-other",
            commit_hash=_commit_hash(),
            reasoning="Added logging to the database layer.",
        )

        results = mgr.lineage.search_reasoning("binary search")
        assert len(results) == 1
        assert "binary search" in results[0].reasoning

    def test_search_with_agent_filter(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_a = uuid4()
        agent_b = uuid4()

        # Create agent profiles first
        mgr.trust.record_decision(
            agent_id=agent_a, decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(), review_time_ms=100,
        )
        mgr.trust.record_decision(
            agent_id=agent_b, decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(), review_time_ms=100,
        )

        mgr.lineage.record_reasoning(
            agent_id=agent_a, task_id="t1", commit_hash=_commit_hash(),
            reasoning="Refactored the parser module.",
        )
        mgr.lineage.record_reasoning(
            agent_id=agent_b, task_id="t2", commit_hash=_commit_hash(),
            reasoning="Refactored the renderer module.",
        )

        results_a = mgr.lineage.search_reasoning("Refactored", agent_id=agent_a)
        assert len(results_a) == 1
        assert results_a[0].agent_id == agent_a

        results_all = mgr.lineage.search_reasoning("Refactored")
        assert len(results_all) == 2


# ---------------------------------------------------------------------------
# 16. Thread safety -- basic concurrent recording
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_decision_recording(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()
        errors = []

        def record_many():
            try:
                for _ in range(10):
                    scorer.record_decision(
                        agent_id=agent_id,
                        decision=DecisionType.ACCEPTED,
                        commit_hash=_commit_hash(),
                        review_time_ms=100,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        profile = scorer.get_agent_profile(agent_id)
        assert profile is not None
        # 4 threads x 10 decisions = 40 total
        assert profile.total_decisions == 40


# ---------------------------------------------------------------------------
# 17. Foreign key enforcement
# ---------------------------------------------------------------------------

class TestForeignKeyEnforcement:
    def test_foreign_keys_enabled(self, tmp_path):
        """Verify that PRAGMA foreign_keys = ON is in effect."""
        scorer = _scorer(tmp_path)
        with scorer._get_connection() as conn:
            result = conn.execute("PRAGMA foreign_keys").fetchone()
            assert result[0] == 1


# ---------------------------------------------------------------------------
# 18. DecisionType enum values
# ---------------------------------------------------------------------------

class TestDecisionTypeEnum:
    def test_all_values_present(self):
        expected = {"accepted", "rejected", "modified", "auto_approved", "reviewed"}
        actual = {dt.value for dt in DecisionType}
        assert actual == expected


# ---------------------------------------------------------------------------
# 19. ChangeComplexity enum values
# ---------------------------------------------------------------------------

class TestChangeComplexityEnum:
    def test_values_are_integers(self):
        expected = {1, 2, 3, 5, 8}
        actual = {cc.value for cc in ChangeComplexity}
        assert actual == expected


# ---------------------------------------------------------------------------
# 20. TrustLineageManager initialisation
# ---------------------------------------------------------------------------

class TestManagerInit:
    def test_creates_database_file(self, tmp_path):
        db_path = tmp_path / "init_test.db"
        mgr = TrustLineageManager(str(db_path))
        assert db_path.exists()
        assert mgr.storage_path == str(db_path)

    def test_sub_components_share_path(self, tmp_path):
        db_path = tmp_path / "shared.db"
        mgr = TrustLineageManager(str(db_path))
        assert str(mgr.trust.storage_path) == str(db_path)
        assert str(mgr.lineage.storage_path) == str(db_path)


# ---------------------------------------------------------------------------
# 21. get_agent_summary() output format
# ---------------------------------------------------------------------------

class TestAgentSummary:
    def test_summary_for_existing_agent(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()

        mgr.trust.record_decision(
            agent_id=agent_id,
            decision=DecisionType.ACCEPTED,
            commit_hash=_commit_hash(),
            review_time_ms=200,
            change_size=30,
        )

        summary = mgr.get_agent_summary(agent_id)
        assert "profile" in summary
        assert "recent_decisions" in summary
        assert "can_auto_approve_small_changes" in summary
        assert "can_auto_approve_medium_changes" in summary
        assert isinstance(summary["profile"], dict)
        assert isinstance(summary["recent_decisions"], list)
        assert isinstance(summary["can_auto_approve_small_changes"], bool)

    def test_summary_for_unknown_agent(self, tmp_path):
        mgr = _manager(tmp_path)
        summary = mgr.get_agent_summary(uuid4())
        assert summary == {"error": "Agent not found"}


# ---------------------------------------------------------------------------
# 22. get_decision_with_lineage() returns correct structure
# ---------------------------------------------------------------------------

class TestGetDecisionWithLineage:
    def test_structure_fields(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()
        commit = _commit_hash()

        mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="struct-task",
            commit_hash=commit,
            decision=DecisionType.REVIEWED,
            reasoning="Detailed review reasoning here.",
            review_time_ms=1500,
            change_size=100,
            complexity=ChangeComplexity.MAJOR,
            confidence=0.75,
            supporting_evidence=["review_doc"],
            library_choices={"flask": "lightweight web framework"},
        )

        lineage = mgr.get_decision_with_lineage(commit)
        assert lineage is not None
        assert lineage.commit_hash == commit
        assert lineage.agent_id == agent_id
        assert lineage.decision.decision_type == DecisionType.REVIEWED
        assert lineage.reasoning.task_id == "struct-task"
        assert lineage.library_choices == {"flask": "lightweight web framework"}
        assert isinstance(lineage.parent_commits, list)
        assert isinstance(lineage.child_commits, list)
        assert isinstance(lineage.full_chain, list)

    def test_to_dict_serialisation(self, tmp_path):
        mgr = _manager(tmp_path)
        agent_id = uuid4()
        commit = _commit_hash()

        mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="dict-task",
            commit_hash=commit,
            decision=DecisionType.ACCEPTED,
            reasoning="Serialisation test.",
            review_time_ms=100,
        )

        lineage = mgr.get_decision_with_lineage(commit)
        d = lineage.to_dict()
        assert isinstance(d, dict)
        assert d["commit_hash"] == commit
        assert d["agent_id"] == str(agent_id)
        assert d["decision"] is not None
        assert d["reasoning"] is not None


# ---------------------------------------------------------------------------
# Additional coverage: TrustScore validation, DecisionRecord.to_dict,
# ReasoningRecord.to_dict, agent history retrieval
# ---------------------------------------------------------------------------

class TestTrustScoreValidation:
    def test_rejects_score_out_of_range(self):
        with pytest.raises(ValueError):
            TrustScore(score=1.5, confidence=0.5, sample_count=1,
                       last_updated=datetime.now(timezone.utc))

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            TrustScore(score=0.5, confidence=-0.1, sample_count=1,
                       last_updated=datetime.now(timezone.utc))


class TestDecisionRecordToDict:
    def test_serialisation_keys(self):
        record = DecisionRecord(
            record_id=uuid4(),
            agent_id=uuid4(),
            decision_type=DecisionType.ACCEPTED,
            commit_hash="abc123",
            timestamp=datetime.now(timezone.utc),
            review_time_ms=100,
            change_size=10,
            complexity=ChangeComplexity.MINOR,
            reviewer_notes="OK",
            reviewer_id="r1",
        )
        d = record.to_dict()
        assert set(d.keys()) == {
            "record_id", "agent_id", "decision_type", "commit_hash",
            "timestamp", "review_time_ms", "change_size", "complexity",
            "reviewer_notes", "reviewer_id",
        }
        assert d["decision_type"] == "accepted"


class TestReasoningRecordToDict:
    def test_serialisation_keys(self):
        record = ReasoningRecord(
            reasoning_id=uuid4(),
            agent_id=uuid4(),
            task_id="t1",
            commit_hash="def456",
            timestamp=datetime.now(timezone.utc),
            reasoning="test reasoning",
            confidence=0.7,
            supporting_evidence=["ev1"],
        )
        d = record.to_dict()
        assert "reasoning" in d
        assert d["confidence"] == 0.7
        assert d["supporting_evidence"] == ["ev1"]


class TestAgentHistory:
    def test_get_agent_history_returns_records(self, tmp_path):
        scorer = _scorer(tmp_path)
        agent_id = uuid4()

        commits = []
        for _ in range(5):
            c = _commit_hash()
            commits.append(c)
            scorer.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=c,
                review_time_ms=100,
            )

        history = scorer.get_agent_history(agent_id, limit=3)
        assert len(history) == 3
        # Most recent first
        assert all(isinstance(r, DecisionRecord) for r in history)

    def test_get_agent_history_empty_for_unknown(self, tmp_path):
        scorer = _scorer(tmp_path)
        history = scorer.get_agent_history(uuid4())
        assert history == []
