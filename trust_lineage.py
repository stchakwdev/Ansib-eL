"""
Trust Scoring and Lineage Tracking System for Ansib-eL
AI-Native Version Control System

This module provides:
- Agent trust score calculation with exponential moving average
- Decision lineage tracking with Chain-of-Thought reasoning
- Trust-based review requirement adjustment
- Immutable provenance records

Author: Ansib-eL Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
from contextlib import contextmanager
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class TrustTier(Enum):
    """Trust tiers based on cumulative agent performance."""
    UNTRUSTED = "untrusted"      # New agents or poor performers
    LOW = "low"                  # Some history but below threshold
    MEDIUM = "medium"            # Acceptable performance
    HIGH = "high"                # Good track record
    VERIFIED = "verified"        # Exceptional, long-term performance


class DecisionType(Enum):
    """Types of decisions that can be recorded."""
    ACCEPTED = "accepted"        # Change was accepted as-is
    REJECTED = "rejected"        # Change was rejected
    MODIFIED = "modified"        # Change was accepted with modifications
    AUTO_APPROVED = "auto_approved"  # Automatically approved based on trust
    REVIEWED = "reviewed"        # Human reviewed the change


class ChangeComplexity(Enum):
    """Complexity levels for changes."""
    TRIVIAL = 1      # Documentation, comments
    MINOR = 2        # Small fixes, config changes
    MODERATE = 3     # Feature additions
    MAJOR = 5        # Refactoring, architecture changes
    CRITICAL = 8     # Security, core system changes


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class TrustScore:
    """Immutable trust score with confidence metric.
    
    Attributes:
        score: Float between 0.0 and 1.0 representing trust level
        confidence: Float between 0.0 and 1.0 based on sample size
        sample_count: Number of decisions contributing to score
        last_updated: Timestamp of last score update
    """
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0 based on sample size
    sample_count: int
    last_updated: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass(frozen=True)
class DecisionRecord:
    """Immutable record of a single decision.
    
    Attributes:
        record_id: Unique identifier for this record
        agent_id: UUID of the agent making the decision
        decision_type: Type of decision (ACCEPTED, REJECTED, etc.)
        commit_hash: Git commit hash associated with decision
        timestamp: When the decision was recorded
        review_time_ms: Time taken for review in milliseconds
        change_size: Lines of code changed
        complexity: Complexity level of the change
        reviewer_notes: Optional human reviewer notes
        reviewer_id: Optional reviewer identifier
    """
    record_id: UUID
    agent_id: UUID
    decision_type: DecisionType
    commit_hash: str
    timestamp: datetime
    review_time_ms: int
    change_size: int  # Lines of code changed
    complexity: ChangeComplexity
    reviewer_notes: Optional[str] = None
    reviewer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": str(self.record_id),
            "agent_id": str(self.agent_id),
            "decision_type": self.decision_type.value,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp.isoformat(),
            "review_time_ms": self.review_time_ms,
            "change_size": self.change_size,
            "complexity": self.complexity.value,
            "reviewer_notes": self.reviewer_notes,
            "reviewer_id": self.reviewer_id
        }


@dataclass(frozen=True)
class ReasoningRecord:
    """Immutable Chain-of-Thought reasoning record.
    
    Attributes:
        reasoning_id: Unique identifier for this reasoning
        agent_id: UUID of the agent providing reasoning
        task_id: Task identifier
        commit_hash: Git commit hash
        timestamp: When reasoning was recorded
        reasoning: Full reasoning text (Chain-of-Thought)
        confidence: Confidence in the reasoning (0.0 to 1.0)
        supporting_evidence: List of evidence references
    """
    reasoning_id: UUID
    agent_id: UUID
    task_id: str
    commit_hash: str
    timestamp: datetime
    reasoning: str
    confidence: float
    supporting_evidence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reasoning_id": str(self.reasoning_id),
            "agent_id": str(self.agent_id),
            "task_id": self.task_id,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass
class DecisionLineage:
    """Complete decision chain with reasoning for a commit.
    
    Attributes:
        commit_hash: Git commit hash
        agent_id: UUID of the agent
        decision: Associated decision record
        reasoning: Associated reasoning record
        parent_commits: List of parent commit hashes
        child_commits: List of child commit hashes
        library_choices: Dict mapping library names to reasoning
        full_chain: Complete trace of decision chain
    """
    commit_hash: str
    agent_id: UUID
    decision: Optional[DecisionRecord]
    reasoning: Optional[ReasoningRecord]
    parent_commits: List[str]
    child_commits: List[str]
    library_choices: Dict[str, str]  # library_name -> reasoning
    full_chain: List[Dict[str, Any]]  # Complete trace
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "commit_hash": self.commit_hash,
            "agent_id": str(self.agent_id),
            "decision": self.decision.to_dict() if self.decision else None,
            "reasoning": self.reasoning.to_dict() if self.reasoning else None,
            "parent_commits": self.parent_commits,
            "child_commits": self.child_commits,
            "library_choices": self.library_choices,
            "full_chain": self.full_chain
        }


@dataclass
class AgentProfile:
    """Complete agent profile with trust metrics.
    
    Attributes:
        agent_id: UUID of the agent
        name: Human-readable agent name
        created_at: When the agent was first seen
        trust_score: Current trust score
        tier: Trust tier classification
        total_decisions: Total number of decisions made
        accepted_count: Number of accepted decisions
        rejected_count: Number of rejected decisions
        modified_count: Number of modified decisions
        avg_review_time_ms: Average review time in milliseconds
        last_activity: Timestamp of last activity
    """
    agent_id: UUID
    name: str
    created_at: datetime
    trust_score: TrustScore
    tier: TrustTier
    total_decisions: int
    accepted_count: int
    rejected_count: int
    modified_count: int
    avg_review_time_ms: float
    last_activity: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "trust_score": {
                "score": self.trust_score.score,
                "confidence": self.trust_score.confidence,
                "sample_count": self.trust_score.sample_count
            },
            "tier": self.tier.value,
            "total_decisions": self.total_decisions,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "modified_count": self.modified_count,
            "avg_review_time_ms": self.avg_review_time_ms,
            "last_activity": self.last_activity.isoformat()
        }


# ============================================================================
# DATABASE SCHEMA
# ============================================================================

SCHEMA_SQL = """
-- Agent profiles table
CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    trust_score REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'untrusted',
    total_decisions INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    modified_count INTEGER NOT NULL DEFAULT 0,
    avg_review_time_ms REAL DEFAULT 0.0,
    last_activity TEXT,
    last_updated TEXT NOT NULL
);

-- Decision records table (immutable)
CREATE TABLE IF NOT EXISTS decision_records (
    record_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    review_time_ms INTEGER NOT NULL,
    change_size INTEGER NOT NULL DEFAULT 0,
    complexity INTEGER NOT NULL DEFAULT 2,
    reviewer_notes TEXT,
    reviewer_id TEXT,
    weight REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id)
);

-- Reasoning records table (immutable)
CREATE TABLE IF NOT EXISTS reasoning_records (
    reasoning_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    supporting_evidence TEXT,  -- JSON array
    FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id)
);

-- Lineage chain table
CREATE TABLE IF NOT EXISTS lineage_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash TEXT NOT NULL,
    parent_commit TEXT,
    child_commit TEXT,
    agent_id TEXT NOT NULL,
    record_id TEXT,
    reasoning_id TEXT,
    FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id),
    FOREIGN KEY (record_id) REFERENCES decision_records(record_id),
    FOREIGN KEY (reasoning_id) REFERENCES reasoning_records(reasoning_id)
);

-- Library choices table
CREATE TABLE IF NOT EXISTS library_choices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash TEXT NOT NULL,
    library_name TEXT NOT NULL,
    reasoning_id TEXT NOT NULL,
    reasoning_summary TEXT,
    FOREIGN KEY (reasoning_id) REFERENCES reasoning_records(reasoning_id)
);

-- Trust decay log
CREATE TABLE IF NOT EXISTS trust_decay_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    old_score REAL NOT NULL,
    new_score REAL NOT NULL,
    decay_reason TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agent_profiles(agent_id)
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decision_records(agent_id);
CREATE INDEX IF NOT EXISTS idx_decisions_commit ON decision_records(commit_hash);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decision_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_reasoning_agent ON reasoning_records(agent_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_commit ON reasoning_records(commit_hash);
CREATE INDEX IF NOT EXISTS idx_lineage_commit ON lineage_chain(commit_hash);
CREATE INDEX IF NOT EXISTS idx_library_commit ON library_choices(commit_hash);
"""


# ============================================================================
# TRUST SCORER CLASS
# ============================================================================

class TrustScorer:
    """
    Calculates and manages agent trust scores with exponential moving average,
    time decay, and recovery mechanisms.
    
    The trust algorithm uses:
    - Exponential Moving Average (EMA) for score smoothing
    - Time decay for inactive agents
    - Weighted decisions based on change size and complexity
    - Recovery mechanism for improving agents
    """
    
    # Configuration constants
    EMA_ALPHA = 0.3  # Exponential moving average smoothing factor
    DECAY_HALF_LIFE_DAYS = 30  # Trust decays by half after 30 days of inactivity
    RECOVERY_RATE = 0.05  # Rate of trust recovery per positive decision
    MIN_SAMPLES_FOR_CONFIDENCE = 10
    MAX_SAMPLES_FOR_FULL_CONFIDENCE = 100
    
    # Tier thresholds
    TIER_THRESHOLDS = {
        TrustTier.VERIFIED: 0.95,
        TrustTier.HIGH: 0.80,
        TrustTier.MEDIUM: 0.60,
        TrustTier.LOW: 0.40,
        TrustTier.UNTRUSTED: 0.0
    }
    
    # Auto-approval thresholds by tier (lines of code)
    AUTO_APPROVE_LIMITS = {
        TrustTier.VERIFIED: 500,   # Lines of code
        TrustTier.HIGH: 200,
        TrustTier.MEDIUM: 100,
        TrustTier.LOW: 50,
        TrustTier.UNTRUSTED: 0
    }
    
    def __init__(self, storage_path: str):
        """Initialize TrustScorer with SQLite storage.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.storage_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _calculate_weight(self, change_size: int, complexity: ChangeComplexity, 
                          review_time_ms: int) -> float:
        """Calculate decision weight based on change characteristics.
        
        Larger changes and higher complexity decisions get more weight
        in the trust calculation.
        
        Args:
            change_size: Lines of code changed
            complexity: Complexity level of the change
            review_time_ms: Time taken for review
            
        Returns:
            Weight factor (1.0 to ~5.0)
        """
        # Larger changes get more weight
        size_factor = min(change_size / 100, 3.0)  # Cap at 3x
        # Higher complexity gets more weight
        complexity_factor = complexity.value
        # Longer review time indicates more scrutiny
        time_factor = min(review_time_ms / 10000, 2.0)  # Cap at 2x
        
        return 1.0 + (size_factor * 0.3) + (complexity_factor * 0.2) + (time_factor * 0.1)
    
    def _calculate_confidence(self, sample_count: int) -> float:
        """Calculate confidence based on sample size.
        
        Confidence increases with more samples, reaching full
        confidence at MAX_SAMPLES_FOR_FULL_CONFIDENCE.
        
        Args:
            sample_count: Number of decisions
            
        Returns:
            Confidence value between 0.0 and 1.0
        """
        if sample_count >= self.MAX_SAMPLES_FOR_FULL_CONFIDENCE:
            return 1.0
        elif sample_count <= self.MIN_SAMPLES_FOR_CONFIDENCE:
            return sample_count / self.MIN_SAMPLES_FOR_CONFIDENCE * 0.5
        else:
            # Linear interpolation between min and max
            progress = (sample_count - self.MIN_SAMPLES_FOR_CONFIDENCE) / \
                      (self.MAX_SAMPLES_FOR_FULL_CONFIDENCE - self.MIN_SAMPLES_FOR_CONFIDENCE)
            return 0.5 + (progress * 0.5)
    
    def _apply_time_decay(self, current_score: float, last_activity: datetime) -> float:
        """Apply time-based decay to trust score.
        
        Agents that are inactive have their trust scores decay over time.
        This encourages continued good performance.
        
        Args:
            current_score: Current trust score
            last_activity: Timestamp of last activity
            
        Returns:
            Decayed trust score
        """
        days_inactive = (datetime.utcnow() - last_activity).days
        if days_inactive <= 0:
            return current_score
        
        # Exponential decay: score * (0.5 ^ (days / half_life))
        decay_factor = 0.5 ** (days_inactive / self.DECAY_HALF_LIFE_DAYS)
        decayed_score = current_score * decay_factor
        
        # Ensure score doesn't drop below 0.1 due to inactivity alone
        return max(decayed_score, min(current_score, 0.1))
    
    def _get_tier(self, score: float) -> TrustTier:
        """Determine trust tier based on score.
        
        Args:
            score: Trust score (0.0 to 1.0)
            
        Returns:
            TrustTier classification
        """
        for tier, threshold in sorted(self.TIER_THRESHOLDS.items(), 
                                       key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return tier
        return TrustTier.UNTRUSTED
    
    def _update_agent_profile(self, conn: sqlite3.Connection, agent_id: UUID,
                              decision_value: float, weight: float,
                              review_time_ms: int) -> Tuple[float, TrustTier]:
        """Update agent profile with new decision.
        
        Args:
            conn: Database connection
            agent_id: UUID of the agent
            decision_value: Numeric value of decision (0.0 to 1.0)
            weight: Calculated weight for this decision
            review_time_ms: Time taken for review
            
        Returns:
            Tuple of (new_score, new_tier)
        """
        cursor = conn.cursor()
        
        # Get current profile
        cursor.execute(
            """SELECT trust_score, sample_count, accepted_count, rejected_count, 
                      modified_count, avg_review_time_ms, last_activity
               FROM agent_profiles WHERE agent_id = ?""",
            (str(agent_id),)
        )
        row = cursor.fetchone()
        
        if row:
            current_score = row["trust_score"]
            sample_count = row["sample_count"]
            avg_time = row["avg_review_time_ms"] or 0.0
            last_activity = datetime.fromisoformat(row["last_activity"]) if row["last_activity"] else datetime.utcnow()
            
            # Apply time decay
            current_score = self._apply_time_decay(current_score, last_activity)
            
            # Update EMA with new decision
            # Weight the new decision by its calculated weight
            weighted_decision = decision_value * min(weight, 2.0)
            new_score = (self.EMA_ALPHA * weighted_decision + 
                        (1 - self.EMA_ALPHA) * current_score)
            
            # Clamp score to valid range [0, 1]
            new_score = max(0.0, min(1.0, new_score))
            
            new_sample_count = sample_count + 1
            new_confidence = self._calculate_confidence(new_sample_count)
            new_tier = self._get_tier(new_score)
            
            # Update moving average review time
            new_avg_time = ((avg_time * sample_count) + review_time_ms) / new_sample_count
            
            cursor.execute(
                """UPDATE agent_profiles 
                   SET trust_score = ?, confidence = ?, sample_count = ?,
                       tier = ?, avg_review_time_ms = ?, last_activity = ?, last_updated = ?
                   WHERE agent_id = ?""",
                (new_score, new_confidence, new_sample_count, new_tier.value,
                 new_avg_time, datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
                 str(agent_id))
            )
        else:
            # New agent - initialize with first decision
            new_score = decision_value
            new_confidence = self._calculate_confidence(1)
            new_tier = self._get_tier(new_score)
            
            cursor.execute(
                """INSERT INTO agent_profiles 
                   (agent_id, name, created_at, trust_score, confidence, sample_count,
                    tier, total_decisions, avg_review_time_ms, last_activity, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(agent_id), f"agent_{str(agent_id)[:8]}", datetime.utcnow().isoformat(),
                 new_score, new_confidence, 1, new_tier.value, 1, review_time_ms,
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )
        
        conn.commit()
        return new_score, new_tier
    
    def record_decision(
        self,
        agent_id: UUID,
        decision: DecisionType,
        commit_hash: str,
        review_time_ms: int,
        change_size: int = 0,
        complexity: ChangeComplexity = ChangeComplexity.MODERATE,
        reviewer_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ) -> DecisionRecord:
        """
        Record a decision and update agent trust score.
        
        Args:
            agent_id: UUID of the agent
            decision: Type of decision (ACCEPTED, REJECTED, etc.)
            commit_hash: Git commit hash
            review_time_ms: Time taken for review in milliseconds
            change_size: Lines of code changed
            complexity: Complexity of the change
            reviewer_notes: Optional human reviewer notes
            reviewer_id: Optional reviewer identifier
            
        Returns:
            DecisionRecord: The recorded decision
        """
        with self._lock:
            with self._get_connection() as conn:
                record_id = uuid4()
                timestamp = datetime.utcnow()
                
                # Map decision to numeric value for scoring
                decision_values = {
                    DecisionType.ACCEPTED: 1.0,
                    DecisionType.AUTO_APPROVED: 1.0,
                    DecisionType.REVIEWED: 0.8,
                    DecisionType.MODIFIED: 0.5,
                    DecisionType.REJECTED: 0.0
                }
                decision_value = decision_values.get(decision, 0.5)
                
                # Calculate weight
                weight = self._calculate_weight(change_size, complexity, review_time_ms)
                
                # Update agent profile
                new_score, new_tier = self._update_agent_profile(
                    conn, agent_id, decision_value, weight, review_time_ms
                )
                
                # Insert decision record
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO decision_records 
                       (record_id, agent_id, decision_type, commit_hash, timestamp,
                        review_time_ms, change_size, complexity, reviewer_notes, 
                        reviewer_id, weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(record_id), str(agent_id), decision.value, commit_hash,
                     timestamp.isoformat(), review_time_ms, change_size, 
                     complexity.value, reviewer_notes, reviewer_id, weight)
                )
                
                # Update decision counts
                if decision == DecisionType.ACCEPTED or decision == DecisionType.AUTO_APPROVED:
                    conn.execute(
                        "UPDATE agent_profiles SET accepted_count = accepted_count + 1 WHERE agent_id = ?",
                        (str(agent_id),)
                    )
                elif decision == DecisionType.REJECTED:
                    conn.execute(
                        "UPDATE agent_profiles SET rejected_count = rejected_count + 1 WHERE agent_id = ?",
                        (str(agent_id),)
                    )
                elif decision == DecisionType.MODIFIED:
                    conn.execute(
                        "UPDATE agent_profiles SET modified_count = modified_count + 1 WHERE agent_id = ?",
                        (str(agent_id),)
                    )
                
                conn.execute(
                    "UPDATE agent_profiles SET total_decisions = total_decisions + 1 WHERE agent_id = ?",
                    (str(agent_id),)
                )
                
                conn.commit()
                
                logger.info(f"Recorded {decision.value} for agent {agent_id} on commit {commit_hash[:8]}")
                
                return DecisionRecord(
                    record_id=record_id,
                    agent_id=agent_id,
                    decision_type=decision,
                    commit_hash=commit_hash,
                    timestamp=timestamp,
                    review_time_ms=review_time_ms,
                    change_size=change_size,
                    complexity=complexity,
                    reviewer_notes=reviewer_notes,
                    reviewer_id=reviewer_id
                )
    
    def get_trust_score(self, agent_id: UUID) -> TrustScore:
        """Get current trust score for an agent.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            TrustScore with current metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT trust_score, confidence, sample_count, last_activity, last_updated
                   FROM agent_profiles WHERE agent_id = ?""",
                (str(agent_id),)
            )
            row = cursor.fetchone()
            
            if not row:
                # Return default score for unknown agents
                return TrustScore(
                    score=0.5,
                    confidence=0.0,
                    sample_count=0,
                    last_updated=datetime.utcnow()
                )
            
            score = row["trust_score"]
            last_activity = datetime.fromisoformat(row["last_activity"]) if row["last_activity"] else datetime.utcnow()
            
            # Apply time decay to returned score
            decayed_score = self._apply_time_decay(score, last_activity)
            
            return TrustScore(
                score=decayed_score,
                confidence=row["confidence"],
                sample_count=row["sample_count"],
                last_updated=datetime.fromisoformat(row["last_updated"])
            )
    
    def get_trust_tier(self, agent_id: UUID) -> TrustTier:
        """Get trust tier for an agent.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            TrustTier classification
        """
        score = self.get_trust_score(agent_id)
        return self._get_tier(score.score)
    
    def should_auto_approve(self, agent_id: UUID, change_size: int) -> bool:
        """
        Determine if a change should be auto-approved based on agent trust.
        
        Args:
            agent_id: UUID of the agent
            change_size: Lines of code in the change
            
        Returns:
            bool: True if change can be auto-approved
        """
        tier = self.get_trust_tier(agent_id)
        limit = self.AUTO_APPROVE_LIMITS.get(tier, 0)
        return change_size <= limit
    
    def get_agent_history(self, agent_id: UUID, limit: int = 100) -> List[DecisionRecord]:
        """Get decision history for an agent.
        
        Args:
            agent_id: UUID of the agent
            limit: Maximum number of records to return
            
        Returns:
            List of DecisionRecord objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM decision_records 
                   WHERE agent_id = ? 
                   ORDER BY timestamp DESC 
                   LIMIT ?""",
                (str(agent_id), limit)
            )
            
            records = []
            for row in cursor.fetchall():
                records.append(DecisionRecord(
                    record_id=UUID(row["record_id"]),
                    agent_id=UUID(row["agent_id"]),
                    decision_type=DecisionType(row["decision_type"]),
                    commit_hash=row["commit_hash"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    review_time_ms=row["review_time_ms"],
                    change_size=row["change_size"],
                    complexity=ChangeComplexity(row["complexity"]),
                    reviewer_notes=row["reviewer_notes"],
                    reviewer_id=row["reviewer_id"]
                ))
            
            return records
    
    def get_agent_profile(self, agent_id: UUID) -> Optional[AgentProfile]:
        """Get complete agent profile.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            AgentProfile or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM agent_profiles WHERE agent_id = ?""",
                (str(agent_id),)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return AgentProfile(
                agent_id=UUID(row["agent_id"]),
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                trust_score=TrustScore(
                    score=row["trust_score"],
                    confidence=row["confidence"],
                    sample_count=row["sample_count"],
                    last_updated=datetime.fromisoformat(row["last_updated"])
                ),
                tier=TrustTier(row["tier"]),
                total_decisions=row["total_decisions"],
                accepted_count=row["accepted_count"],
                rejected_count=row["rejected_count"],
                modified_count=row["modified_count"],
                avg_review_time_ms=row["avg_review_time_ms"] or 0.0,
                last_activity=datetime.fromisoformat(row["last_activity"]) if row["last_activity"] else datetime.utcnow()
            )
    
    def get_all_agents(self) -> List[AgentProfile]:
        """Get profiles for all agents.
        
        Returns:
            List of AgentProfile objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT agent_id FROM agent_profiles")
            return [self.get_agent_profile(UUID(row["agent_id"])) for row in cursor.fetchall()]
    
    def apply_recovery(self, agent_id: UUID, boost_amount: float = 0.1) -> TrustScore:
        """
        Apply manual trust recovery boost.
        
        This can be used to manually boost an agent's trust score
        after remediation or training.
        
        Args:
            agent_id: UUID of the agent
            boost_amount: Amount to boost trust score (0.0 to 1.0)
            
        Returns:
            Updated TrustScore
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trust_score FROM agent_profiles WHERE agent_id = ?",
                    (str(agent_id),)
                )
                row = cursor.fetchone()
                
                if not row:
                    raise ValueError(f"Agent {agent_id} not found")
                
                new_score = min(row["trust_score"] + boost_amount, 1.0)
                new_tier = self._get_tier(new_score)
                
                cursor.execute(
                    """UPDATE agent_profiles 
                       SET trust_score = ?, tier = ?, last_updated = ?
                       WHERE agent_id = ?""",
                    (new_score, new_tier.value, datetime.utcnow().isoformat(), str(agent_id))
                )
                conn.commit()
                
                logger.info(f"Applied recovery boost to agent {agent_id}: +{boost_amount}")
                
                return self.get_trust_score(agent_id)


# ============================================================================
# LINEAGE TRACKER CLASS
# ============================================================================

class LineageTracker:
    """
    Tracks decision lineage with Chain-of-Thought reasoning capture.
    Provides immutable provenance records for audit and analysis.
    """
    
    def __init__(self, storage_path: str):
        """Initialize LineageTracker with SQLite storage.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database (schema already created by TrustScorer)."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.storage_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def record_reasoning(
        self,
        agent_id: UUID,
        task_id: str,
        commit_hash: str,
        reasoning: str,
        confidence: float = 0.5,
        supporting_evidence: Optional[List[str]] = None,
        parent_commits: Optional[List[str]] = None,
        library_choices: Optional[Dict[str, str]] = None
    ) -> ReasoningRecord:
        """
        Record Chain-of-Thought reasoning for a decision.
        
        Args:
            agent_id: UUID of the agent
            task_id: Task identifier
            commit_hash: Git commit hash
            reasoning: Full reasoning text
            confidence: Confidence in the reasoning (0.0 to 1.0)
            supporting_evidence: List of evidence references
            parent_commits: Parent commit hashes
            library_choices: Dict of library_name -> reasoning
            
        Returns:
            ReasoningRecord: The recorded reasoning
        """
        with self._lock:
            with self._get_connection() as conn:
                reasoning_id = uuid4()
                timestamp = datetime.utcnow()
                evidence = supporting_evidence or []
                
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO reasoning_records 
                       (reasoning_id, agent_id, task_id, commit_hash, timestamp,
                        reasoning, confidence, supporting_evidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(reasoning_id), str(agent_id), task_id, commit_hash,
                     timestamp.isoformat(), reasoning, confidence, 
                     json.dumps(evidence))
                )
                
                # Record lineage chain
                if parent_commits:
                    for parent in parent_commits:
                        cursor.execute(
                            """INSERT INTO lineage_chain 
                               (commit_hash, parent_commit, agent_id, reasoning_id)
                               VALUES (?, ?, ?, ?)""",
                            (commit_hash, parent, str(agent_id), str(reasoning_id))
                        )
                
                # Record library choices
                if library_choices:
                    for lib_name, lib_reasoning in library_choices.items():
                        cursor.execute(
                            """INSERT INTO library_choices 
                               (commit_hash, library_name, reasoning_id, reasoning_summary)
                               VALUES (?, ?, ?, ?)""",
                            (commit_hash, lib_name, str(reasoning_id), 
                             lib_reasoning[:500])  # Truncate for storage
                        )
                
                conn.commit()
                
                logger.info(f"Recorded reasoning for commit {commit_hash[:8]}")
                
                return ReasoningRecord(
                    reasoning_id=reasoning_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    commit_hash=commit_hash,
                    timestamp=timestamp,
                    reasoning=reasoning,
                    confidence=confidence,
                    supporting_evidence=evidence
                )
    
    def get_lineage(self, commit_hash: str) -> Optional[DecisionLineage]:
        """
        Get complete lineage for a commit.
        
        Args:
            commit_hash: Git commit hash
            
        Returns:
            DecisionLineage with full decision chain
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get reasoning record
            cursor.execute(
                """SELECT * FROM reasoning_records WHERE commit_hash = ?""",
                (commit_hash,)
            )
            reasoning_row = cursor.fetchone()
            
            # Get decision record
            cursor.execute(
                """SELECT * FROM decision_records WHERE commit_hash = ?""",
                (commit_hash,)
            )
            decision_row = cursor.fetchone()
            
            # Get parent commits
            cursor.execute(
                """SELECT parent_commit FROM lineage_chain WHERE commit_hash = ?""",
                (commit_hash,)
            )
            parent_commits = [row["parent_commit"] for row in cursor.fetchall() if row["parent_commit"]]
            
            # Get child commits
            cursor.execute(
                """SELECT commit_hash FROM lineage_chain WHERE parent_commit = ?""",
                (commit_hash,)
            )
            child_commits = [row["commit_hash"] for row in cursor.fetchall()]
            
            # Get library choices
            cursor.execute(
                """SELECT library_name, reasoning_summary FROM library_choices 
                   WHERE commit_hash = ?""",
                (commit_hash,)
            )
            library_choices = {row["library_name"]: row["reasoning_summary"] 
                              for row in cursor.fetchall()}
            
            # Build decision record
            decision = None
            if decision_row:
                decision = DecisionRecord(
                    record_id=UUID(decision_row["record_id"]),
                    agent_id=UUID(decision_row["agent_id"]),
                    decision_type=DecisionType(decision_row["decision_type"]),
                    commit_hash=decision_row["commit_hash"],
                    timestamp=datetime.fromisoformat(decision_row["timestamp"]),
                    review_time_ms=decision_row["review_time_ms"],
                    change_size=decision_row["change_size"],
                    complexity=ChangeComplexity(decision_row["complexity"]),
                    reviewer_notes=decision_row["reviewer_notes"],
                    reviewer_id=decision_row["reviewer_id"]
                )
            
            # Build reasoning record
            reasoning = None
            if reasoning_row:
                reasoning = ReasoningRecord(
                    reasoning_id=UUID(reasoning_row["reasoning_id"]),
                    agent_id=UUID(reasoning_row["agent_id"]),
                    task_id=reasoning_row["task_id"],
                    commit_hash=reasoning_row["commit_hash"],
                    timestamp=datetime.fromisoformat(reasoning_row["timestamp"]),
                    reasoning=reasoning_row["reasoning"],
                    confidence=reasoning_row["confidence"],
                    supporting_evidence=json.loads(reasoning_row["supporting_evidence"] or "[]")
                )
            
            # Build full chain trace
            full_chain = self._trace_chain_recursive(conn, commit_hash, max_depth=10)
            
            agent_id = UUID(reasoning_row["agent_id"]) if reasoning_row else \
                      (UUID(decision_row["agent_id"]) if decision_row else uuid4())
            
            return DecisionLineage(
                commit_hash=commit_hash,
                agent_id=agent_id,
                decision=decision,
                reasoning=reasoning,
                parent_commits=parent_commits,
                child_commits=child_commits,
                library_choices=library_choices,
                full_chain=full_chain
            )
    
    def _trace_chain_recursive(self, conn: sqlite3.Connection, commit_hash: str, 
                                max_depth: int, current_depth: int = 0) -> List[Dict[str, Any]]:
        """Recursively trace the decision chain."""
        if current_depth >= max_depth:
            return [{"commit_hash": commit_hash, "truncated": True}]
        
        cursor = conn.cursor()
        
        # Get this commit's data
        cursor.execute(
            """SELECT r.reasoning, r.confidence, r.agent_id, r.timestamp,
                      d.decision_type, d.review_time_ms
               FROM reasoning_records r
               LEFT JOIN decision_records d ON r.commit_hash = d.commit_hash
               WHERE r.commit_hash = ?""",
            (commit_hash,)
        )
        row = cursor.fetchone()
        
        if not row:
            return [{"commit_hash": commit_hash, "missing": True}]
        
        chain_entry = {
            "commit_hash": commit_hash,
            "agent_id": row["agent_id"],
            "timestamp": row["timestamp"],
            "reasoning_preview": row["reasoning"][:200] if row["reasoning"] else None,
            "confidence": row["confidence"],
            "decision_type": row["decision_type"],
            "depth": current_depth
        }
        
        # Get parents and recurse
        cursor.execute(
            "SELECT parent_commit FROM lineage_chain WHERE commit_hash = ?",
            (commit_hash,)
        )
        parents = [r["parent_commit"] for r in cursor.fetchall() if r["parent_commit"]]
        
        result = [chain_entry]
        for parent in parents:
            result.extend(self._trace_chain_recursive(conn, parent, max_depth, current_depth + 1))
        
        return result
    
    def trace_decision_chain(self, start_commit: str, max_depth: int = 10) -> List[DecisionLineage]:
        """
        Trace the full decision chain starting from a commit.
        
        Args:
            start_commit: Starting commit hash
            max_depth: Maximum chain depth to trace
            
        Returns:
            List of DecisionLineage in chain order
        """
        chain = []
        current = start_commit
        visited = set()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            while current and len(chain) < max_depth:
                if current in visited:
                    break
                visited.add(current)
                
                lineage = self.get_lineage(current)
                if lineage:
                    chain.append(lineage)
                    # Follow first parent
                    current = lineage.parent_commits[0] if lineage.parent_commits else None
                else:
                    break
        
        return chain
    
    def get_reasoning_for_library_choice(self, commit_hash: str, library: str) -> Optional[str]:
        """
        Get reasoning for a specific library choice in a commit.
        
        Args:
            commit_hash: Git commit hash
            library: Library name
            
        Returns:
            Reasoning string or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT r.reasoning FROM reasoning_records r
                   JOIN library_choices l ON r.reasoning_id = l.reasoning_id
                   WHERE l.commit_hash = ? AND l.library_name = ?""",
                (commit_hash, library)
            )
            row = cursor.fetchone()
            return row["reasoning"] if row else None
    
    def search_reasoning(self, keyword: str, agent_id: Optional[UUID] = None,
                         limit: int = 50) -> List[ReasoningRecord]:
        """
        Search reasoning records by keyword.
        
        Args:
            keyword: Search term
            agent_id: Optional agent filter
            limit: Maximum results
            
        Returns:
            List of matching ReasoningRecords
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if agent_id:
                cursor.execute(
                    """SELECT * FROM reasoning_records 
                       WHERE reasoning LIKE ? AND agent_id = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{keyword}%", str(agent_id), limit)
                )
            else:
                cursor.execute(
                    """SELECT * FROM reasoning_records 
                       WHERE reasoning LIKE ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (f"%{keyword}%", limit)
                )
            
            records = []
            for row in cursor.fetchall():
                records.append(ReasoningRecord(
                    reasoning_id=UUID(row["reasoning_id"]),
                    agent_id=UUID(row["agent_id"]),
                    task_id=row["task_id"],
                    commit_hash=row["commit_hash"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    reasoning=row["reasoning"],
                    confidence=row["confidence"],
                    supporting_evidence=json.loads(row["supporting_evidence"] or "[]")
                ))
            
            return records


# ============================================================================
# INTEGRATED TRUST & LINEAGE MANAGER
# ============================================================================

class TrustLineageManager:
    """
    Integrated manager combining TrustScorer and LineageTracker.
    Provides unified interface for trust and lineage operations.
    """
    
    def __init__(self, storage_path: str):
        """Initialize with shared storage.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = storage_path
        self.trust = TrustScorer(storage_path)
        self.lineage = LineageTracker(storage_path)
    
    def record_complete_decision(
        self,
        agent_id: UUID,
        task_id: str,
        commit_hash: str,
        decision: DecisionType,
        reasoning: str,
        review_time_ms: int,
        change_size: int = 0,
        complexity: ChangeComplexity = ChangeComplexity.MODERATE,
        confidence: float = 0.5,
        supporting_evidence: Optional[List[str]] = None,
        parent_commits: Optional[List[str]] = None,
        library_choices: Optional[Dict[str, str]] = None,
        reviewer_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ) -> Tuple[DecisionRecord, ReasoningRecord]:
        """
        Record a complete decision with both trust and lineage data.
        
        Args:
            agent_id: UUID of the agent
            task_id: Task identifier
            commit_hash: Git commit hash
            decision: Type of decision
            reasoning: Full reasoning text
            review_time_ms: Time taken for review
            change_size: Lines of code changed
            complexity: Complexity level
            confidence: Confidence in reasoning
            supporting_evidence: List of evidence
            parent_commits: Parent commit hashes
            library_choices: Library choice reasoning
            reviewer_notes: Optional reviewer notes
            reviewer_id: Optional reviewer ID
            
        Returns:
            Tuple of (DecisionRecord, ReasoningRecord)
        """
        # Record reasoning first
        reasoning_record = self.lineage.record_reasoning(
            agent_id=agent_id,
            task_id=task_id,
            commit_hash=commit_hash,
            reasoning=reasoning,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            parent_commits=parent_commits,
            library_choices=library_choices
        )
        
        # Record decision
        decision_record = self.trust.record_decision(
            agent_id=agent_id,
            decision=decision,
            commit_hash=commit_hash,
            review_time_ms=review_time_ms,
            change_size=change_size,
            complexity=complexity,
            reviewer_notes=reviewer_notes,
            reviewer_id=reviewer_id
        )
        
        return decision_record, reasoning_record
    
    def get_decision_with_lineage(self, commit_hash: str) -> Optional[DecisionLineage]:
        """Get complete decision with full lineage.
        
        Args:
            commit_hash: Git commit hash
            
        Returns:
            DecisionLineage or None
        """
        return self.lineage.get_lineage(commit_hash)
    
    def get_agent_summary(self, agent_id: UUID) -> Dict[str, Any]:
        """Get comprehensive agent summary.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            Dictionary with profile and recent history
        """
        profile = self.trust.get_agent_profile(agent_id)
        recent_history = self.trust.get_agent_history(agent_id, limit=10)
        
        if not profile:
            return {"error": "Agent not found"}
        
        return {
            "profile": profile.to_dict(),
            "recent_decisions": [d.to_dict() for d in recent_history],
            "can_auto_approve_small_changes": self.trust.should_auto_approve(agent_id, 50),
            "can_auto_approve_medium_changes": self.trust.should_auto_approve(agent_id, 150)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    
    # Create temporary database for demonstration
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "trust_lineage.db")
        
        print("=" * 70)
        print("Ansib-eL Trust Scoring & Lineage Tracking Demo")
        print("=" * 70)
        
        # Initialize manager
        manager = TrustLineageManager(db_path)
        
        # Create sample agents
        agent_1 = uuid4()
        agent_2 = uuid4()
        
        print(f"\nCreated agents:")
        print(f"  Agent 1: {agent_1}")
        print(f"  Agent 2: {agent_2}")
        
        # Simulate decisions for agent_1 (good performer)
        print("\n--- Simulating Agent 1 (Good Performer) ---")
        
        for i in range(10):
            decision = DecisionType.ACCEPTED if i < 8 else DecisionType.MODIFIED
            commit = hashlib.sha256(f"commit_a1_{i}".encode()).hexdigest()[:16]
            
            decision_rec, reasoning_rec = manager.record_complete_decision(
                agent_id=agent_1,
                task_id=f"task_{i}",
                commit_hash=commit,
                decision=decision,
                reasoning=f"Analyzed code change {i}. "
                         f"{'All tests passed, no issues found.' if decision == DecisionType.ACCEPTED else 'Minor style issues fixed.'}",
                review_time_ms=5000 + (i * 100),
                change_size=50 + (i * 10),
                complexity=ChangeComplexity.MODERATE,
                confidence=0.85,
                supporting_evidence=["test_results.json", "lint_report.txt"],
                library_choices={"requests": "Standard HTTP library, well-maintained"}
            )
            
            print(f"  Commit {commit[:8]}: {decision.value} "
                  f"(score: {manager.trust.get_trust_score(agent_1).score:.3f})")
        
        # Simulate decisions for agent_2 (poor performer)
        print("\n--- Simulating Agent 2 (Poor Performer) ---")
        
        for i in range(5):
            decision = DecisionType.REJECTED if i < 3 else DecisionType.MODIFIED
            commit = hashlib.sha256(f"commit_a2_{i}".encode()).hexdigest()[:16]
            
            manager.record_complete_decision(
                agent_id=agent_2,
                task_id=f"task_{i}",
                commit_hash=commit,
                decision=decision,
                reasoning=f"Analyzed code change {i}. "
                         f"{'Critical security vulnerability detected.' if decision == DecisionType.REJECTED else 'Issues fixed after review.'}",
                review_time_ms=15000,
                change_size=200,
                complexity=ChangeComplexity.MAJOR,
                confidence=0.6,
                supporting_evidence=["security_scan.json"]
            )
            
            print(f"  Commit {commit[:8]}: {decision.value} "
                  f"(score: {manager.trust.get_trust_score(agent_2).score:.3f})")
        
        # Display final results
        print("\n--- Final Agent Profiles ---")
        
        for agent_id, name in [(agent_1, "Agent 1 (Good)"), (agent_2, "Agent 2 (Poor)")]:
            profile = manager.trust.get_agent_profile(agent_id)
            print(f"\n{name}:")
            print(f"  Trust Score: {profile.trust_score.score:.3f}")
            print(f"  Confidence: {profile.trust_score.confidence:.3f}")
            print(f"  Tier: {profile.tier.value}")
            print(f"  Total Decisions: {profile.total_decisions}")
            print(f"  Accepted: {profile.accepted_count}")
            print(f"  Rejected: {profile.rejected_count}")
            print(f"  Modified: {profile.modified_count}")
            print(f"  Auto-approve 50 lines: {manager.trust.should_auto_approve(agent_id, 50)}")
            print(f"  Auto-approve 150 lines: {manager.trust.should_auto_approve(agent_id, 150)}")
        
        # Show lineage for one commit
        print("\n--- Lineage Example ---")
        sample_commit = hashlib.sha256("commit_a1_5".encode()).hexdigest()[:16]
        lineage = manager.get_decision_with_lineage(sample_commit)
        if lineage:
            print(f"Commit: {lineage.commit_hash[:8]}")
            print(f"Agent: {lineage.agent_id}")
            if lineage.reasoning:
                print(f"Reasoning: {lineage.reasoning.reasoning[:100]}...")
            print(f"Library Choices: {lineage.library_choices}")
        
        # Show agent summary
        print("\n--- Agent 1 Summary ---")
        summary = manager.get_agent_summary(agent_1)
        print(json.dumps(summary, indent=2, default=str)[:1000] + "...")
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
