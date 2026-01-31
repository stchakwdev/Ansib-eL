# Trust and Reputation System

## Overview

Ansib-eL assigns every agent a trust score that reflects its track record of producing acceptable work. Trust scores determine whether an agent's changes require human review or can be auto-approved, creating a graduated autonomy model: new agents always need review, while agents with a proven record can merge small changes independently.

## Trust Tiers

Trust scores map to five tiers:

| Tier | Score Range | Behavior |
|------|-------------|----------|
| `UNTRUSTED` | 0.0 - 0.19 | All changes require human review |
| `LOW` | 0.2 - 0.39 | Trivial changes (< 10 lines) may auto-approve |
| `MEDIUM` | 0.4 - 0.59 | Minor changes (< 50 lines) may auto-approve |
| `HIGH` | 0.6 - 0.79 | Moderate changes (< 200 lines) may auto-approve |
| `VERIFIED` | 0.8 - 1.0 | Major changes (< 500 lines) may auto-approve |

New agents start with a score of 0.5 (MEDIUM tier) and 0 confidence until they accumulate decision history.

## Scoring Algorithm

Trust scores use an **Exponential Moving Average (EMA)** that weights recent decisions more heavily than older ones.

When a decision is recorded for an agent:

1. **Compute the decision value**: Accepted = 1.0, Modified = 0.5, Rejected = 0.0
2. **Compute the weight** based on change size, complexity, and review time:
   ```
   weight = base_value * complexity_multiplier * review_time_factor
   ```
3. **Update the score** using EMA with smoothing factor alpha = 0.3:
   ```
   new_score = alpha * weighted_value + (1 - alpha) * current_score
   ```
4. **Update confidence** based on sample count:
   ```
   confidence = min(sample_count / MAX_SAMPLES, 1.0)
   ```
   Confidence reaches 1.0 at 100 samples. Minimum 10 samples for meaningful confidence.

### Weight Factors

The weight of each decision depends on the stakes involved:

- **Change size**: Larger changes carry more weight. A 500-line change has more impact on the score than a 5-line change.
- **Complexity**: Uses `ChangeComplexity` enum values (TRIVIAL=1, MINOR=2, MODERATE=3, MAJOR=5, CRITICAL=8) as multipliers.
- **Review time**: Faster reviews slightly increase weight, reflecting reviewer confidence in the outcome.

## Time Decay

Idle agents see their scores decay toward 0.5 (neutral) over time, implementing the principle that trust must be continuously maintained.

The decay follows an exponential curve with a **30-day half-life**:

```
decayed_score = neutral + (current_score - neutral) * 2^(-days_since_last_activity / 30)
```

An agent with score 0.9 that has been idle for 30 days decays to approximately 0.7. After 60 days of inactivity, it would be around 0.6.

Decay is applied lazily when a score is queried, not on a background timer.

## Auto-Approval Logic

The `should_auto_approve(agent_id, change_size)` method checks whether a change can skip human review:

1. Look up the agent's current trust tier
2. Look up the maximum auto-approve line count for that tier
3. If `change_size <= max_lines`, return `True`

The limits per tier:

| Tier | Max Lines for Auto-Approval |
|------|----------------------------|
| UNTRUSTED | 0 (never auto-approve) |
| LOW | 10 |
| MEDIUM | 50 |
| HIGH | 200 |
| VERIFIED | 500 |

Changes above these thresholds always require human review, regardless of trust level.

## Trust Recovery

Agents that have lost trust (due to rejections or inactivity decay) can recover through the `apply_recovery(agent_id, boost_amount)` method:

- Adds `boost_amount` to the current score
- The recovery rate constant is 0.05 per recovery event
- The total score is capped at 1.0
- Recovery requires the agent to have an existing profile (no bootstrapping new agents with high trust)

## API Examples

### Recording a Decision

```python
from ansibel.trust_lineage import TrustLineageManager, DecisionType, ChangeComplexity

manager = TrustLineageManager(storage_path=".ai-git")

# Record an accepted decision
record = manager.record_decision(
    agent_id="agent-001",
    decision=DecisionType.ACCEPTED,
    commit_hash="abc1234",
    review_time_ms=5000,
    change_size=45,
    complexity=ChangeComplexity.MINOR
)
```

### Querying Trust

```python
# Get trust score
score = manager.get_trust_score("agent-001")
print(f"Score: {score.score}, Confidence: {score.confidence}")

# Get trust tier
tier = manager.get_trust_tier("agent-001")
print(f"Tier: {tier.name}")

# Check auto-approval
can_auto = manager.should_auto_approve("agent-001", change_size=30)
print(f"Auto-approve 30 lines: {can_auto}")
```

### Agent History

```python
# Get decision history
history = manager.get_agent_history("agent-001", limit=20)
for record in history:
    print(f"{record.decision_type.name}: {record.commit_hash[:8]}")

# Get full agent profile
profile = manager.get_agent_profile("agent-001")
print(f"Total decisions: {profile.total_decisions}")
print(f"Accept rate: {profile.accepted_count / profile.total_decisions:.0%}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANSIBEL_TRUST_THRESHOLD_HIGH` | Score threshold for HIGH tier | `0.8` |
| `ANSIBEL_TRUST_THRESHOLD_MEDIUM` | Score threshold for MEDIUM tier | `0.5` |

### Config File (`.ai-git/config.yaml`)

```yaml
trust:
  decay_half_life_days: 30
  recovery_rate: 0.05
  ema_alpha: 0.3
```
