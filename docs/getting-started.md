# Getting Started

This guide walks you through installing Ansib-eL, initializing your first AI-native repository, and running basic operations with both the CLI and Python API.

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **Git 2.30+** (check with `git --version`)
- **pip** (included with Python)

## Installation

Ansib-eL is installed from source:

```bash
git clone <repo-url>
cd ansib-el

# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package (registers the ai-git CLI command)
pip install -e .
```

### Verify Installation

```bash
# CLI should print version
ai-git --version

# Python import should succeed
python -c "from ansibel import GitWrapper; print('Ansib-eL ready')"
```

## Your First AI-Git Repository

### Step 1: Initialize

```bash
mkdir my-project && cd my-project
ai-git init
```

This creates:
- A standard Git repository (if one does not exist)
- A `.ai-git/` directory containing metadata storage, agent records, and configuration

### Step 2: Create an Agent Branch

```bash
ai-git branch agent-001 --purpose "Add user authentication"
```

This creates an isolated branch named `agent/<agent-id>/<timestamp>` and checks it out. Each agent works in its own branch to prevent interference.

### Step 3: Make Changes and Commit with Metadata

After making code changes on the agent branch:

```bash
ai-git commit "Add OAuth login endpoint" \
  --agent-id agent-001 \
  --model-version gpt-4 \
  --confidence 0.85
```

The commit stores standard Git data plus AI metadata (agent identity, model version, prompt hash, confidence score) in `.ai-git/metadata/`.

### Step 4: Review and Merge

Switch back to the main branch and review the agent's work:

```bash
git checkout main
ai-git review
```

The interactive review mode shows diffs and lets you approve, skip, or inspect each pending agent branch. When you approve:

```bash
ai-git merge agent/agent-001/<timestamp> --strategy merge
```

### Step 5: View AI-Enhanced History

```bash
ai-git history --limit 10
```

Each commit shows the standard Git log plus agent attribution, model version, and confidence score when metadata is available.

## Programmatic Usage

The `AnsibElSystem` class provides a unified Python API:

```python
from ansibel.ansib_el import AnsibElSystem

# Initialize
system = AnsibElSystem("./my-project")
system.initialize()

# Process a prompt (single agent)
result = system.process_prompt(
    "Add input validation to the signup form",
    use_tournament=False
)

# Check system status
status = system.get_status()
print(f"Active agents: {status.active_agents}")
print(f"Pending approvals: {status.pending_approvals}")
```

## Using Tournament Mode

Tournament mode spawns multiple agents to solve the same task in parallel, then lets you pick the best solution:

```python
result = system.process_prompt(
    "Refactor the database layer for connection pooling",
    use_tournament=True,
    num_agents=3
)

# Review and approve the winning solution
pending = system.list_pending_approvals()
system.review_and_approve(
    approval_id=pending[0]["id"],
    approve=True,
    comments="Selected for cleaner connection lifecycle"
)
```

For details, see [Tournament Mode](concepts/tournament-mode.md).

## Checking Trust Scores

Every agent builds a reputation over time. Trust scores influence auto-approval thresholds:

```bash
# Update trust via CLI
ai-git trust agent-001 0.1
```

```python
# Query trust programmatically
info = system.get_agent_info("agent-001")
print(f"Trust: {info['trust_score']}")
```

For details, see [Trust and Reputation](concepts/trust-and-reputation.md).

## Configuration

Ansib-eL reads configuration from environment variables and `.ai-git/config.yaml`:

```bash
# Set defaults via environment
export ANSIBEL_DEFAULT_MODEL=gpt-4
export ANSIBEL_TOURNAMENT_AGENTS=3
export ANSIBEL_TOURNAMENT_TIMEOUT=300
```

```yaml
# .ai-git/config.yaml
orchestrator:
  main_branch_protection: human_approval
tournament:
  default_agents: 3
  selection_mode: human_choice
trust:
  decay_half_life_days: 30
```

See the full list of environment variables in the [CLI Reference](cli-reference.md).

## Next Steps

- [CLI Reference](cli-reference.md) -- All `ai-git` commands and options
- [API Reference](api-reference.md) -- Python class and method documentation
- [Architecture](architecture.md) -- System design and data flow
- [Trust and Reputation](concepts/trust-and-reputation.md) -- Scoring algorithm details
- [Tournament Mode](concepts/tournament-mode.md) -- Parallel execution deep dive
- [Decision Lineage](concepts/decision-lineage.md) -- Audit trail and chain-of-thought
