# CLI Reference

The `ai-git` command extends Git with AI-native version control operations. It requires Click (`pip install click`); if Click is unavailable, a built-in argparse fallback provides the same commands with reduced formatting.

## Synopsis

```
ai-git [OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Print version and exit |
| `--help` | Show help message and exit |

---

## Commands

### init

Initialize an ai-git repository in the current directory.

```
ai-git init [OPTIONS]
```

Creates a `.ai-git/` directory structure alongside the Git repository. If no Git repository exists, one is created first.

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Force reinitialization of an existing repository |

**Examples:**

```bash
ai-git init
ai-git init --force
```

---

### status

Show repository status with AI-specific information.

```
ai-git status [OPTIONS]
```

Displays the current branch, working tree state, staged/modified/untracked files, and active agent branches with their trust scores.

| Option | Short | Description |
|--------|-------|-------------|
| `--json-output` | `-j` | Output as JSON |

**Examples:**

```bash
ai-git status
ai-git status --json-output
```

---

### branch

Create a new agent branch.

```
ai-git branch AGENT_ID [OPTIONS]
```

Creates an isolated branch named `agent/<agent-id>/<timestamp>` and checks it out. The branch is tracked in `.ai-git/` metadata.

| Argument/Option | Short | Description |
|-----------------|-------|-------------|
| `AGENT_ID` | | Agent identifier (required) |
| `--purpose` | `-p` | Purpose description (default: "Agent workspace") |

**Examples:**

```bash
ai-git branch agent-001 --purpose "Implement authentication"
ai-git branch my-agent -p "Fix database connection pooling"
```

---

### commit

Create a commit with AI metadata attached.

```
ai-git commit MESSAGE [OPTIONS]
```

Commits staged changes (or specified files) and stores agent metadata in `.ai-git/metadata/`. The metadata includes the agent identity, model version, prompt hash, and optional confidence score.

| Argument/Option | Short | Description |
|-----------------|-------|-------------|
| `MESSAGE` | | Commit message (required) |
| `--agent-id` | `-a` | Agent identifier (required) |
| `--model-version` | `-m` | Model version string (default: "unknown") |
| `--prompt-hash` | `-p` | SHA-256 hash of the prompt (auto-generated if omitted) |
| `--parent-task` | `-t` | Parent task ID |
| `--confidence` | `-c` | Confidence score (0.0-1.0) |
| `--files` | `-f` | Specific files to commit (repeatable) |

**Examples:**

```bash
ai-git commit "Add OAuth login endpoint" --agent-id agent-001 --model-version gpt-5.2
ai-git commit "Fix null check" -a agent-002 -m claude-opus-4.5 -c 0.95
ai-git commit "Update schema" -a agent-001 -f schema.sql -f models.py
```

---

### merge

Merge an agent branch into the current branch.

```
ai-git merge BRANCH_NAME [OPTIONS]
```

Shows a diff preview before merging. Supports three merge strategies.

| Argument/Option | Short | Description |
|-----------------|-------|-------------|
| `BRANCH_NAME` | | Branch to merge (required) |
| `--strategy` | `-s` | Merge strategy: `merge`, `squash`, or `rebase` (default: "merge") |
| `--target` | `-t` | Target branch (default: current branch) |
| `--yes` | `-y` | Skip confirmation prompt |

**Examples:**

```bash
ai-git merge agent/agent-001/20240115 --strategy squash
ai-git merge agent/agent-002/20240116 -s rebase -t main -y
```

---

### history

Show AI-enhanced commit history.

```
ai-git history [OPTIONS]
```

Displays commit log annotated with agent metadata (agent ID, model version, confidence score) when available.

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Number of commits to show (default: 20) |
| `--agent` | `-a` | Filter by agent ID |
| `--json-output` | `-j` | Output as JSON |

**Examples:**

```bash
ai-git history
ai-git history --limit 50 --agent agent-001
ai-git history -j | jq '.[] | select(.metadata)'
```

---

### agents

List agents and their branches.

```
ai-git agents [OPTIONS]
```

Groups agents by status and displays branch name, agent ID, purpose, creation date, and trust score.

| Option | Short | Description |
|--------|-------|-------------|
| `--status` | `-s` | Filter: `active`, `merged`, `closed`, or `all` (default: "all") |
| `--json-output` | `-j` | Output as JSON |

**Examples:**

```bash
ai-git agents
ai-git agents --status active
ai-git agents -j
```

---

### review

Interactive review mode for pending agent merges.

```
ai-git review [OPTIONS]
```

Iterates through active agent branches, showing diffs and prompting for action. Available actions: `merge`, `skip`, `diff` (full diff), `details` (commit history), `quit`.

| Option | Short | Description |
|--------|-------|-------------|
| `--branch` | `-b` | Review a specific branch only |

**Examples:**

```bash
ai-git review
ai-git review --branch agent/agent-001/20240115
```

---

### trust

Update the trust score for an agent.

```
ai-git trust AGENT_ID SCORE_DELTA
```

Adjusts the agent's trust score by the given delta. Scores are clamped to the range [0.0, 1.0]. Color-coded output indicates the resulting tier.

| Argument | Description |
|----------|-------------|
| `AGENT_ID` | Agent identifier |
| `SCORE_DELTA` | Amount to add (positive) or subtract (negative) |

**Examples:**

```bash
ai-git trust agent-001 0.1    # Increase trust
ai-git trust agent-002 -0.2   # Decrease trust
```

---

### diff

Show diff between two branches.

```
ai-git diff BRANCH_A [BRANCH_B]
```

If only one branch is given, compares it against the current branch.

| Argument | Description |
|----------|-------------|
| `BRANCH_A` | First branch |
| `BRANCH_B` | Second branch (default: current branch) |

**Examples:**

```bash
ai-git diff agent/agent-001/20240115
ai-git diff main agent/agent-001/20240115
```

---

### metadata

Show AI metadata for a specific commit.

```
ai-git metadata COMMIT_HASH
```

Retrieves and displays the `AgentMetadata` stored in `.ai-git/metadata/` for the given commit, including agent ID, model version, prompt hash, timestamp, parent task, and confidence score.

| Argument | Description |
|----------|-------------|
| `COMMIT_HASH` | Full or abbreviated commit hash |

**Examples:**

```bash
ai-git metadata abc1234
ai-git metadata $(git rev-parse HEAD)
```

---

## Environment Variables

These variables configure default behavior across all commands:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANSIBEL_DEFAULT_MODEL` | Default model for new agents | `gpt-5.2` |
| `ANSIBEL_TOURNAMENT_AGENTS` | Number of agents per tournament | `3` |
| `ANSIBEL_TOURNAMENT_TIMEOUT` | Tournament timeout in seconds | `300` |
| `ANSIBEL_TRUST_THRESHOLD_HIGH` | High trust tier threshold | `0.8` |
| `ANSIBEL_TRUST_THRESHOLD_MEDIUM` | Medium trust tier threshold | `0.5` |
| `ANSIBEL_STORAGE_PATH` | Metadata storage directory | `.ai-git` |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (uninitialized repo, Git failure, merge conflict) |
