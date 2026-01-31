# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Ansib-eL, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email **security@ansibel.dev** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days, including next steps and an expected timeline for a fix.

## Security Architecture

Ansib-eL implements several layers of defense:

### Git Ref Validation

All branch names and ref inputs are validated against a strict regex pattern. Path traversal sequences (`..`, `~`) and shell metacharacters are rejected before any Git operation executes. See `GitWrapper._validate_git_ref()` in `src/ansibel/git_wrapper.py`.

### Atomic File Operations

Metadata and configuration writes use a tempfile-then-rename pattern (`tempfile.NamedTemporaryFile` + `os.replace`) to prevent partial writes and data corruption.

### Subprocess Isolation

Git commands execute via `subprocess.run` with `shell=False` and explicit argument lists. Agent contexts provide isolated environment variables through `AgentContext.get_subprocess_env()`, preventing cross-agent environment leakage.

### Branch Protection

The orchestrator enforces configurable protection levels on branches:

| Level | Behavior |
|-------|----------|
| `NONE` | Direct merge allowed |
| `REVIEW_REQUIRED` | At least one review before merge |
| `HUMAN_APPROVAL` | Explicit human approval required (default for main) |
| `LOCKED` | No merges permitted |

### Trust-Based Access Control

Auto-approval decisions combine agent trust tier with change complexity:

| Trust Tier | Max Auto-Approve Complexity |
|------------|----------------------------|
| UNTRUSTED | None (always requires review) |
| LOW | Trivial changes only (< 10 lines) |
| MEDIUM | Minor changes (< 50 lines) |
| HIGH | Moderate changes (< 200 lines) |
| VERIFIED | Major changes (< 500 lines) |

### Data Integrity

- Frozen dataclasses for immutable records (`DecisionRecord`, `ReasoningRecord`, `TrustScore`)
- SHA-256 prompt hashing for tamper-evident agent provenance
- SQLite with WAL mode for concurrent-safe trust and lineage storage

## Scope

This security policy covers:

- The core Ansib-eL Python package (`src/ansibel/`)
- The `ai-git` CLI
- The `.ai-git/` metadata storage format

Out of scope:

- Security of AI models used by agents (model providers are responsible for their own security)
- Third-party dependencies (report upstream)
- Deployment infrastructure
