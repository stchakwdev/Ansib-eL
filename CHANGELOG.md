# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-beta] - 2026-01-28

### Added

- Documentation suite: Getting Started guide, CLI Reference, API Reference, Architecture overview
- Concept guides: Trust and Reputation, Tournament Mode, Decision Lineage
- CONTRIBUTING.md with development setup, workflow, and coding standards
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md with vulnerability reporting and security architecture summary
- GitHub issue templates (bug report, feature request) and PR template
- `commit_changes()` method on GitWrapper for plain commits without AI metadata
- `workspace_branch` field on orchestrator Solution dataclass
- `get_trust_tier()`, `get_agent_history()`, `should_auto_approve()` delegate methods on TrustLineageManager
- `ApprovalResult` dataclass for structured approve/reject responses

### Changed

- README restructured as documentation hub with link table to all docs
- README: removed emoji prefixes from headers, updated project structure to reflect `src/ansibel/` layout, fixed Python version to 3.10+
- `approve_solution()` and `reject_solution()` now return `ApprovalResult` instead of raw dict
- `_execute_merge()` uses real git merge via GitWrapper instead of placeholder

### Fixed

- Tournament `Task` type adapter for compatibility with orchestrator Task
- Async wrappers in tournament for synchronous AgentManager methods
- `AgentManager` storage path resolution for `src/` layout
- Trust lineage time decay calculation for edge case with zero elapsed time

## [1.0.0-alpha] - 2026-01-28

### Added
- Initial release of Ansib-eL AI-Native Version Control System
- GitWrapper: Git operations with AI metadata stored in `.ai-git/`
- Orchestrator: Task breakdown, delegation, and approval gatekeeper
- AgentManager: Agent lifecycle management with isolated workspace branches
- TournamentOrchestrator: Async parallel agent execution with configurable selection modes
- TrustLineageManager: EMA-based reputation scoring with SQLite persistence
- CLI (`ai-git`): Click-based command-line interface with init, status, commit, branch, merge, history, trust subcommands
- Unified exception hierarchy (`AnsibElError` base)
- Security: Git ref validation, atomic JSON writes, path traversal guards, subprocess env isolation
- Package restructured to `src/ansibel/` layout with `pyproject.toml`
