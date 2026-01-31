"""
Unified error hierarchy for Ansib-eL.
"""


class AnsibElError(Exception):
    """Base exception for all Ansib-eL errors."""


class GitWrapperError(AnsibElError):
    """Errors from Git wrapper operations."""


class TournamentError(AnsibElError):
    """Errors from tournament operations."""


class TrustError(AnsibElError):
    """Errors from trust/lineage operations."""


class AgentError(AnsibElError):
    """Errors from agent operations."""


class AgentNotFoundError(AgentError):
    """Raised when an agent is not found."""


class AgentValidationError(AgentError):
    """Raised when agent validation fails."""


class AgentContextError(AgentError):
    """Raised when context operations fail."""
