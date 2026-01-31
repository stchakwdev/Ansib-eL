#!/usr/bin/env python3
"""
Ansib-eL: AI-Native Version Control System

This package provides AI-native extensions to Git for tracking agent decisions
and maintaining branch isolation for ephemeral agent workspaces.
"""

__version__ = "1.0.0"
__author__ = "Ansib-eL Team"

from ansibel.cli import (
    main,
)
from ansibel.exceptions import (
    AgentError,
    AnsibElError,
    GitWrapperError,
    TournamentError,
    TrustError,
)
from ansibel.git_wrapper import (
    AgentMetadata,
    GitWrapper,
    MergeResult,
)

__all__ = [
    # Exceptions
    "AnsibElError",
    "GitWrapperError",
    "TournamentError",
    "TrustError",
    "AgentError",
    # Git
    "GitWrapper",
    "AgentMetadata",
    "MergeResult",
    # CLI
    "main",
]
