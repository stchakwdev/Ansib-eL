#!/usr/bin/env python3
"""
Ansib-eL: AI-Native Version Control System

This package provides AI-native extensions to Git for tracking agent decisions
and maintaining branch isolation for ephemeral agent workspaces.
"""

__version__ = '1.0.0'
__author__ = 'Ansib-eL Team'

from ansibel.exceptions import (
    AnsibElError,
    GitWrapperError,
    TournamentError,
    TrustError,
    AgentError,
)

from ansibel.git_wrapper import (
    GitWrapper,
    AgentMetadata,
    MergeResult,
)

from ansibel.cli import (
    main,
)

__all__ = [
    # Exceptions
    'AnsibElError',
    'GitWrapperError',
    'TournamentError',
    'TrustError',
    'AgentError',
    # Git
    'GitWrapper',
    'AgentMetadata',
    'MergeResult',
    # CLI
    'main',
]
