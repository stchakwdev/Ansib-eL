"""
Ansib-eL Agent Management System
================================

AI-Native Version Control System - Agent Lifecycle Management

This module provides complete agent lifecycle management including:
- Unique agent identity with UUID generation
- Purpose tracking and context isolation
- Agent spawn/track/terminate operations
- Metadata serialization and persistence
"""

import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field, asdict
from uuid import UUID, uuid4
import logging

from ansibel.exceptions import AgentError, AgentNotFoundError, AgentValidationError, AgentContextError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Validation Models
# ============================================================================

try:
    from pydantic import BaseModel, Field, validator, root_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    logger.warning("Pydantic not installed. Running without validation.")

if HAS_PYDANTIC:
    from pydantic import field_validator
    
    class AgentPydanticModel(BaseModel):
        """Pydantic model for agent data validation."""
        agent_id: str = Field(..., description="Unique UUID for the agent")
        purpose: str = Field(..., min_length=1, max_length=500, description="Agent's purpose/task")
        model_version: str = Field(..., min_length=1, description="AI model version")
        prompt_hash: str = Field(..., min_length=32, max_length=64, description="SHA-256 hash of the prompt")
        created_at: str = Field(..., description="ISO format timestamp")
        status: str = Field(..., description="Current agent status")
        workspace_branch: str = Field(..., description="Git branch for agent workspace")
        parent_task_id: Optional[str] = Field(None, description="Parent task ID if spawned from another task")
        
        @field_validator('agent_id')
        @classmethod
        def validate_uuid(cls, v: str) -> str:
            try:
                UUID(v)
            except ValueError:
                raise ValueError(f"Invalid UUID format: {v}")
            return v
        
        @field_validator('status')
        @classmethod
        def validate_status(cls, v: str) -> str:
            valid_statuses = {'IDLE', 'WORKING', 'COMPLETED', 'FAILED', 'TERMINATED'}
            if v.upper() not in valid_statuses:
                raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
            return v.upper()
        
        @field_validator('created_at')
        @classmethod
        def validate_timestamp(cls, v: str) -> str:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"Invalid ISO timestamp: {v}")
            return v

    class AgentMetadataPydanticModel(BaseModel):
        """Pydantic model for agent metadata validation."""
        agent_id: str
        model_version: str
        prompt_hash: str
        timestamp: str
        task_signature: Optional[str] = None
        
        model_config = {"extra": "allow"}


# ============================================================================
# Enums
# ============================================================================

class AgentStatus(Enum):
    """
    Agent lifecycle status enumeration.
    
    - IDLE: Agent created but not yet assigned work
    - WORKING: Agent actively processing a task
    - COMPLETED: Agent finished task successfully
    - FAILED: Agent encountered an error
    - TERMINATED: Agent manually terminated
    """
    IDLE = "IDLE"
    WORKING = "WORKING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AgentMetadata:
    """
    Metadata for agent commit tagging and tracking.
    
    Used when agents need to be referenced in version control commits.
    """
    agent_id: UUID
    model_version: str
    prompt_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_signature: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'agent_id': str(self.agent_id),
            'model_version': self.model_version,
            'prompt_hash': self.prompt_hash,
            'timestamp': self.timestamp.isoformat(),
            'task_signature': self.task_signature,
            'additional_data': self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetadata':
        """Create metadata from dictionary."""
        return cls(
            agent_id=UUID(data['agent_id']),
            model_version=data['model_version'],
            prompt_hash=data['prompt_hash'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            task_signature=data.get('task_signature'),
            additional_data=data.get('additional_data', {})
        )
    
    def generate_task_signature(self, task_description: str) -> str:
        """Generate a unique task signature based on task description."""
        signature = hashlib.sha256(
            f"{str(self.agent_id)}:{task_description}".encode()
        ).hexdigest()[:16]
        self.task_signature = signature
        return signature


@dataclass
class Agent:
    """
    Core Agent class representing an AI agent in the Ansib-eL system.
    
    Each agent has a unique identity, purpose, and isolated workspace context.
    
    Attributes:
        agent_id: Unique UUID for the agent
        purpose: Description of the agent's task/purpose
        model_version: Version of the AI model being used
        prompt_hash: SHA-256 hash of the system prompt
        created_at: Timestamp when agent was created
        status: Current lifecycle status
        workspace_branch: Git branch for agent's isolated workspace
        parent_task_id: Optional parent task ID for task hierarchies
    """
    agent_id: UUID
    purpose: str
    model_version: str
    prompt_hash: str
    created_at: datetime
    status: AgentStatus
    workspace_branch: str
    parent_task_id: Optional[str] = None
    metadata: AgentMetadata = field(default=None)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = AgentMetadata(
                agent_id=self.agent_id,
                model_version=self.model_version,
                prompt_hash=self.prompt_hash
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            'agent_id': str(self.agent_id),
            'purpose': self.purpose,
            'model_version': self.model_version,
            'prompt_hash': self.prompt_hash,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'workspace_branch': self.workspace_branch,
            'parent_task_id': self.parent_task_id,
            'metadata': self.metadata.to_dict() if self.metadata else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserialize agent from dictionary."""
        agent = cls(
            agent_id=UUID(data['agent_id']),
            purpose=data['purpose'],
            model_version=data['model_version'],
            prompt_hash=data['prompt_hash'],
            created_at=datetime.fromisoformat(data['created_at']),
            status=AgentStatus(data['status']),
            workspace_branch=data['workspace_branch'],
            parent_task_id=data.get('parent_task_id')
        )
        if data.get('metadata'):
            agent.metadata = AgentMetadata.from_dict(data['metadata'])
        return agent
    
    def to_json(self) -> str:
        """Serialize agent to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Agent':
        """Deserialize agent from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def validate(self) -> bool:
        """Validate agent data using Pydantic if available."""
        if not HAS_PYDANTIC:
            return True
        try:
            AgentPydanticModel(**self.to_dict())
            return True
        except Exception as e:
            logger.error(f"Agent validation failed: {e}")
            return False


# ============================================================================
# Context Isolation
# ============================================================================

class AgentContext:
    """
    Manages isolated workspace context for an agent.
    
    Provides:
    - Environment variable isolation
    - Workspace path management
    - Context-specific configuration
    """
    
    def __init__(self, agent_id: UUID, base_workspace_path: Optional[str] = None):
        """
        Initialize agent context.
        
        Args:
            agent_id: UUID of the agent
            base_workspace_path: Base directory for agent workspaces
        """
        self.agent_id = agent_id
        if base_workspace_path is None:
            base_workspace_path = os.path.join(tempfile.gettempdir(), "ansibel", "agents")
        self.base_workspace_path = Path(base_workspace_path)
        self.workspace_path = self.base_workspace_path / str(agent_id)

        # Path traversal guard: ensure workspace stays under base
        resolved_base = self.base_workspace_path.resolve()
        resolved_workspace = self.workspace_path.resolve()
        if not str(resolved_workspace).startswith(str(resolved_base)):
            raise AgentContextError(
                f"Workspace path {resolved_workspace} escapes base {resolved_base}"
            )
        self.env_vars: Dict[str, str] = {}
        self._original_env: Dict[str, str] = {}
        self._is_active = False
        
    def initialize(self) -> None:
        """Create workspace directory and initialize context."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.workspace_path / "workspace").mkdir(exist_ok=True)
        (self.workspace_path / "logs").mkdir(exist_ok=True)
        (self.workspace_path / "artifacts").mkdir(exist_ok=True)
        
        logger.info(f"Initialized workspace for agent {self.agent_id} at {self.workspace_path}")
    
    def set_env_var(self, key: str, value: str) -> None:
        """Set an environment variable for this agent context."""
        self.env_vars[key] = value
        
    def set_env_vars(self, vars_dict: Dict[str, str]) -> None:
        """Set multiple environment variables."""
        self.env_vars.update(vars_dict)
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable value."""
        return self.env_vars.get(key, default)
    
    def activate(self) -> None:
        """Activate the agent context (isolate environment)."""
        if self._is_active:
            return
            
        # Save original environment
        self._original_env = dict(os.environ)
        
        # Set agent-specific environment variables
        os.environ['ANSIBEL_AGENT_ID'] = str(self.agent_id)
        os.environ['ANSIBEL_WORKSPACE'] = str(self.workspace_path)
        os.environ['ANSIBEL_WORKSPACE_WORK'] = str(self.workspace_path / "workspace")
        os.environ['ANSIBEL_WORKSPACE_LOGS'] = str(self.workspace_path / "logs")
        os.environ['ANSIBEL_WORKSPACE_ARTIFACTS'] = str(self.workspace_path / "artifacts")
        
        # Apply custom environment variables
        for key, value in self.env_vars.items():
            os.environ[key] = value
            
        self._is_active = True
        logger.debug(f"Activated context for agent {self.agent_id}")
    
    def deactivate(self) -> None:
        """Deactivate the agent context (restore environment)."""
        if not self._is_active:
            return
            
        # Restore original environment
        os.environ.clear()
        os.environ.update(self._original_env)
        
        self._is_active = False
        logger.debug(f"Deactivated context for agent {self.agent_id}")
    
    def get_subprocess_env(self) -> Dict[str, str]:
        """Return a copy of os.environ with agent-specific variables set.

        Use this instead of activate() when spawning subprocesses to avoid
        mutating global state.
        """
        env = dict(os.environ)
        env['ANSIBEL_AGENT_ID'] = str(self.agent_id)
        env['ANSIBEL_WORKSPACE'] = str(self.workspace_path)
        env['ANSIBEL_WORKSPACE_WORK'] = str(self.workspace_path / "workspace")
        env['ANSIBEL_WORKSPACE_LOGS'] = str(self.workspace_path / "logs")
        env['ANSIBEL_WORKSPACE_ARTIFACTS'] = str(self.workspace_path / "artifacts")
        for key, value in self.env_vars.items():
            env[key] = value
        return env

    def get_workspace_file_path(self, filename: str, subdir: str = "workspace") -> Path:
        """Get full path for a file in the agent's workspace."""
        return self.workspace_path / subdir / filename
    
    def cleanup(self) -> None:
        """Clean up the workspace directory."""
        import shutil
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
            logger.info(f"Cleaned up workspace for agent {self.agent_id}")
    
    def __enter__(self):
        """Context manager entry."""
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.deactivate()


# ============================================================================
# Agent Manager
# ============================================================================

class AgentManager:
    """
    Central manager for agent lifecycle operations.
    
    Handles:
    - Agent spawning with unique IDs
    - Agent tracking and status management
    - Agent termination and cleanup
    - Persistence to JSON storage
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the agent manager.
        
        Args:
            storage_path: Path to JSON file for agent persistence
        """
        self.storage_path = Path(storage_path)
        self.agents: Dict[UUID, Agent] = {}
        self.contexts: Dict[UUID, AgentContext] = {}
        self._agents_by_task: Dict[str, Set[UUID]] = {}
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing agents
        self._load_agents()
        
        logger.info(f"AgentManager initialized with storage at {storage_path}")
    
    @staticmethod
    def _compute_prompt_hash(prompt: str) -> str:
        """Compute SHA-256 hash of a prompt."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _generate_workspace_branch(agent_id: UUID, purpose: str) -> str:
        """Generate a unique workspace branch name."""
        purpose_hash = hashlib.sha256(purpose.encode()).hexdigest()[:8]
        return f"agent/{str(agent_id)[:8]}/{purpose_hash}"
    
    def spawn_agent(
        self,
        purpose: str,
        model_version: str,
        prompt: str,
        task_id: str,
        parent_task_id: Optional[str] = None
    ) -> Agent:
        """
        Spawn a new agent with unique identity.
        
        Args:
            purpose: Description of the agent's task
            model_version: AI model version to use
            prompt: System prompt for the agent
            task_id: Associated task ID
            parent_task_id: Optional parent task ID
            
        Returns:
            Newly created Agent instance
        """
        agent_id = uuid4()
        prompt_hash = self._compute_prompt_hash(prompt)
        workspace_branch = self._generate_workspace_branch(agent_id, purpose)
        
        agent = Agent(
            agent_id=agent_id,
            purpose=purpose,
            model_version=model_version,
            prompt_hash=prompt_hash,
            created_at=datetime.now(timezone.utc),
            status=AgentStatus.IDLE,
            workspace_branch=workspace_branch,
            parent_task_id=parent_task_id
        )
        
        # Validate agent data
        if HAS_PYDANTIC and not agent.validate():
            raise ValueError("Agent validation failed")
        
        # Store agent
        self.agents[agent_id] = agent
        
        # Track by task
        if task_id not in self._agents_by_task:
            self._agents_by_task[task_id] = set()
        self._agents_by_task[task_id].add(agent_id)
        
        # Create isolated context
        context = AgentContext(agent_id)
        context.initialize()
        self.contexts[agent_id] = context
        
        # Persist to storage
        self._save_agents()
        
        logger.info(f"Spawned agent {agent_id} for task {task_id}")
        return agent
    
    def get_agent(self, agent_id: UUID) -> Optional[Agent]:
        """
        Retrieve an agent by ID.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agent_by_string_id(self, agent_id_str: str) -> Optional[Agent]:
        """
        Retrieve an agent by string ID.
        
        Args:
            agent_id_str: String representation of agent UUID
            
        Returns:
            Agent instance or None if not found/invalid
        """
        try:
            agent_id = UUID(agent_id_str)
            return self.get_agent(agent_id)
        except ValueError:
            logger.error(f"Invalid UUID string: {agent_id_str}")
            return None
    
    def list_active_agents(self) -> List[Agent]:
        """
        List all non-terminated agents.
        
        Returns:
            List of active agents
        """
        return [
            agent for agent in self.agents.values()
            if agent.status != AgentStatus.TERMINATED
        ]
    
    def list_all_agents(self) -> List[Agent]:
        """
        List all agents including terminated.
        
        Returns:
            List of all agents
        """
        return list(self.agents.values())
    
    def list_agents_by_status(self, status: AgentStatus) -> List[Agent]:
        """
        List agents filtered by status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of agents with matching status
        """
        return [
            agent for agent in self.agents.values()
            if agent.status == status
        ]
    
    def get_agents_by_task(self, task_id: str) -> List[Agent]:
        """
        Get all agents associated with a task.
        
        Args:
            task_id: Task ID to search for
            
        Returns:
            List of agents for the task
        """
        agent_ids = self._agents_by_task.get(task_id, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def update_agent_status(self, agent_id: UUID, status: AgentStatus) -> bool:
        """
        Update an agent's status.
        
        Args:
            agent_id: UUID of the agent
            status: New status value
            
        Returns:
            True if successful, False if agent not found
        """
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found for status update")
            return False
        
        old_status = agent.status
        agent.status = status
        
        # Persist changes
        self._save_agents()
        
        logger.info(f"Agent {agent_id} status changed: {old_status.value} -> {status.value}")
        return True
    
    def terminate_agent(self, agent_id: UUID, cleanup: bool = True) -> bool:
        """
        Terminate an agent and optionally clean up resources.
        
        Args:
            agent_id: UUID of the agent to terminate
            cleanup: Whether to clean up workspace
            
        Returns:
            True if successful, False if agent not found
        """
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found for termination")
            return False
        
        # Update status
        agent.status = AgentStatus.TERMINATED
        
        # Clean up context if requested
        if cleanup and agent_id in self.contexts:
            self.contexts[agent_id].cleanup()
            del self.contexts[agent_id]
        
        # Persist changes
        self._save_agents()
        
        logger.info(f"Terminated agent {agent_id}")
        return True
    
    def get_agent_context(self, agent_id: UUID) -> Optional[AgentContext]:
        """
        Get the context for an agent.
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            AgentContext or None if not found
        """
        return self.contexts.get(agent_id)
    
    def _save_agents(self) -> None:
        """Persist all agents to storage."""
        data = {
            'agents': {str(k): v.to_dict() for k, v in self.agents.items()},
            'agents_by_task': {k: [str(aid) for aid in v] for k, v in self._agents_by_task.items()}
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_agents(self) -> None:
        """Load agents from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load agents
            for agent_id_str, agent_data in data.get('agents', {}).items():
                agent = Agent.from_dict(agent_data)
                self.agents[agent.agent_id] = agent
            
            # Load task mappings
            for task_id, agent_ids in data.get('agents_by_task', {}).items():
                self._agents_by_task[task_id] = {
                    UUID(aid) for aid in agent_ids
                }
            
            logger.info(f"Loaded {len(self.agents)} agents from storage")
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load agents from storage: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_agents': len(self.agents),
            'active_agents': len(self.list_active_agents()),
            'by_status': {
                status.value: len(self.list_agents_by_status(status))
                for status in AgentStatus
            },
            'total_tasks': len(self._agents_by_task)
        }
    
    def cleanup_terminated(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up terminated agents older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours, None for all terminated
            
        Returns:
            Number of agents cleaned up
        """
        now = datetime.now(timezone.utc)
        to_remove = []
        
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.TERMINATED:
                if max_age_hours is None:
                    to_remove.append(agent_id)
                else:
                    age = (now - agent.created_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(agent_id)
        
        for agent_id in to_remove:
            if agent_id in self.contexts:
                self.contexts[agent_id].cleanup()
                del self.contexts[agent_id]
            del self.agents[agent_id]
        
        if to_remove:
            self._save_agents()
        
        logger.info(f"Cleaned up {len(to_remove)} terminated agents")
        return len(to_remove)


# ============================================================================
# Agent Communication Protocol
# ============================================================================

@dataclass
class AgentMessage:
    """
    Message for agent-to-agent communication.
    
    Enables structured communication between agents in the system.
    """
    message_id: UUID
    sender_id: UUID
    recipient_id: UUID
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[UUID] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': str(self.message_id),
            'sender_id': str(self.sender_id),
            'recipient_id': str(self.recipient_id),
            'message_type': self.message_type,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': str(self.correlation_id) if self.correlation_id else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            message_id=UUID(data['message_id']),
            sender_id=UUID(data['sender_id']),
            recipient_id=UUID(data['recipient_id']),
            message_type=data['message_type'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=UUID(data['correlation_id']) if data.get('correlation_id') else None
        )


class AgentCommunicationBus:
    """
    Simple in-memory message bus for agent communication.
    
    Provides publish/subscribe pattern for agent messaging.
    """
    
    def __init__(self):
        self._messages: List[AgentMessage] = []
        self._subscribers: Dict[UUID, List[callable]] = {}
    
    def send_message(
        self,
        sender_id: UUID,
        recipient_id: UUID,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[UUID] = None
    ) -> AgentMessage:
        """Send a message from one agent to another."""
        message = AgentMessage(
            message_id=uuid4(),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        
        self._messages.append(message)
        
        # Notify subscribers
        if recipient_id in self._subscribers:
            for callback in self._subscribers[recipient_id]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
        
        return message
    
    def subscribe(self, agent_id: UUID, callback: callable) -> None:
        """Subscribe an agent to receive messages."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)
    
    def unsubscribe(self, agent_id: UUID, callback: callable) -> None:
        """Unsubscribe a callback."""
        if agent_id in self._subscribers:
            self._subscribers[agent_id] = [
                cb for cb in self._subscribers[agent_id]
                if cb != callback
            ]
    
    def get_messages_for_agent(
        self,
        agent_id: UUID,
        message_type: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get messages for a specific agent."""
        messages = [
            m for m in self._messages
            if m.recipient_id == agent_id
        ]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages
    
    def clear_messages(self, agent_id: Optional[UUID] = None) -> None:
        """Clear messages, optionally for a specific agent."""
        if agent_id:
            self._messages = [
                m for m in self._messages
                if m.recipient_id != agent_id
            ]
        else:
            self._messages.clear()


# Exceptions are imported from ansibel.exceptions
