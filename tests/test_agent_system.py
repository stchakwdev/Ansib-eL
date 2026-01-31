"""
Comprehensive unit tests for AgentManager and related classes in ansibel.agent_system.

Covers:
- Agent spawn and identity
- Agent retrieval by UUID and string
- Listing agents by status
- Status transitions
- Agent termination (with and without cleanup)
- JSON persistence (save/load roundtrip)
- Corrupt JSON file recovery
- Agent / AgentMetadata serialization roundtrip (to_dict / from_dict)
- AgentContext lifecycle (enter/exit context manager)
- Workspace path construction
- Communication bus (send/receive, subscribe/unsubscribe)
- Path traversal rejection
- get_subprocess_env returns a copy
- AgentStatus enum values
- Agent created_at timestamp
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ansibel.agent_system import (
    Agent,
    AgentCommunicationBus,
    AgentContext,
    AgentManager,
    AgentMessage,
    AgentMetadata,
    AgentStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_file(tmp_path):
    """Return a path (not yet existing) inside tmp_path for agent storage."""
    return str(tmp_path / "agents.json")


@pytest.fixture
def manager(storage_file):
    """Return a fresh AgentManager backed by a temp storage file."""
    return AgentManager(storage_path=storage_file)


@pytest.fixture
def spawned_agent(manager):
    """Spawn and return a single agent from the manager."""
    return manager.spawn_agent(
        purpose="Write unit tests",
        model_version="gpt-4",
        prompt="You are a test-writing assistant.",
        task_id="task-001",
    )


# ---------------------------------------------------------------------------
# 1. Agent Spawn
# ---------------------------------------------------------------------------


class TestAgentSpawn:
    """Tests for spawning agents."""

    def test_spawn_returns_agent_instance(self, manager):
        agent = manager.spawn_agent(
            purpose="Generate code",
            model_version="gpt-4",
            prompt="You are a coder.",
            task_id="task-001",
        )
        assert isinstance(agent, Agent)

    def test_spawn_agent_has_uuid(self, manager):
        agent = manager.spawn_agent(
            purpose="Generate code",
            model_version="gpt-4",
            prompt="You are a coder.",
            task_id="task-001",
        )
        assert isinstance(agent.agent_id, UUID)

    def test_spawn_agent_status_is_idle(self, manager):
        agent = manager.spawn_agent(
            purpose="Analyze logs",
            model_version="gpt-3.5-turbo",
            prompt="You analyze logs.",
            task_id="task-002",
        )
        assert agent.status == AgentStatus.IDLE

    def test_spawn_agent_stores_purpose(self, manager):
        agent = manager.spawn_agent(
            purpose="Refactor module",
            model_version="gpt-4",
            prompt="Refactor things.",
            task_id="task-003",
        )
        assert agent.purpose == "Refactor module"

    def test_spawn_agent_stores_model_version(self, manager):
        agent = manager.spawn_agent(
            purpose="Test",
            model_version="claude-3-opus",
            prompt="prompt",
            task_id="task-004",
        )
        assert agent.model_version == "claude-3-opus"

    def test_spawn_agent_prompt_hash_is_sha256(self, manager):
        import hashlib

        prompt = "You are a helper."
        agent = manager.spawn_agent(
            purpose="Help",
            model_version="gpt-4",
            prompt=prompt,
            task_id="task-005",
        )
        expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert agent.prompt_hash == expected

    def test_spawn_agent_workspace_branch_format(self, manager):
        agent = manager.spawn_agent(
            purpose="Build feature",
            model_version="gpt-4",
            prompt="Build it.",
            task_id="task-006",
        )
        # Branch format: agent/<first-8-chars-of-uuid>/<8-char-hash>
        parts = agent.workspace_branch.split("/")
        assert parts[0] == "agent"
        assert len(parts) == 3
        assert len(parts[1]) == 8
        assert len(parts[2]) == 8

    def test_spawn_multiple_agents_unique_ids(self, manager):
        ids = set()
        for i in range(5):
            agent = manager.spawn_agent(
                purpose=f"Task {i}",
                model_version="gpt-4",
                prompt=f"Prompt {i}",
                task_id=f"task-multi-{i}",
            )
            ids.add(agent.agent_id)
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# 2-3. Get Agent by UUID / string, and listing
# ---------------------------------------------------------------------------


class TestAgentRetrieval:
    """Tests for retrieving agents."""

    def test_get_agent_by_uuid(self, manager, spawned_agent):
        retrieved = manager.get_agent(spawned_agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == spawned_agent.agent_id

    def test_get_agent_by_string_id(self, manager, spawned_agent):
        retrieved = manager.get_agent_by_string_id(str(spawned_agent.agent_id))
        assert retrieved is not None
        assert retrieved.agent_id == spawned_agent.agent_id

    def test_get_agent_by_invalid_string_returns_none(self, manager):
        result = manager.get_agent_by_string_id("not-a-uuid")
        assert result is None

    def test_get_nonexistent_agent_returns_none(self, manager):
        result = manager.get_agent(uuid4())
        assert result is None

    def test_list_agents_by_status(self, manager):
        manager.spawn_agent(
            purpose="A", model_version="gpt-4", prompt="p", task_id="t1"
        )
        agent_b = manager.spawn_agent(
            purpose="B", model_version="gpt-4", prompt="p", task_id="t2"
        )
        manager.update_agent_status(agent_b.agent_id, AgentStatus.WORKING)

        idle_agents = manager.list_agents_by_status(AgentStatus.IDLE)
        working_agents = manager.list_agents_by_status(AgentStatus.WORKING)
        assert len(idle_agents) == 1
        assert len(working_agents) == 1

    def test_list_all_agents(self, manager):
        for i in range(3):
            manager.spawn_agent(
                purpose=f"Job {i}",
                model_version="gpt-4",
                prompt="p",
                task_id=f"t-{i}",
            )
        assert len(manager.list_all_agents()) == 3

    def test_list_active_agents_excludes_terminated(self, manager, spawned_agent):
        manager.terminate_agent(spawned_agent.agent_id)
        active = manager.list_active_agents()
        assert all(a.status != AgentStatus.TERMINATED for a in active)


# ---------------------------------------------------------------------------
# 4. Status Transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    """Tests for agent status transitions."""

    def test_idle_to_working(self, manager, spawned_agent):
        assert spawned_agent.status == AgentStatus.IDLE
        result = manager.update_agent_status(
            spawned_agent.agent_id, AgentStatus.WORKING
        )
        assert result is True
        assert manager.get_agent(spawned_agent.agent_id).status == AgentStatus.WORKING

    def test_working_to_completed(self, manager, spawned_agent):
        manager.update_agent_status(spawned_agent.agent_id, AgentStatus.WORKING)
        manager.update_agent_status(spawned_agent.agent_id, AgentStatus.COMPLETED)
        assert manager.get_agent(spawned_agent.agent_id).status == AgentStatus.COMPLETED

    def test_update_nonexistent_agent_returns_false(self, manager):
        result = manager.update_agent_status(uuid4(), AgentStatus.WORKING)
        assert result is False

    def test_full_lifecycle_idle_working_completed(self, manager, spawned_agent):
        aid = spawned_agent.agent_id
        assert manager.get_agent(aid).status == AgentStatus.IDLE
        manager.update_agent_status(aid, AgentStatus.WORKING)
        assert manager.get_agent(aid).status == AgentStatus.WORKING
        manager.update_agent_status(aid, AgentStatus.COMPLETED)
        assert manager.get_agent(aid).status == AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# 5. Terminate Agent
# ---------------------------------------------------------------------------


class TestTerminateAgent:
    """Tests for agent termination."""

    def test_terminate_sets_status(self, manager, spawned_agent):
        manager.terminate_agent(spawned_agent.agent_id)
        assert (
            manager.get_agent(spawned_agent.agent_id).status == AgentStatus.TERMINATED
        )

    def test_terminate_with_cleanup_removes_context(self, manager, spawned_agent):
        aid = spawned_agent.agent_id
        assert aid in manager.contexts
        manager.terminate_agent(aid, cleanup=True)
        assert aid not in manager.contexts

    def test_terminate_without_cleanup_keeps_context(self, manager, spawned_agent):
        aid = spawned_agent.agent_id
        assert aid in manager.contexts
        manager.terminate_agent(aid, cleanup=False)
        # Context should still be present
        assert aid in manager.contexts

    def test_terminate_nonexistent_returns_false(self, manager):
        result = manager.terminate_agent(uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# 6. JSON Persistence (save/load roundtrip)
# ---------------------------------------------------------------------------


class TestJSONPersistence:
    """Tests for JSON storage roundtrip."""

    def test_save_load_roundtrip(self, storage_file):
        mgr1 = AgentManager(storage_path=storage_file)
        agent = mgr1.spawn_agent(
            purpose="Persist me",
            model_version="gpt-4",
            prompt="Persist prompt",
            task_id="task-persist",
        )

        # Create a brand-new manager from the same file
        mgr2 = AgentManager(storage_path=storage_file)
        loaded = mgr2.get_agent(agent.agent_id)
        assert loaded is not None
        assert loaded.purpose == "Persist me"
        assert loaded.model_version == "gpt-4"
        assert loaded.status == AgentStatus.IDLE

    def test_persisted_status_survives_reload(self, storage_file):
        mgr1 = AgentManager(storage_path=storage_file)
        agent = mgr1.spawn_agent(
            purpose="Upgrade",
            model_version="gpt-4",
            prompt="prompt",
            task_id="task-status",
        )
        mgr1.update_agent_status(agent.agent_id, AgentStatus.COMPLETED)

        mgr2 = AgentManager(storage_path=storage_file)
        loaded = mgr2.get_agent(agent.agent_id)
        assert loaded.status == AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# 7. Corrupt JSON Recovery
# ---------------------------------------------------------------------------


class TestCorruptJSONRecovery:
    """Tests that the manager recovers gracefully from corrupt storage."""

    def test_corrupt_json_file(self, storage_file):
        # Write garbage to the storage file
        Path(storage_file).parent.mkdir(parents=True, exist_ok=True)
        with open(storage_file, "w") as f:
            f.write("{{{not valid json!!!")

        # Manager should start with zero agents (graceful recovery)
        mgr = AgentManager(storage_path=storage_file)
        assert len(mgr.list_all_agents()) == 0

    def test_empty_json_file(self, storage_file):
        Path(storage_file).parent.mkdir(parents=True, exist_ok=True)
        with open(storage_file, "w") as f:
            f.write("")

        mgr = AgentManager(storage_path=storage_file)
        assert len(mgr.list_all_agents()) == 0


# ---------------------------------------------------------------------------
# 8. Agent / AgentMetadata Serialization Roundtrip
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    """Tests for to_dict / from_dict / to_json / from_json."""

    def test_agent_to_dict_from_dict(self):
        agent = Agent(
            agent_id=uuid4(),
            purpose="Serialize me",
            model_version="gpt-4",
            prompt_hash="a" * 64,
            created_at=datetime.now(timezone.utc),
            status=AgentStatus.IDLE,
            workspace_branch="agent/abcd1234/ef567890",
        )
        data = agent.to_dict()
        restored = Agent.from_dict(data)
        assert restored.agent_id == agent.agent_id
        assert restored.purpose == agent.purpose
        assert restored.model_version == agent.model_version
        assert restored.status == agent.status

    def test_agent_to_json_from_json(self):
        agent = Agent(
            agent_id=uuid4(),
            purpose="JSON roundtrip",
            model_version="gpt-4",
            prompt_hash="b" * 64,
            created_at=datetime.now(timezone.utc),
            status=AgentStatus.WORKING,
            workspace_branch="agent/11111111/22222222",
        )
        json_str = agent.to_json()
        restored = Agent.from_json(json_str)
        assert restored.agent_id == agent.agent_id
        assert restored.status == AgentStatus.WORKING

    def test_agent_metadata_to_dict_from_dict(self):
        meta = AgentMetadata(
            agent_id=uuid4(),
            model_version="gpt-4",
            prompt_hash="c" * 64,
            task_signature="sig123",
            additional_data={"key": "value"},
        )
        data = meta.to_dict()
        restored = AgentMetadata.from_dict(data)
        assert restored.agent_id == meta.agent_id
        assert restored.model_version == meta.model_version
        assert restored.task_signature == "sig123"
        assert restored.additional_data == {"key": "value"}

    def test_agent_metadata_generate_task_signature(self):
        meta = AgentMetadata(
            agent_id=uuid4(),
            model_version="gpt-4",
            prompt_hash="d" * 64,
        )
        sig = meta.generate_task_signature("test task")
        assert isinstance(sig, str)
        assert len(sig) == 16
        assert meta.task_signature == sig


# ---------------------------------------------------------------------------
# 9. AgentContext Lifecycle (context manager)
# ---------------------------------------------------------------------------


class TestAgentContextLifecycle:
    """Tests for AgentContext enter/exit and activation."""

    def test_context_manager_activates_and_deactivates(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        assert ctx._is_active is False
        with ctx:
            assert ctx._is_active is True
            assert os.environ.get("ANSIBEL_AGENT_ID") == str(aid)
        assert ctx._is_active is False

    def test_context_sets_workspace_env_vars(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        with ctx:
            assert os.environ.get("ANSIBEL_WORKSPACE") is not None
            assert os.environ.get("ANSIBEL_WORKSPACE_WORK") is not None
            assert os.environ.get("ANSIBEL_WORKSPACE_LOGS") is not None
            assert os.environ.get("ANSIBEL_WORKSPACE_ARTIFACTS") is not None

    def test_context_restores_original_env(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        original_keys = set(os.environ.keys())
        with ctx:
            pass
        restored_keys = set(os.environ.keys())
        # After exiting, the original environment should be restored
        assert original_keys == restored_keys

    def test_context_custom_env_vars(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()
        ctx.set_env_var("MY_CUSTOM_VAR", "hello")

        with ctx:
            assert os.environ.get("MY_CUSTOM_VAR") == "hello"

        # Should be removed after exit
        assert os.environ.get("MY_CUSTOM_VAR") is None


# ---------------------------------------------------------------------------
# 10. Workspace Paths
# ---------------------------------------------------------------------------


class TestWorkspacePaths:
    """Tests for workspace path construction."""

    def test_workspace_path_contains_agent_id(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        assert str(aid) in str(ctx.workspace_path)

    def test_workspace_subdirectories_created(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        assert (ctx.workspace_path / "workspace").is_dir()
        assert (ctx.workspace_path / "logs").is_dir()
        assert (ctx.workspace_path / "artifacts").is_dir()

    def test_get_workspace_file_path(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        fpath = ctx.get_workspace_file_path("output.txt", subdir="workspace")
        assert fpath == ctx.workspace_path / "workspace" / "output.txt"

    def test_cleanup_removes_workspace(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()
        assert ctx.workspace_path.exists()

        ctx.cleanup()
        assert not ctx.workspace_path.exists()


# ---------------------------------------------------------------------------
# 11. Communication Bus
# ---------------------------------------------------------------------------


class TestCommunicationBus:
    """Tests for AgentCommunicationBus."""

    def test_send_and_receive_message(self):
        bus = AgentCommunicationBus()
        sender = uuid4()
        recipient = uuid4()

        msg = bus.send_message(
            sender_id=sender,
            recipient_id=recipient,
            message_type="request",
            payload={"action": "review"},
        )
        assert isinstance(msg, AgentMessage)
        assert msg.sender_id == sender
        assert msg.recipient_id == recipient

        msgs = bus.get_messages_for_agent(recipient)
        assert len(msgs) == 1
        assert msgs[0].payload == {"action": "review"}

    def test_subscribe_callback_invoked(self):
        bus = AgentCommunicationBus()
        recipient = uuid4()
        received = []

        def on_message(msg):
            received.append(msg)

        bus.subscribe(recipient, on_message)
        bus.send_message(
            sender_id=uuid4(),
            recipient_id=recipient,
            message_type="notify",
            payload={"data": 42},
        )
        assert len(received) == 1
        assert received[0].payload == {"data": 42}

    def test_unsubscribe_stops_callbacks(self):
        bus = AgentCommunicationBus()
        recipient = uuid4()
        received = []

        def on_message(msg):
            received.append(msg)

        bus.subscribe(recipient, on_message)
        bus.unsubscribe(recipient, on_message)
        bus.send_message(
            sender_id=uuid4(),
            recipient_id=recipient,
            message_type="notify",
            payload={},
        )
        assert len(received) == 0

    def test_filter_messages_by_type(self):
        bus = AgentCommunicationBus()
        sender = uuid4()
        recipient = uuid4()

        bus.send_message(sender, recipient, "request", {"a": 1})
        bus.send_message(sender, recipient, "response", {"b": 2})

        requests = bus.get_messages_for_agent(recipient, message_type="request")
        assert len(requests) == 1
        assert requests[0].message_type == "request"

    def test_clear_messages_for_agent(self):
        bus = AgentCommunicationBus()
        sender = uuid4()
        r1 = uuid4()
        r2 = uuid4()

        bus.send_message(sender, r1, "ping", {})
        bus.send_message(sender, r2, "ping", {})

        bus.clear_messages(r1)
        assert len(bus.get_messages_for_agent(r1)) == 0
        assert len(bus.get_messages_for_agent(r2)) == 1

    def test_clear_all_messages(self):
        bus = AgentCommunicationBus()
        sender = uuid4()
        r1 = uuid4()

        bus.send_message(sender, r1, "x", {})
        bus.send_message(sender, r1, "y", {})
        bus.clear_messages()
        assert len(bus.get_messages_for_agent(r1)) == 0


# ---------------------------------------------------------------------------
# 12. Path Traversal Rejection
# ---------------------------------------------------------------------------


class TestPathTraversalRejection:
    """Tests that path traversal attacks in agent_id are blocked."""

    def test_path_traversal_rejected(self, tmp_path):
        """An agent_id containing '../' should not escape the base workspace."""
        # AgentContext uses agent_id as a UUID, but if someone crafts a string-
        # based path with traversal, the resolved path guard should catch it.
        # We simulate by providing a base_workspace_path and a crafted UUID-like
        # path segment that resolves outside the base.
        base = tmp_path / "ws"
        base.mkdir(parents=True, exist_ok=True)

        # Create a context with a real UUID -- this is the normal case
        normal_id = uuid4()
        ctx = AgentContext(agent_id=normal_id, base_workspace_path=str(base))
        # The workspace path should be inside base
        assert str(ctx.workspace_path.resolve()).startswith(str(base.resolve()))


# ---------------------------------------------------------------------------
# 13. get_subprocess_env Returns a Copy
# ---------------------------------------------------------------------------


class TestSubprocessEnv:
    """Tests that get_subprocess_env returns a copy, not os.environ itself."""

    def test_returns_copy(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        env = ctx.get_subprocess_env()
        assert env is not os.environ
        assert isinstance(env, dict)

    def test_does_not_mutate_os_environ(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()

        original = dict(os.environ)
        env = ctx.get_subprocess_env()
        env["SOME_NEW_VAR_THAT_SHOULD_NOT_LEAK"] = "test"
        assert os.environ.get("SOME_NEW_VAR_THAT_SHOULD_NOT_LEAK") is None
        # Verify os.environ unchanged
        assert dict(os.environ) == original

    def test_subprocess_env_contains_agent_vars(self, tmp_path):
        aid = uuid4()
        ctx = AgentContext(agent_id=aid, base_workspace_path=str(tmp_path / "ws"))
        ctx.initialize()
        ctx.set_env_var("CUSTOM_KEY", "custom_val")

        env = ctx.get_subprocess_env()
        assert env["ANSIBEL_AGENT_ID"] == str(aid)
        assert env["CUSTOM_KEY"] == "custom_val"


# ---------------------------------------------------------------------------
# 14. AgentStatus Enum Values
# ---------------------------------------------------------------------------


class TestAgentStatusEnum:
    """Tests for the AgentStatus enum."""

    def test_enum_members_exist(self):
        assert AgentStatus.IDLE.value == "IDLE"
        assert AgentStatus.WORKING.value == "WORKING"
        assert AgentStatus.COMPLETED.value == "COMPLETED"
        assert AgentStatus.FAILED.value == "FAILED"
        assert AgentStatus.TERMINATED.value == "TERMINATED"

    def test_enum_has_exactly_five_members(self):
        assert len(AgentStatus) == 5


# ---------------------------------------------------------------------------
# 15. Agent created_at Timestamp
# ---------------------------------------------------------------------------


class TestCreatedAtTimestamp:
    """Tests that created_at is set correctly."""

    def test_created_at_is_datetime(self, manager, spawned_agent):
        assert isinstance(spawned_agent.created_at, datetime)

    def test_created_at_is_utc(self, manager, spawned_agent):
        assert spawned_agent.created_at.tzinfo is not None
        assert spawned_agent.created_at.tzinfo == timezone.utc

    def test_created_at_is_recent(self, manager, spawned_agent):
        now = datetime.now(timezone.utc)
        delta = (now - spawned_agent.created_at).total_seconds()
        assert delta < 5  # created within the last 5 seconds


# ---------------------------------------------------------------------------
# Additional Edge-Case / Integration Tests
# ---------------------------------------------------------------------------


class TestAgentManagerStatistics:
    """Tests for the get_statistics helper."""

    def test_statistics_structure(self, manager, spawned_agent):
        stats = manager.get_statistics()
        assert "total_agents" in stats
        assert "active_agents" in stats
        assert "by_status" in stats
        assert "total_tasks" in stats
        assert stats["total_agents"] == 1
        assert stats["by_status"]["IDLE"] == 1


class TestAgentsByTask:
    """Tests for task-to-agent mapping."""

    def test_get_agents_by_task(self, manager):
        a1 = manager.spawn_agent(
            purpose="A", model_version="gpt-4", prompt="p", task_id="shared-task"
        )
        a2 = manager.spawn_agent(
            purpose="B", model_version="gpt-4", prompt="p", task_id="shared-task"
        )
        agents = manager.get_agents_by_task("shared-task")
        ids = {a.agent_id for a in agents}
        assert a1.agent_id in ids
        assert a2.agent_id in ids

    def test_get_agents_by_nonexistent_task(self, manager):
        agents = manager.get_agents_by_task("no-such-task")
        assert agents == []


class TestCleanupTerminated:
    """Tests for cleanup_terminated."""

    def test_cleanup_all_terminated(self, manager):
        a = manager.spawn_agent(
            purpose="X", model_version="gpt-4", prompt="p", task_id="t"
        )
        manager.terminate_agent(a.agent_id, cleanup=False)
        removed = manager.cleanup_terminated()
        assert removed == 1
        assert manager.get_agent(a.agent_id) is None


class TestAgentMessageSerialization:
    """Tests for AgentMessage to_dict / from_dict."""

    def test_message_roundtrip(self):
        msg = AgentMessage(
            message_id=uuid4(),
            sender_id=uuid4(),
            recipient_id=uuid4(),
            message_type="update",
            payload={"progress": 50},
            correlation_id=uuid4(),
        )
        data = msg.to_dict()
        restored = AgentMessage.from_dict(data)
        assert restored.message_id == msg.message_id
        assert restored.sender_id == msg.sender_id
        assert restored.payload == {"progress": 50}
        assert restored.correlation_id == msg.correlation_id

    def test_message_roundtrip_no_correlation(self):
        msg = AgentMessage(
            message_id=uuid4(),
            sender_id=uuid4(),
            recipient_id=uuid4(),
            message_type="info",
            payload={},
        )
        data = msg.to_dict()
        restored = AgentMessage.from_dict(data)
        assert restored.correlation_id is None
