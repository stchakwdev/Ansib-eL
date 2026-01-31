"""
Comprehensive async tests for the TournamentOrchestrator and related components.

Tests cover:
- Tournament creation and lifecycle
- Parallel agent execution (success, failure, timeout, cancellation)
- Selection modes (AUTO_BEST, THRESHOLD, HUMAN_CHOICE)
- Review presentation and markdown output
- Winner selection and loser archiving
- Evaluation strategies (complexity, test-pass, requirement-match, composite)
- DiffPresenter (unified and side-by-side modes)
- Semaphore concurrency limits
- Leaderboard ranking
- Progress callbacks
- Dataclass and enum validation
- AsyncAgentAdapter
"""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from ansibel.tournament import (
    AgentConfig,
    ArchivedSolution,
    AsyncAgentAdapter,
    ComplexityEvaluator,
    CompositeEvaluator,
    DiffPresenter,
    RequirementMatchEvaluator,
    ReviewPresentation,
    SelectionMode,
    Solution,
    SolutionStatus,
    Task,
    TestPassEvaluator,
    Tournament,
    TournamentOrchestrator,
    TournamentResult,
    TournamentStatus,
)

# =============================================================================
# Mock Agent Manager
# =============================================================================


class MockAgentManager:
    """Configurable mock implementing the AgentManager Protocol.

    Parameters:
        execution_results: mapping of ``agent_id`` to the Solution that
            ``execute_task`` should return.  When an agent_id is absent the
            default behaviour is to return a COMPLETED solution.
        fail_agents: set of agent_ids that should raise an exception.
        timeout_agents: set of agent_ids that should sleep forever (exceed timeout).
        execution_delay: artificial delay (seconds) added to every execution.
    """

    def __init__(
        self,
        execution_results: dict[str, Solution] | None = None,
        fail_agents: set | None = None,
        timeout_agents: set | None = None,
        execution_delay: float = 0.0,
    ):
        self.execution_results = execution_results or {}
        self.fail_agents = fail_agents or set()
        self.timeout_agents = timeout_agents or set()
        self.execution_delay = execution_delay
        self.spawned: list[str] = []
        self.terminated: list[str] = []

    async def spawn_agent(self, config: AgentConfig) -> str:
        self.spawned.append(config.agent_id)
        return config.agent_id

    async def execute_task(self, agent_id: str, task: Task) -> Solution:
        if self.execution_delay > 0:
            await asyncio.sleep(self.execution_delay)

        if agent_id in self.timeout_agents:
            # Sleep long enough to exceed any reasonable timeout
            await asyncio.sleep(9999)

        if agent_id in self.fail_agents:
            raise RuntimeError(f"Agent {agent_id} intentionally failed")

        if agent_id in self.execution_results:
            return self.execution_results[agent_id]

        # Default: return a completed solution with some files changed
        return Solution(
            solution_id=str(uuid.uuid4()),
            agent_id=agent_id,
            task_id=task.task_id,
            status=SolutionStatus.COMPLETED,
            files_changed={"main.py": "print('hello')"},
            diff="--- a/main.py\n+++ b/main.py\n+print('hello')",
            explanation="Implemented the solution successfully.",
            metrics={"quality": 0.9},
            test_results={"passed": 10, "total": 10},
        )

    async def terminate_agent(self, agent_id: str) -> None:
        self.terminated.append(agent_id)


# =============================================================================
# Helper factories
# =============================================================================


def make_task(**overrides) -> Task:
    defaults = dict(
        task_id="task-1",
        description="Implement a hello world function",
        requirements=["must print hello", "must return string"],
    )
    defaults.update(overrides)
    return Task(**defaults)


def make_config(agent_id: str, **overrides) -> AgentConfig:
    defaults = dict(agent_id=agent_id, agent_type="gpt-4", timeout_seconds=5.0)
    defaults.update(overrides)
    return AgentConfig(**defaults)


def make_solution(agent_id: str, task_id: str = "task-1", **overrides) -> Solution:
    defaults = dict(
        solution_id=str(uuid.uuid4()),
        agent_id=agent_id,
        task_id=task_id,
        status=SolutionStatus.COMPLETED,
        files_changed={"main.py": "print('hello')"},
        diff="--- a/main.py\n+++ b/main.py\n+print('hello')",
        explanation="Implemented hello.",
        metrics={"quality": 0.85},
        test_results={"passed": 8, "total": 10},
    )
    defaults.update(overrides)
    return Solution(**defaults)


# =============================================================================
# Tests: Tournament Creation
# =============================================================================


class TestTournamentCreation:
    """Tests for TournamentOrchestrator.create_tournament."""

    async def test_create_tournament_returns_tournament_object(self):
        orch = TournamentOrchestrator(MockAgentManager())
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)

        assert isinstance(t, Tournament)
        assert t.task is task
        assert len(t.agent_configs) == 2
        assert t.selection_mode == SelectionMode.HUMAN_CHOICE
        assert t.status == TournamentStatus.CREATED

    async def test_create_tournament_generates_unique_id(self):
        orch = TournamentOrchestrator(MockAgentManager())
        t1 = orch.create_tournament(make_task(), [make_config("a1")])
        t2 = orch.create_tournament(make_task(), [make_config("a2")])
        assert t1.tournament_id != t2.tournament_id

    async def test_create_tournament_sorts_by_priority(self):
        orch = TournamentOrchestrator(MockAgentManager())
        configs = [
            make_config("low", priority=1),
            make_config("high", priority=10),
            make_config("mid", priority=5),
        ]
        t = orch.create_tournament(make_task(), configs)
        priorities = [c.priority for c in t.agent_configs]
        assert priorities == [10, 5, 1]


# =============================================================================
# Tests: Running Tournaments
# =============================================================================


class TestRunTournament:
    """Tests for full tournament execution."""

    async def test_run_all_agents_succeed(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2"), make_config("a3")]
        t = orch.create_tournament(task, configs, SelectionMode.AUTO_BEST)

        result = await orch.run_tournament(t.tournament_id)

        assert isinstance(result, TournamentResult)
        assert result.execution_summary["total_agents"] == 3
        assert result.execution_summary["successful"] == 3
        assert result.execution_summary["failed"] == 0
        assert t.status == TournamentStatus.COMPLETED

    async def test_run_partial_failures(self):
        mock = MockAgentManager(fail_agents={"a2"})
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2"), make_config("a3")]
        t = orch.create_tournament(task, configs, SelectionMode.AUTO_BEST)

        result = await orch.run_tournament(t.tournament_id)

        assert result.execution_summary["successful"] == 2
        assert result.execution_summary["failed"] == 1
        # The failed solution should still be recorded
        assert "a2" in t.solutions
        assert t.solutions["a2"].status == SolutionStatus.FAILED

    async def test_run_with_timeout(self):
        mock = MockAgentManager(timeout_agents={"a1"})
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [
            make_config("a1", timeout_seconds=0.1),
            make_config("a2"),
        ]
        t = orch.create_tournament(task, configs, SelectionMode.AUTO_BEST)

        result = await orch.run_tournament(t.tournament_id)

        assert t.solutions["a1"].status == SolutionStatus.TIMEOUT
        assert t.solutions["a2"].status == SolutionStatus.COMPLETED
        assert result.execution_summary["timeouts"] == 1

    async def test_run_nonexistent_tournament_raises(self):
        orch = TournamentOrchestrator(MockAgentManager())
        with pytest.raises(ValueError, match="not found"):
            await orch.run_tournament("nonexistent-id")


# =============================================================================
# Tests: Cancellation
# =============================================================================


class TestCancellation:
    """Tests for tournament cancellation."""

    async def test_cancel_running_tournament(self):
        mock = MockAgentManager(execution_delay=5.0)
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs)

        # Start in background and cancel immediately
        run_task = asyncio.create_task(orch.run_tournament(t.tournament_id))
        await asyncio.sleep(0.05)
        cancelled = await orch.cancel_tournament(t.tournament_id)

        assert cancelled is True
        assert t.status == TournamentStatus.CANCELLED

        # Let the background task finish (gather will still complete)
        result = await run_task
        assert isinstance(result, TournamentResult)

    async def test_cancel_nonexistent_tournament(self):
        orch = TournamentOrchestrator(MockAgentManager())
        result = await orch.cancel_tournament("does-not-exist")
        assert result is False


# =============================================================================
# Tests: Selection Modes
# =============================================================================


class TestSelectionModes:
    """Tests for AUTO_BEST, THRESHOLD, and HUMAN_CHOICE selection."""

    async def test_auto_best_selects_highest_score(self):
        # Build solutions with known scores
        sol_a = make_solution("a1", test_results={"passed": 10, "total": 10})
        sol_b = make_solution("a2", test_results={"passed": 5, "total": 10})

        mock = MockAgentManager(execution_results={"a1": sol_a, "a2": sol_b})
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.AUTO_BEST)

        result = await orch.run_tournament(t.tournament_id)

        assert result.winner is not None
        assert result.winner.agent_id == "a1"

    async def test_threshold_selects_first_above_threshold(self):
        sol_a = make_solution(
            "a1",
            files_changed={"main.py": "x = 1"},
            test_results={"passed": 10, "total": 10},
            explanation="must print hello must return string",
        )
        sol_b = make_solution(
            "a2",
            files_changed={"main.py": "x = 1"},
            test_results={"passed": 2, "total": 10},
        )

        mock = MockAgentManager(execution_results={"a1": sol_a, "a2": sol_b})
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.THRESHOLD)

        result = await orch.run_tournament(t.tournament_id)

        assert result.winner is not None
        # At least one solution should meet threshold
        assert result.winner.status == SolutionStatus.COMPLETED

    async def test_human_choice_no_auto_winner(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)

        result = await orch.run_tournament(t.tournament_id)

        # HUMAN_CHOICE should not auto-select winner
        assert result.winner is None


# =============================================================================
# Tests: Review Presentation
# =============================================================================


class TestReviewPresentation:
    """Tests for present_for_review and markdown output."""

    async def test_present_for_review_returns_review(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)
        await orch.run_tournament(t.tournament_id)

        review = await orch.present_for_review(t.tournament_id)

        assert isinstance(review, ReviewPresentation)
        assert review.tournament_id == t.tournament_id
        assert review.task_description == task.description
        assert len(review.solution_comparisons) == 2
        assert len(review.diffs) == 2
        assert len(review.agent_metadata) == 2

    async def test_review_markdown_output(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)
        await orch.run_tournament(t.tournament_id)

        review = await orch.present_for_review(t.tournament_id)
        md = review.to_markdown()

        assert "# Tournament Review:" in md
        assert "## Task:" in md
        assert "## Solutions Overview" in md
        assert "## Detailed Diffs" in md

    async def test_present_nonexistent_raises(self):
        orch = TournamentOrchestrator(MockAgentManager())
        with pytest.raises(ValueError, match="not found"):
            await orch.present_for_review("bad-id")


# =============================================================================
# Tests: Winner Selection and Loser Archiving
# =============================================================================


class TestWinnerAndArchiving:
    """Tests for select_winner and archive_losers."""

    async def test_select_winner_by_id(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)
        await orch.run_tournament(t.tournament_id)

        target_id = t.solutions["a1"].solution_id
        winner = await orch.select_winner(t.tournament_id, winner_id=target_id)

        assert winner.solution_id == target_id
        assert t.winner_id == target_id

    async def test_select_winner_auto_fallback(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs, SelectionMode.AUTO_BEST)
        await orch.run_tournament(t.tournament_id)

        winner = await orch.select_winner(t.tournament_id)
        assert winner.agent_id == "a1"

    async def test_select_invalid_winner_raises(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs)
        await orch.run_tournament(t.tournament_id)

        with pytest.raises(ValueError, match="not found"):
            await orch.select_winner(t.tournament_id, winner_id="nonexistent")

    async def test_archive_losers(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1"), make_config("a2"), make_config("a3")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)
        await orch.run_tournament(t.tournament_id)

        # Select a1 as winner
        target_id = t.solutions["a1"].solution_id
        await orch.select_winner(t.tournament_id, winner_id=target_id)

        archived = await orch.archive_losers(t.tournament_id)

        assert len(archived) == 2
        assert all(isinstance(a, ArchivedSolution) for a in archived)
        archived_agent_ids = {a.solution.agent_id for a in archived}
        assert "a1" not in archived_agent_ids
        assert "a2" in archived_agent_ids
        assert "a3" in archived_agent_ids

    async def test_archive_without_winner_raises(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs, SelectionMode.HUMAN_CHOICE)
        await orch.run_tournament(t.tournament_id)

        with pytest.raises(ValueError, match="No winner"):
            await orch.archive_losers(t.tournament_id)


# =============================================================================
# Tests: Evaluation Strategies
# =============================================================================


class TestEvaluationStrategies:
    """Tests for ComplexityEvaluator, TestPassEvaluator, RequirementMatchEvaluator, CompositeEvaluator."""

    async def test_complexity_evaluator_moderate_code(self):
        evaluator = ComplexityEvaluator()
        solution = make_solution(
            "a1", files_changed={"app.py": "x = 1\ny = 2\nz = 3\nw = 4\nv = 5\n"}
        )
        score = await evaluator.evaluate(solution, make_task())
        assert 0.0 <= score <= 1.0
        assert evaluator.name == "complexity"

    async def test_complexity_evaluator_high_control_flow(self):
        evaluator = ComplexityEvaluator()
        code = "\n".join([f"if x == {i}: pass" for i in range(50)])
        solution = make_solution("a1", files_changed={"app.py": code})
        score = await evaluator.evaluate(solution, make_task())
        # High control flow density should reduce the score
        assert score < 1.0

    async def test_test_pass_evaluator_perfect_score(self):
        evaluator = TestPassEvaluator()
        solution = make_solution("a1", test_results={"passed": 10, "total": 10})
        score = await evaluator.evaluate(solution, make_task())
        assert score == 1.0
        assert evaluator.name == "test_pass"

    async def test_test_pass_evaluator_partial(self):
        evaluator = TestPassEvaluator()
        solution = make_solution("a1", test_results={"passed": 7, "total": 10})
        score = await evaluator.evaluate(solution, make_task())
        # 7/10 * 0.9 = 0.63
        assert 0.5 < score < 1.0

    async def test_test_pass_evaluator_no_tests(self):
        evaluator = TestPassEvaluator()
        solution = make_solution("a1", test_results={})
        score = await evaluator.evaluate(solution, make_task())
        assert score == 0.5

    async def test_requirement_match_evaluator(self):
        evaluator = RequirementMatchEvaluator()
        task = make_task(requirements=["print hello", "return string"])
        solution = make_solution(
            "a1",
            explanation="We print hello and return string properly.",
            diff="print hello return string",
        )
        score = await evaluator.evaluate(solution, task)
        assert score > 0.0
        assert evaluator.name == "requirement_match"

    async def test_requirement_match_no_requirements(self):
        evaluator = RequirementMatchEvaluator()
        task = make_task(requirements=[])
        solution = make_solution("a1")
        score = await evaluator.evaluate(solution, task)
        assert score == 1.0

    async def test_composite_evaluator(self):
        strategies = [
            (TestPassEvaluator(), 0.5),
            (ComplexityEvaluator(), 0.3),
            (RequirementMatchEvaluator(), 0.2),
        ]
        evaluator = CompositeEvaluator(strategies)
        solution = make_solution(
            "a1",
            files_changed={"main.py": "print('hello')\nreturn 'world'\n"},
            test_results={"passed": 10, "total": 10},
            explanation="print hello return string",
        )
        task = make_task(requirements=["print hello"])
        score = await evaluator.evaluate(solution, task)
        assert 0.0 <= score <= 1.0
        assert evaluator.name == "composite"

    async def test_composite_evaluator_empty(self):
        evaluator = CompositeEvaluator([])
        score = await evaluator.evaluate(make_solution("a1"), make_task())
        assert score == 0.5


# =============================================================================
# Tests: DiffPresenter
# =============================================================================


class TestDiffPresenter:
    """Tests for unified diff and side-by-side presentation."""

    def test_unified_diff_shows_changes(self):
        presenter = DiffPresenter(context_lines=2)
        original = {"file.py": "line1\nline2\nline3\n"}
        modified = {"file.py": "line1\nmodified\nline3\n"}

        diff = presenter.format_unified_diff(original, modified)

        assert "---" in diff
        assert "+++" in diff
        assert "-line2" in diff
        assert "+modified" in diff

    def test_unified_diff_no_changes(self):
        presenter = DiffPresenter()
        content = {"file.py": "same content"}
        diff = presenter.format_unified_diff(content, content)
        assert diff == "No changes"

    def test_side_by_side_format(self):
        presenter = DiffPresenter()
        original = {"file.py": "line1\nline2"}
        modified = {"file.py": "line1\nchanged"}

        output = presenter.format_side_by_side(original, modified, width=40)

        assert "file.py" in output
        assert "*" in output  # Difference marker

    def test_solution_summary(self):
        presenter = DiffPresenter()
        solution = make_solution("a1")
        summary = presenter.format_solution_summary(solution)

        assert "Agent: a1" in summary
        assert "Status: completed" in summary

    def test_highlight_differences_multiple_solutions(self):
        presenter = DiffPresenter()
        sol1 = make_solution("a1", files_changed={"f.py": "version1"})
        sol2 = make_solution("a2", files_changed={"f.py": "version2"})

        diffs = presenter.highlight_differences([sol1, sol2])

        assert "f.py" in diffs
        assert diffs["f.py"]["num_unique_versions"] == 2

    def test_highlight_differences_with_focus_files(self):
        presenter = DiffPresenter()
        sol1 = make_solution("a1", files_changed={"a.py": "x", "b.py": "y"})
        sol2 = make_solution("a2", files_changed={"a.py": "z", "b.py": "y"})

        diffs = presenter.highlight_differences([sol1, sol2], focus_files=["a.py"])

        assert "a.py" in diffs
        assert "b.py" not in diffs

    def test_highlight_differences_empty(self):
        presenter = DiffPresenter()
        assert presenter.highlight_differences([]) == {}


# =============================================================================
# Tests: Semaphore Concurrency Limits
# =============================================================================


class TestConcurrencyLimits:
    """Tests for max_concurrent_agents semaphore enforcement."""

    async def test_semaphore_limits_concurrency(self):
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        class ConcurrencyTrackingManager(MockAgentManager):
            async def execute_task(self, agent_id: str, task: Task) -> Solution:
                nonlocal peak_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > peak_concurrent:
                        peak_concurrent = current_concurrent
                await asyncio.sleep(0.05)
                async with lock:
                    current_concurrent -= 1
                return await super().execute_task(agent_id, task)

        mock = ConcurrencyTrackingManager()
        orch = TournamentOrchestrator(mock, max_concurrent_agents=2)
        task = make_task()
        configs = [make_config(f"a{i}") for i in range(5)]
        t = orch.create_tournament(task, configs)

        await orch.run_tournament(t.tournament_id)

        assert peak_concurrent <= 2


# =============================================================================
# Tests: Leaderboard
# =============================================================================


class TestLeaderboard:
    """Tests for Tournament.get_leaderboard."""

    async def test_leaderboard_ranking(self):
        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        task = make_task()

        sol_a = make_solution("a1", test_results={"passed": 10, "total": 10})
        sol_b = make_solution("a2", test_results={"passed": 5, "total": 10})
        sol_c = make_solution("a3", test_results={"passed": 8, "total": 10})

        mock.execution_results = {"a1": sol_a, "a2": sol_b, "a3": sol_c}
        configs = [make_config("a1"), make_config("a2"), make_config("a3")]
        t = orch.create_tournament(task, configs)

        await orch.run_tournament(t.tournament_id)

        lb = t.get_leaderboard()
        assert len(lb) >= 2
        # The leaderboard is sorted descending by composite score
        scores = [score for _, score in lb]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Tests: Progress Callbacks
# =============================================================================


class TestProgressCallbacks:
    """Tests for register/unregister progress callbacks."""

    async def test_progress_callback_receives_events(self):
        events = []

        def on_progress(tournament_id: str, info: dict):
            events.append(info)

        mock = MockAgentManager()
        orch = TournamentOrchestrator(mock)
        orch.register_progress_callback(on_progress)

        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs)

        await orch.run_tournament(t.tournament_id)

        statuses = [e["status"] for e in events]
        assert "started" in statuses
        assert "agent_started" in statuses
        assert "agent_completed" in statuses
        assert "completed" in statuses

    async def test_unregister_callback(self):
        events = []

        def on_progress(tournament_id, info):
            events.append(info)

        orch = TournamentOrchestrator(MockAgentManager())
        orch.register_progress_callback(on_progress)
        orch.unregister_progress_callback(on_progress)

        task = make_task()
        configs = [make_config("a1")]
        t = orch.create_tournament(task, configs)
        await orch.run_tournament(t.tournament_id)

        assert len(events) == 0


# =============================================================================
# Tests: Dataclass and Enum Validation
# =============================================================================


class TestDataclassAndEnumValidation:
    """Tests for AgentConfig, SolutionStatus, TournamentStatus enums."""

    def test_agent_config_defaults(self):
        config = AgentConfig(agent_id="x", agent_type="gpt-4")
        assert config.timeout_seconds == 300.0
        assert config.priority == 0
        assert config.model_config == {}
        assert config.system_prompt is None

    def test_agent_config_auto_id(self):
        config = AgentConfig(agent_id="", agent_type="gpt-4")
        assert config.agent_id != ""  # __post_init__ generates UUID

    def test_solution_status_values(self):
        assert SolutionStatus.PENDING.value == "pending"
        assert SolutionStatus.RUNNING.value == "running"
        assert SolutionStatus.COMPLETED.value == "completed"
        assert SolutionStatus.FAILED.value == "failed"
        assert SolutionStatus.TIMEOUT.value == "timeout"
        assert SolutionStatus.CANCELLED.value == "cancelled"

    def test_tournament_status_values(self):
        assert TournamentStatus.CREATED.value == "created"
        assert TournamentStatus.RUNNING.value == "running"
        assert TournamentStatus.COMPLETED.value == "completed"
        assert TournamentStatus.CANCELLED.value == "cancelled"
        assert TournamentStatus.ERROR.value == "error"

    def test_selection_mode_variants(self):
        assert SelectionMode.HUMAN_CHOICE is not None
        assert SelectionMode.AUTO_BEST is not None
        assert SelectionMode.THRESHOLD is not None

    def test_solution_to_dict(self):
        sol = make_solution("a1")
        d = sol.to_dict()
        assert d["agent_id"] == "a1"
        assert d["status"] == "completed"
        assert "solution_id" in d
        assert "created_at" in d

    def test_tournament_result_to_dict(self):
        sol = make_solution("a1")
        result = TournamentResult(
            tournament_id="t1",
            solutions=[sol],
            winner=sol,
            execution_summary={"total_agents": 1},
        )
        d = result.to_dict()
        assert d["tournament_id"] == "t1"
        assert len(d["solutions"]) == 1
        assert d["winner"]["agent_id"] == "a1"

    def test_archived_solution_to_dict(self):
        sol = make_solution("a1")
        archive = ArchivedSolution(
            archive_id="arc-1",
            solution=sol,
            tournament_id="t1",
            rejection_reason="Lower score",
        )
        d = archive.to_dict()
        assert d["archive_id"] == "arc-1"
        assert d["rejection_reason"] == "Lower score"

    def test_task_auto_id(self):
        task = Task(task_id="", description="test")
        assert task.task_id != ""


# =============================================================================
# Tests: AsyncAgentAdapter
# =============================================================================


class TestAsyncAgentAdapter:
    """Tests for AsyncAgentAdapter wrapping sync agent manager."""

    async def test_spawn_agent(self):
        sync_mgr = MagicMock()
        mock_agent = MagicMock()
        mock_agent.agent_id = uuid.uuid4()
        sync_mgr.spawn_agent.return_value = mock_agent

        adapter = AsyncAgentAdapter(sync_mgr)
        config = AgentConfig(
            agent_id="a1",
            agent_type="gpt-4",
            system_prompt="Be helpful",
            metadata={"purpose": "coding", "task_id": "t1"},
        )
        result = await adapter.spawn_agent(config)

        assert result == str(mock_agent.agent_id)
        sync_mgr.spawn_agent.assert_called_once()

    async def test_execute_task_returns_completed_solution(self):
        sync_mgr = MagicMock()
        adapter = AsyncAgentAdapter(sync_mgr)
        task = make_task()

        solution = await adapter.execute_task("agent-1", task)

        assert isinstance(solution, Solution)
        assert solution.status == SolutionStatus.COMPLETED
        assert solution.agent_id == "agent-1"

    async def test_terminate_agent(self):
        sync_mgr = MagicMock()
        adapter = AsyncAgentAdapter(sync_mgr)

        agent_uuid = uuid.uuid4()
        await adapter.terminate_agent(str(agent_uuid))

        sync_mgr.terminate_agent.assert_called_once()


# =============================================================================
# Tests: Tournament helper methods
# =============================================================================


class TestTournamentHelpers:
    """Tests for Tournament dataclass helper methods."""

    def test_get_completed_solutions(self):
        t = Tournament(
            tournament_id="t1",
            task=make_task(),
            agent_configs=[],
            selection_mode=SelectionMode.HUMAN_CHOICE,
        )
        t.solutions["a1"] = make_solution("a1", status=SolutionStatus.COMPLETED)
        t.solutions["a2"] = make_solution("a2", status=SolutionStatus.FAILED)
        t.solutions["a3"] = make_solution("a3", status=SolutionStatus.COMPLETED)

        completed = t.get_completed_solutions()
        assert len(completed) == 2

    def test_get_failed_solutions(self):
        t = Tournament(
            tournament_id="t1",
            task=make_task(),
            agent_configs=[],
            selection_mode=SelectionMode.HUMAN_CHOICE,
        )
        t.solutions["a1"] = make_solution("a1", status=SolutionStatus.FAILED)
        t.solutions["a2"] = make_solution("a2", status=SolutionStatus.TIMEOUT)
        t.solutions["a3"] = make_solution("a3", status=SolutionStatus.COMPLETED)

        failed = t.get_failed_solutions()
        assert len(failed) == 2

    def test_get_leaderboard_empty(self):
        t = Tournament(
            tournament_id="t1",
            task=make_task(),
            agent_configs=[],
            selection_mode=SelectionMode.HUMAN_CHOICE,
        )
        assert t.get_leaderboard() == []

    def test_get_leaderboard_sorted(self):
        t = Tournament(
            tournament_id="t1",
            task=make_task(),
            agent_configs=[],
            selection_mode=SelectionMode.HUMAN_CHOICE,
        )
        t.evaluation_scores = {
            "a1": {"complexity": 0.8, "test_pass": 0.9},
            "a2": {"complexity": 0.5, "test_pass": 0.6},
            "a3": {"complexity": 0.95, "test_pass": 1.0},
        }
        lb = t.get_leaderboard()
        assert lb[0][0] == "a3"  # Highest avg score
        assert lb[-1][0] == "a2"  # Lowest avg score
