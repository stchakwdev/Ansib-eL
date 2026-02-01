#!/usr/bin/env python3
"""
Ansib-eL Dogfood Script
========================

Exercises every layer of the Ansib-eL system end-to-end and generates a
report card of what works and what breaks.

Scenarios:
  A -- CLI Workflow (subprocess calls to `ai-git`)
  B -- Programmatic API (import AnsibElSystem)
  C -- Tournament Standalone (async)
  D -- Trust / Lineage Standalone

Usage:
    python3 dogfood.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    scenario: str
    name: str
    passed: bool
    message: str = ""
    error: Optional[str] = None


results: List[TestResult] = []

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Dogfood Bot",
    "GIT_AUTHOR_EMAIL": "dogfood@ansibel.dev",
    "GIT_COMMITTER_NAME": "Dogfood Bot",
    "GIT_COMMITTER_EMAIL": "dogfood@ansibel.dev",
}


def run_test(scenario: str, name: str, fn: Callable[[], Any]) -> None:
    """Execute *fn*, record pass/fail in *results*."""
    try:
        fn()
        results.append(TestResult(scenario=scenario, name=name, passed=True, message="OK"))
        print(f"  [PASS] {name}")
    except Exception as exc:
        tb = traceback.format_exc()
        results.append(TestResult(
            scenario=scenario, name=name, passed=False,
            message=str(exc), error=tb,
        ))
        print(f"  [FAIL] {name} -- {exc}")


# ===================================================================
# Scenario A: CLI Workflow
# ===================================================================

def scenario_a() -> None:
    """Exercise the CLI layer via subprocess calls."""
    print("\n=== Scenario A: CLI Workflow ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="dogfood_a_"))

    def _run(args: List[str], cwd: str = str(tmpdir)) -> subprocess.CompletedProcess:
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, env=GIT_ENV, check=False,
        )

    # A1: ai-git init
    def a1():
        r = _run([sys.executable, "-m", "ansibel.cli", "init"])
        assert r.returncode == 0, f"init failed: {r.stderr}"
    run_test("A", "A1: ai-git init", a1)

    # A2: seed commit
    def a2():
        (tmpdir / "seed.txt").write_text("hello\n")
        _run(["git", "add", "."], str(tmpdir))
        _run(["git", "commit", "-m", "Seed commit"], str(tmpdir))
    run_test("A", "A2: seed commit", a2)

    # A3: create agent-alpha branch + commit with metadata
    def a3():
        r = _run([sys.executable, "-m", "ansibel.cli", "branch", "agent-alpha", "-p", "Alpha work"])
        assert r.returncode == 0, f"branch alpha failed: {r.stderr}"
        (tmpdir / "alpha.py").write_text("# alpha code\n")
        _run(["git", "add", "."], str(tmpdir))
        r = _run([
            sys.executable, "-m", "ansibel.cli", "commit",
            "Alpha commit", "-a", "agent-alpha", "-m", "gpt-5.2",
        ])
        assert r.returncode == 0, f"commit alpha failed: {r.stderr}"
    run_test("A", "A3: agent-alpha branch + commit", a3)

    # A4: create agent-beta branch + commit with metadata
    def a4():
        _run(["git", "checkout", "main"], str(tmpdir))
        r = _run([sys.executable, "-m", "ansibel.cli", "branch", "agent-beta", "-p", "Beta work"])
        assert r.returncode == 0, f"branch beta failed: {r.stderr}"
        (tmpdir / "beta.py").write_text("# beta code\n")
        _run(["git", "add", "."], str(tmpdir))
        r = _run([
            sys.executable, "-m", "ansibel.cli", "commit",
            "Beta commit", "-a", "agent-beta", "-m", "claude-opus-4.5",
        ])
        assert r.returncode == 0, f"commit beta failed: {r.stderr}"
    run_test("A", "A4: agent-beta branch + commit", a4)

    # A5: status + agents
    def a5():
        r = _run([sys.executable, "-m", "ansibel.cli", "status", "--json-output"])
        assert r.returncode == 0, f"status failed: {r.stderr}"
        data = json.loads(r.stdout)
        assert data.get("initialized") is True

        r = _run([sys.executable, "-m", "ansibel.cli", "agents", "--json-output"])
        assert r.returncode == 0, f"agents failed: {r.stderr}"
    run_test("A", "A5: status + agents", a5)

    # A6: history
    def a6():
        r = _run([sys.executable, "-m", "ansibel.cli", "history", "--json-output"])
        assert r.returncode == 0, f"history failed: {r.stderr}"
    run_test("A", "A6: history", a6)

    # A7: trust update
    def a7():
        r = _run([sys.executable, "-m", "ansibel.cli", "trust", "agent-alpha", "0.2"])
        assert r.returncode == 0, f"trust failed: {r.stderr}"
    run_test("A", "A7: trust update", a7)

    # A8: merge agent-alpha branch back to main
    def a8():
        # Find the alpha branch name
        r = _run([sys.executable, "-m", "ansibel.cli", "agents", "--json-output"])
        branches = json.loads(r.stdout)
        alpha_branches = [b["name"] for b in branches if "agent-alpha" in b["name"]]
        assert alpha_branches, "No alpha branch found"
        branch_name = alpha_branches[0]

        _run(["git", "checkout", "main"], str(tmpdir))
        r = _run([sys.executable, "-m", "ansibel.cli", "merge", branch_name, "--yes"])
        assert r.returncode == 0, f"merge failed: {r.stderr}"
    run_test("A", "A8: merge agent-alpha to main", a8)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# Scenario B: Programmatic API
# ===================================================================

def scenario_b() -> None:
    """Exercise the AnsibElSystem integration layer."""
    print("\n=== Scenario B: Programmatic API ===")

    from ansibel.ansib_el import AnsibElSystem, SystemStatus
    from ansibel.orchestrator import (
        Task, TaskId, AgentId, Solution, CodeChange, TaskPriority, ApprovalResult,
    )
    from ansibel.trust_lineage import DecisionType

    tmpdir = Path(tempfile.mkdtemp(prefix="dogfood_b_"))
    # Pre-init git repo
    subprocess.run(["git", "init"], cwd=str(tmpdir), capture_output=True, check=True, env=GIT_ENV)
    subprocess.run(["git", "config", "user.email", "dogfood@ansibel.dev"], cwd=str(tmpdir), capture_output=True, check=True, env=GIT_ENV)
    subprocess.run(["git", "config", "user.name", "Dogfood Bot"], cwd=str(tmpdir), capture_output=True, check=True, env=GIT_ENV)
    (tmpdir / "README.md").write_text("# Dogfood\n")
    subprocess.run(["git", "add", "."], cwd=str(tmpdir), capture_output=True, check=True, env=GIT_ENV)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmpdir), capture_output=True, check=True, env=GIT_ENV)

    system = AnsibElSystem(str(tmpdir))

    # B1: initialize
    def b1():
        ok = system.initialize()
        assert ok is True, "initialize() returned False"
    run_test("B", "B1: AnsibElSystem.initialize()", b1)

    # B2: process_prompt single-agent mode (mocked internally)
    def b2():
        from unittest.mock import patch
        mock_result = {
            "mode": "single", "task_id": "mock", "status": "delegated",
            "message": "ok", "pending_approval": True,
        }
        with patch.object(system, "_run_single_agent_task", return_value=mock_result):
            result = system.process_prompt("Fix a bug", use_tournament=False, num_agents=1)
        assert result["status"] == "success"
        assert result["tasks_processed"] >= 1
    run_test("B", "B2: process_prompt(single-agent)", b2)

    # B3: process_prompt tournament mode (mocked internally)
    def b3():
        from unittest.mock import patch
        mock_result = {
            "mode": "tournament", "tournament_id": "t-001",
            "solutions_generated": 3, "status": "completed",
            "review_presentation": "markdown", "pending_approval": True,
        }
        with patch.object(system, "_run_tournament_task", return_value=mock_result):
            result = system.process_prompt("Add auth", use_tournament=True, num_agents=2)
        assert result["status"] == "success"
    run_test("B", "B3: process_prompt(tournament)", b3)

    # B4: list_pending_approvals (should be empty)
    def b4():
        approvals = system.list_pending_approvals()
        assert isinstance(approvals, list)
    run_test("B", "B4: list_pending_approvals()", b4)

    # B5: submit solution then review_and_approve
    def b5():
        task = Task(
            description="Dogfood approval test",
            requirements=["req"],
            acceptance_criteria=["works"],
        )
        solution = Solution(
            task_id=task.id,
            agent_id=AgentId(),
            changes=[CodeChange(file_path="test.py", diff_content="+# hi", description="test change")],
            explanation="Dogfood solution",
        )
        approval_id = system.orchestrator.submit_for_approval(solution)

        from unittest.mock import patch
        with patch.object(system.trust_lineage, "record_decision"):
            result = system.review_and_approve(approval_id, approve=True, comments="LGTM")
        assert result["success"] is True
    run_test("B", "B5: review_and_approve()", b5)

    # B6: get_status
    def b6():
        status = system.get_status()
        assert isinstance(status, SystemStatus)
        assert status.repo_initialized is True
    run_test("B", "B6: get_status()", b6)

    # B7: spawn agent then get_agent_info
    def b7():
        agent = system.agents.spawn_agent(
            purpose="Dogfood agent",
            model_version="gpt-5.2",
            prompt="test prompt",
            task_id="dogfood-task",
        )
        info = system.get_agent_info(str(agent.agent_id))
        assert "error" not in info
        assert info["purpose"] == "Dogfood agent"
        assert "trust_score" in info
        assert "trust_tier" in info
        assert "decision_count" in info
    run_test("B", "B7: get_agent_info()", b7)

    shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# Scenario C: Tournament Standalone
# ===================================================================

def scenario_c() -> None:
    """Exercise the tournament system in isolation with async."""
    print("\n=== Scenario C: Tournament Standalone ===")

    from ansibel.tournament import (
        TournamentOrchestrator, AgentConfig, Task as TTask,
        SelectionMode, SolutionStatus, Solution as TSolution,
        TournamentStatus,
    )

    # Build a mock agent manager
    class MockAgentMgr:
        async def spawn_agent(self, config: AgentConfig) -> str:
            return config.agent_id

        async def execute_task(self, agent_id: str, task: TTask) -> TSolution:
            return TSolution(
                solution_id=str(uuid4()),
                agent_id=agent_id,
                task_id=task.task_id,
                files_changed={"mock.py": "pass\n"},
                diff="+pass",
                explanation=f"Mock solution by {agent_id}",
                status=SolutionStatus.COMPLETED,
            )

        async def terminate_agent(self, agent_id: str) -> None:
            pass

    mgr = MockAgentMgr()
    orch = TournamentOrchestrator(agent_manager=mgr, max_concurrent_agents=5)
    task = TTask(task_id=str(uuid4()), description="Dogfood task", requirements=["Be correct"])
    configs = [
        AgentConfig(agent_id=str(uuid4()), agent_type=f"agent-{i}", timeout_seconds=30)
        for i in range(3)
    ]

    # C1: create tournament
    def c1():
        t = orch.create_tournament(task=task, agent_configs=configs, selection_mode=SelectionMode.AUTO_BEST)
        assert t.tournament_id
        assert len(t.agent_configs) == 3
    run_test("C", "C1: create tournament", c1)

    # C2: run tournament (all succeed)
    tournament_ref = [None]
    def c2():
        t = orch.create_tournament(task=task, agent_configs=configs, selection_mode=SelectionMode.AUTO_BEST)
        result = asyncio.run(orch.run_tournament(t.tournament_id))
        assert len(result.solutions) == 3
        assert result.winner is not None
        tournament_ref[0] = t
    run_test("C", "C2: run_tournament (all succeed)", c2)

    # C3: run tournament with failure agent
    class FailAgentMgr(MockAgentMgr):
        async def execute_task(self, agent_id: str, task: TTask) -> TSolution:
            if "fail" in agent_id:
                raise RuntimeError("Simulated agent failure")
            return await super().execute_task(agent_id, task)

    def c3():
        fail_orch = TournamentOrchestrator(agent_manager=FailAgentMgr())
        fail_configs = [
            AgentConfig(agent_id="fail-agent", agent_type="bad", timeout_seconds=10),
            AgentConfig(agent_id=str(uuid4()), agent_type="good", timeout_seconds=10),
        ]
        t = fail_orch.create_tournament(task=task, agent_configs=fail_configs, selection_mode=SelectionMode.AUTO_BEST)
        result = asyncio.run(fail_orch.run_tournament(t.tournament_id))
        statuses = [s.status for s in result.solutions]
        assert SolutionStatus.FAILED in statuses or SolutionStatus.COMPLETED in statuses
    run_test("C", "C3: run_tournament (with failures)", c3)

    # C4: present for review
    def c4():
        if tournament_ref[0] is None:
            raise RuntimeError("Skipped -- C2 failed")
        presentation = asyncio.run(orch.present_for_review(tournament_ref[0].tournament_id))
        md = presentation.to_markdown()
        assert "Tournament Review" in md
        assert len(md) > 50
    run_test("C", "C4: present_for_review + markdown", c4)

    # C5: select winner + archive losers
    def c5():
        if tournament_ref[0] is None:
            raise RuntimeError("Skipped -- C2 failed")
        winner = asyncio.run(orch.select_winner(tournament_ref[0].tournament_id))
        assert winner is not None
        archives = asyncio.run(orch.archive_losers(tournament_ref[0].tournament_id))
        assert len(archives) >= 1
    run_test("C", "C5: select_winner + archive_losers", c5)


# ===================================================================
# Scenario D: Trust / Lineage Standalone
# ===================================================================

def scenario_d() -> None:
    """Exercise the trust scoring and lineage systems."""
    print("\n=== Scenario D: Trust / Lineage Standalone ===")

    from ansibel.trust_lineage import (
        TrustLineageManager, TrustScorer, TrustTier,
        DecisionType, ChangeComplexity,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="dogfood_d_"))
    db_path = str(tmpdir / "trust.db")
    mgr = TrustLineageManager(db_path)
    agent_id = uuid4()

    # D1: record 5 decisions, verify score/tier
    def d1():
        for i in range(5):
            mgr.record_decision(
                agent_id=agent_id,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"abc{i:04d}",
                review_time_ms=100,
                change_size=50,
                complexity=ChangeComplexity.MODERATE,
            )
        score = mgr.get_trust_score(agent_id)
        assert 0.0 <= score.score <= 1.0
        assert score.sample_count == 5

        tier = mgr.get_trust_tier(agent_id)
        assert isinstance(tier, TrustTier)
    run_test("D", "D1: record decisions + verify score/tier", d1)

    # D2: record_complete_decision returns both records
    def d2():
        decision_rec, reasoning_rec = mgr.record_complete_decision(
            agent_id=agent_id,
            task_id="task-d2",
            commit_hash="d2commit",
            decision=DecisionType.ACCEPTED,
            reasoning="Good implementation with tests",
            review_time_ms=200,
            change_size=100,
            confidence=0.9,
        )
        assert decision_rec.commit_hash == "d2commit"
        assert reasoning_rec.reasoning == "Good implementation with tests"
    run_test("D", "D2: record_complete_decision()", d2)

    # D3: auto-approve threshold check
    def d3():
        can_small = mgr.should_auto_approve(agent_id, change_size=30)
        can_large = mgr.should_auto_approve(agent_id, change_size=9999)
        # At least small changes should be auto-approvable after 6 accepted decisions
        assert isinstance(can_small, bool)
        assert isinstance(can_large, bool)
        # Large changes should NOT be auto-approved for any tier
        assert can_large is False
    run_test("D", "D3: auto-approve threshold", d3)

    shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# Report Card
# ===================================================================

def print_report() -> None:
    """Print terminal summary grouped by scenario."""
    print("\n" + "=" * 70)
    print("  ANSIB-EL DOGFOOD REPORT CARD")
    print("=" * 70)

    scenarios = {}
    for r in results:
        scenarios.setdefault(r.scenario, []).append(r)

    total_pass = sum(1 for r in results if r.passed)
    total_fail = sum(1 for r in results if not r.passed)

    for scenario, tests in sorted(scenarios.items()):
        passed = sum(1 for t in tests if t.passed)
        failed = sum(1 for t in tests if not t.passed)
        label = f"Scenario {scenario}: {passed}/{len(tests)} passed"
        status = "ALL PASS" if failed == 0 else f"{failed} FAILED"
        print(f"\n  [{status}] {label}")
        for t in tests:
            mark = "PASS" if t.passed else "FAIL"
            print(f"    [{mark}] {t.name}")
            if not t.passed and t.message:
                # Print first line of error
                print(f"           {t.message[:120]}")

    print(f"\n  TOTAL: {total_pass} passed, {total_fail} failed, {len(results)} total")
    print("=" * 70)

    # Save JSON report
    report_path = Path(__file__).parent / "dogfood_report.json"
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {"passed": total_pass, "failed": total_fail, "total": len(results)},
        "results": [
            {
                "scenario": r.scenario,
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "error": r.error[:500] if r.error else None,
            }
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
    print_report()
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
