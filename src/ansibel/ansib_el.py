#!/usr/bin/env python3
"""
Ansib-eL - AI-Native Version Control System
============================================

Main integration module that ties together all components:
- Orchestrator: Central repo manager
- GitWrapper: Git operations with AI metadata
- AgentSystem: Agent lifecycle management
- Tournament: Parallel execution and evaluation
- TrustLineage: Reputation and provenance tracking

Usage:
    from ansib_el import AnsibElSystem

    system = AnsibElSystem("/path/to/repo")
    system.initialize()

    # Process a human prompt
    result = system.process_prompt("Add a login page with OAuth support")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ansibel.agent_system import AgentManager
from ansibel.git_wrapper import GitWrapper

# Import all components
from ansibel.orchestrator import Orchestrator, Task
from ansibel.tournament import SelectionMode, TournamentOrchestrator
from ansibel.trust_lineage import DecisionType, TrustLineageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ansib-el")


@dataclass
class SystemStatus:
    """Overall system status report."""

    repo_initialized: bool
    active_agents: int
    pending_approvals: int
    total_commits: int
    trust_scores: dict[str, float]
    recent_tournaments: list[str]


class AnsibElSystem:
    """
    Main Ansib-eL System integrating all components.

    This is the primary interface for interacting with the AI-Native
    Version Control System. It orchestrates all subsystems and provides
    a unified API for human operators.

    Attributes:
        repo_path: Path to the git repository
        git: GitWrapper instance for git operations
        agents: AgentManager for agent lifecycle
        orchestrator: Orchestrator for task management
        tournament: TournamentOrchestrator for parallel execution
        trust_lineage: TrustLineageManager for reputation tracking
    """

    def __init__(self, repo_path: str = "."):
        """
        Initialize the Ansib-eL system.

        Args:
            repo_path: Path to the git repository
        """
        self.repo_path = Path(repo_path).resolve()
        self._initialized = False

        # Initialize components
        logger.info(f"Initializing Ansib-eL system at {self.repo_path}")

        self.git = GitWrapper(str(self.repo_path))

        # Ensure .ai-git directory exists before initializing components that need it
        ai_git_dir = self.repo_path / ".ai-git"
        ai_git_dir.mkdir(parents=True, exist_ok=True)

        self.agents = AgentManager(str(ai_git_dir / "agents.json"))
        self.trust_lineage = TrustLineageManager(str(ai_git_dir / "trust.db"))

        # These will be initialized after git setup
        self.orchestrator: Orchestrator | None = None
        self.tournament: TournamentOrchestrator | None = None

    def initialize(self) -> bool:
        """
        Initialize the repository for Ansib-eL.

        This sets up:
        - Git repository (if not exists)
        - AI-Git metadata structure
        - All subsystem integrations

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Ansib-eL repository...")

            # Initialize git wrapper
            if not self.git.init_repo():
                logger.error("Failed to initialize git repository")
                return False

            # Initialize orchestrator with integration protocols
            self.orchestrator = Orchestrator(
                repo_path=str(self.repo_path),
                git_wrapper=self.git,  # type: ignore[arg-type]
                human_interface=self._create_human_interface(),
                tournament_system=self._create_tournament_interface(),
            )

            # Initialize tournament system
            self.tournament = TournamentOrchestrator(
                agent_manager=self.agents,  # type: ignore[arg-type]
                git_wrapper=self.git,  # type: ignore[arg-type]
            )

            self._initialized = True
            logger.info("Ansib-eL system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def process_prompt(
        self, prompt: str, use_tournament: bool = True, num_agents: int = 2
    ) -> dict[str, Any]:
        """
        Process a human prompt through the system.

        This is the main entry point for human operators to request
        AI-generated code changes.

        Args:
            prompt: Human-readable task description
            use_tournament: Whether to use tournament mode (multiple agents)
            num_agents: Number of agents to spawn in tournament mode

        Returns:
            Dictionary with result status and details

        Example:
            >>> system = AnsibElSystem("./my-project")
            >>> system.initialize()
            >>> result = system.process_prompt("Add user authentication")
            >>> print(result['status'])
            'pending_approval'
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        logger.info(f"Processing prompt: {prompt}")

        assert self.orchestrator is not None, "System not initialized"
        # Step 1: Break down the prompt into tasks
        task_breakdown = self.orchestrator.process_human_prompt(prompt)

        results = []

        for task in task_breakdown.tasks:
            logger.info(f"Processing task: {task.description}")

            if use_tournament and num_agents > 1:
                # Tournament mode: multiple agents compete
                result = self._run_tournament_task(task, num_agents)
            else:
                # Single agent mode
                result = self._run_single_agent_task(task)

            results.append(result)

        return {
            "status": "success",
            "tasks_processed": len(results),
            "results": results,
            "pending_approvals": len(self.orchestrator.get_pending_approvals()),
        }

    def review_and_approve(
        self, approval_id: str, approve: bool, comments: str = ""
    ) -> dict[str, Any]:
        """
        Review and approve/reject a pending solution.

        Args:
            approval_id: ID of the approval request
            approve: True to approve, False to reject
            comments: Optional review comments

        Returns:
            Result of the approval action
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")

        assert self.orchestrator is not None

        from uuid import UUID

        if approve:
            result = self.orchestrator.approve_solution(approval_id, "human", comments)

            # Record in trust system
            if result.success and result.solution:
                agent_uuid = UUID(str(result.solution.agent_id))
                self.trust_lineage.record_decision(
                    agent_id=agent_uuid,
                    decision=DecisionType.ACCEPTED,
                    commit_hash=result.merged_commit or "",
                    review_time_ms=0,
                )
        else:
            result = self.orchestrator.reject_solution(approval_id, "human", comments)

            # Record in trust system
            if result.solution:
                agent_uuid = UUID(str(result.solution.agent_id))
                self.trust_lineage.record_decision(
                    agent_id=agent_uuid,
                    decision=DecisionType.REJECTED,
                    commit_hash="",
                    review_time_ms=0,
                )

        return {
            "success": result.success,
            "message": result.message,
            "commit_hash": (
                result.merged_commit if hasattr(result, "merged_commit") else None
            ),
        }

    def get_status(self) -> SystemStatus:
        """
        Get overall system status.

        Returns:
            SystemStatus with current state information
        """
        if not self._initialized:
            return SystemStatus(
                repo_initialized=False,
                active_agents=0,
                pending_approvals=0,
                total_commits=0,
                trust_scores={},
                recent_tournaments=[],
            )

        assert self.orchestrator is not None

        # Get trust scores for all agents
        trust_scores = {}
        for agent in self.agents.list_active_agents():
            score = self.trust_lineage.get_trust_score(agent.agent_id)
            trust_scores[str(agent.agent_id)] = score.score

        return SystemStatus(
            repo_initialized=self._initialized,
            active_agents=len(self.agents.list_active_agents()),
            pending_approvals=len(self.orchestrator.get_pending_approvals()),
            total_commits=(
                len(list(self.git.repo.iter_commits())) if self.git.repo else 0
            ),
            trust_scores=trust_scores,
            recent_tournaments=[],  # Would be populated from tournament history
        )

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        """
        List all pending approval requests.

        Returns:
            List of pending approval details
        """
        if not self._initialized:
            return []

        assert self.orchestrator is not None
        approvals = self.orchestrator.get_pending_approvals()
        return [
            {
                "id": str(a.id),
                "title": str(a.solution.task_id),
                "agent_id": str(a.solution.agent_id),
                "submitted_at": a.submitted_at.isoformat(),
                "priority": a.priority.name,
            }
            for a in approvals
        ]

    def get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """
        Get detailed information about an agent.

        Args:
            agent_id: UUID of the agent

        Returns:
            Agent information including trust score and history
        """
        from uuid import UUID

        agent = self.agents.get_agent(UUID(agent_id))
        if not agent:
            return {"error": "Agent not found"}

        trust_score = self.trust_lineage.get_trust_score(UUID(agent_id))
        trust_tier = self.trust_lineage.get_trust_tier(UUID(agent_id))
        history = self.trust_lineage.get_agent_history(UUID(agent_id))

        return {
            "agent_id": str(agent.agent_id),
            "purpose": agent.purpose,
            "model_version": agent.model_version,
            "status": agent.status.name,
            "created_at": agent.created_at,
            "trust_score": trust_score.score,
            "trust_tier": trust_tier.name,
            "decision_count": len(history),
            "workspace_branch": agent.workspace_branch,
        }

    def _run_tournament_task(self, task: Task, num_agents: int) -> dict[str, Any]:
        """Run a task in tournament mode with multiple agents."""
        import asyncio
        from uuid import uuid4

        from ansibel.tournament import AgentConfig
        from ansibel.tournament import Task as TournamentTask

        assert self.tournament is not None, "Tournament system not initialized"

        # Create agent configurations
        agent_configs = [
            AgentConfig(
                agent_id=str(uuid4()),
                agent_type=f"agent-{i + 1}",
                model_config={"temperature": 0.7 + (i * 0.1)},
                timeout_seconds=300,
            )
            for i in range(num_agents)
        ]

        # Convert orchestrator.Task -> tournament.Task
        tournament_task = TournamentTask(
            task_id=str(task.id),
            description=task.description,
            requirements=task.requirements,
        )

        # Create and run tournament
        tournament = self.tournament.create_tournament(
            task=tournament_task,
            agent_configs=agent_configs,
            selection_mode=SelectionMode.HUMAN_CHOICE,
        )

        # Wrap async calls with asyncio.run()
        result = asyncio.run(self.tournament.run_tournament(tournament.tournament_id))

        # Present for human review
        presentation = asyncio.run(
            self.tournament.present_for_review(tournament.tournament_id)
        )

        return {
            "mode": "tournament",
            "tournament_id": tournament.tournament_id,
            "solutions_generated": len(result.solutions),
            "status": "completed",
            "review_presentation": presentation.to_markdown(),
            "pending_approval": True,
        }

    def _run_single_agent_task(self, task: Task) -> dict[str, Any]:
        """Run a task with a single agent."""
        assert self.orchestrator is not None, "System not initialized"

        # Spawn an agent for this task
        agent = self.agents.spawn_agent(
            purpose=task.description,
            model_version="gpt-5.2",
            prompt=task.description,
            task_id=str(task.id),
        )
        agent_pool = [agent] if agent else []

        result = self.orchestrator.delegate_task(
            task=task,
            agent_pool=agent_pool,  # type: ignore[arg-type]
            use_tournament=False,
        )

        return {
            "mode": "single",
            "task_id": str(task.id),
            "status": "delegated" if result.success else "failed",
            "message": result.message,
            "pending_approval": (
                result.requires_approval
                if hasattr(result, "requires_approval")
                else True
            ),
        }

    def _create_human_interface(self):
        """Create human interface protocol implementation."""
        from ansibel.orchestrator import ApprovalRequest, HumanInterface, TaskId

        class ConsoleHumanInterface(HumanInterface):
            def request_approval(self, request: ApprovalRequest) -> bool:
                print(f"\n{'=' * 60}")
                print("APPROVAL REQUIRED")
                print(f"{'=' * 60}")
                print(f"Solution from agent: {request.solution.agent_id}")
                print(f"Explanation: {request.solution.explanation}")
                choice = input("Approve? (y/n): ")
                return choice.lower().startswith("y")

            def provide_feedback(self, task_id: TaskId, feedback: str) -> None:
                print(f"[FEEDBACK] Task {task_id}: {feedback}")
                return None

            def is_available(self) -> bool:
                return True

            def prompt_for_decision(self, message: str, options: list[str]) -> str:
                print(f"\n{'=' * 60}")
                print("HUMAN REVIEW REQUIRED")
                print(f"{'=' * 60}")
                print(message)
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
                choice = input("\nSelect option: ")
                return (
                    options[int(choice) - 1]
                    if choice.isdigit() and 0 < int(choice) <= len(options)
                    else options[0]
                )

            def display_diff(self, diff: str, metadata: dict[str, Any]):
                print(f"\n{'=' * 60}")
                print(f"DIFF - Agent: {metadata.get('agent_id', 'Unknown')}")
                print(f"{'=' * 60}")
                print(diff)

            def notify(self, message: str, level: str = "info"):
                print(f"[{level.upper()}] {message}")

        return ConsoleHumanInterface()

    def _create_tournament_interface(self):
        """Create tournament interface protocol implementation using lazy binding."""
        from ansibel.orchestrator import TournamentInterface

        system = self  # Capture reference for lazy access

        from ansibel.orchestrator import AgentInterface
        from ansibel.orchestrator import Solution as OrchestratorSolution

        class TournamentSystemInterface(TournamentInterface):
            def __init__(self):
                pass

            @property
            def tournament(self):
                return system.tournament

            def register_agent(self, agent: AgentInterface) -> None:
                return None

            def execute_parallel(
                self, task: Task, agents: list[AgentInterface]
            ) -> list[OrchestratorSolution]:
                return []

            def select_winner(
                self, solutions: list[OrchestratorSolution]
            ) -> OrchestratorSolution:
                if not solutions:
                    raise ValueError("No solutions to select from")
                return solutions[0]

            def run_tournament(self, task: Task, agent_pool: list[Any]) -> Any:
                from ansibel.tournament import AgentConfig

                configs = [
                    AgentConfig(agent_id=str(a.agent_id), agent_type=a.model_version)
                    for a in agent_pool
                ]

                tournament = self.tournament.create_tournament(
                    task=task,
                    agent_configs=configs,
                    selection_mode=SelectionMode.HUMAN_CHOICE,
                )

                return self.tournament.run_tournament(tournament.tournament_id)

        return TournamentSystemInterface()
