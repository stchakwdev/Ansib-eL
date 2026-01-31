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
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import all components
from orchestrator import Orchestrator, Task, TaskBreakdown
from git_wrapper import GitWrapper, AgentMetadata
from agent_system import AgentManager, Agent, AgentStatus
from tournament import TournamentOrchestrator, Tournament, SelectionMode
from trust_lineage import TrustLineageManager, TrustTier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ansib-el')


@dataclass
class SystemStatus:
    """Overall system status report."""
    repo_initialized: bool
    active_agents: int
    pending_approvals: int
    total_commits: int
    trust_scores: Dict[str, float]
    recent_tournaments: List[str]


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
        self.agents = AgentManager(str(self.repo_path / ".ai-git" / "agents"))
        self.trust_lineage = TrustLineageManager(str(self.repo_path / ".ai-git"))
        
        # These will be initialized after git setup
        self.orchestrator: Optional[Orchestrator] = None
        self.tournament: Optional[TournamentOrchestrator] = None
        
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
                git_wrapper=self.git,
                human_interface=self._create_human_interface(),
                tournament_system=self._create_tournament_interface()
            )
            
            # Initialize tournament system
            self.tournament = TournamentOrchestrator(
                agent_manager=self.agents,
                git_wrapper=self.git
            )
            
            self._initialized = True
            logger.info("Ansib-eL system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def process_prompt(
        self, 
        prompt: str, 
        use_tournament: bool = True,
        num_agents: int = 2
    ) -> Dict[str, Any]:
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
        
        # Step 1: Break down the prompt into tasks
        task_breakdown = self.orchestrator.process_human_prompt(prompt)
        
        results = []
        
        for task in task_breakdown.tasks:
            logger.info(f"Processing task: {task.title}")
            
            if use_tournament and num_agents > 1:
                # Tournament mode: multiple agents compete
                result = self._run_tournament_task(task, num_agents)
            else:
                # Single agent mode
                result = self._run_single_agent_task(task)
            
            results.append(result)
        
        return {
            'status': 'success',
            'tasks_processed': len(results),
            'results': results,
            'pending_approvals': len(self.orchestrator.get_pending_approvals())
        }
    
    def review_and_approve(self, approval_id: str, approve: bool, 
                          comments: str = "") -> Dict[str, Any]:
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
        
        if approve:
            result = self.orchestrator.approve_solution(approval_id, "human", comments)
            
            # Record in trust system
            if result.success and result.solution:
                self.trust_lineage.record_decision(
                    agent_id=result.solution.agent_id,
                    decision="ACCEPTED",
                    commit_hash=result.merged_commit or "",
                    review_time_ms=0
                )
        else:
            result = self.orchestrator.reject_solution(approval_id, "human", comments)
            
            # Record in trust system
            if result.solution:
                self.trust_lineage.record_decision(
                    agent_id=result.solution.agent_id,
                    decision="REJECTED",
                    commit_hash="",
                    review_time_ms=0
                )
        
        return {
            'success': result.success,
            'message': result.message,
            'commit_hash': result.merged_commit if hasattr(result, 'merged_commit') else None
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
                recent_tournaments=[]
            )
        
        # Get trust scores for all agents
        trust_scores = {}
        for agent in self.agents.list_active_agents():
            score = self.trust_lineage.get_trust_score(agent.agent_id)
            trust_scores[str(agent.agent_id)] = score.score
        
        return SystemStatus(
            repo_initialized=self._initialized,
            active_agents=len(self.agents.list_active_agents()),
            pending_approvals=len(self.orchestrator.get_pending_approvals()),
            total_commits=len(list(self.git.repo.iter_commits())) if self.git.repo else 0,
            trust_scores=trust_scores,
            recent_tournaments=[]  # Would be populated from tournament history
        )
    
    def list_pending_approvals(self) -> List[Dict[str, Any]]:
        """
        List all pending approval requests.
        
        Returns:
            List of pending approval details
        """
        if not self._initialized:
            return []
        
        approvals = self.orchestrator.get_pending_approvals()
        return [
            {
                'id': str(a.id),
                'title': a.solution.task.title if a.solution.task else "Unknown",
                'agent_id': str(a.solution.agent_id) if a.solution else "Unknown",
                'submitted_at': a.submitted_at.isoformat(),
                'priority': a.priority.name
            }
            for a in approvals
        ]
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
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
            return {'error': 'Agent not found'}
        
        trust_score = self.trust_lineage.get_trust_score(UUID(agent_id))
        trust_tier = self.trust_lineage.get_trust_tier(UUID(agent_id))
        history = self.trust_lineage.get_agent_history(UUID(agent_id))
        
        return {
            'agent_id': str(agent.agent_id),
            'purpose': agent.purpose,
            'model_version': agent.model_version,
            'status': agent.status.name,
            'created_at': agent.created_at,
            'trust_score': trust_score.score,
            'trust_tier': trust_tier.name,
            'decision_count': len(history),
            'workspace_branch': agent.workspace_branch
        }
    
    def _run_tournament_task(self, task: Task, num_agents: int) -> Dict[str, Any]:
        """Run a task in tournament mode with multiple agents."""
        from uuid import uuid4
        from tournament import AgentConfig
        
        # Create agent configurations
        agent_configs = [
            AgentConfig(
                agent_id=str(uuid4()),
                agent_type=f"agent-{i+1}",
                model_config={"temperature": 0.7 + (i * 0.1)},
                timeout_seconds=300
            )
            for i in range(num_agents)
        ]
        
        # Create and run tournament
        tournament = self.tournament.create_tournament(
            task=task,
            agent_configs=agent_configs,
            selection_mode=SelectionMode.HUMAN_CHOICE
        )
        
        result = self.tournament.run_tournament(tournament.tournament_id)
        
        # Present for human review
        presentation = self.tournament.present_for_review(tournament.tournament_id)
        
        return {
            'mode': 'tournament',
            'tournament_id': tournament.tournament_id,
            'solutions_generated': len(result.solutions),
            'status': result.status.value,
            'review_presentation': presentation.to_markdown(),
            'pending_approval': True
        }
    
    def _run_single_agent_task(self, task: Task) -> Dict[str, Any]:
        """Run a task with a single agent."""
        # Delegate to single agent
        agent_pool = []  # Would be populated from available agents
        
        result = self.orchestrator.delegate_task(
            task=task,
            agent_pool=agent_pool,
            use_tournament=False
        )
        
        return {
            'mode': 'single',
            'task_id': str(task.id),
            'status': 'delegated' if result.success else 'failed',
            'message': result.message,
            'pending_approval': result.requires_approval if hasattr(result, 'requires_approval') else True
        }
    
    def _create_human_interface(self):
        """Create human interface protocol implementation."""
        from orchestrator import HumanInterface
        
        class ConsoleHumanInterface(HumanInterface):
            def prompt_for_decision(self, message: str, options: List[str]) -> str:
                print(f"\n{'='*60}")
                print(f"HUMAN REVIEW REQUIRED")
                print(f"{'='*60}")
                print(message)
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
                choice = input("\nSelect option: ")
                return options[int(choice) - 1] if choice.isdigit() and 0 < int(choice) <= len(options) else options[0]
            
            def display_diff(self, diff: str, metadata: Dict[str, Any]):
                print(f"\n{'='*60}")
                print(f"DIFF - Agent: {metadata.get('agent_id', 'Unknown')}")
                print(f"{'='*60}")
                print(diff)
            
            def notify(self, message: str, level: str = "info"):
                print(f"[{level.upper()}] {message}")
        
        return ConsoleHumanInterface()
    
    def _create_tournament_interface(self):
        """Create tournament interface protocol implementation."""
        from orchestrator import TournamentInterface
        
        class TournamentSystemInterface(TournamentInterface):
            def __init__(self, tournament_orchestrator):
                self.tournament = tournament_orchestrator
            
            def run_tournament(self, task: Task, agent_pool: List[Any]) -> Any:
                from tournament import AgentConfig
                
                configs = [
                    AgentConfig(agent_id=str(a.agent_id), agent_type=a.model_version)
                    for a in agent_pool
                ]
                
                tournament = self.tournament.create_tournament(
                    task=task,
                    agent_configs=configs,
                    selection_mode=SelectionMode.HUMAN_CHOICE
                )
                
                return self.tournament.run_tournament(tournament.tournament_id)
        
        return TournamentSystemInterface(self.tournament)


def main():
    """Main entry point for demonstration."""
    import tempfile
    import os
    
    print("="*70)
    print("Ansib-eL: AI-Native Version Control System")
    print("="*70)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nDemo repository: {tmpdir}")
        
        # Initialize system
        system = AnsibElSystem(tmpdir)
        
        if not system.initialize():
            print("Failed to initialize system")
            return
        
        print("✓ System initialized")
        
        # Show initial status
        status = system.get_status()
        print(f"\nInitial Status:")
        print(f"  - Repo initialized: {status.repo_initialized}")
        print(f"  - Active agents: {status.active_agents}")
        print(f"  - Pending approvals: {status.pending_approvals}")
        
        print("\n" + "="*70)
        print("Demo complete! System is ready for use.")
        print("="*70)


if __name__ == "__main__":
    main()
