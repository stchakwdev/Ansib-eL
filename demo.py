#!/usr/bin/env python3
"""
Ansib-eL Demo Script
====================

This script demonstrates the key features of the AI-Native Version Control System:
1. System initialization
2. Agent spawning with UUIDs
3. Tournament mode execution
4. Trust scoring
5. Lineage tracking
6. Human-in-the-loop approval

Run: python demo.py
"""

import sys
import tempfile
from pathlib import Path
from uuid import uuid4

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print("─"*70)

def demo_agent_system():
    """Demonstrate agent management system."""
    print_header("DEMO 1: Agent Management System")
    
    from agent_system import AgentManager, AgentStatus
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize agent manager
        manager = AgentManager(tmpdir)
        print(f"✓ Agent manager initialized at: {tmpdir}")
        
        # Spawn an agent
        agent = manager.spawn_agent(
            purpose="Implement user authentication",
            model_version="gpt-5.2",
            prompt="Create a secure login system with OAuth support",
            task_id="task-001"
        )
        
        print(f"\n✓ Agent spawned:")
        print(f"  - Agent ID: {agent.agent_id}")
        print(f"  - Purpose: {agent.purpose}")
        print(f"  - Model: {agent.model_version}")
        print(f"  - Prompt Hash: {agent.prompt_hash[:16]}...")
        print(f"  - Status: {agent.status.name}")
        print(f"  - Workspace: {agent.workspace_branch}")
        
        # List active agents
        active = manager.list_active_agents()
        print(f"\n✓ Active agents: {len(active)}")
        
        # Update status
        manager.update_agent_status(agent.agent_id, AgentStatus.WORKING)
        print(f"✓ Agent status updated to: WORKING")
        
        # Terminate agent
        manager.terminate_agent(agent.agent_id)
        print(f"✓ Agent terminated")

def demo_git_wrapper():
    """Demonstrate git wrapper functionality."""
    print_header("DEMO 2: Git Wrapper with AI Metadata")
    
    from git_wrapper import GitWrapper, AgentMetadata
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git wrapper
        git = GitWrapper(tmpdir)
        
        # Initialize repository
        if git.init_repo():
            print(f"✓ Repository initialized at: {tmpdir}")
        else:
            print("✗ Failed to initialize repository")
            return
        
        # Create a test file
        test_file = Path(tmpdir) / "hello.txt"
        test_file.write_text("Hello, Ansib-eL!")
        
        # Create agent branch
        branch_name = git.create_agent_branch(
            agent_id="agent-001",
            purpose="Add greeting file"
        )
        print(f"✓ Agent branch created: {branch_name}")
        
        # Commit with metadata
        metadata = AgentMetadata(
            agent_id="agent-001",
            model_version="gpt-5.2",
            prompt_hash="sha256:abc123...",
            timestamp="2024-01-15T10:30:00Z",
            parent_task="task-001",
            reasoning="Added greeting as per user request"
        )
        
        commit_hash = git.commit_with_metadata(
            message="Add greeting file",
            metadata=metadata,
            files=[str(test_file)]
        )
        print(f"✓ Commit created: {commit_hash[:8]}")
        
        # Retrieve metadata
        retrieved = git.get_commit_metadata(commit_hash)
        print(f"\n✓ Retrieved metadata:")
        print(f"  - Agent ID: {retrieved.agent_id}")
        print(f"  - Model: {retrieved.model_version}")
        print(f"  - Reasoning: {retrieved.reasoning}")

def demo_trust_lineage():
    """Demonstrate trust scoring and lineage tracking."""
    print_header("DEMO 3: Trust Scoring & Lineage Tracking")
    
    from trust_lineage import TrustLineageManager, DecisionType
    from uuid import uuid4
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize trust system
        tl = TrustLineageManager(tmpdir)
        print(f"✓ Trust system initialized")
        
        # Create test agents
        good_agent = uuid4()
        bad_agent = uuid4()
        
        print_section("Recording Decisions")
        
        # Record good agent decisions (all accepted)
        for i in range(5):
            tl.record_decision(
                agent_id=good_agent,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"commit-good-{i}",
                review_time_ms=5000
            )
        print(f"✓ Recorded 5 ACCEPTED decisions for good_agent")
        
        # Record bad agent decisions (mixed)
        for i in range(3):
            tl.record_decision(
                agent_id=bad_agent,
                decision=DecisionType.ACCEPTED,
                commit_hash=f"commit-bad-ok-{i}",
                review_time_ms=10000
            )
        for i in range(4):
            tl.record_decision(
                agent_id=bad_agent,
                decision=DecisionType.REJECTED,
                commit_hash=f"commit-bad-fail-{i}",
                review_time_ms=5000
            )
        print(f"✓ Recorded 3 ACCEPTED, 4 REJECTED for bad_agent")
        
        print_section("Trust Scores")
        
        # Get trust scores
        good_score = tl.get_trust_score(good_agent)
        bad_score = tl.get_trust_score(bad_agent)
        
        print(f"Good Agent:")
        print(f"  - Trust Score: {good_score.score:.3f}")
        print(f"  - Confidence: {good_score.confidence:.3f}")
        print(f"  - Tier: {tl.get_trust_tier(good_agent).name}")
        print(f"  - Auto-approve 100 lines: {tl.should_auto_approve(good_agent, 100)}")
        
        print(f"\nBad Agent:")
        print(f"  - Trust Score: {bad_score.score:.3f}")
        print(f"  - Confidence: {bad_score.confidence:.3f}")
        print(f"  - Tier: {tl.get_trust_tier(bad_agent).name}")
        print(f"  - Auto-approve 100 lines: {tl.should_auto_approve(bad_agent, 100)}")
        
        print_section("Lineage Tracking")
        
        # Record reasoning
        tl.record_reasoning(
            agent_id=good_agent,
            task_id="task-001",
            reasoning="""
            Chose PostgreSQL over MySQL because:
            1. Better JSON support for flexible schemas
            2. Stronger consistency guarantees
            3. Better geospatial extensions if needed later
            """
        )
        print(f"✓ Recorded reasoning for library choice")
        
        # Query reasoning
        lineage = tl.get_lineage("commit-good-0")
        if lineage:
            print(f"✓ Retrieved lineage for commit")

def demo_tournament():
    """Demonstrate tournament/parallel execution."""
    print_header("DEMO 4: Tournament Mode (Parallel Execution)")
    
    from tournament import (
        TournamentOrchestrator, AgentConfig, Task, 
        SelectionMode, SolutionStatus
    )
    from agent_system import AgentManager
    from git_wrapper import GitWrapper
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize components
        git = GitWrapper(tmpdir)
        git.init_repo()
        
        agents = AgentManager(tmpdir + "/agents")
        
        # Create tournament orchestrator
        tournament = TournamentOrchestrator(agents, git)
        print(f"✓ Tournament orchestrator initialized")
        
        # Create a task
        task = Task(
            task_id="task-001",
            description="Implement a factorial function",
            context_files=[],
            requirements=[
                "Handle edge cases (0, 1, negative)",
                "Use recursion or iteration",
                "Include docstring"
            ],
            test_commands=["python -c \"assert factorial(5) == 120\""],
            timeout_seconds=60
        )
        print(f"✓ Task created: {task.description}")
        
        # Create agent configs for tournament
        agent_configs = [
            AgentConfig(
                agent_id="agent-iterative",
                agent_type="gpt-5.2",
                model_config={"temperature": 0.7},
                timeout_seconds=30
            ),
            AgentConfig(
                agent_id="agent-recursive",
                agent_type="claude",
                model_config={"temperature": 0.8},
                timeout_seconds=30
            ),
            AgentConfig(
                agent_id="agent-mathlib",
                agent_type="gpt-5.2",
                model_config={"temperature": 0.5},
                timeout_seconds=30
            )
        ]
        print(f"✓ Created {len(agent_configs)} agent configurations")
        
        # Create tournament
        tourney = tournament.create_tournament(
            task=task,
            agent_configs=agent_configs,
            selection_mode=SelectionMode.HUMAN_CHOICE
        )
        print(f"✓ Tournament created: {tourney.tournament_id}")
        print(f"  - Agents: {len(tourney.agent_configs)}")
        print(f"  - Status: {tourney.status.value}")
        
        print_section("Tournament Results")
        
        # Note: In a real scenario, we'd run actual agents
        # Here we just show the structure
        print("Tournament structure ready for execution:")
        print(f"  - Tournament ID: {tourney.tournament_id}")
        print(f"  - Selection Mode: {tourney.selection_mode.name}")
        print(f"  - Max Concurrent: {tournament.max_concurrent_agents}")
        
        # Show review presentation format
        print("\n✓ Review presentation would include:")
        print("  - Side-by-side diff comparison")
        print("  - Agent metadata for each solution")
        print("  - Evaluation scores")
        print("  - Human selection interface")

def demo_orchestrator():
    """Demonstrate the main orchestrator."""
    print_header("DEMO 5: Core Orchestrator Integration")
    
    from ansib_el import AnsibElSystem
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Repository: {tmpdir}")
        
        # Initialize system
        system = AnsibElSystem(tmpdir)
        success = system.initialize()
        
        if success:
            print("✓ System initialized successfully")
        else:
            print("✗ Failed to initialize system")
            return
        
        # Show initial status
        status = system.get_status()
        print(f"\nInitial Status:")
        print(f"  - Repo initialized: {status.repo_initialized}")
        print(f"  - Active agents: {status.active_agents}")
        print(f"  - Pending approvals: {status.pending_approvals}")
        print(f"  - Total commits: {status.total_commits}")
        
        print_section("Processing a Prompt")
        
        # Note: In a real scenario with actual LLM agents
        # this would spawn agents and execute tasks
        print("Prompt: 'Add a login page'")
        print("\nWorkflow:")
        print("  1. Orchestrator breaks down prompt into tasks")
        print("  2. Tasks delegated to agent pool")
        print("  3. Tournament mode: 3 agents work in parallel")
        print("  4. Solutions queued for human review")
        print("  5. Human approves/rejects")
        print("  6. Winner merged, trust scores updated")
        
        print("\n✓ Orchestrator integration complete")

def main():
    """Run all demos."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "Ansib-eL: AI-Native Version Control" + " "*16 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        demo_agent_system()
    except Exception as e:
        print(f"\n✗ Agent system demo failed: {e}")
    
    try:
        demo_git_wrapper()
    except Exception as e:
        print(f"\n✗ Git wrapper demo failed: {e}")
    
    try:
        demo_trust_lineage()
    except Exception as e:
        print(f"\n✗ Trust/lineage demo failed: {e}")
    
    try:
        demo_tournament()
    except Exception as e:
        print(f"\n✗ Tournament demo failed: {e}")
    
    try:
        demo_orchestrator()
    except Exception as e:
        print(f"\n✗ Orchestrator demo failed: {e}")
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "All demos completed!" + " "*23 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

if __name__ == "__main__":
    main()
