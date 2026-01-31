#!/usr/bin/env python3
"""
CLI for Ansib-eL (AI-Native Version Control System)

This module provides the command-line interface for ai-git,
extending standard git with AI-native commands.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

# Try to import Click, fall back to argparse
try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    import argparse

# Import the git wrapper
from ansibel.git_wrapper import GitWrapper, AgentMetadata
from ansibel.exceptions import GitWrapperError


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def print_header(text: str):
    """Print a formatted header."""
    print(colorize(f"\n{'='*60}", Colors.HEADER))
    print(colorize(f"  {text}", Colors.BOLD + Colors.HEADER))
    print(colorize(f"{'='*60}\n", Colors.HEADER))


def print_success(text: str):
    """Print a success message."""
    print(colorize(f"✓ {text}", Colors.GREEN))


def print_error(text: str):
    """Print an error message."""
    print(colorize(f"✗ {text}", Colors.RED), file=sys.stderr)


def print_warning(text: str):
    """Print a warning message."""
    print(colorize(f"⚠ {text}", Colors.YELLOW))


def print_info(text: str):
    """Print an info message."""
    print(colorize(f"ℹ {text}", Colors.CYAN))


def get_wrapper() -> GitWrapper:
    """Get a GitWrapper instance for the current directory."""
    return GitWrapper('.')


# =============================================================================
# Click-based CLI Implementation
# =============================================================================

if CLICK_AVAILABLE:
    
    @click.group()
    @click.version_option(version='1.0.0', prog_name='ai-git')
    def cli():
        """AI-Native Version Control System - Extending Git for AI Agents."""
        pass

    # -------------------------------------------------------------------------
    # Init Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.option('--force', '-f', is_flag=True, help='Force reinitialization')
    def init(force: bool):
        """Initialize an ai-git repository."""
        wrapper = get_wrapper()
        
        if wrapper.is_initialized() and not force:
            print_warning("Repository already initialized. Use --force to reinitialize.")
            return
        
        print_header("Initializing AI-Git Repository")
        
        if wrapper.init_repo():
            print_success("Successfully initialized ai-git repository")
            print_info(f"Metadata directory: {wrapper.AI_GIT_DIR}/")
            print_info("Git repository ready for AI-native version control")
        else:
            print_error("Failed to initialize repository")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Status Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
    def status(json_output: bool):
        """Show repository status with AI information."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        repo_status = wrapper.get_status()
        
        if json_output:
            print(json.dumps(repo_status, indent=2))
            return
        
        print_header("AI-Git Repository Status")
        
        # Basic info
        print(colorize(f"Branch: {repo_status.get('branch', 'unknown')}", Colors.BOLD))
        print(f"Working tree: {'dirty' if repo_status.get('is_dirty') else 'clean'}")
        print()
        
        # File status
        if repo_status.get('staged_files'):
            print(colorize("Staged files:", Colors.GREEN))
            for f in repo_status['staged_files']:
                print(f"  + {f}")
            print()
        
        if repo_status.get('modified_files'):
            print(colorize("Modified files:", Colors.YELLOW))
            for f in repo_status['modified_files']:
                print(f"  M {f}")
            print()
        
        if repo_status.get('untracked_files'):
            print(colorize("Untracked files:", Colors.RED))
            for f in repo_status['untracked_files']:
                print(f"  ? {f}")
            print()
        
        # Agent info
        active_agents = repo_status.get('active_agents', [])
        if active_agents:
            print(colorize(f"Active Agent Branches ({len(active_agents)}):", Colors.CYAN))
            for agent in active_agents:
                print(f"  → {agent['name']}")
                print(f"    Agent: {agent.get('agent_id', 'unknown')}")
                print(f"    Purpose: {agent.get('purpose', 'N/A')}")
                print()
        else:
            print_info("No active agent branches")

    # -------------------------------------------------------------------------
    # Branch Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('agent_id')
    @click.option('--purpose', '-p', default='Agent workspace', help='Purpose of the branch')
    def branch(agent_id: str, purpose: str):
        """Create a new agent branch."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        print_header(f"Creating Agent Branch for {agent_id}")
        
        try:
            branch_name = wrapper.create_agent_branch(agent_id, purpose)
            print_success(f"Created branch: {branch_name}")
            print_info(f"Purpose: {purpose}")
            print_info("You are now on the new branch. Make your changes and commit.")
        except GitWrapperError as e:
            print_error(str(e))
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Commit Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('message')
    @click.option('--agent-id', '-a', required=True, help='Agent identifier')
    @click.option('--model-version', '-m', default='unknown', help='Model version')
    @click.option('--prompt-hash', '-p', default='', help='Prompt hash')
    @click.option('--parent-task', '-t', default=None, help='Parent task ID')
    @click.option('--confidence', '-c', type=float, default=None, help='Confidence score')
    @click.option('--files', '-f', multiple=True, help='Specific files to commit')
    def commit(message: str, agent_id: str, model_version: str, prompt_hash: str,
               parent_task: Optional[str], confidence: Optional[float], files):
        """Create a commit with AI metadata."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        # Generate prompt hash if not provided
        if not prompt_hash:
            prompt_hash = f"auto-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        
        metadata = AgentMetadata(
            agent_id=agent_id,
            model_version=model_version,
            prompt_hash=prompt_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_task=parent_task,
            confidence_score=confidence
        )
        
        print_header("Creating Commit with AI Metadata")
        
        try:
            file_list = list(files) if files else None
            commit_hash = wrapper.commit_with_metadata(message, metadata, file_list)
            print_success(f"Created commit: {commit_hash[:8]}")
            print_info(f"Agent: {agent_id}")
            print_info(f"Model: {model_version}")
        except GitWrapperError as e:
            print_error(str(e))
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Merge Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('branch_name')
    @click.option('--strategy', '-s', 
                  type=click.Choice(['merge', 'squash', 'rebase'], case_sensitive=False),
                  default='merge', help='Merge strategy')
    @click.option('--target', '-t', default=None, help='Target branch (default: current)')
    @click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
    def merge(branch_name: str, strategy: str, target: Optional[str], yes: bool):
        """Merge an agent branch."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        print_header(f"Merging Branch: {branch_name}")
        
        # Show diff preview
        try:
            current_branch = wrapper.get_status().get('branch', 'main')
            diff = wrapper.get_diff(current_branch, branch_name)
            if diff:
                print(colorize("Changes to be merged:", Colors.CYAN))
                print(diff[:2000] + "..." if len(diff) > 2000 else diff)
                print()
        except (GitWrapperError, Exception):
            pass

        if not yes:
            if not click.confirm("Proceed with merge?"):
                print_info("Merge cancelled")
                return
        
        result = wrapper.merge_agent_branch(branch_name, strategy, target)
        
        if result.success:
            print_success(result.message)
            if result.merged_commit:
                print_info(f"Merged commit: {result.merged_commit[:8]}")
        else:
            print_error(result.message)
            if result.conflicts:
                print_warning(f"Conflicts in: {', '.join(result.conflicts)}")
                print_info("Run 'ai-git review' to resolve conflicts")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # History Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.option('--limit', '-n', default=20, help='Number of commits to show')
    @click.option('--agent', '-a', default=None, help='Filter by agent ID')
    @click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
    def history(limit: int, agent: Optional[str], json_output: bool):
        """Show AI-enhanced commit history."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        commits = wrapper.get_ai_enhanced_history(limit)
        
        if agent:
            commits = [c for c in commits 
                      if c.get('metadata') and c['metadata'].get('agent_id') == agent]
        
        if json_output:
            print(json.dumps(commits, indent=2))
            return
        
        print_header(f"AI-Enhanced Commit History (last {len(commits)} commits)")
        
        for commit in commits:
            # Format commit line
            hash_str = colorize(commit['hash'], Colors.YELLOW)
            message = commit['message']
            author = colorize(commit['author'], Colors.CYAN)
            
            print(f"{hash_str} - {message}")
            print(f"  Author: {author} | {commit['date'][:10]}")
            
            # Show metadata if available
            metadata = commit.get('metadata')
            if metadata:
                agent_id = metadata.get('agent_id', 'unknown')
                model = metadata.get('model_version', 'unknown')
                confidence = metadata.get('confidence_score')
                
                meta_str = f"  AI: {colorize(agent_id, Colors.GREEN)} | Model: {model}"
                if confidence is not None:
                    meta_str += f" | Confidence: {confidence:.2f}"
                print(meta_str)
            
            print()

    # -------------------------------------------------------------------------
    # Agents Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.option('--status', '-s', 
                  type=click.Choice(['active', 'merged', 'closed', 'all'], case_sensitive=False),
                  default='all', help='Filter by status')
    @click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
    def agents(status: str, json_output: bool):
        """List active agents and their branches."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        status_filter = None if status == 'all' else status
        agent_branches = wrapper.list_agent_branches(status=status_filter)
        
        if json_output:
            print(json.dumps(agent_branches, indent=2))
            return
        
        print_header(f"Agent Branches ({len(agent_branches)} total)")
        
        # Group by status
        by_status = {}
        for branch in agent_branches:
            s = branch.get('status', 'unknown')
            by_status.setdefault(s, []).append(branch)
        
        for status_name, branches in by_status.items():
            status_color = Colors.GREEN if status_name == 'active' else Colors.YELLOW
            print(colorize(f"\n[{status_name.upper()}] ({len(branches)})", status_color + Colors.BOLD))
            
            for branch in branches:
                print(f"  → {colorize(branch['name'], Colors.CYAN)}")
                print(f"    Agent ID: {branch.get('agent_id', 'unknown')}")
                print(f"    Purpose: {branch.get('purpose', 'N/A')}")
                print(f"    Created: {branch.get('created_at', 'unknown')[:10]}")
                
                # Show trust score
                agent_id = branch.get('agent_id')
                if agent_id:
                    trust = wrapper.get_agent_trust_score(agent_id)
                    trust_color = Colors.GREEN if trust > 0.7 else (Colors.YELLOW if trust > 0.4 else Colors.RED)
                    print(f"    Trust Score: {colorize(f'{trust:.2f}', trust_color)}")

    # -------------------------------------------------------------------------
    # Review Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.option('--branch', '-b', default=None, help='Specific branch to review')
    def review(branch: Optional[str]):
        """Enter review mode for pending merges."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        print_header("AI-Git Review Mode")
        
        active_branches = wrapper.list_agent_branches(status='active')
        
        if not active_branches:
            print_info("No pending agent branches to review")
            return
        
        if branch:
            # Review specific branch
            target_branch = next((b for b in active_branches if b['name'] == branch), None)
            if not target_branch:
                print_error(f"Branch '{branch}' not found or not active")
                sys.exit(1)
            branches_to_review = [target_branch]
        else:
            branches_to_review = active_branches
        
        for idx, branch_info in enumerate(branches_to_review, 1):
            branch_name = branch_info['name']
            
            print(colorize(f"\n[{idx}/{len(branches_to_review)}] Reviewing: {branch_name}", Colors.BOLD))
            print(f"Agent: {branch_info.get('agent_id', 'unknown')}")
            print(f"Purpose: {branch_info.get('purpose', 'N/A')}")
            print()
            
            # Show diff
            try:
                current = wrapper.get_status().get('branch', 'main')
                diff = wrapper.get_diff(current, branch_name)
                if diff:
                    print(colorize("Changes:", Colors.CYAN))
                    print(diff[:3000] + "..." if len(diff) > 3000 else diff)
                else:
                    print_info("No changes to display")
            except Exception as e:
                print_warning(f"Could not show diff: {e}")
            
            print()
            
            # Interactive review
            action = click.prompt(
                "Action",
                type=click.Choice(['merge', 'skip', 'diff', 'details', 'quit'], case_sensitive=False),
                default='details'
            )
            
            if action == 'merge':
                result = wrapper.merge_agent_branch(branch_name)
                if result.success:
                    print_success(f"Merged {branch_name}")
                else:
                    print_error(f"Merge failed: {result.message}")
            
            elif action == 'skip':
                print_info(f"Skipped {branch_name}")
            
            elif action == 'diff':
                try:
                    full_diff = wrapper.get_diff(current, branch_name)
                    print(full_diff)
                except Exception as e:
                    print_error(f"Could not show diff: {e}")
            
            elif action == 'details':
                # Show commit history for this branch
                commits = wrapper.get_ai_enhanced_history(limit=10)
                branch_commits = [c for c in commits if branch_name in c.get('branches', [])]
                for c in branch_commits:
                    print(f"  {c['hash']} - {c['message']}")
            
            elif action == 'quit':
                print_info("Exiting review mode")
                return

    # -------------------------------------------------------------------------
    # Trust Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('agent_id')
    @click.argument('score_delta', type=float)
    def trust(agent_id: str, score_delta: float):
        """Update trust score for an agent."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        new_score = wrapper.update_agent_trust_score(agent_id, score_delta)
        
        score_color = Colors.GREEN if new_score > 0.7 else (Colors.YELLOW if new_score > 0.4 else Colors.RED)
        print_success(f"Updated trust score for {agent_id}: {colorize(f'{new_score:.2f}', score_color)}")

    # -------------------------------------------------------------------------
    # Diff Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('branch_a')
    @click.argument('branch_b', required=False)
    def diff(branch_a: str, branch_b: Optional[str]):
        """Show diff between branches."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        if branch_b is None:
            # Compare with current branch
            branch_b = wrapper.get_status().get('branch', 'main')
            branch_a, branch_b = branch_b, branch_a
        
        try:
            diff_output = wrapper.get_diff(branch_a, branch_b)
            print(diff_output)
        except GitWrapperError as e:
            print_error(str(e))
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Metadata Command
    # -------------------------------------------------------------------------
    @cli.command()
    @click.argument('commit_hash')
    def metadata(commit_hash: str):
        """Show metadata for a specific commit."""
        wrapper = get_wrapper()
        
        if not wrapper.is_initialized():
            print_error("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        metadata = wrapper.get_commit_metadata(commit_hash)
        
        if metadata:
            print_header(f"Metadata for {commit_hash}")
            print(json.dumps(metadata.to_dict(), indent=2))
        else:
            print_warning(f"No metadata found for commit {commit_hash}")


# =============================================================================
# Argparse-based CLI Implementation (Fallback)
# =============================================================================

def create_argparse_cli():
    """Create CLI using argparse as fallback."""
    parser = argparse.ArgumentParser(
        prog='ai-git',
        description='AI-Native Version Control System - Extending Git for AI Agents'
    )
    parser.add_argument('--version', action='version', version='ai-git 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize an ai-git repository')
    init_parser.add_argument('--force', '-f', action='store_true', help='Force reinitialization')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show repository status')
    status_parser.add_argument('--json-output', '-j', action='store_true', help='Output as JSON')
    
    # Branch command
    branch_parser = subparsers.add_parser('branch', help='Create a new agent branch')
    branch_parser.add_argument('agent_id', help='Agent identifier')
    branch_parser.add_argument('--purpose', '-p', default='Agent workspace', help='Purpose of the branch')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Create a commit with AI metadata')
    commit_parser.add_argument('message', help='Commit message')
    commit_parser.add_argument('--agent-id', '-a', required=True, help='Agent identifier')
    commit_parser.add_argument('--model-version', '-m', default='unknown', help='Model version')
    commit_parser.add_argument('--prompt-hash', '-p', default='', help='Prompt hash')
    commit_parser.add_argument('--parent-task', '-t', default=None, help='Parent task ID')
    commit_parser.add_argument('--confidence', '-c', type=float, default=None, help='Confidence score')
    commit_parser.add_argument('--files', '-f', nargs='+', help='Specific files to commit')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge an agent branch')
    merge_parser.add_argument('branch_name', help='Branch to merge')
    merge_parser.add_argument('--strategy', '-s', choices=['merge', 'squash', 'rebase'],
                             default='merge', help='Merge strategy')
    merge_parser.add_argument('--target', '-t', default=None, help='Target branch')
    merge_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show AI-enhanced commit history')
    history_parser.add_argument('--limit', '-n', type=int, default=20, help='Number of commits')
    history_parser.add_argument('--agent', '-a', default=None, help='Filter by agent ID')
    history_parser.add_argument('--json-output', '-j', action='store_true', help='Output as JSON')
    
    # Agents command
    agents_parser = subparsers.add_parser('agents', help='List active agents')
    agents_parser.add_argument('--status', '-s', choices=['active', 'merged', 'closed', 'all'],
                              default='all', help='Filter by status')
    agents_parser.add_argument('--json-output', '-j', action='store_true', help='Output as JSON')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Enter review mode')
    review_parser.add_argument('--branch', '-b', default=None, help='Specific branch to review')
    
    # Trust command
    trust_parser = subparsers.add_parser('trust', help='Update trust score for an agent')
    trust_parser.add_argument('agent_id', help='Agent identifier')
    trust_parser.add_argument('score_delta', type=float, help='Trust score change')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Show diff between branches')
    diff_parser.add_argument('branch_a', help='First branch')
    diff_parser.add_argument('branch_b', nargs='?', help='Second branch (default: current)')
    
    # Metadata command
    meta_parser = subparsers.add_parser('metadata', help='Show commit metadata')
    meta_parser.add_argument('commit_hash', help='Commit hash')
    
    return parser


def run_argparse_cli():
    """Run the argparse-based CLI."""
    parser = create_argparse_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    wrapper = get_wrapper()
    
    # Init command
    if args.command == 'init':
        if wrapper.is_initialized() and not args.force:
            print("Repository already initialized. Use --force to reinitialize.")
            return
        
        if wrapper.init_repo():
            print("Successfully initialized ai-git repository")
        else:
            print("Failed to initialize repository")
            sys.exit(1)
    
    # Status command
    elif args.command == 'status':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        status = wrapper.get_status()
        if args.json_output:
            print(json.dumps(status, indent=2))
        else:
            print(f"Branch: {status.get('branch')}")
            print(f"Dirty: {status.get('is_dirty')}")
            print(f"Active agents: {len(status.get('active_agents', []))}")
    
    # Branch command
    elif args.command == 'branch':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        try:
            branch_name = wrapper.create_agent_branch(args.agent_id, args.purpose)
            print(f"Created branch: {branch_name}")
        except GitWrapperError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Commit command
    elif args.command == 'commit':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        metadata = AgentMetadata(
            agent_id=args.agent_id,
            model_version=args.model_version,
            prompt_hash=args.prompt_hash or f"auto-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_task=args.parent_task,
            confidence_score=args.confidence
        )
        
        try:
            commit_hash = wrapper.commit_with_metadata(
                args.message, metadata, args.files
            )
            print(f"Created commit: {commit_hash[:8]}")
        except GitWrapperError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Merge command
    elif args.command == 'merge':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        if not args.yes:
            confirm = input(f"Merge {args.branch_name}? [y/N]: ")
            if confirm.lower() != 'y':
                print("Merge cancelled")
                return
        
        result = wrapper.merge_agent_branch(args.branch_name, args.strategy, args.target)
        
        if result.success:
            print(f"Success: {result.message}")
        else:
            print(f"Error: {result.message}")
            sys.exit(1)
    
    # History command
    elif args.command == 'history':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        commits = wrapper.get_ai_enhanced_history(args.limit)
        
        if args.agent:
            commits = [c for c in commits 
                      if c.get('metadata') and c['metadata'].get('agent_id') == args.agent]
        
        if args.json_output:
            print(json.dumps(commits, indent=2))
        else:
            for commit in commits:
                print(f"{commit['hash']} - {commit['message']}")
                if commit.get('metadata'):
                    print(f"  Agent: {commit['metadata'].get('agent_id')}")
    
    # Agents command
    elif args.command == 'agents':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        status_filter = None if args.status == 'all' else args.status
        branches = wrapper.list_agent_branches(status=status_filter)
        
        if args.json_output:
            print(json.dumps(branches, indent=2))
        else:
            for branch in branches:
                print(f"{branch['name']} - {branch.get('status')} - {branch.get('agent_id')}")
    
    # Review command
    elif args.command == 'review':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        active = wrapper.list_agent_branches(status='active')
        
        if not active:
            print("No pending agent branches to review")
            return
        
        for branch in active:
            print(f"\nBranch: {branch['name']}")
            print(f"Agent: {branch.get('agent_id')}")
            print(f"Purpose: {branch.get('purpose')}")
    
    # Trust command
    elif args.command == 'trust':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        new_score = wrapper.update_agent_trust_score(args.agent_id, args.score_delta)
        print(f"Updated trust score for {args.agent_id}: {new_score:.2f}")
    
    # Diff command
    elif args.command == 'diff':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        branch_b = args.branch_b or wrapper.get_status().get('branch', 'main')
        
        try:
            diff_output = wrapper.get_diff(args.branch_a, branch_b)
            print(diff_output)
        except GitWrapperError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Metadata command
    elif args.command == 'metadata':
        if not wrapper.is_initialized():
            print("Not an ai-git repository. Run 'ai-git init' first.")
            sys.exit(1)
        
        metadata = wrapper.get_commit_metadata(args.commit_hash)
        
        if metadata:
            print(json.dumps(metadata.to_dict(), indent=2))
        else:
            print(f"No metadata found for commit {args.commit_hash}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI."""
    if CLICK_AVAILABLE:
        cli()
    else:
        run_argparse_cli()


if __name__ == '__main__':
    main()
