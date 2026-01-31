#!/usr/bin/env python3
"""
Tests for the Git Wrapper and CLI modules.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from git_wrapper import GitWrapper, AgentMetadata, MergeResult, GitWrapperError


class TestGitWrapper:
    """Test suite for GitWrapper class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.wrapper = GitWrapper(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init_repo(self):
        """Test repository initialization."""
        assert self.wrapper.init_repo() is True
        assert self.wrapper.is_initialized() is True
        assert (Path(self.test_dir) / '.ai-git').exists()
        assert (Path(self.test_dir) / '.ai-git' / 'metadata.json').exists()
    
    def test_is_initialized_false(self):
        """Test is_initialized returns False before init."""
        assert self.wrapper.is_initialized() is False
    
    def test_create_agent_branch(self):
        """Test creating an agent branch."""
        self.wrapper.init_repo()
        
        branch_name = self.wrapper.create_agent_branch(
            agent_id="test-agent",
            purpose="Test purpose"
        )
        
        assert branch_name.startswith("agent/test-agent/")
        
        # Check branch metadata was saved
        branch_file = Path(self.test_dir) / '.ai-git' / 'agents' / f"{branch_name.replace('/', '_')}.json"
        assert branch_file.exists()
        
        with open(branch_file) as f:
            data = json.load(f)
            assert data['agent_id'] == 'test-agent'
            assert data['purpose'] == 'Test purpose'
            assert data['status'] == 'active'
    
    def test_commit_with_metadata(self):
        """Test committing with AI metadata."""
        self.wrapper.init_repo()
        
        # Create a test file
        test_file = Path(self.test_dir) / 'test.txt'
        test_file.write_text('Hello, World!')
        
        # Stage the file using git
        import subprocess
        subprocess.run(['git', 'add', 'test.txt'], cwd=self.test_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.test_dir, check=True)
        
        # Create agent branch
        branch_name = self.wrapper.create_agent_branch('test-agent', 'Testing')
        
        # Modify file
        test_file.write_text('Modified content')
        subprocess.run(['git', 'add', 'test.txt'], cwd=self.test_dir, check=True)
        
        # Create commit with metadata
        metadata = AgentMetadata(
            agent_id="test-agent",
            model_version="gpt-4",
            prompt_hash="abc123",
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence_score=0.95
        )
        
        commit_hash = self.wrapper.commit_with_metadata("Test commit", metadata)
        
        assert len(commit_hash) == 40  # Full SHA-1 hash
        
        # Verify metadata was saved
        retrieved = self.wrapper.get_commit_metadata(commit_hash)
        assert retrieved is not None
        assert retrieved.agent_id == "test-agent"
        assert retrieved.model_version == "gpt-4"
        assert retrieved.confidence_score == 0.95
    
    def test_list_agent_branches(self):
        """Test listing agent branches."""
        self.wrapper.init_repo()
        
        # Create some branches
        self.wrapper.create_agent_branch('agent-1', 'Purpose 1')
        self.wrapper.create_agent_branch('agent-2', 'Purpose 2')
        
        branches = self.wrapper.list_agent_branches()
        
        assert len(branches) == 2
        assert all(b['status'] == 'active' for b in branches)
    
    def test_get_status(self):
        """Test getting repository status."""
        self.wrapper.init_repo()
        
        status = self.wrapper.get_status()
        
        assert status['initialized'] is True
        assert 'branch' in status
        assert 'is_dirty' in status
        assert 'active_agents' in status
    
    def test_agent_trust_score(self):
        """Test trust score management."""
        self.wrapper.init_repo()
        
        # Initial score should be 0.5
        score = self.wrapper.get_agent_trust_score('test-agent')
        assert score == 0.5
        
        # Update score
        new_score = self.wrapper.update_agent_trust_score('test-agent', 0.1)
        assert new_score == 0.6
        
        # Verify it persists
        score = self.wrapper.get_agent_trust_score('test-agent')
        assert score == 0.6
        
        # Test clamping
        score = self.wrapper.update_agent_trust_score('test-agent', 0.5)
        assert score == 1.0  # Clamped to max
        
        score = self.wrapper.update_agent_trust_score('test-agent', -2.0)
        assert score == 0.0  # Clamped to min


class TestAgentMetadata:
    """Test suite for AgentMetadata class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = AgentMetadata(
            agent_id="test",
            model_version="gpt-4",
            prompt_hash="abc",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        d = metadata.to_dict()
        
        assert d['agent_id'] == 'test'
        assert d['model_version'] == 'gpt-4'
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'agent_id': 'test',
            'model_version': 'gpt-4',
            'prompt_hash': 'abc',
            'timestamp': '2024-01-01T00:00:00Z',
            'confidence_score': 0.9
        }
        
        metadata = AgentMetadata.from_dict(data)
        
        assert metadata.agent_id == 'test'
        assert metadata.confidence_score == 0.9


def run_tests():
    """Run all tests."""
    import subprocess
    
    # Run pytest if available
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', __file__, '-v'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode
    except FileNotFoundError:
        print("pytest not available, running basic tests...")
        
        # Run basic tests
        test_wrapper = TestGitWrapper()
        test_wrapper.setup_method()
        
        try:
            test_wrapper.test_init_repo()
            print("✓ test_init_repo passed")
        except Exception as e:
            print(f"✗ test_init_repo failed: {e}")
        
        try:
            test_wrapper.test_is_initialized_false()
            print("✓ test_is_initialized_false passed")
        except Exception as e:
            print(f"✗ test_is_initialized_false failed: {e}")
        
        try:
            test_wrapper.test_create_agent_branch()
            print("✓ test_create_agent_branch passed")
        except Exception as e:
            print(f"✗ test_create_agent_branch failed: {e}")
        
        test_wrapper.teardown_method()
        
        # Test AgentMetadata
        test_meta = TestAgentMetadata()
        
        try:
            test_meta.test_to_dict()
            print("✓ test_to_dict passed")
        except Exception as e:
            print(f"✗ test_to_dict failed: {e}")
        
        try:
            test_meta.test_from_dict()
            print("✓ test_from_dict passed")
        except Exception as e:
            print(f"✗ test_from_dict failed: {e}")
        
        return 0


if __name__ == '__main__':
    sys.exit(run_tests())
