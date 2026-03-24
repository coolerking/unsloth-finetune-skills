"""Pytest configuration."""
import sys
from pathlib import Path

# Add .claude to Python path so 'skills' package is recognized
claude_dir = Path(__file__).parent.parent / ".claude"
sys.path.insert(0, str(claude_dir))

# Also add .claude/skills for direct imports
skills_dir = claude_dir / "skills"
sys.path.insert(0, str(skills_dir))
