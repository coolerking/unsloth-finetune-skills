"""Pytest configuration."""
import sys
from pathlib import Path

# Add .claude/skills to Python path for tests
skills_dir = Path(__file__).parent.parent / ".claude" / "skills"
sys.path.insert(0, str(skills_dir))
