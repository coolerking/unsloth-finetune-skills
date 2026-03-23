"""Utilities for generating unique run IDs."""
import random
import string
from datetime import datetime

def generate_run_id() -> str:
    """Generate unique run ID in format: YYYYMMDD_HHMMSS_suffix"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{suffix}"
