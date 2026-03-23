import re
from datetime import datetime
from skills.shared.run_id import generate_run_id

def test_generate_run_id_format():
    run_id = generate_run_id()
    # Format: YYYYMMDD_HHMMSS_suffix
    pattern = r'^\d{8}_\d{6}_[a-z0-9]{4}$'
    assert re.match(pattern, run_id), f"Invalid format: {run_id}"

def test_generate_run_id_unique():
    run_id1 = generate_run_id()
    run_id2 = generate_run_id()
    assert run_id1 != run_id2

def test_generate_run_id_prefix():
    run_id = generate_run_id()
    today = datetime.now().strftime('%Y%m%d')
    assert run_id.startswith(today)
