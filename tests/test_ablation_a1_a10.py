import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_table5_recomputes_from_released_files():
    r = subprocess.run([sys.executable, 'scripts/analysis/ablation_a1_a10.py', '--check'], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'ABLATION A1-A10 = PASS' in r.stdout, r.stdout + r.stderr
