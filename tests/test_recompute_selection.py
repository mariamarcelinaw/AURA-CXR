import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_selection_recomputes_from_development_oof(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/recompute_selection.py'), '--check',
                        '--out', str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'RECOMPUTE SELECTION = PASS' in r.stdout
    assert 'Selected pair: xrv+eva_x' in r.stdout
