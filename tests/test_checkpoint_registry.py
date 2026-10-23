import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_deployment_hashes_locked():
 d=json.loads((ROOT/'configs/checkpoints.json').read_text()); assert d['xrv']['sha256']=='ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a'; assert d['eva_x']['sha256']=='83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6'
