import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_output_provenance_present():
 g=json.loads((ROOT/'reproducibility/golden_reference.json').read_text()); c=json.loads((ROOT/'configs/checkpoints.json').read_text()); assert c['xrv']['sha256']; assert c['eva_x']['sha256']; assert (ROOT/'predictions/standardized/locked_rsna.csv').exists()
