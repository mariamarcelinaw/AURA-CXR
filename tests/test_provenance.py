from pathlib import Path
import json
def test_provenance():
 r=Path(__file__).parents[1]; assert json.loads((r/'reproducibility/deployment_manifest.json').read_text())['deployment_lock_id']=='bdd68083c49a27bf29eda552eb5e8c40b75e8a124046dce3440c6488abde803b'
