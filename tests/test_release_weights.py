import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_weight_hashes_match_registry():
    ck = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    listed = dict(l.split()[::-1] for l in (ROOT / 'configs/release_weights.sha256').read_text().splitlines() if l.strip())
    assert len(listed) == 3
    for k in ('xrv', 'eva_x', 'radiomics'):
        assert listed[ck[k]['local_filename']] == ck[k]['sha256']
        assert ck[k]['redistribution_status'] == 'RELEASED'
        assert 'releases/download/v1.0/' + ck[k]['local_filename'] in ck[k]['download_instructions']
