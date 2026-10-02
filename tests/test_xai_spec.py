import json
from pathlib import Path

def test_xai_spec():
    p=Path(__file__).resolve().parents[1]/"configs"/"xai_frozen.json"
    d=json.loads(p.read_text())
    assert d["scope"]["xai_scope_constant"]=="HYBRID_CNN_VIT_BRANCH_ONLY"
    assert d["eva"]["token_structure"]["num_prefix_tokens"]==1
    assert d["eva"]["token_structure"]["num_patch_tokens"]==196
    assert d["eva"]["token_structure"]["ordering"]=="row-major raster"
    assert d["comparator"]["method"]=="Integrated Gradients"
    assert d["comparator"]["n_valid"]==1203
    assert d["protocol_integrity"]["classifier_retrained"] is False
    assert d["protocol_integrity"]["a8_modified"] is False
