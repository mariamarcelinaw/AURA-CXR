import json
from pathlib import Path
def root(): return Path(__file__).resolve().parents[1]
def test_lightgbm_config():
    p=json.loads((root()/"reproducibility/radiomics_specification.json").read_text())["lightgbm"]["full_development_effective_params"]
    assert p["learning_rate"]==0.03401 and p["num_leaves"]==39
    assert p["search_depth"]==12 and p["max_depth"]==-1
    assert p["reg_lambda"]==10.0 and p["min_child_samples"]==21
    assert p["n_estimators"]==400 and p["subsample"]==0.9 and p["subsample_freq"]==1
    assert p["colsample_bytree"]==0.9 and p["class_weight"]=="balanced" and p["random_state"]==42
    assert p["feature_scaler"] is None
