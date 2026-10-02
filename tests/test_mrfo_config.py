import csv,json
from pathlib import Path
def root(): return Path(__file__).resolve().parents[1]
def test_mrfo_config():
    rows=list(csv.DictReader((root()/"reproducibility/mrfo_search_space.csv").open()))
    l=[r for r in rows if r["estimator"]=="lgbm"]
    assert len(l)==5
    d={r["parameter"]:r for r in l}
    assert float(d["log10_learning_rate"]["lower_bound"])==-2.3
    assert float(d["search_depth"]["upper_bound"])==12
    s=json.loads((root()/"reproducibility/radiomics_specification.json").read_text())["mrfo"]
    assert s["final"]["population"]==15 and s["final"]["iterations"]==15 and s["final"]["inner_cv"]==3
    assert s["final"]["seed"]==42 and s["optimizer_direction"].startswith("MINIMIZE")
