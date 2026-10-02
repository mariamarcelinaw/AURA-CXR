from pathlib import Path
from aura_cxr.evaluation import reproduce_locked
def test_locked_golden():
 r=Path(__file__).parents[1]; m=reproduce_locked(r/'predictions/standardized/locked_rsna.csv'); assert abs(m['A4']['AUC']-0.8995679885333019)<1e-12; assert abs(m['A8']['AUC']-0.8987752304044404)<1e-12
