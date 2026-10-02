import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_phase_budgets():
 d=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text()); p=d['models']['efficientnetv2s']['training_phases']; assert [x['max_epochs'] for x in p]==[40,20]; assert d['models']['densenet121_xrv']['training_phases'][0]['max_epochs']==60
