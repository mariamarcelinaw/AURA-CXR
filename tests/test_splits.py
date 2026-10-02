from pathlib import Path
import pandas as pd
from aura_cxr.data import assert_split_integrity
def test_splits():
 r=Path(__file__).parents[1]/'splits'; assert_split_integrity(pd.read_csv(r/'train.csv'),pd.read_csv(r/'val.csv'),pd.read_csv(r/'test.csv'),pd.read_csv(r/'development_oof_folds.csv'))
