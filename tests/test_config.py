from pathlib import Path
from aura_cxr.config import load_config
def test_config(): load_config(Path(__file__).parents[1]/'configs/frozen_config.yaml')
