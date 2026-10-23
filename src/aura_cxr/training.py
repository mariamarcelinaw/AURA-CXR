"""Training safety helpers and deterministic seed setup.
The full training loops are the scripts in scripts/ and are not rewritten here.
"""
import random, numpy as np
from .data import require_optimization_safe

def set_global_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception: pass
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass

def guard_training_split(split_name:str): require_optimization_safe(split_name)
