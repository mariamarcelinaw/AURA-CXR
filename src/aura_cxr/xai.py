"""Branch-scoped Hybrid CNN-ViT LayerCAM configuration and pure helpers.
This module does NOT claim to explain LightGBM or the complete A8 Logistic Regression decision.
"""
import math, numpy as np
XAI_SCOPE='HYBRID_CNN_VIT_BRANCH_ONLY'
XRV_LAYERS=('backbone.features.denseblock4.denselayer16.conv1','backbone.features.denseblock4.denselayer15.conv1')
EVA_LAYERS=('blocks.11.mlp','blocks.11')
LAYER_WEIGHTS=(0.75,0.25); INTER_MODEL_WEIGHTS={'xrv':0.25,'eva_x':0.75}; ACTIVATION_THRESHOLD=0.25; PATCH_SIZE=16; GRID=(14,14)

def token_grid_size(n_tokens:int):
    if int(math.isqrt(max(n_tokens-1,0)))**2==n_tokens-1: return True,int(math.isqrt(n_tokens-1))
    if int(math.isqrt(n_tokens))**2==n_tokens: return False,int(math.isqrt(n_tokens))
    raise ValueError('token sequence cannot form square spatial grid')

def layercam_from_activation_gradient(activation,gradient,embedding_axis=-1):
    a=np.asarray(activation); g=np.asarray(gradient); return np.sum(a*np.maximum(g,0),axis=embedding_axis)

def normalize_map(h):
    h=np.maximum(np.asarray(h,dtype=np.float32),0); m=float(h.max())
    return h/m if m>0 else np.zeros_like(h)

def fuse_maps(xrv,eva): return normalize_map(INTER_MODEL_WEIGHTS['xrv']*normalize_map(xrv)+INTER_MODEL_WEIGHTS['eva_x']*normalize_map(eva))
def assert_xai_config():
    assert XAI_SCOPE=='HYBRID_CNN_VIT_BRANCH_ONLY'; assert abs(sum(LAYER_WEIGHTS)-1)<1e-12; assert abs(sum(INTER_MODEL_WEIGHTS.values())-1)<1e-12; assert ACTIVATION_THRESHOLD==0.25; return True
