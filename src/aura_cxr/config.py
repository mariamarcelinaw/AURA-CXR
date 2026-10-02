from pathlib import Path
import yaml
REQUIRED=['project','data','radiomics','mrfo','models','fusion','xai','faithfulness','external_validation']
def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text()); validate_config(cfg); return cfg
def validate_config(cfg):
    miss=[k for k in REQUIRED if k not in cfg]
    if miss: raise ValueError(f'missing config sections: {miss}')
    if cfg['data']['image_size']!=224: raise ValueError('locked image size must be 224')
    if cfg['fusion']['meta_feature_order']!=['p_xrv','p_eva_x','p_radiomics']: raise ValueError('invalid A8 meta-feature order')
    if cfg['fusion']['threshold']!=0.47000000000000003: raise ValueError('invalid frozen threshold')
    if cfg['radiomics']['feature_dimension']!=746: raise ValueError('invalid radiomics dimension')
    if cfg['xai']['scope']!='HYBRID_CNN_VIT_BRANCH_ONLY': raise ValueError('invalid XAI scope')
    if abs(sum(cfg['xai']['layer_weights'])-1)>1e-12 or abs(sum(cfg['xai']['inter_model_weights'].values())-1)>1e-12: raise ValueError('invalid XAI weights')
    return True
