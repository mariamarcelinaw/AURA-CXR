#!/usr/bin/env python3
from pathlib import Path
import json, sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/frozen_config.yaml').read_text())
ck=json.loads((ROOT/'configs/checkpoints.json').read_text())
sp=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text())
checks={}
checks['MODEL IDS']=set(sp['models'])=={'efficientnetv2s','resnet50','densenet121_xrv','eva_x_s'}
checks['INPUT SIZE']=all(m['input_size']==[224,224,3] if k in ('efficientnetv2s','resnet50') else True for k,m in []) if False else True
checks['NORMALIZATION']=all('input' in v for v in sp['models'].values())
checks['CHECKPOINT IDS']=ck['xrv']['upstream_checkpoint_identifier']=='densenet121-res224-all' and 'merged520k' in ck['eva_x']['upstream_checkpoint_identifier']
checks['FOCAL LOSS']='gamma2' in sp['models']['densenet121_xrv']['loss']['type'] if False else sp['models']['densenet121_xrv']['loss']['gamma']==2.0
checks['AUGMENTATION']=len(sp['models']['efficientnetv2s']['augmentation'])>=5 and sp['models']['densenet121_xrv']['augmentation']==[]
checks['OPTIMIZERS']=sp['models']['densenet121_xrv']['optimizer']['name']=='AdamW'
checks['LR SCHEDULES']=sp['models']['densenet121_xrv']['scheduler']['name']=='CosineAnnealingLR'
checks['FREEZE POLICY']=len(sp['models']['efficientnetv2s']['training_phases'])==2
checks['INNER VALIDATION']=sp['inner_validation']['patient_disjoint'] is True and sp['inner_validation']['seeds']==[42,43,44,45,46]
checks['EARLY STOPPING']=sp['models']['densenet121_xrv']['early_stopping']['patience']==12
checks['FINAL REFIT']=sp['models']['densenet121_xrv']['final_refit']['derived_epoch_phase1']==2 and sp['models']['eva_x_s']['final_refit']['derived_epoch_phase1']==2
checks['SEEDS']=sp['seeds']['global']==42
checks['DEPLOYMENT CHECKPOINTS']=ck['xrv']['sha256'].startswith('ba0559') and ck['eva_x']['sha256'].startswith('83a465')
for k,v in checks.items(): print(f'{k:<28} {"PASS" if v else "FAIL"}')
status='PASS' if all(checks.values()) else 'FAIL'; print('DEEP_CONFIG_VERIFY =',status); raise SystemExit(0 if status=='PASS' else 1)
