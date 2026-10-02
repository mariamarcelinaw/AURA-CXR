#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    statuses={}
    try:
        spec=json.loads((ROOT/'configs/xai_frozen.json').read_text())
        statuses['CHECKPOINT HASHES']='PASS' if spec['checkpoints']['xrv_sha256'] and spec['checkpoints']['eva_sha256'] and spec['checkpoints']['unchanged_after_run'] else 'FAIL'
        statuses['XRV HOOKS']='PASS' if len(spec['xrv']['hooks'])==2 and spec['xrv']['activation_shapes']==['[1, 128, 7, 7]','[1, 128, 7, 7]'] else 'FAIL'
        statuses['EVA HOOKS']='PASS' if len(spec['eva']['hooks'])==2 and spec['eva']['activation_shapes']==['[1, 197, 384]','[1, 197, 384]'] else 'FAIL'
        t=spec['eva']['token_structure']
        statuses['TOKEN COUNT']='PASS' if t['num_patch_tokens']==196 else 'FAIL'
        statuses['PREFIX TOKENS']='PASS' if t['num_prefix_tokens']==1 and t['class_token'] is True and t['other_prefix_tokens']==0 else 'FAIL'
        statuses['PATCH ORDER']='PASS' if t['ordering']=='row-major raster' else 'FAIL'
        statuses['GRID SHAPE']='PASS' if t['grid']==[14,14] and t['patch_size']==[16,16] else 'FAIL'
        statuses['TARGET CLASS']='PASS' if spec['target']['class_index']==1 and spec['target']['scalar']=='pre-softmax class-1 logit' else 'FAIL'
        statuses['LAYER WEIGHTS']='PASS' if spec['fusion']['layer_weights']==[0.75,0.25] else 'FAIL'
        statuses['MODEL WEIGHTS']='PASS' if spec['fusion']['xrv_weight']==0.25 and spec['fusion']['eva_weight']==0.75 else 'FAIL'
        statuses['MAP POLICY']='PASS' if spec['postprocessing']['map_policy']=='raw' else 'FAIL'
        statuses['THRESHOLD']='PASS' if abs(spec['postprocessing']['threshold']-0.25)<1e-12 else 'FAIL'
        statuses['FAITHFULNESS CONFIG']='PASS' if spec['faithfulness']['sample_n']==256 and spec['faithfulness']['top_fraction']==0.10 and spec['faithfulness']['explains_final_A8'] is False else 'FAIL'
        statuses['BRANCH SCOPE']='PASS' if spec['scope']['xai_scope_constant']=='HYBRID_CNN_VIT_BRANCH_ONLY' and not spec['scope']['explains_radiomics'] and not spec['scope']['explains_meta_learner'] and not spec['scope']['explains_final_A8'] else 'FAIL'
    except Exception as e:
        statuses['SPECIFICATION']='FAIL:'+str(e)
    try:
        exp=ROOT/'analysis/transformer_xai/results'
        cmp=pd.read_csv(exp/'comparator_results.csv')
        case=pd.read_csv(exp/'case_level_sensitivity.csv')
        statuses['ORIGINAL METRICS']='PASS' if len(case)==1203 else 'FAIL'
        statuses['COMPARATOR']='PASS' if len(cmp[cmp.analysis_level=='EVA_branch'])==4 and case.eva_ig_valid.astype(bool).all() else 'FAIL'
    except Exception as e:
        statuses['RESULTS']='FAIL:'+str(e)
    for key in ['CHECKPOINT HASHES','XRV HOOKS','EVA HOOKS','TOKEN COUNT','PREFIX TOKENS','PATCH ORDER','GRID SHAPE','TARGET CLASS','LAYER WEIGHTS','MODEL WEIGHTS','MAP POLICY','THRESHOLD','ORIGINAL METRICS','FAITHFULNESS CONFIG','BRANCH SCOPE','COMPARATOR']:
        if key in statuses: print(f'{key:<24} {statuses[key]}')
    for key,val in statuses.items():
        if key not in ['CHECKPOINT HASHES','XRV HOOKS','EVA HOOKS','TOKEN COUNT','PREFIX TOKENS','PATCH ORDER','GRID SHAPE','TARGET CLASS','LAYER WEIGHTS','MODEL WEIGHTS','MAP POLICY','THRESHOLD','ORIGINAL METRICS','FAITHFULNESS CONFIG','BRANCH SCOPE','COMPARATOR']:
            print(f'{key:<24} {val}')
    fail=any(str(v).startswith('FAIL') for v in statuses.values())
    print('XAI_VERIFY =','FAIL' if fail else 'PASS')
    return 1 if fail else 0
if __name__=='__main__': raise SystemExit(main())
