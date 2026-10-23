#!/usr/bin/env python3
"""Reproduce core Stage-9 XAI multiplicity tables from canonical case-level artifacts.

This script does not regenerate heatmaps, alter thresholds, modify GT boxes, or change classifiers.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRICS = ["pointing_hit", "iou_at_thr", "localization_score", "energy_inside_gt"]
BASELINES = ["image_center", "lesion_prevalence_prior", "lung_prior", "random_noise"]

def paired_boot_ci(diff, n_boot, seed, alpha=.05):
    diff=np.asarray(diff,float); rng=np.random.default_rng(seed); b=np.empty(n_boot)
    for i in range(n_boot): b[i]=rng.choice(diff,size=len(diff),replace=True).mean()
    return np.percentile(b,[100*alpha/2,100*(1-alpha/2)])

def signflip_p(diff, n_perm, seed):
    diff=np.asarray(diff,float); obs=abs(diff.mean()); rng=np.random.default_rng(seed)
    extreme=0; done=0
    while done<n_perm:
        k=min(500,n_perm-done)
        s=rng.integers(0,2,size=(k,len(diff)),dtype=np.int8)*2-1
        null=(s.astype(np.float32) @ diff.astype(np.float32))/len(diff)
        extreme += int(np.count_nonzero(np.abs(null)>=obs-1e-12)); done += k
    return (extreme+1)/(n_perm+1), extreme

def cliffs(u,n1,n2): return 2*u/(n1*n2)-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reports-dir',required=True,type=Path)
    ap.add_argument('--stage7-dir',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--bootstrap-seed',type=int,default=42)
    ap.add_argument('--permutations',type=int,default=20000)
    ap.add_argument('--permutation-seed',type=int,default=20261002)
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)

    model=pd.read_csv(args.reports_dir/'xai_localization_per_sample.csv')
    base=pd.read_csv(args.reports_dir/'lung_prior_completion'/'xai_localization_baselines_per_sample.csv')
    orig=pd.read_csv(args.reports_dir/'lung_prior_completion'/'xai_localization_baseline_comparison.csv')
    tpfn=pd.read_csv(args.reports_dir/'xai_tp_vs_fn_statistics.csv')
    st7=pd.read_csv(args.stage7_dir/'12_TRANSFORMER_COMPARATOR_RESULTS.csv')

    if not (len(model)==1203 and model.patientId.nunique()==1203): raise RuntimeError('canonical 1203-case model file not verified')
    if set(base.method)!=set(BASELINES) or len(base)!=1203*4: raise RuntimeError('completed four-baseline case file not verified')
    if len(orig)!=16: raise RuntimeError('original 16-contrast family not reconstructable')

    A=[]; R=[]
    for bi,bn in enumerate(BASELINES):
        m=model[['sample_index','patientId']+METRICS].merge(base[base.method==bn][['sample_index','patientId']+METRICS],on=['sample_index','patientId'],suffixes=('_model','_baseline'),validate='one_to_one')
        for mi,metric in enumerate(METRICS):
            d=(m[f'{metric}_model']-m[f'{metric}_baseline']).to_numpy(float)
            effect=float(d.mean()); lo,hi=paired_boot_ci(d,args.bootstrap,args.bootstrap_seed+1000+mi)
            old=orig[(orig.baseline==bn)&(orig.metric==metric)].iloc[0]
            ok=max(abs(effect-old.mean_difference_model_minus_baseline),abs(lo-old.ci95_low),abs(hi-old.ci95_high))<1e-12
            R.append({'comparison_id':f'A_{bn}_{metric}','reproduction_pass':ok,'effect':effect,'CI95_low':lo,'CI95_high':hi})
            p,ext=signflip_p(d,args.permutations,args.permutation_seed+bi*100+mi)
            A.append({'comparison_id':f'A_{bn}_{metric}','baseline':bn,'metric':metric,'n':len(d),'delta_model_minus_baseline':effect,'CI95_low':lo,'CI95_high':hi,'p_raw':p,'extreme_count':ext})
    A=pd.DataFrame(A); _,A['p_Holm'],_,_=multipletests(A.p_raw,method='holm'); _,A['q_BH'],_,_=multipletests(A.p_raw,method='fdr_bh'); A['Holm_significant']=A.p_Holm<.05
    A.to_csv(out/'02_FAMILY_A_LOCALIZATION_BASELINE_RESULTS.csv',index=False); pd.DataFrame(R).to_csv(out/'05_ORIGINAL_XAI_CONTRAST_REPRODUCTION.csv',index=False)

    B=[]
    for _,r in tpfn.iterrows():
        metric=r.metric; tp=model.loc[model.category=='TP',metric].dropna().to_numpy(float); fn=model.loc[model.category=='FN',metric].dropna().to_numpy(float)
        res=stats.mannwhitneyu(tp,fn,alternative='two-sided')
        B.append({'comparison_id':f'B_TP_vs_FN_{metric}','metric':metric,'n_tp':len(tp),'n_fn':len(fn),'p_raw':res.pvalue,'cliffs_delta':cliffs(res.statistic,len(tp),len(fn))})
    B=pd.DataFrame(B); _,B['p_Holm'],_,_=multipletests(B.p_raw,method='holm'); B['Holm_significant']=B.p_Holm<.05; B.to_csv(out/'15_FAMILY_B_TP_FN_RESULTS.csv',index=False)

    C=st7[st7.analysis_level=='EVA_branch'].copy(); _,reh,_,_=multipletests(C.raw_p,method='holm'); C['holm_recomputed']=reh; C['holm_verification_pass']=np.abs(C.adjusted_p-reh)<1e-12; C.to_csv(out/'16_FAMILY_C_TRANSFORMER_SENSITIVITY_RESULTS.csv',index=False)
    print(f"Family A: {int(A.Holm_significant.sum())}/16 Holm-significant")
    print(f"Family B: {int(B.Holm_significant.sum())}/{len(B)} Holm-significant")
    print(f"Family C Holm verification: {'PASS' if C.holm_verification_pass.all() else 'FAIL'}")

if __name__=='__main__': main()
