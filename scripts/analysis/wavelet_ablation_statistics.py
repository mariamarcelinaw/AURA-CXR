#!/usr/bin/env python3
"""Statistics of the post-hoc adaptive-versus-fixed wavelet ablation (Section 5.3, Table 6).

Inputs are read below the work directory given by the WORK_DIR environment variable
(default: the current directory):
  $WORK_DIR/stage4_server_results/wavelet_ablation_resumable/   output of wavelet_ablation_run.py --out
  $WORK_DIR/stage4_probs_extract/probs/oof_radiomics_lgbm.npz    radiomics LightGBM OOF probabilities
                                                                  written by train_aura.py (<run>/probs/)
Outputs are written to $WORK_DIR/STAGE_4_WAVELET_ABLATION/; the released copies are in
results/analysis/wavelet_ablation/.

Example:
    WORK_DIR=runs/wavelet python scripts/analysis/wavelet_ablation_statistics.py
"""
from __future__ import annotations
import json, math, hashlib, os, time, resource
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

ROOT=Path(os.environ.get('WORK_DIR', '.'))
SRC=ROOT/'stage4_server_results'/'wavelet_ablation_resumable'
OUT=ROOT/'STAGE_4_WAVELET_ABLATION'
OUT.mkdir(exist_ok=True)
ORIG_NPZ=ROOT/'stage4_probs_extract'/'probs'/'oof_radiomics_lgbm.npz'
POLICIES=['adaptive','fixed_haar','fixed_db4','fixed_sym4','fixed_coif3']
FIXED=['fixed_haar','fixed_db4','fixed_sym4','fixed_coif3']
DISPLAY={'adaptive':'Adaptive entropy-guided','fixed_haar':'Fixed Haar','fixed_db4':'Fixed db4','fixed_sym4':'Fixed sym4','fixed_coif3':'Fixed coif3'}
BOOT_N=2000; SEED=42; MIN_SENS=0.85
START=time.perf_counter()

def sha256(p):
    p=Path(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    out=np.empty(N,float); out[J]=T; return out

def _fast_delong(preds,m):
    n=preds.shape[1]-m; pos=preds[:,:m]; neg=preds[:,m:]; k=preds.shape[0]
    tx=np.vstack([_midrank(pos[r]) for r in range(k)])
    ty=np.vstack([_midrank(neg[r]) for r in range(k)])
    tz=np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs=tz[:,:m].sum(axis=1)/m/n-(m+1.0)/(2*n)
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n
    return aucs,cov

def delong_compare(y,p_adaptive,p_fixed):
    # orientation adaptive - fixed
    y=np.asarray(y,int); pa=np.asarray(p_adaptive,float); pf=np.asarray(p_fixed,float)
    order=np.argsort(-y,kind='stable'); m=int(y.sum())
    aucs,cov=_fast_delong(np.vstack([pa[order],pf[order]]),m)
    l=np.array([1.0,-1.0]); delta=float(l@aucs); var=float(l@cov@l.T); se=math.sqrt(max(var,0.0))
    z=delta/se if se>0 else 0.0; p=float(2*stats.norm.sf(abs(z))) if se>0 else 1.0
    zcrit=float(stats.norm.ppf(0.975))
    return {'adaptive_auc':float(aucs[0]),'fixed_auc':float(aucs[1]),'delta':delta,'se':se,
            'ci_low':delta-zcrit*se,'ci_high':delta+zcrit*se,'z':z,'p_raw':p}

def holm_adjust(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj

def tune_threshold(y,p,min_sensitivity=0.85):
    y=np.asarray(y,int); p=np.asarray(p,float); best=None
    for th in np.linspace(0.01,0.99,197):
        pred=p>=th
        tp=np.sum((y==1)&pred); fn=np.sum((y==1)&~pred); tn=np.sum((y==0)&~pred); fp=np.sum((y==0)&pred)
        sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
        if sens<min_sensitivity: continue
        cand=(sens+spec-1,spec,sens,-abs(th-0.5),th)
        if best is None or cand>best: best=cand
    if best is None:
        for th in np.linspace(0.01,0.99,197):
            pred=p>=th
            tp=np.sum((y==1)&pred); fn=np.sum((y==1)&~pred); tn=np.sum((y==0)&~pred); fp=np.sum((y==0)&pred)
            sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
            cand=(sens+spec-1,spec,sens,-abs(th-0.5),th)
            if best is None or cand>best: best=cand
    return float(best[-1])

def crossfitted_macro_f1(y,p,folds):
    pred=np.zeros(len(y),dtype=int); thresholds=[]
    for k in sorted(np.unique(folds)):
        hold=folds==k; train=~hold
        th=tune_threshold(y[train],p[train],MIN_SENS)
        pred[hold]=(p[hold]>=th).astype(int)
        thresholds.append({'outer_fold':int(k),'threshold_from_other_outer_folds':th})
    return float(f1_score(y,pred,average='macro',zero_division=0)), thresholds

# load and validate
D={p:pd.read_csv(SRC/f'{p}.csv') for p in POLICIES}
base=D['adaptive']
assert len(base)==21346 and base.patient_id.nunique()==21346
assert int(base.true_label.sum())==4809 and int((base.true_label==0).sum())==16537
for pol,df in D.items():
    assert len(df)==len(base) and df.patient_id.is_unique
    assert df.patient_id.astype(str).tolist()==base.patient_id.astype(str).tolist()
    assert np.array_equal(df.fold.to_numpy(int),base.fold.to_numpy(int))
    assert np.array_equal(df.true_label.to_numpy(int),base.true_label.to_numpy(int))
    pp=df.probability.to_numpy(float)
    assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
    if pol!='adaptive':
        exp=pol.replace('fixed_','')
        assert set(df.selected_wavelet.astype(str).unique())=={exp}

y=base.true_label.to_numpy(int); folds=base.fold.to_numpy(int); pred={p:D[p].probability.to_numpy(float) for p in POLICIES}
# adaptive golden equivalence to original OOF
orig=np.load(ORIG_NPZ,allow_pickle=False)
assert np.array_equal(base.patient_id.astype(str).to_numpy(),orig['patient_id'].astype(str))
assert np.array_equal(y,orig['label'].astype(int)); assert np.array_equal(folds,orig['fold_id'].astype(int))
max_abs=float(np.max(np.abs(pred['adaptive']-orig['proba'].astype(float))))
adaptive_float32_exact=bool(np.array_equal(pred['adaptive'].astype(np.float32),orig['proba']))
assert adaptive_float32_exact

# pooled metrics and canonical cross-fitted macro F1
pooled=[]; thresholds_by_policy={}
for pol in POLICIES:
    p=pred[pol]
    cf_f1,ths=crossfitted_macro_f1(y,p,folds); thresholds_by_policy[pol]=ths
    fold_aucs=[roc_auc_score(y[folds==k],p[folds==k]) for k in sorted(np.unique(folds))]
    pooled.append({'policy':pol,'display_name':DISPLAY[pol],'n':len(y),'positive_n':int(y.sum()),'negative_n':int((y==0).sum()),
                   'ROC_AUC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),
                   'Brier':float(brier_score_loss(y,p)),'crossfitted_macro_F1':cf_f1,
                   'fold_auc_mean':float(np.mean(fold_aucs)),'fold_auc_sd':float(np.std(fold_aucs,ddof=1))})
POOLED=pd.DataFrame(pooled)
POOLED.to_csv(OUT/'05_POOLED_OOF_RESULTS.csv',index=False)

# fold metrics
fold_rows=[]
for pol in POLICIES:
    p=pred[pol]
    for k in sorted(np.unique(folds)):
        m=folds==k; yy=y[m]; pp=p[m]
        fold_rows.append({'policy':pol,'display_name':DISPLAY[pol],'fold':int(k),'n':int(m.sum()),'positive_n':int(yy.sum()),'negative_n':int((yy==0).sum()),
                          'ROC_AUC':float(roc_auc_score(yy,pp)),'AUPRC':float(average_precision_score(yy,pp)),
                          'Brier':float(brier_score_loss(yy,pp)),'macro_F1_at_0_5':float(f1_score(yy,pp>=0.5,average='macro',zero_division=0))})
FOLD=pd.DataFrame(fold_rows); FOLD.to_csv(OUT/'04_FOLD_LEVEL_RESULTS.csv',index=False)

# fold difference matrix AUC
fd=[]
for fixed in FIXED:
    for k in range(5):
        aa=float(FOLD[(FOLD.policy=='adaptive')&(FOLD.fold==k)].ROC_AUC.iloc[0])
        ff=float(FOLD[(FOLD.policy==fixed)&(FOLD.fold==k)].ROC_AUC.iloc[0])
        fd.append({'fixed_policy':fixed,'fold':k,'adaptive_auc':aa,'fixed_auc':ff,'delta_adaptive_minus_fixed':aa-ff,
                   'direction':'adaptive_higher' if aa>ff else ('fixed_higher' if aa<ff else 'equal')})
FD=pd.DataFrame(fd); FD.to_csv(OUT/'07_FOLD_DIFFERENCE_MATRIX.csv',index=False)

# DeLong primary comparisons
DL={f:delong_compare(y,pred['adaptive'],pred[f]) for f in FIXED}
adj=holm_adjust([DL[f]['p_raw'] for f in FIXED])
for f,a in zip(FIXED,adj): DL[f]['p_holm']=float(a)

# Paired class-stratified bootstrap, same resample weights for all 5 policies.
# Multinomial counts are an exact count-representation of sampling within each class with replacement.
pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; rng=np.random.default_rng(SEED)
boot_records=[]
boot_arrays={f:{'ROC_AUC':np.empty(BOOT_N), 'AUPRC':np.empty(BOOT_N), 'Brier_raw':np.empty(BOOT_N), 'Brier_improvement':np.empty(BOOT_N)} for f in FIXED}
# Pre-sort once per policy; weighted AUC/AP exactly match explicit replication of bootstrap indices.
def _prep_weighted_metric(score):
    order=np.argsort(score,kind='mergesort'); ss=score[order]; yy=y[order]
    starts=np.r_[0,np.where(np.diff(ss)!=0)[0]+1]
    return order,yy,starts
_PREP={p:_prep_weighted_metric(pred[p]) for p in POLICIES}
def _weighted_auc_ap(prep,w):
    order,yy,starts=prep; ww=w[order]
    posg=np.add.reduceat(ww*(yy==1),starts); negg=np.add.reduceat(ww*(yy==0),starts)
    cumneg=np.cumsum(negg)-negg; P=posg.sum(); N=negg.sum()
    auc=float((posg*(cumneg+0.5*negg)).sum()/(P*N))
    pg=posg[::-1]; ng=negg[::-1]; ctp=np.cumsum(pg); ctot=np.cumsum(pg+ng)
    precision=np.divide(ctp,ctot,out=np.zeros_like(ctp,dtype=float),where=ctot>0)
    ap=float((pg/P*precision).sum())
    return auc,ap
ppos=np.full(len(pos),1/len(pos)); pneg=np.full(len(neg),1/len(neg))
for i in range(BOOT_N):
    w=np.zeros(len(y),dtype=np.int32)
    w[pos]=rng.multinomial(len(pos),ppos); w[neg]=rng.multinomial(len(neg),pneg)
    aa,aap=_weighted_auc_ap(_PREP['adaptive'],w)
    ab=float(np.sum(w*(pred['adaptive']-y)**2)/np.sum(w))
    for f in FIXED:
        fa,fap=_weighted_auc_ap(_PREP[f],w)
        fb=float(np.sum(w*(pred[f]-y)**2)/np.sum(w))
        boot_arrays[f]['ROC_AUC'][i]=aa-fa
        boot_arrays[f]['AUPRC'][i]=aap-fap
        boot_arrays[f]['Brier_raw'][i]=ab-fb
        boot_arrays[f]['Brier_improvement'][i]=fb-ab
        boot_records.append({'bootstrap_iteration':i,'fixed_policy':f,'delta_auc_adaptive_minus_fixed':aa-fa,
                             'delta_auprc_adaptive_minus_fixed':aap-fap,'delta_brier_raw_adaptive_minus_fixed':ab-fb,
                             'brier_improvement_fixed_minus_adaptive':fb-ab})
BOOT=pd.DataFrame(boot_records); BOOT.to_csv(OUT/'bootstrap_differences.csv',index=False)

# comparisons long table
comp=[]; boot_summary=[]
pa=POOLED.set_index('policy')
for f in FIXED:
    # AUC: DeLong as primary CI/p; bootstrap CI retained separately
    arr=boot_arrays[f]['ROC_AUC']; blo,bhi=np.percentile(arr,[2.5,97.5])
    comp.append({'fixed_policy':f,'metric':'ROC_AUC','adaptive_value':pa.loc['adaptive','ROC_AUC'],'fixed_value':pa.loc[f,'ROC_AUC'],
                 'delta_adaptive_minus_fixed':pa.loc['adaptive','ROC_AUC']-pa.loc[f,'ROC_AUC'],'CI_low':DL[f]['ci_low'],'CI_high':DL[f]['ci_high'],
                 'raw_p':DL[f]['p_raw'],'adjusted_p':DL[f]['p_holm'],'statistical_method':'paired_DeLong_correlated_ROC',
                 'multiplicity_family':'four_primary_AUC_contrasts_Holm','interpretation':'ADAPTIVE_HIGHER' if DL[f]['ci_low']>0 else ('FIXED_HIGHER' if DL[f]['ci_high']<0 else 'UNRESOLVED')})
    boot_summary.append({'fixed_policy':f,'metric':'ROC_AUC','delta':float(np.mean(arr)),'CI_low':float(blo),'CI_high':float(bhi),'resamples':BOOT_N,'seed':SEED,'valid':BOOT_N})
    for metric,key in [('AUPRC','AUPRC'),('Brier','Brier_raw')]:
        arr=boot_arrays[f][key]; lo,hi=np.percentile(arr,[2.5,97.5])
        av=float(pa.loc['adaptive',metric]); fv=float(pa.loc[f,metric]); delta=av-fv
        if metric=='Brier': interp='ADAPTIVE_LOWER_BRIER' if hi<0 else ('FIXED_LOWER_BRIER' if lo>0 else 'UNRESOLVED')
        else: interp='ADAPTIVE_HIGHER' if lo>0 else ('FIXED_HIGHER' if hi<0 else 'UNRESOLVED')
        comp.append({'fixed_policy':f,'metric':metric,'adaptive_value':av,'fixed_value':fv,'delta_adaptive_minus_fixed':delta,
                     'CI_low':float(lo),'CI_high':float(hi),'raw_p':np.nan,'adjusted_p':np.nan,
                     'statistical_method':'paired_class_stratified_percentile_bootstrap','multiplicity_family':'secondary_CI_supporting_no_formal_p',
                     'interpretation':interp})
        boot_summary.append({'fixed_policy':f,'metric':metric,'delta':float(np.mean(arr)),'CI_low':float(lo),'CI_high':float(hi),'resamples':BOOT_N,'seed':SEED,'valid':BOOT_N})
COMP=pd.DataFrame(comp); COMP.to_csv(OUT/'08_ADAPTIVE_FIXED_COMPARISONS.csv',index=False)
pd.DataFrame(boot_summary).to_csv(OUT/'bootstrap_summary.csv',index=False)

# selection distribution + entropy/tie behavior
C=pd.read_csv(SRC/'adaptive_wavelet_choices.csv')
assert C.patient_id.astype(str).tolist()==base.patient_id.astype(str).tolist()
assert np.array_equal(C.fold.to_numpy(int),folds); assert np.array_equal(C.label.to_numpy(int),y)
sel_rows=[]
def add_dist(stype,sval,sub):
    n=len(sub)
    for w in ['haar','db4','sym4','coif3']:
        cnt=int((sub.selected_wavelet==w).sum())
        sel_rows.append({'stratum_type':stype,'stratum_value':str(sval),'wavelet':w,'count':cnt,'denominator':n,'percentage':100*cnt/n})
add_dist('overall','all',C)
for lab in [0,1]: add_dist('class',lab,C[C.label==lab])
for k in range(5): add_dist('fold',k,C[C.fold==k])
SEL=pd.DataFrame(sel_rows); SEL.to_csv(OUT/'06_ADAPTIVE_SELECTION_DISTRIBUTION.csv',index=False)
ents=C[['entropy_haar','entropy_db4','entropy_sym4','entropy_coif3']].to_numpy(float)
mx=ents.max(axis=1); ties=(ents==mx[:,None]).sum(axis=1)
exact_ties=int(np.sum(ties>1))
margin=C.margin.to_numpy(float)
q=np.percentile(margin,[0,25,50,75,100])
ent_summary={
 'n':len(C),'exact_max_entropy_ties':exact_ties,'near_tie_tolerance':'NOT_PRESPECIFIED_NO_POST_HOC_THRESHOLD_APPLIED',
 'margin_mean':float(np.mean(margin)),'margin_sd':float(np.std(margin,ddof=1)),'margin_min':float(q[0]),'margin_q1':float(q[1]),
 'margin_median':float(q[2]),'margin_q3':float(q[3]),'margin_max':float(q[4]),
 'winner_entropy_mean':float(C.winner_entropy.mean()),'second_best_entropy_mean':float(C.second_best_entropy.mean())}
json.dump(ent_summary,open(OUT/'entropy_margin_summary.json','w'),indent=2)
# Copy adaptive choices and OOFs
C.to_csv(OUT/'adaptive_wavelet_choices.csv',index=False)
for p in POLICIES: D[p].to_csv(OUT/f'{p}.csv',index=False)

# point-estimate verification vs server-provided quick metrics
server_pm=pd.read_csv(SRC/'stage4_policy_point_metrics.csv').set_index('policy')
for p in POLICIES:
    assert abs(server_pm.loc[p,'roc_auc']-pa.loc[p,'ROC_AUC'])<1e-14
    assert abs(server_pm.loc[p,'auprc']-pa.loc[p,'AUPRC'])<1e-14
    assert abs(server_pm.loc[p,'brier']-pa.loc[p,'Brier'])<1e-14

# forest plot data AUC
forest=[]
for f in FIXED:
    forest.append({'fixed_policy':f,'comparison':f'Adaptive - {DISPLAY[f]}','delta_auc':DL[f]['delta'],'CI_low':DL[f]['ci_low'],'CI_high':DL[f]['ci_high'],
                   'p_raw':DL[f]['p_raw'],'p_holm':DL[f]['p_holm'],'method':'paired_DeLong'})
pd.DataFrame(forest).to_csv(OUT/'forest_plot_data.csv',index=False)

# publication candidate table
pub=POOLED[['display_name','ROC_AUC','AUPRC','Brier','crossfitted_macro_F1','fold_auc_mean','fold_auc_sd']].copy()
pub.columns=['Wavelet policy','OOF ROC-AUC','AUPRC','Brier','Cross-fitted Macro-F1','Fold AUC Mean','Fold AUC SD']
pub.to_csv(OUT/'publication_candidate_table.csv',index=False)
pub2=pd.DataFrame([{'Comparator':DISPLAY[f],'Delta AUC Adaptive-Fixed':DL[f]['delta'],'95% CI low':DL[f]['ci_low'],'95% CI high':DL[f]['ci_high'],'Raw p':DL[f]['p_raw'],'Holm-adjusted p':DL[f]['p_holm']} for f in FIXED])
pub2.to_csv(OUT/'publication_candidate_paired_table.csv',index=False)

# hashes inputs
input_files=[SRC/f'{p}.csv' for p in POLICIES]+[SRC/'adaptive_wavelet_choices.csv',SRC/'run_complete.json',SRC/'stage4_wavelet_ablation.log',ORIG_NPZ,
             OUT/'stage4_wavelet_ablation_config.json']
input_hashes={str(p):sha256(p) for p in input_files if Path(p).exists()}
json.dump(input_hashes,open(OUT/'input_hashes.json','w'),indent=2)

# validation and decision fields
all_fixed_auc_higher=all(pa.loc[f,'ROC_AUC']>pa.loc['adaptive','ROC_AUC'] for f in FIXED)
any_fixed_sig=any((DL[f]['ci_high']<0 and DL[f]['p_holm']<0.05) for f in FIXED)
if all_fixed_auc_higher and any_fixed_sig:
    novelty='FIXED_WAVELET_SUPERIOR_IN_SOME_COMPARISONS'
elif all(abs(DL[f]['delta'])<0.002 for f in FIXED): novelty='FIXED_WAVELET_COMPARABLE'
else: novelty='NO_CLEAR_ADAPTIVE_BENEFIT'
best_fixed=max(FIXED,key=lambda f:pa.loc[f,'ROC_AUC'])

# write machine data object for downstream report builder
results={
 'adaptive_golden_equivalence':{'float32_exact':adaptive_float32_exact,'max_abs_difference_vs_original_float32':max_abs,'level':'E1_after_float32_roundtrip'},
 'cohort':{'n':len(y),'positive':int(y.sum()),'negative':int((y==0).sum()),'outer_folds':5},
 'pooled':POOLED.set_index('policy').to_dict(orient='index'),
 'delong':DL,'best_fixed_by_oof_auc':best_fixed,'novelty_classification':novelty,
 'selection_overall':SEL[SEL.stratum_type=='overall'].set_index('wavelet')[['count','percentage']].to_dict(orient='index'),
 'entropy':ent_summary,'thresholds_by_policy':thresholds_by_policy,
 'controls':{'same_patients':True,'same_folds':True,'same_preprocessing':True,'same_feature_schema':True,'feature_dimension':746,
             'same_glcm':True,'same_lbp':True,'same_lightgbm_within_outer_fold_across_policies':True,'mrfo_retuned_per_policy':False,
             'locked_test_used':False,'external_used':False,'a8_reselected':False},
 'bootstrap':{'resamples':BOOT_N,'seed':SEED,'valid_rate':1.0},
 'stage_4_gate':'PASS'}
json.dump(results,open(OUT/'stage4_results_core.json','w'),indent=2)

elapsed=time.perf_counter()-START
print(json.dumps({'elapsed_stats_seconds':elapsed,'novelty':novelty,'best_fixed':best_fixed,'delong':{f:{k:v for k,v in DL[f].items() if k in ['delta','ci_low','ci_high','p_raw','p_holm']} for f in FIXED}},indent=2))
