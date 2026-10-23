#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
MODEL_ID='convnextv2_tiny.fcmae_ft_in1k'; BASELINE_ID='C1_CONVNEXTV2_TINY'; ATTEMPT_ID='STAGE8_CORRECTED_FP32_V2'; SEED=42
MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)[:,None,None]; STD=np.array([0.229,0.224,0.225],dtype=np.float32)[:,None,None]
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy(); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
def dicom_path(root,pid):
    a=Path(root)/str(pid); b=Path(root)/(str(pid)+'.dcm')
    if a.exists(): return a
    if b.exists(): return b
    raise FileNotFoundError(f'DICOM missing for {pid}')
def load_base_image(path,size=224):
    import pydicom
    from PIL import Image
    d=pydicom.dcmread(str(path)); a=d.pixel_array.astype(np.float32); a=a*float(getattr(d,'RescaleSlope',1.0))+float(getattr(d,'RescaleIntercept',0.0))
    if getattr(d,'PhotometricInterpretation','MONOCHROME2')=='MONOCHROME1': a=a.max()-a
    lo=float(a.min()); hi=float(a.max()); a=(a-lo)/(hi-lo) if hi>lo else np.zeros_like(a,dtype=np.float32)
    im=Image.fromarray(np.clip(a*255,0,255).astype(np.uint8),mode='L').resize((size,size),Image.Resampling.BILINEAR)
    return np.asarray(im,dtype=np.float32)/255.0
def augment(a):
    if np.random.random()<0.5: a=np.ascontiguousarray(a[:,::-1])
    a=np.clip(a+np.random.uniform(-0.15,0.15),0,1); c=np.random.uniform(0.85,1.15); m=float(a.mean()); return np.clip((a-m)*c+m,0,1)
class RSNADataset:
    def __init__(self,df,dicom_dir,train=False): self.df=df.reset_index(drop=True); self.dicom_dir=Path(dicom_dir); self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        import torch
        r=self.df.iloc[i]; a=load_base_image(dicom_path(self.dicom_dir,r.patientId),224); a=augment(a) if self.train else a; x=np.repeat(a[None,:,:],3,axis=0).astype(np.float32); x=(x-MEAN)/STD
        return torch.from_numpy(x),torch.tensor(int(r.label),dtype=torch.long),str(r.patientId)
def worker_init(worker_id):
    seed=(torch_seed := (SEED+worker_id)) % (2**32-1); np.random.seed(seed); random.seed(seed)
def build_model(pretrained=True,drop_path=0.2):
    import timm
    return timm.create_model(MODEL_ID,pretrained=pretrained,num_classes=2,drop_path_rate=drop_path)
def class_weights(df):
    import torch
    n=len(df); c=df.label.value_counts().to_dict(); return torch.tensor([n/(2*c[0]),n/(2*c[1])],dtype=torch.float32)
def focal_loss(logits,y,w,gamma=2.0):
    import torch.nn.functional as F
    logp=F.log_softmax(logits.float(),dim=1); p=logp.exp(); pt=p.gather(1,y[:,None]).squeeze(1); ce=F.nll_loss(logp,y,weight=w,reduction='none'); return (((1-pt)**gamma)*ce).mean()
def loader(df,dicom_dir,train,batch,workers,seed):
    import torch
    g=torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(RSNADataset(df,dicom_dir,train),batch_size=batch,shuffle=train,num_workers=workers,pin_memory=True,drop_last=train,worker_init_fn=worker_init if workers else None,generator=g,persistent_workers=bool(workers))
def predict(model,ld,device):
    import torch
    model.eval(); ps=[]; ys=[]; ids=[]
    with torch.no_grad():
        for x,y,pid in ld:
            p=torch.softmax(model(x.to(device,non_blocking=True)).float(),dim=1)[:,1]; ps.extend(p.cpu().numpy()); ys.extend(y.numpy()); ids.extend(pid)
    return np.asarray(ys,int),np.asarray(ps,float),ids
def auc(y,p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,p))
def train_epoch(model,ld,opt,device,w,clip):
    import torch
    # Corrective Stage-8 attempt: full FP32 training.
    # No autocast / GradScaler is used. This is the only training-semantic
    # change from attempt 1 and was locked after a development-only diagnostic
    # reproduced Fold-4 AMP instability with non-finite gradients.
    model.train(); total=n=0
    for x,y,_ in ld:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
        logits=model(x)
        loss=focal_loss(logits,y,w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        opt.step()
        total+=float(loss.detach())*len(y); n+=len(y)
    return total/max(n,1)
def split_inner(outer,fold):
    from sklearn.model_selection import train_test_split
    idx=np.arange(len(outer)); tr,va=train_test_split(idx,test_size=.10,random_state=SEED+int(fold),stratify=outer.label.to_numpy()); return outer.iloc[tr].copy(),outer.iloc[va].copy()
def atomic_save(obj,path):
    import torch
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def train_fold(fold,dev,dicom_dir,out,cfg,device):
    import torch
    fold=int(fold); fdir=out/'checkpoints'/f'fold_{fold}'; fdir.mkdir(parents=True,exist_ok=True); pred_path=out/'oof'/f'fold_{fold}_predictions.csv'; pred_path.parent.mkdir(parents=True,exist_ok=True)
    hold=dev[dev.oof_fold==fold].copy(); outer=dev[dev.oof_fold!=fold].copy(); itr,iva=split_inner(outer,fold)
    if pred_path.exists():
        d=pd.read_csv(pred_path)
        if len(d)==len(hold) and set(d.patient_id)==set(hold.patientId): print(f'[FOLD {fold}] complete; skip',flush=True); return json.loads((fdir/'summary.json').read_text())
    set_seed(SEED+fold); model=build_model(True,cfg['drop_path_rate']).to(device); ph=state_hash(model); opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_epochs'],eta_min=cfg['scheduler']['eta_min']); w=class_weights(itr).to(device); trl=loader(itr,dicom_dir,True,cfg['batch_size'],cfg['num_workers'],SEED+fold); val=loader(iva,dicom_dir,False,cfg['batch_size'],cfg['num_workers'],SEED+fold)
    latest=fdir/'latest.pt'; best=fdir/'best.pt'; start=1; best_auc=-1.; best_epoch=0; stale=0
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); start=ck['epoch']+1; best_auc=ck['best_auc']; best_epoch=ck['best_epoch']; stale=ck['stale']; print(f'[FOLD {fold}] FP32 resume {start}',flush=True)
    t0=time.time(); stop=0; logp=out/'logs'/f'fold_{fold}.csv'; logp.parent.mkdir(parents=True,exist_ok=True)
    if not logp.exists(): logp.write_text('epoch,train_loss,val_auc,lr,best_auc,stale\n')
    for ep in range(start,cfg['max_epochs']+1):
        loss=train_epoch(model,trl,opt,device,w,cfg['gradient_clip_norm']); yy,pp,_=predict(model,val,device); va=auc(yy,pp); imp=va>best_auc+cfg['early_stopping']['min_delta']
        if imp: best_auc=va; best_epoch=ep; stale=0; atomic_save({'model':model.state_dict(),'epoch':ep,'val_auc':va,'pretrained_state_hash':ph},best)
        else: stale+=1
        with logp.open('a') as f: f.write(f'{ep},{loss:.8f},{va:.8f},{opt.param_groups[0]["lr"]:.10g},{best_auc:.8f},{stale}\n')
        atomic_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep,'best_auc':best_auc,'best_epoch':best_epoch,'stale':stale},latest); sched.step(); stop=ep; print(f'[FOLD {fold}] epoch={ep} loss={loss:.5f} val_auc={va:.5f} best={best_auc:.5f}@{best_epoch} stale={stale}',flush=True)
        if stale>=cfg['early_stopping']['patience']: break
    ck=torch.load(best,map_location=device,weights_only=False); model.load_state_dict(ck['model']); yy,pp,ids=predict(model,loader(hold,dicom_dir,False,cfg['batch_size'],cfg['num_workers'],SEED+fold),device); pd.DataFrame({'patient_id':ids,'fold':fold,'true_label':yy,'probability':pp,'model_id':BASELINE_ID}).to_csv(pred_path,index=False)
    s={'fold':fold,'n_outer_train':len(outer),'n_inner_train':len(itr),'n_inner_val':len(iva),'n_holdout':len(hold),'best_epoch':best_epoch,'stopping_epoch':stop,'best_inner_val_auc':best_auc,'holdout_auc':auc(yy,pp),'runtime_seconds':time.time()-t0,'pretrained_state_hash':ph,'best_checkpoint_sha256':sha256(best),'precision':'FP32','attempt_id':ATTEMPT_ID}; (fdir/'summary.json').write_text(json.dumps(s,indent=2)); return s
def tune_threshold(y,p,min_sensitivity=.85):
    best=None
    for th in np.linspace(.01,.99,197):
        pred=p>=th; tp=np.sum((y==1)&pred); fn=np.sum((y==1)&~pred); tn=np.sum((y==0)&~pred); fp=np.sum((y==0)&pred); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
        if sens<min_sensitivity: continue
        cand=(sens+spec-1,spec,sens,-abs(th-.5),th); best=cand if best is None or cand>best else best
    if best is None:
        for th in np.linspace(.01,.99,197):
            pred=p>=th; tp=np.sum((y==1)&pred); fn=np.sum((y==1)&~pred); tn=np.sum((y==0)&~pred); fp=np.sum((y==0)&pred); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); cand=(sens+spec-1,spec,sens,-abs(th-.5),th); best=cand if best is None or cand>best else best
    return float(best[-1])
def metrics(y,p,t):
    from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,accuracy_score,balanced_accuracy_score,matthews_corrcoef,f1_score,brier_score_loss
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); return {'ROC_AUC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'Sensitivity':float(tp/(tp+fn)),'Specificity':float(tn/(tn+fp)),'Accuracy':float(accuracy_score(y,pred)),'F1':float(f1_score(y,pred,zero_division=0)),'Balanced_Accuracy':float(balanced_accuracy_score(y,pred)),'MCC':float(matthews_corrcoef(y,pred)),'Brier':float(brier_score_loss(y,p)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=.5*(i+j-1)+1; i=j
    out=np.empty(N,float); out[J]=T; return out
def fast_delong(preds,m):
    n=preds.shape[1]-m; pos=preds[:,:m]; neg=preds[:,m:]; k=preds.shape[0]; tx=np.vstack([midrank(pos[r]) for r in range(k)]); ty=np.vstack([midrank(neg[r]) for r in range(k)]); tz=np.vstack([midrank(preds[r]) for r in range(k)]); aucs=tz[:,:m].sum(axis=1)/m/n-(m+1.)/(2*n); v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m; return aucs,np.cov(v01)/m+np.cov(v10)/n
def delong(y,a8,base):
    from scipy import stats
    order=np.argsort(-y,kind='stable'); m=int(y.sum()); aucs,cov=fast_delong(np.vstack([a8[order],base[order]]),m); l=np.array([1.,-1.]); d=float(l@aucs); var=float(l@cov@l.T); se=math.sqrt(max(var,0)); z=d/se if se else 0.; p=float(2*stats.norm.sf(abs(z))) if se else 1.; zc=float(stats.norm.ppf(.975)); return {'A8_auc':float(aucs[0]),'baseline_auc':float(aucs[1]),'delta_A8_minus_baseline':d,'se':se,'ci_low':d-zc*se,'ci_high':d+zc*se,'p_value':p}
def bootstrap(y,a8,ta8,base,tbase,n=2000,seed=42):
    rng=np.random.default_rng(seed); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; names=['ROC_AUC','AUPRC','Sensitivity','Specificity','Accuracy','F1','Balanced_Accuracy','MCC','Brier']; arr=[]
    for _ in range(n):
        idx=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]); yy=y[idx]; m8=metrics(yy,a8[idx],ta8); mb=metrics(yy,base[idx],tbase); arr.append([m8[k]-mb[k] for k in names])
    return pd.DataFrame(arr,columns=names)
def assemble_oof(dev,out):
    parts=[pd.read_csv(out/'oof'/f'fold_{int(f)}_predictions.csv') for f in sorted(dev.oof_fold.unique())]; d=pd.concat(parts,ignore_index=True); assert len(d)==len(dev) and d.patient_id.nunique()==len(dev); m=dev[['patientId','label','oof_fold']].merge(d,left_on='patientId',right_on='patient_id',how='left'); assert m.probability.notna().all() and (m.label==m.true_label).all() and (m.oof_fold==m.fold).all(); d.to_csv(out/'08_BASELINE_OOF_PREDICTIONS.csv',index=False); return d
def final_refit(dev,dicom_dir,out,cfg,device,epochs):
    import torch
    p=out/'checkpoints'/'final_convnextv2_tiny.pt'; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): return p
    set_seed(SEED); model=build_model(True,cfg['drop_path_rate']).to(device); ph=state_hash(model); opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs,1),eta_min=cfg['scheduler']['eta_min']); w=class_weights(dev).to(device); ld=loader(dev,dicom_dir,True,cfg['batch_size'],cfg['num_workers'],SEED); t0=time.time()
    for ep in range(1,epochs+1): loss=train_epoch(model,ld,opt,device,w,cfg['gradient_clip_norm']); sched.step(); print(f'[REFIT] {ep}/{epochs} loss={loss:.5f}',flush=True)
    atomic_save({'model':model.state_dict(),'epochs':epochs,'pretrained_state_hash':ph,'model_id':MODEL_ID,'precision':'FP32','attempt_id':ATTEMPT_ID},p); (out/'final_refit_runtime.json').write_text(json.dumps({'epochs':epochs,'runtime_seconds':time.time()-t0,'checkpoint_sha256':sha256(p),'precision':'FP32','attempt_id':ATTEMPT_ID},indent=2)); return p
def locked_eval(test,dicom_dir,out,cfg,device,ckpt,threshold,locked_a8):
    import torch
    if not (out/'05_BASELINE_DEPLOYMENT_LOCK.json').exists(): raise RuntimeError('deployment lock missing')
    model=build_model(False,cfg['drop_path_rate']).to(device); st=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(st['model'],strict=True); y,p,ids=predict(model,loader(test,dicom_dir,False,cfg['batch_size'],cfg['num_workers'],SEED),device); pred=(p>=threshold).astype(int); pd.DataFrame({'patient_id':ids,'true_label':y,'probability':p,'predicted_class':pred,'threshold':threshold,'model_id':BASELINE_ID,'split':'locked_test'}).to_csv(out/'06_BASELINE_LOCKED_PREDICTIONS.csv',index=False)
    a=pd.read_csv(locked_a8); z=pd.DataFrame({'patient_id':ids,'y':y,'base':p}).merge(a[['case_id','true_label','a8_probability']],left_on='patient_id',right_on='case_id',how='inner'); assert len(z)==len(test) and (z.y==z.true_label).all(); yy=z.y.to_numpy(int); base=z.base.to_numpy(float); a8=z.a8_probability.to_numpy(float); mb=metrics(yy,base,threshold); m8=metrics(yy,a8,.47); dl=delong(yy,a8,base); boot=bootstrap(yy,a8,.47,base,threshold); rows=[]
    for k in ['ROC_AUC','AUPRC','Sensitivity','Specificity','Accuracy','F1','Balanced_Accuracy','MCC','Brier']:
        lo,hi=np.percentile(boot[k],[2.5,97.5]); rows.append({'metric':k,'A8_estimate':m8[k],'baseline_estimate':mb[k],'delta_A8_minus_baseline':m8[k]-mb[k],'CI_low':dl['ci_low'] if k=='ROC_AUC' else float(lo),'CI_high':dl['ci_high'] if k=='ROC_AUC' else float(hi),'p_value':dl['p_value'] if k=='ROC_AUC' else np.nan,'adjusted_p_if_applicable':np.nan,'statistical_method':'paired_DeLong' if k=='ROC_AUC' else 'paired_class_stratified_bootstrap','analysis_role':'PRIMARY' if k=='ROC_AUC' else 'SECONDARY'})
    pd.DataFrame(rows).to_csv(out/'12_A8_BASELINE_PAIRED_COMPARISON.csv',index=False); boot.to_csv(out/'bootstrap_differences.csv',index_label='bootstrap_index'); return mb,m8,dl
def run(args):
    import torch,timm
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); cfg=json.loads(Path(args.config).read_text()); dev=pd.read_csv(args.development_folds); assert len(dev)==21346 and dev.patientId.nunique()==21346 and dev.oof_fold.nunique()==5; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device=',device,'torch=',torch.__version__,'timm=',timm.__version__,flush=True)
    set_seed(SEED); probe=build_model(True,cfg['drop_path_rate']); ph=state_hash(probe); pc=sum(p.numel() for p in probe.parameters()); del probe; sel=json.loads(Path(args.selection_lock).read_text()); sel['pretrained_checkpoint_state_hash']=ph; sel['instantiated_parameter_count']=pc; sel['server_preflight_timestamp_unix']=time.time(); (out/'02_PRIMARY_BASELINE_SELECTION_LOCK.json').write_text(json.dumps(sel,indent=2)); shutil.copy2(args.config,out/'04_BASELINE_TRAINING_CONFIG.json'); print('STAGE8_CORRECTED_FP32_PREFLIGHT = PASS',flush=True)
    if args.mode=='preflight': return
    summaries=[train_fold(int(f),dev,args.dicom_dir,out,cfg,device) for f in sorted(dev.oof_fold.unique())]; oof=assemble_oof(dev,out); y=oof.true_label.to_numpy(int); p=oof.probability.to_numpy(float); th=tune_threshold(y,p,cfg['threshold']['min_sensitivity']); mo=metrics(y,p,th); fr=pd.DataFrame(summaries); best_epochs=fr.best_epoch.to_numpy(int); refit_epochs=max(1,int(round(float(np.median(best_epochs))))); fr.to_csv(out/'09_BASELINE_FOLD_RESULTS_RAW.csv',index=False); pd.DataFrame([{'scope':'pooled_oof',**mo,'threshold':th,'fold_auc_mean':float(fr.holdout_auc.mean()),'fold_auc_sd':float(fr.holdout_auc.std(ddof=1)),'refit_epochs':refit_epochs}]).to_csv(out/'10_BASELINE_DEVELOPMENT_RESULTS.csv',index=False)
    ck=final_refit(dev,args.dicom_dir,out,cfg,device,refit_epochs); dep={'baseline_id':BASELINE_ID,'oof_sha256':sha256(out/'08_BASELINE_OOF_PREDICTIONS.csv'),'threshold':th,'threshold_source':'development_oof','final_refit_epochs':refit_epochs,'final_training_population':len(dev),'final_checkpoint':str(ck),'final_checkpoint_sha256':sha256(ck),'training_config_sha256':sha256(args.config),'locked_test_patient_list_sha256':sha256(args.test_split),'created_before_locked_test':True,'locked_test_tuning':False,'attempt_id':ATTEMPT_ID,'precision':'FP32','corrective_rerun':True,'attempt1_locked_test_already_seen':True,'correction_basis':'development-only Fold-4 AMP-vs-FP32 diagnostic','correction_locked_before_attempt2_locked_test':True}; (out/'05_BASELINE_DEPLOYMENT_LOCK.json').write_text(json.dumps(dep,indent=2))
    test=pd.read_csv(args.test_split); assert len(test)==5338; t0=time.time(); mb,m8,dl=locked_eval(test,args.dicom_dir,out,cfg,device,ck,th,args.locked_a8); lock_seconds=time.time()-t0; pd.DataFrame([{'metric':k,'point_estimate':v,'threshold':th} for k,v in mb.items() if k not in {'TN','FP','FN','TP'}]).to_csv(out/'11_BASELINE_LOCKED_RESULTS.csv',index=False)
    folds=[]
    for f in sorted(dev.oof_fold.unique()):
        d=pd.read_csv(out/'oof'/f'fold_{int(f)}_predictions.csv'); mm=metrics(d.true_label.to_numpy(int),d.probability.to_numpy(float),th); s=json.loads((out/'checkpoints'/f'fold_{int(f)}'/'summary.json').read_text()); folds.append({'fold':int(f),'n':len(d),'positive':int(d.true_label.sum()),'negative':int((d.true_label==0).sum()),**mm,'best_epoch':s['best_epoch']})
    pd.DataFrame(folds).to_csv(out/'09_BASELINE_FOLD_RESULTS.csv',index=False); refit=json.loads((out/'final_refit_runtime.json').read_text()); comp={'model':BASELINE_ID,'parameters':pc,'trainable_parameters':pc,'input_size':'224x224','FLOPs_if_verified':None,'checkpoint_size_bytes':Path(ck).stat().st_size,'peak_VRAM_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,'inference_locked_seconds_total':lock_seconds,'training_GPU_hours_if_measured':sum(float(s['runtime_seconds']) for s in summaries)/3600+refit['runtime_seconds']/3600}; pd.DataFrame([comp]).to_csv(out/'14_MODEL_COMPLEXITY_COMPARISON.csv',index=False)
    sm={'project':'AURA-CXR','stage':8,'analysis_origin':'contemporary_baseline_analysis','baseline':{'name':'ConvNeXt V2 Tiny','architecture_family':'ConvNeXt V2','publication':'CVPR 2023','year':2023,'implementation':MODEL_ID,'pretrained_checkpoint':'timm/convnextv2_tiny.fcmae_ft_in1k','pretrained_state_hash':ph,'checkpoint_hash':sha256(ck),'parameter_count':pc,'input_size':[224,224],'pretraining_overlap_status':'NO_KNOWN_RSNA_OVERLAP'},'development':{'n':len(dev),'oof_auc':mo['ROC_AUC'],'oof_auprc':mo['AUPRC'],'threshold':th,'fold_auc_mean':float(fr.holdout_auc.mean()),'fold_auc_sd':float(fr.holdout_auc.std(ddof=1))},'locked_test':{'n':len(test),'positive':int(test.label.sum()),'negative':int((test.label==0).sum()),'auc':mb['ROC_AUC'],'auprc':mb['AUPRC'],'sensitivity':mb['Sensitivity'],'specificity':mb['Specificity'],'accuracy':mb['Accuracy'],'f1':mb['F1'],'balanced_accuracy':mb['Balanced_Accuracy'],'mcc':mb['MCC']},'paired_A8_vs_baseline':{'difference_orientation':'A8_minus_baseline','auc':{'delta':dl['delta_A8_minus_baseline'],'ci_low':dl['ci_low'],'ci_high':dl['ci_high'],'p_value':dl['p_value'],'method':'paired_DeLong'}},'complexity':comp,'protocol_integrity':{'a8_retrained':False,'a8_reselected':False,'baseline_selected_on_test':False,'threshold_tuned_on_test':False,'external_tuning':False},'attempt_id':ATTEMPT_ID,'precision':'FP32','corrective_rerun':True,'attempt1_locked_test_already_seen':True,'correction_basis':'development-only Fold-4 diagnostic','stage_8_dynamic_run':'COMPLETE'}; (out/'stage8_dynamic_summary.json').write_text(json.dumps(sm,indent=2)); print('STAGE8_CORRECTED_FP32_DYNAMIC_RUN = COMPLETE',flush=True); print(json.dumps(sm['locked_test'],indent=2),flush=True); print(json.dumps(sm['paired_A8_vs_baseline']['auc'],indent=2),flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['preflight','full'],default='full'); ap.add_argument('--dicom-dir',required=True); ap.add_argument('--development-folds',required=True); ap.add_argument('--test-split',required=True); ap.add_argument('--locked-a8',required=True); ap.add_argument('--config',required=True); ap.add_argument('--selection-lock',required=True); ap.add_argument('--out',required=True); run(ap.parse_args())
if __name__=='__main__': main()
