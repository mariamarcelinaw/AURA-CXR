"""Evaluation utilities and audit-mode reproduction from released predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, accuracy_score, f1_score, brier_score_loss
from .fusion import a4_probability,a8_probability,threshold,A4_THRESHOLD,A8_THRESHOLD,weighted_soft_vote,SOFT_THRESHOLD

def binary_metrics(y,p,tau):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=threshold(p,tau); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'AUC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'accuracy':float(accuracy_score(y,pred)),'balanced_acc':float(balanced_accuracy_score(y,pred)),'MCC':float(matthews_corrcoef(y,pred)),'F1':float(f1_score(y,pred,zero_division=0)),'Brier':float(brier_score_loss(y,p)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def reproduce_locked(locked_csv):
    d=pd.read_csv(locked_csv); y=d.true_label.to_numpy(int)
    a4=d.a4_probability.to_numpy(float); a8=d.a8_probability.to_numpy(float)
    return {'A4':binary_metrics(y,a4,A4_THRESHOLD),'A8':binary_metrics(y,a8,A8_THRESHOLD)}

def reproduce_external(kermany_csv):
    d=pd.read_csv(kermany_csv); y=d.true_label.to_numpy(int)
    return {'XRV':binary_metrics(y,d.xrv_probability,0.495),'EVA-X-S':binary_metrics(y,d.eva_x_probability,0.47),'LightGBM':binary_metrics(y,d.radiomics_probability,0.225),'A8':binary_metrics(y,d.a8_probability,A8_THRESHOLD),'Soft':binary_metrics(y,d.soft_probability,SOFT_THRESHOLD)}
