"""Data, DICOM preprocessing, split loading and split-integrity guards.
Source lineage: train_aura.py, train_xrv_backbone.py, train_eva_x_backbone.py.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    image=image.astype(np.float32); lo,hi=float(image.min()),float(image.max())
    if math.isclose(lo,hi): return np.zeros_like(image,dtype=np.float32)
    return (image-lo)/(hi-lo)

def to_uint8(image: np.ndarray)->np.ndarray:
    return np.clip(normalize_minmax(image)*255,0,255).astype(np.uint8)

def load_dicom_grayscale(path, image_size:int=224)->np.ndarray:
    import pydicom
    from PIL import Image
    ds=pydicom.dcmread(str(path)); img=ds.pixel_array.astype(np.float32)
    img=img*float(getattr(ds,'RescaleSlope',1.0))+float(getattr(ds,'RescaleIntercept',0.0))
    if getattr(ds,'PhotometricInterpretation','MONOCHROME2')=='MONOCHROME1': img=img.max()-img
    img=normalize_minmax(img)
    pil=Image.fromarray((img*255).astype(np.uint8),mode='L').resize((image_size,image_size),resample=Image.Resampling.BILINEAR)
    return np.asarray(pil,dtype=np.float32)/255.0

def load_dicom_rgb(path,image_size:int=224)->np.ndarray:
    g=load_dicom_grayscale(path,image_size); return np.repeat(g[...,None],3,axis=-1).astype(np.float32)

def load_split(path)->pd.DataFrame: return pd.read_csv(path)

def assert_split_integrity(train,val,test,folds=None,expected=(18678,2668,5338)):
    for name,df,n in [('train',train,expected[0]),('val',val,expected[1]),('test',test,expected[2])]:
        if len(df)!=n: raise ValueError(f'{name} count {len(df)} != {n}')
        if df.patientId.duplicated().any(): raise ValueError(f'duplicate patientId in {name}')
    a,b,c=set(train.patientId),set(val.patientId),set(test.patientId)
    if a&b or a&c or b&c: raise ValueError('patient leakage across train/val/test')
    if folds is not None:
        dev=a|b
        if len(folds)!=len(dev) or set(folds.patientId)!=dev: raise ValueError('OOF fold assignment does not cover development exactly')
        if folds.patientId.duplicated().any(): raise ValueError('duplicate OOF patient')
        if set(folds.oof_fold.unique())!={0,1,2,3,4}: raise ValueError('OOF folds must be 0..4')
    return True

def require_optimization_safe(split_name:str):
    if split_name in {'locked_test','external','kermany'}: raise ValueError(f'Optimization/model selection is forbidden on {split_name}')
