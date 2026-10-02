"""Canonical adaptive radiomics primitives extracted from train_aura.py.
Scientific behavior intentionally mirrors the verified implementation.
"""
from typing import List, Tuple
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from .data import to_uint8
WAVELET_CANDIDATES=['haar','db4','sym4','coif3']

def shannon_entropy(coeffs:np.ndarray)->float:
    c=np.abs(coeffs.ravel()); c=c[c>0]
    if len(c)==0:return 0.0
    p=c/c.sum(); return float(-np.sum(p*np.log2(p+1e-12)))

def select_best_wavelet(gray_image:np.ndarray,candidates:List[str]=WAVELET_CANDIDATES)->str:
    best_wavelet=candidates[0]; best_entropy=-np.inf
    for w in candidates:
        try:
            _,(lh,hl,hh)=pywt.dwt2(gray_image,w); ent=shannon_entropy(np.concatenate([lh.ravel(),hl.ravel(),hh.ravel()]))
            if ent>best_entropy: best_entropy=ent; best_wavelet=w
        except Exception: continue
    return best_wavelet

def wavelet_subbands(gray_image:np.ndarray,wavelet:str,wavelet_levels:int=3)->List[np.ndarray]:
    details=[]; current=gray_image
    for _ in range(wavelet_levels):
        ll,(lh,hl,hh)=pywt.dwt2(current,wavelet); details.extend([lh,hl,hh]); current=ll
    return details+[current]

def extract_radiomics(gray_image:np.ndarray,distances=[1,2,3],angles_deg=[0,45,90,135],wavelet_levels=3,lbp_radius=3,lbp_n_points=24,use_wavelet=True,adaptive=True,fixed_wavelet='db4',use_glcm=True,use_lbp=True)->Tuple[np.ndarray,str]:
    if not use_glcm and not use_lbp: raise ValueError('At least one of use_glcm/use_lbp must be True')
    angles_rad=[np.deg2rad(a) for a in angles_deg]; features=[]
    if use_wavelet:
        chosen=select_best_wavelet(gray_image) if adaptive else fixed_wavelet; coeffs=wavelet_subbands(gray_image,chosen,wavelet_levels)
    else: chosen='none'; coeffs=[gray_image]
    if use_glcm:
        for coeff in coeffs:
            glcm=graycomatrix(to_uint8(coeff),distances=distances,angles=angles_rad,levels=256,symmetric=True,normed=True)
            for prop in ['contrast','energy','homogeneity','dissimilarity','correlation','ASM']:
                features.extend(graycoprops(glcm,prop).flatten().tolist())
    if use_lbp:
        lbp=local_binary_pattern(to_uint8(gray_image),lbp_n_points,lbp_radius,method='uniform'); n_bins=lbp_n_points+2
        hist,_=np.histogram(lbp.ravel(),bins=n_bins,range=(0,n_bins),density=True); features.extend(hist.tolist())
    return np.asarray(features,dtype=np.float32),chosen

MRFO_BOUNDS={'knn':{'k_odd':list(range(3,102,2)),'weights':['uniform','distance'],'metrics':['euclidean','manhattan','minkowski','chebyshev']},'svm':{'lb':[-2.,-4.],'ub':[3.,0.]},'histgb_lgbm':{'lb':[-2.3,15.,2.,-4.,0.001],'ub':[-0.4,255.,12.,1.,0.05]}}
