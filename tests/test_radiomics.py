import math, json, csv
from pathlib import Path
import numpy as np
import pytest
def root(): return Path(__file__).resolve().parents[1]
def entropy(x):
    c=np.abs(np.asarray(x,dtype=float).ravel()); c=c[c>0]
    if len(c)==0: return 0.0
    p=c/c.sum(); return float(-np.sum(p*np.log2(p+1e-12)))
def test_candidate_wavelets():
    s=json.loads((root()/"reproducibility/radiomics_specification.json").read_text())
    assert s["wavelets"]["candidates"]==["haar","db4","sym4","coif3"]
def test_entropy_normalization():
    expected=-2*(0.5*np.log2(0.5+1e-12))
    assert math.isclose(entropy([1,-1,0]),expected,rel_tol=1e-12,abs_tol=1e-12)
def test_entropy_zero_case(): assert entropy([0,0,0])==0.0
def test_feature_schema():
    rows=list(csv.DictReader((root()/"reproducibility/feature_schema_746.csv").open()))
    assert len(rows)==746
    assert rows[0]["feature_name"]=="GLCM_LH1_contrast_d1_a0"
    assert rows[719]["feature_name"]=="GLCM_LL3_ASM_d3_a135"
    assert rows[720]["feature_name"]=="LBP_uniform_P24_R3_bin00"
    assert rows[-1]["feature_name"]=="LBP_uniform_P24_R3_bin25"
def test_glcm_order():
    rows=list(csv.DictReader((root()/"reproducibility/feature_schema_746.csv").open()))[:720]
    assert rows[1]["distance"]=="1" and rows[1]["angle_deg"]=="45"
    assert rows[4]["distance"]=="2" and rows[4]["angle_deg"]=="0"
def test_dynamic_feature_if_dependencies_available():
    pytest.importorskip("pywt"); pytest.importorskip("skimage")
    import sys; sys.path.insert(0,str(root()/"src"))
    from aura_cxr.radiomics import select_best_wavelet, extract_radiomics
    x=np.zeros((224,224),dtype=np.float32)
    assert select_best_wavelet(x)=="haar"
    f,_=extract_radiomics(x)
    assert f.shape==(746,) and f.dtype==np.float32 and np.isfinite(f).all()
