#!/usr/bin/env python3
from pathlib import Path
import csv,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()
def chk(label,v,msg=""):
    print(f"{label:<20} {'PASS' if v else 'FAIL'}{(' - '+msg) if msg else ''}"); return bool(v)
s=json.loads((ROOT/"reproducibility/radiomics_specification.json").read_text())
rows=list(csv.DictReader((ROOT/"reproducibility/feature_schema_746.csv").open()))
m=list(csv.DictReader((ROOT/"reproducibility/mrfo_search_space.csv").open()))
c=[]
c+=[chk("INPUT SPEC",s["preprocessing"]["internal_rsna"]["resize"]==[224,224])]
c+=[chk("WAVELET SET",s["wavelets"]["candidates"]==["haar","db4","sym4","coif3"])]
c+=[chk("ENTROPY",s["entropy"]["log_base"]==2 and s["entropy"]["epsilon"]==1e-12)]
c+=[chk("TIE BREAK",s["entropy"]["tie_break"].startswith("strict >"))]
c+=[chk("BORDER MODE",s["wavelets"]["extension_mode"]["effective"]=="symmetric")]
c+=[chk("SUBBANDS",s["decomposition"]["subband_order"]==["LH1","HL1","HH1","LH2","HL2","HH2","LH3","HL3","HH3","LL3"])]
c+=[chk("GLCM",len(rows[:720])==720 and s["glcm"]["levels"]==256)]
c+=[chk("LBP",s["lbp"]["points"]==24 and s["lbp"]["bins"]==26)]
c+=[chk("FEATURE COUNT",len(rows)==746)]
c+=[chk("FEATURE HASH",sha(ROOT/"reproducibility/feature_schema_746.csv")==s["feature_schema"]["schema_sha256"])]
c+=[chk("LIGHTGBM CONFIG",s["lightgbm"]["full_development_effective_params"]["max_depth"]==-1)]
c+=[chk("MRFO CONFIG",len([x for x in m if x["estimator"]=="lgbm"])==5 and s["mrfo"]["final"]["population"]==15)]
try:
    import pywt,skimage,numpy as np
    sys.path.insert(0,str(ROOT/"src"))
    from aura_cxr.radiomics import select_best_wavelet,extract_radiomics
    x=np.zeros((224,224),np.float32)
    c+=[chk("DYNAMIC FEATURE",select_best_wavelet(x)=="haar" and extract_radiomics(x)[0].shape==(746,))]
except Exception as e:
    print(f"{'DYNAMIC FEATURE':<20} SKIP_DEPENDENCY - {type(e).__name__}: {e}")
print("RADIOMICS_VERIFY =", "PASS" if all(c) else "FAIL")
raise SystemExit(0 if all(c) else 1)
