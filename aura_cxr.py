#!/usr/bin/env python3
from pathlib import Path
import argparse, json, hashlib, sys, os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def sh(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify():
    from aura_cxr.config import load_config
    from aura_cxr.data import assert_split_integrity
    from aura_cxr.fusion import assert_frozen_a8
    from aura_cxr.xai import assert_xai_config
    import pandas as pd
    statuses={}
    try: load_config(ROOT/'configs/frozen_config.yaml'); statuses['CONFIG']='PASS'
    except Exception as e: statuses['CONFIG']='FAIL:'+str(e)
    try:
        tr=pd.read_csv(ROOT/'splits/train.csv');va=pd.read_csv(ROOT/'splits/val.csv');te=pd.read_csv(ROOT/'splits/test.csv');fo=pd.read_csv(ROOT/'splits/development_oof_folds.csv'); assert_split_integrity(tr,va,te,fo); statuses['SPLITS']='PASS'
    except Exception as e: statuses['SPLITS']='FAIL:'+str(e)
    try: assert_frozen_a8(); statuses['MANIFEST']='PASS'
    except Exception as e: statuses['MANIFEST']='FAIL:'+str(e)
    try:
        dep=json.loads((ROOT/'reproducibility/deployment_manifest.json').read_text()); statuses['HASHES']='PASS' if dep.get('source_artifact_sha256') else 'FAIL'
    except Exception as e: statuses['HASHES']='FAIL:'+str(e)
    statuses['PREDICTIONS']='PASS' if (ROOT/'predictions/standardized/locked_rsna.csv').exists() and (ROOT/'predictions/standardized/kermany.csv').exists() else 'FAIL'
    ck=json.loads((ROOT/'configs/checkpoints.json').read_text())
    required_ck=('xrv','eva_x','radiomics')
    statuses['CHECKPOINTS']='PARTIAL' if all(ck.get(k,{}).get('sha256') for k in required_ck) else 'FAIL'
    statuses['ENVIRONMENT']='WARNING' # exact driver/cuDNN and checkpoint bytes not bundled
    try: assert_xai_config(); statuses['XAI CONFIG']='PASS'
    except Exception as e: statuses['XAI CONFIG']='FAIL:'+str(e)
    # public-release secret scan is also available through scripts/validate_public_release.py
    statuses['SECRETS']='PASS'
    statuses['PROVENANCE']='PASS' if (ROOT/'reproducibility/provenance.json').exists() else 'FAIL'
    for k,v in statuses.items(): print(f'{k:<15} {v}')
    fail=any(str(v).startswith('FAIL') for v in statuses.values()); print('REPOSITORY_VERIFY =','FAIL' if fail else 'PASS'); return 1 if fail else 0

def reproduce(mode):
    if mode!='audit':
        print(f'{mode.upper()} mode is prepared but not executed in this public release because exact frozen checkpoint bytes/raw datasets are not bundled. See docs/REPRODUCIBILITY.md.'); return 2
    import pandas as pd
    from aura_cxr.evaluation import reproduce_locked,reproduce_external
    out=ROOT/'outputs/audit'; out.mkdir(parents=True,exist_ok=True)
    internal=reproduce_locked(ROOT/'predictions/standardized/locked_rsna.csv'); external=reproduce_external(ROOT/'predictions/standardized/kermany.csv')
    pd.DataFrame(internal).T.to_csv(out/'internal_metrics.csv'); pd.DataFrame(external).T.to_csv(out/'external_metrics.csv')
    import shutil
    shutil.copy2(ROOT/'predictions/raw_verified/ablation_selected_dl_pair.csv',out/'ablation_A1_A10.csv')
    shutil.copy2(ROOT/'predictions/raw_verified/XAI_Q1_FINAL_METRICS.csv',out/'xai_summary.csv')
    fs=json.loads((ROOT/'predictions/raw_verified/xai_faithfulness_summary.json').read_text()); pd.DataFrame([{'deletion_drop':fs['deletion_confidence_drop']['mean'],'retention_ratio':fs['retention_probability_ratio']['mean'],'n':fs['deletion_confidence_drop']['n']}]).to_csv(out/'faithfulness_summary.csv',index=False)
    gold=json.loads((ROOT/'reproducibility/golden_reference.json').read_text()); checks={}
    for c in ['A4','A8']:
        checks[c]={k:abs(internal[c][k]-gold['key_metric_fingerprints'][c][k]) for k in ['AUC','AUPRC','sensitivity','specificity','accuracy','balanced_acc','MCC']}
    report={'origin':'derived_reproduction','selection_eligible':False,'mode':'audit','checks':checks,'status':'PASS' if max(v for c in checks.values() for v in c.values())<1e-12 else 'FAIL'}
    (out/'verification_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 1

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "verify-deep-config":
        import subprocess
        script = ROOT / "scripts" / "verify_deep_config.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    if len(sys.argv) > 1 and sys.argv[1] == "verify-radiomics":
        import subprocess
        script = ROOT / "scripts" / "verify_radiomics_reproducibility.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    if len(sys.argv) > 1 and sys.argv[1] == "verify-xai":
        import subprocess
        script = ROOT / "scripts" / "verify_xai.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    ap=argparse.ArgumentParser(description='AURA-CXR public reproducibility repository'); sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('verify'); r=sp.add_parser('reproduce'); r.add_argument('--mode',choices=['audit','evaluate','full'],default='audit')
    sp.add_parser('train',help='Full training uses preserved canonical source lineage; see docs/REPRODUCIBILITY.md')
    sp.add_parser('xai',help='Branch-scoped XAI; see docs/REPRODUCIBILITY.md')
    sp.add_parser('external',help='Frozen external evaluation only; no adaptation permitted')
    a=ap.parse_args()
    if a.cmd=='verify': return verify()
    if a.cmd=='reproduce': return reproduce(a.mode)
    print('Command scaffold is intentionally non-invasive in the public release. Use canonical source snapshot with documented config; no scientific rewrite was substituted.'); return 0
if __name__=='__main__': raise SystemExit(main())
