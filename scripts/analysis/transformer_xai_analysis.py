#!/usr/bin/env python3
"""AURA-CXR Stage 7 post-hoc Transformer-XAI sensitivity analysis.

Scientific lock:
- Frozen XRV/EVA deployment checkpoints only.
- No classifier retraining, reselection, or threshold retuning.
- Historical LayerCAM reproduced exactly from the canonical AURA-CXR XAI logic.
- Comparator = Integrated Gradients on the same frozen EVA-X-S classifier.
- Primary comparison = EVA token-LayerCAM vs EVA Integrated Gradients.
- Optional supporting comparison = original hybrid fused LayerCAM vs hybrid map with EVA IG substituted while preserving XRV and 0.25/0.75 fusion weights.

The script is resumable at patient level through an append-only case CSV.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, inspect, json, math, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

EXPECTED_XRV_SHA = "ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a"
EXPECTED_EVA_SHA = "83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6"
XRV_HOOKS = [
    "backbone.features.denseblock4.denselayer16.conv1",
    "backbone.features.denseblock4.denselayer15.conv1",
]
EVA_HOOKS = ["blocks.11.mlp", "blocks.11"]
LAYER_WEIGHTS = [0.75, 0.25]
MODEL_WEIGHTS = {"xrv": 0.25, "eva_x": 0.75}
IMAGE_SIZE = 224
CLASS_INDEX = 1
THRESHOLD = 0.25
SMOOTH_SIGMA_FRAC = 0.005
SEED = 42


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def normalize_map(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        return None
    a = np.maximum(a, 0)
    lo, hi = float(a.min()), float(a.max())
    if not np.isfinite(hi) or hi <= lo + 1e-12:
        return None
    return ((a - lo) / (hi - lo)).astype(np.float32)


def resize_map(a, h=IMAGE_SIZE, w=IMAGE_SIZE):
    from scipy.ndimage import zoom
    a = np.asarray(a, dtype=np.float32)
    if a.shape == (h, w):
        return a.copy()
    f = (h / a.shape[0], w / a.shape[1])
    return np.clip(zoom(a, f, order=1), 0, 1).astype(np.float32)


def fuse_layer_maps(maps, weights=LAYER_WEIGHTS):
    if any(m is None for m in maps):
        return None
    ms = [normalize_map(resize_map(m)) for m in maps]
    if any(m is None for m in ms):
        return None
    w = np.asarray(weights, dtype=np.float64); w /= w.sum()
    return normalize_map(np.tensordot(w, np.stack(ms), axes=(0, 0)))


def fuse_hybrid(xrv_map, eva_map):
    from scipy.ndimage import gaussian_filter
    if xrv_map is None or eva_map is None: return None
    xrv_map = normalize_map(xrv_map); eva_map = normalize_map(eva_map)
    if xrv_map is None or eva_map is None: return None
    z = MODEL_WEIGHTS["xrv"] * xrv_map + MODEL_WEIGHTS["eva_x"] * eva_map
    sigma = max(0.5, SMOOTH_SIGMA_FRAC * IMAGE_SIZE)
    z = gaussian_filter(z.astype(np.float32), sigma=sigma).astype(np.float32)
    return normalize_map(z)


def import_models(args):
    import torch
    import torchxrayvision as xrv

    xrv_script = Path(args.xrv_script)
    eva_script = Path(args.eva_script)
    xrv_mod = load_module(xrv_script, "stage7_xrv_train")
    eva_mod = load_module(eva_script, "stage7_eva_train")

    xrv_ckpt = Path(args.xrv_checkpoint)
    eva_ckpt = Path(args.eva_checkpoint)
    if sha256_file(xrv_ckpt) != EXPECTED_XRV_SHA:
        raise RuntimeError("XRV checkpoint SHA256 mismatch")
    if sha256_file(eva_ckpt) != EXPECTED_EVA_SHA:
        raise RuntimeError("EVA checkpoint SHA256 mismatch")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    xck = torch.load(xrv_ckpt, map_location="cpu")
    xdrop = float(xck.get("dropout", 0.3)) if isinstance(xck, dict) else 0.3
    xmodel = xrv_mod.build_model(xrv, torch, dropout=xdrop, weights=None).to(device)
    # Audited TorchXRayVision compatibility handling: checkpoints created from
    # pretrained DenseNet may contain the non-inference buffer
    # ``backbone.op_threshs`` although weights=None constructors in some XRV
    # releases do not register it. The AURA-CXR binary forward path uses
    # backbone.features + the separately trained binary head, so this buffer is
    # not consumed. Remove ONLY this known extra buffer, then retain strict=True
    # for every trainable parameter and every other persistent buffer.
    xstate = xck.get("state_dict", xck) if isinstance(xck, dict) else xck
    if not isinstance(xstate, dict):
        raise TypeError("XRV checkpoint state_dict must be a mapping")
    xstate = dict(xstate)
    compat_key = "backbone.op_threshs"
    ignored_xrv_buffers = []
    if compat_key in xstate and compat_key not in xmodel.state_dict():
        v = xstate.pop(compat_key)
        ignored_xrv_buffers.append({"key": compat_key, "shape": tuple(v.shape) if hasattr(v, "shape") else None})
    xmodel.load_state_dict(xstate, strict=True)
    if ignored_xrv_buffers:
        print("[XRV][CHECKPOINT_COMPAT] ignored audited non-inference buffer: " + str(ignored_xrv_buffers), flush=True)
    xmodel.eval()

    eva_official = eva_mod.import_eva_x(Path(args.eva_repo))
    eck = torch.load(eva_ckpt, map_location="cpu")
    edrop = float(eck.get("dropout", 0.2)) if isinstance(eck, dict) else 0.2
    ens = SimpleNamespace(pretrained_checkpoint=str(args.eva_upstream_checkpoint), dropout=edrop)
    emodel = eva_mod.build_model(ens, eva_official, torch).to(device)
    emodel.load_state_dict(eck.get("state_dict", eck) if isinstance(eck, dict) else eck)
    emodel.eval()

    return torch, xrv, xrv_mod, eva_mod, xmodel, emodel, device


def raw_to_xrv(raw, xrv, torch, device):
    g = np.asarray(raw, dtype=np.float32)
    g = g - g.min(); g = g / g.max() if g.max() > 0 else np.zeros_like(g)
    z = xrv.datasets.normalize((g * 255).astype(np.float32), 255)
    return torch.from_numpy(z[None, None].astype(np.float32)).to(device)


def raw_to_eva(raw, torch, device):
    rgb = np.repeat(np.asarray(raw, dtype=np.float32)[None, None, ...], 3, axis=1)
    mean = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)[None, :, None, None]
    return torch.from_numpy(((rgb - mean) / std).astype(np.float32)).to(device)


def first_tensor(o):
    if hasattr(o, "shape"):
        return o
    if isinstance(o, (tuple, list)):
        for x in o:
            if hasattr(x, "shape"): return x
    if isinstance(o, dict):
        for x in o.values():
            if hasattr(x, "shape"): return x
    return None


def historical_token_grid(a, g):
    n = int(a.shape[0])
    if int(round((n - 1) ** 0.5)) ** 2 == n - 1:
        return a[1:], g[1:], 1, int(round((n - 1) ** 0.5))
    if int(round(n ** 0.5)) ** 2 == n:
        return a, g, 0, int(round(n ** 0.5))
    return None, None, None, None


def single_layer_layercam(model, module, x, token_mode=False):
    import torch
    captured = {}
    def hook(_m, _i, o):
        t = first_tensor(o)
        captured["act"] = t
        if t is not None and hasattr(t, "retain_grad"): t.retain_grad()
    h = module.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True); model.eval()
        logits = model(x)
        score = logits[:, CLASS_INDEX].sum()
        score.backward()
        act = captured.get("act")
        if act is None or act.grad is None: return None, {}
        grad = act.grad
        meta = {"activation_shape": list(act.shape), "gradient_shape": list(grad.shape), "target": "pre_softmax_logit_class_1"}
        if token_mode:
            if act.ndim != 3: return None, meta
            a, g = act[0], grad[0]
            a, g, dropped, side = historical_token_grid(a, g)
            if a is None: return None, meta
            token = (torch.relu(g) * a).sum(dim=1)
            cam = torch.relu(token.reshape(side, side))
            meta.update({"historical_prefix_tokens_dropped": dropped, "spatial_grid": [side, side], "embedding_dim": int(a.shape[1]), "retained_patch_tokens": int(a.shape[0])})
        else:
            if act.ndim != 4: return None, meta
            cam = torch.relu((torch.relu(grad) * act).sum(dim=1))[0]
            meta.update({"channel_axis": 1, "spatial_grid": [int(act.shape[-2]), int(act.shape[-1])]})
        arr = cam.detach().cpu().numpy().astype(np.float32)
        m = float(np.nanmax(arr)) if arr.size else 0.0
        if not np.isfinite(m) or m <= 1e-8: return None, meta
        return arr / m, meta
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)


def multilayer_layercam(model, modules, x, token_mode=False):
    maps=[]; metas=[]
    for name in (EVA_HOOKS if token_mode else XRV_HOOKS):
        m, meta = single_layer_layercam(model, modules[name], x, token_mode=token_mode)
        meta["module_path"] = name; meta["module_type"] = type(modules[name]).__name__
        maps.append(m); metas.append(meta)
    return fuse_layer_maps(maps), metas


def integrated_gradients_eva(model, x, steps=64, internal_batch=16):
    """64-step trapezoidal IG to EVA pneumonia logit from normalized-zero baseline."""
    import torch
    model.eval(); model.zero_grad(set_to_none=True)
    baseline = torch.zeros_like(x)
    delta = x - baseline
    alphas = torch.linspace(0.0, 1.0, int(steps) + 1, device=x.device, dtype=x.dtype)
    total = torch.zeros_like(x)
    completeness = {}
    for start in range(0, len(alphas), int(internal_batch)):
        aa = alphas[start:start+int(internal_batch)]
        xb = baseline + aa[:, None, None, None] * delta
        xb = xb.detach().requires_grad_(True)
        logits = model(xb)
        score = logits[:, CLASS_INDEX].sum()
        grads = torch.autograd.grad(score, xb, retain_graph=False, create_graph=False)[0]
        weights = torch.ones(len(aa), device=x.device, dtype=x.dtype)
        global_indices = torch.arange(start, start+len(aa), device=x.device)
        weights[(global_indices == 0) | (global_indices == int(steps))] = 0.5
        total += (grads * weights[:, None, None, None]).sum(dim=0, keepdim=True)
    avg_grad = total / float(steps)
    ig = delta * avg_grad
    spatial = torch.relu(ig.sum(dim=1))[0]
    arr = spatial.detach().cpu().numpy().astype(np.float32)
    out = normalize_map(arr)
    with torch.no_grad():
        fx = float(model(x)[:, CLASS_INDEX].item())
        fb = float(model(baseline)[:, CLASS_INDEX].item())
    completeness["target_logit_input"] = fx
    completeness["target_logit_baseline"] = fb
    completeness["delta_logit"] = fx - fb
    completeness["sum_signed_ig"] = float(ig.sum().detach().cpu().item())
    completeness["absolute_completeness_error"] = abs(completeness["sum_signed_ig"] - completeness["delta_logit"])
    model.zero_grad(set_to_none=True)
    return out, completeness


def probability(model, x):
    import torch
    with torch.no_grad():
        model.eval()
        p = torch.softmax(model(x), dim=1)[:, CLASS_INDEX]
        return float(p.item())


def map_similarity(a, b):
    if a is None or b is None: return np.nan, np.nan, np.nan
    x=np.asarray(a,dtype=np.float64).ravel(); y=np.asarray(b,dtype=np.float64).ravel()
    pearson = float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else np.nan
    denom=np.linalg.norm(x)*np.linalg.norm(y); cosine=float(np.dot(x,y)/denom) if denom>0 else np.nan
    k=max(1,int(round(0.10*len(x))))
    ix=np.argpartition(x,-k)[-k:]; iy=np.argpartition(y,-k)[-k:]
    sx=set(ix.tolist()); sy=set(iy.tolist()); j=float(len(sx&sy)/len(sx|sy)) if sx|sy else np.nan
    return pearson, cosine, j


def verify_token_structure(model, device, torch):
    mods=dict(model.named_modules())
    if "blocks.0" not in mods: raise RuntimeError("EVA blocks.0 unavailable")
    captured={}
    def pre(_m, inp):
        t=first_tensor(inp)
        if t is not None: captured["tokens"]=t.detach().cpu()
    h=mods["blocks.0"].register_forward_pre_hook(pre)
    try:
        z=torch.zeros((1,3,IMAGE_SIZE,IMAGE_SIZE),device=device)
        with torch.no_grad(): model.eval(); model(z)
        tok0=captured["tokens"]
        N=int(tok0.shape[1]); D=int(tok0.shape[2])
        patch_embed=getattr(model,"patch_embed",None)
        num_patches=int(getattr(patch_embed,"num_patches",196))
        grid=getattr(patch_embed,"grid_size",(14,14)); grid=[int(grid[0]),int(grid[1])]
        ps=getattr(patch_embed,"patch_size",(16,16));
        if isinstance(ps,int): ps=(ps,ps)
        ps=[int(ps[0]),int(ps[1])]
        prefix=N-num_patches
        cls=getattr(model,"cls_token",None)
        has_cls=cls is not None
        num_prefix_attr=getattr(model,"num_prefix_tokens",None)
        tests=[]
        for r,c in [(0,0),(0,1),(1,0),(grid[0]-1,grid[1]-1)]:
            x=z.clone(); x[:,:,r*ps[0]:(r+1)*ps[0],c*ps[1]:(c+1)*ps[1]] = 1.0
            with torch.no_grad(): model(x)
            t=captured["tokens"]
            diff=torch.linalg.vector_norm(t-tok0,dim=2)[0].numpy()
            idx=int(np.argmax(diff))
            expected=prefix + r*grid[1]+c
            tests.append({"patch_row":r,"patch_col":c,"observed_token_index":idx,"expected_row_major_index":expected,"pass":bool(idx==expected)})
        historical_prefix = 1 if int(round((N-1)**0.5))**2==N-1 else (0 if int(round(N**0.5))**2==N else None)
        return {
            "block0_input_shape": list(tok0.shape), "sequence_length":N,"embedding_dim":D,
            "num_patches":num_patches,"grid":grid,"patch_size":ps,"prefix_tokens_from_shape":prefix,
            "num_prefix_tokens_attribute": int(num_prefix_attr) if num_prefix_attr is not None else None,
            "has_cls_token_parameter":bool(has_cls),"cls_token_shape":list(cls.shape) if has_cls else None,
            "historical_heuristic_prefix_tokens":historical_prefix,
            "historical_heuristic_equivalent":bool(historical_prefix==prefix),
            "row_major_tests":tests,"row_major_verified":bool(all(t["pass"] for t in tests)),
            "other_prefix_tokens": int(max(0,prefix-(1 if has_cls else 0)))
        }
    finally:
        h.remove()


def write_hook_csv(path, xrv_meta, eva_meta):
    rows=[]
    for i,m in enumerate(xrv_meta): rows.append({"branch":"xrv","hook_index":i+1,**m,"layer_weight":LAYER_WEIGHTS[i]})
    for i,m in enumerate(eva_meta): rows.append({"branch":"eva_x","hook_index":i+1,**m,"layer_weight":LAYER_WEIGHTS[i]})
    pd.DataFrame(rows).to_csv(path,index=False)


def bootstrap_delta(a,b,n_boot=2000,seed=42):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    ok=np.isfinite(a)&np.isfinite(b); a=a[ok]; b=b[ok]
    d=b-a; point=float(np.mean(d))
    rng=np.random.default_rng(seed); n=len(d); vals=np.empty(n_boot,float)
    for i in range(n_boot):
        idx=rng.integers(0,n,n); vals[i]=float(np.mean(d[idx]))
    return point,float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))


def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj


def summarize_results(case_df, outdir):
    from scipy.stats import wilcoxon, binomtest, mannwhitneyu
    primary=["pointing_hit","iou_at_thr","localization_score","energy_inside_gt"]
    rows=[]; raw_ps=[]
    pair=case_df[(case_df.eva_layercam_valid==True)&(case_df.eva_ig_valid==True)].copy()
    for metric in primary:
        a=pd.to_numeric(pair[f"eva_layercam_{metric}"],errors="coerce").to_numpy(float)
        b=pd.to_numeric(pair[f"eva_ig_{metric}"],errors="coerce").to_numpy(float)
        ok=np.isfinite(a)&np.isfinite(b); a=a[ok]; b=b[ok]
        delta,lo,hi=bootstrap_delta(a,b,2000,42)
        if metric=="pointing_hit":
            b01=int(np.sum((a==1)&(b==0))); c10=int(np.sum((a==0)&(b==1)))
            pv=float(binomtest(b01,b01+c10,p=0.5,alternative="two-sided").pvalue) if b01+c10 else 1.0
            method="paired proportion + exact McNemar/binomial"
        else:
            try: pv=float(wilcoxon(b-a,alternative="two-sided",zero_method="wilcox").pvalue)
            except Exception: pv=np.nan
            method="paired bootstrap CI + Wilcoxon signed-rank"
        raw_ps.append(pv)
        rows.append({"analysis_level":"EVA_branch","metric":metric,"layercam_value":float(np.mean(a)),"comparator_value":float(np.mean(b)),"delta_comparator_minus_layercam":delta,"CI_low":lo,"CI_high":hi,"raw_p":pv,"adjusted_p":np.nan,"method":method,"n_both_valid":len(a)})
    adj=holm([1 if not np.isfinite(p) else p for p in raw_ps])
    for r,p in zip(rows,adj): r["adjusted_p"]=float(p)

    # Supporting hybrid-level sensitivity (same XRV map, EVA method swapped).
    hpair=case_df[(case_df.hybrid_layercam_valid==True)&(case_df.hybrid_ig_valid==True)].copy()
    for metric in primary:
        a=pd.to_numeric(hpair[f"hybrid_layercam_{metric}"],errors="coerce").to_numpy(float)
        b=pd.to_numeric(hpair[f"hybrid_ig_{metric}"],errors="coerce").to_numpy(float)
        ok=np.isfinite(a)&np.isfinite(b); a=a[ok]; b=b[ok]
        delta,lo,hi=bootstrap_delta(a,b,2000,142)
        rows.append({"analysis_level":"Hybrid_supporting","metric":metric,"layercam_value":float(np.mean(a)),"comparator_value":float(np.mean(b)),"delta_comparator_minus_layercam":delta,"CI_low":lo,"CI_high":hi,"raw_p":np.nan,"adjusted_p":np.nan,"method":"paired bootstrap CI; supporting only","n_both_valid":len(a)})
    pd.DataFrame(rows).to_csv(outdir/"12_TRANSFORMER_COMPARATOR_RESULTS.csv",index=False)

    sim={}
    for col in ["eva_map_pearson","eva_map_cosine","eva_top10_jaccard","hybrid_map_pearson","hybrid_top10_jaccard","ig_completeness_error"]:
        v=pd.to_numeric(case_df[col],errors="coerce").dropna()
        sim[col]={"n":int(len(v)),"mean":float(v.mean()) if len(v) else None,"median":float(v.median()) if len(v) else None,"q1":float(v.quantile(.25)) if len(v) else None,"q3":float(v.quantile(.75)) if len(v) else None}
    (outdir/"map_similarity_summary.json").write_text(json.dumps(sim,indent=2))

    # TP/FN comparator sensitivity using EVA IG.
    tprows=[]
    for metric in primary:
        tp=pd.to_numeric(case_df.loc[case_df.category=="TP",f"eva_ig_{metric}"],errors="coerce").dropna().to_numpy(float)
        fn=pd.to_numeric(case_df.loc[case_df.category=="FN",f"eva_ig_{metric}"],errors="coerce").dropna().to_numpy(float)
        if len(tp) and len(fn):
            u,p=mannwhitneyu(tp,fn,alternative="two-sided")
            gt=sum(float(x)>float(y) for x in tp for y in fn); lt=sum(float(x)<float(y) for x in tp for y in fn)
            cliff=(gt-lt)/(len(tp)*len(fn))
            tprows.append({"metric":metric,"n_tp":len(tp),"n_fn":len(fn),"tp_mean":float(tp.mean()),"fn_mean":float(fn.mean()),"mann_whitney_u":float(u),"p_value":float(p),"cliffs_delta_tp_vs_fn":float(cliff)})
    pd.DataFrame(tprows).to_csv(outdir/"comparator_tp_vs_fn.csv",index=False)

    return rows, sim


def static_reference_metrics(reference_csv):
    df=pd.read_csv(reference_csv)
    return {
        "n":int(len(df)),
        "pointing_game":float(pd.to_numeric(df.pointing_hit,errors="coerce").mean()),
        "iou":float(pd.to_numeric(df.iou_at_thr,errors="coerce").mean()),
        "localization_score":float(pd.to_numeric(df.localization_score,errors="coerce").mean()),
        "energy_inside_gt":float(pd.to_numeric(df.energy_inside_gt,errors="coerce").mean())
    }


def run(args):
    set_seed(SEED)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"maps").mkdir(exist_ok=True)
    ref_csv=Path(args.original_xai_per_sample)
    ref=pd.read_csv(ref_csv)
    if len(ref)!=1203: raise RuntimeError(f"Canonical XAI case list expected 1203, got {len(ref)}")
    if ref.patientId.astype(str).duplicated().any(): raise RuntimeError("Duplicate patient IDs in canonical XAI list")

    xai_mod=load_module(Path(args.canonical_xai_script),"stage7_original_xai")
    torch,xrv,xrv_mod,eva_mod,xmodel,emodel,device=import_models(args)
    xmods=dict(xmodel.named_modules()); emods=dict(emodel.named_modules())
    for h in XRV_HOOKS:
        if h not in xmods: raise RuntimeError(f"Missing XRV hook {h}")
    for h in EVA_HOOKS:
        if h not in emods: raise RuntimeError(f"Missing EVA hook {h}")

    token_info=verify_token_structure(emodel,device,torch)
    (out/"token_structure.json").write_text(json.dumps(token_info,indent=2))
    if not token_info["row_major_verified"]: raise RuntimeError("EVA patch-token row-major order verification failed")
    if not token_info["historical_heuristic_equivalent"]: raise RuntimeError("Historical token heuristic is not equivalent to actual prefix-token structure")

    gt_map=xai_mod.load_radiologist_bboxes(args.labels_csv)
    first_pid=str(ref.iloc[0].patientId); first_path=Path(args.dicom_dir)/f"{first_pid}.dcm"
    raw=xai_mod.load_dicom_grayscale(first_path,IMAGE_SIZE)
    xt=raw_to_xrv(raw,xrv,torch,device); et=raw_to_eva(raw,torch,device)

    # Hook non-interference and exact shapes.
    px0=probability(xmodel,xt); pe0=probability(emodel,et)
    xmap,xmeta=multilayer_layercam(xmodel,xmods,xt,False)
    emap,emeta=multilayer_layercam(emodel,emods,et,True)
    px1=probability(xmodel,xt); pe1=probability(emodel,et)
    if abs(px0-px1)>1e-7 or abs(pe0-pe1)>1e-7: raise RuntimeError("Hook non-interference probability test failed")
    write_hook_csv(out/"hook_specification.csv",xmeta,emeta)
    (out/"hook_noninterference.json").write_text(json.dumps({"xrv_before":px0,"xrv_after":px1,"eva_before":pe0,"eva_after":pe1,"tolerance":1e-7,"pass":True},indent=2))

    # Determinism check for first case.
    emap2,_=multilayer_layercam(emodel,emods,et,True)
    if emap is None or emap2 is None or not np.allclose(emap,emap2,rtol=1e-6,atol=1e-7): raise RuntimeError("Repeated EVA LayerCAM determinism check failed")

    if args.mode=="preflight":
        # One-case comparator smoke test before any full locked-test sensitivity run.
        ig0, comp0 = integrated_gradients_eva(emodel, et, args.ig_steps, args.ig_internal_batch)
        if ig0 is None:
            raise RuntimeError("Integrated Gradients preflight produced an invalid map")
        boxes0 = xai_mod.scale_boxes_to_image(first_pid, str(first_path), gt_map, IMAGE_SIZE)
        if not boxes0:
            raise RuntimeError("First canonical Stage-7 case has no GT box")
        h0 = fuse_hybrid(xmap, emap)
        m0 = xai_mod.compute_localization_metrics(h0, boxes0, threshold_pct=THRESHOLD, heatmap_valid=(h0 is not None))
        stored0 = ref.iloc[0]
        first_case_diff = {
            k: abs(float(m0.get(k, np.nan)) - float(stored0.get(k, np.nan)))
            for k in ["pointing_hit","iou_at_thr","localization_score","energy_inside_gt"]
        }
        if any((not np.isfinite(v)) or v > 0.01 for v in first_case_diff.values()):
            raise RuntimeError(f"First-case original LayerCAM reproduction mismatch: {first_case_diff}")
        report={
            "status":"PASS","device":str(device),"token_structure":token_info,
            "xrv_checkpoint_sha256":sha256_file(Path(args.xrv_checkpoint)),"eva_checkpoint_sha256":sha256_file(Path(args.eva_checkpoint)),
            "xrv_hook_meta":xmeta,"eva_hook_meta":emeta,"hook_noninterference":{"xrv":abs(px0-px1),"eva":abs(pe0-pe1)},
            "ig_smoke":{"valid":True,"completeness":comp0},
            "first_case_original_reproduction_abs_diff":first_case_diff,
            "reference_metrics":static_reference_metrics(ref_csv)
        }
        (out/"preflight_report.json").write_text(json.dumps(report,indent=2))
        print("STAGE7_PREFLIGHT = PASS")
        return

    pre=out/"preflight_report.json"
    if not pre.exists() or json.loads(pre.read_text()).get("status")!="PASS":
        raise RuntimeError("Run --mode preflight successfully before --mode full")

    case_path=out/"13_XAI_METHOD_SENSITIVITY_CASE_LEVEL.csv"
    done=set()
    if case_path.exists():
        old=pd.read_csv(case_path); done=set(old.patientId.astype(str)); print(f"[RESUME] {len(done)}/{len(ref)} cases already complete")
    else:
        old=None

    fields=["pointing_hit","iou_at_thr","localization_score","energy_inside_gt","peak_distance_norm"]
    start=time.time(); completed=len(done)
    for ri,row in ref.iterrows():
        pid=str(row.patientId)
        if pid in done: continue
        p=Path(args.dicom_dir)/f"{pid}.dcm"
        if not p.exists(): raise FileNotFoundError(p)
        raw=xai_mod.load_dicom_grayscale(p,IMAGE_SIZE)
        x=raw_to_xrv(raw,xrv,torch,device); e=raw_to_eva(raw,torch,device)
        xcam,_=multilayer_layercam(xmodel,xmods,x,False)
        ecam,_=multilayer_layercam(emodel,emods,e,True)
        eig,comp=integrated_gradients_eva(emodel,e,args.ig_steps,args.ig_internal_batch)
        hcam=fuse_hybrid(xcam,ecam); hig=fuse_hybrid(xcam,eig)
        boxes=xai_mod.scale_boxes_to_image(pid,str(p),gt_map,IMAGE_SIZE)
        if not boxes: raise RuntimeError(f"No GT box for canonical positive case {pid}")
        maps={"eva_layercam":ecam,"eva_ig":eig,"hybrid_layercam":hcam,"hybrid_ig":hig}
        rec={"patientId":pid,"sample_index":int(row.sample_index),"category":str(row.category),"true_label":int(row.label),"classifier_prediction":int(row.pred),
             "eva_layercam_valid":ecam is not None,"eva_ig_valid":eig is not None,"hybrid_layercam_valid":hcam is not None,"hybrid_ig_valid":hig is not None,
             "ig_completeness_error":comp.get("absolute_completeness_error",np.nan)}
        for key,m in maps.items():
            met=xai_mod.compute_localization_metrics(m if m is not None else np.zeros((IMAGE_SIZE,IMAGE_SIZE),np.float32),boxes,threshold_pct=THRESHOLD,heatmap_valid=(m is not None))
            for f in fields: rec[f"{key}_{f}"]=met.get(f,np.nan)
        pr,co,ja=map_similarity(ecam,eig); rec.update({"eva_map_pearson":pr,"eva_map_cosine":co,"eva_top10_jaccard":ja})
        prh,coh,jah=map_similarity(hcam,hig); rec.update({"hybrid_map_pearson":prh,"hybrid_map_cosine":coh,"hybrid_top10_jaccard":jah})
        # Original reproduction reference per case.
        for f in fields: rec[f"original_stored_{f}"]=row.get(f,np.nan)
        if args.save_maps:
            np.savez_compressed(out/"maps"/f"{pid}.npz",eva_layercam=np.asarray(ecam,dtype=np.float16),eva_ig=np.asarray(eig,dtype=np.float16),hybrid_layercam=np.asarray(hcam,dtype=np.float16),hybrid_ig=np.asarray(hig,dtype=np.float16))
        pd.DataFrame([rec]).to_csv(case_path,mode="a",header=not case_path.exists(),index=False)
        completed+=1
        if completed%max(1,args.progress_every)==0 or completed==len(ref):
            elapsed=time.time()-start; rate=(completed-len(done))/max(elapsed,1e-9); rem=(len(ref)-completed)/max(rate,1e-9)
            print(f"[STAGE7] {completed}/{len(ref)} ({100*completed/len(ref):.2f}%) last={pid} ETA(this-run-rate)={rem/3600:.2f} h",flush=True)
        if args.max_cases>0 and completed-len(done)>=args.max_cases: break

    df=pd.read_csv(case_path)
    if len(df)<len(ref):
        print(f"[PARTIAL] completed {len(df)}/{len(ref)}; rerun same command to resume")
        return
    if df.patientId.astype(str).duplicated().any(): raise RuntimeError("Duplicate case rows after resume")

    # Reproduce original hybrid LayerCAM summary before accepting comparator results.
    dyn={f:float(pd.to_numeric(df[f"hybrid_layercam_{f}"],errors="coerce").mean()) for f in ["pointing_hit","iou_at_thr","localization_score","energy_inside_gt"]}
    orig=static_reference_metrics(ref_csv)
    comparison={"dynamic":dyn,"stored":orig,"absolute_difference":{k:abs(dyn[k]-orig[{"pointing_hit":"pointing_game","iou_at_thr":"iou","localization_score":"localization_score","energy_inside_gt":"energy_inside_gt"}[k]]) for k in dyn}}
    comparison["pass"] = bool(all(v <= 0.002 for v in comparison["absolute_difference"].values()))
    (out/"original_reproduction_dynamic.json").write_text(json.dumps(comparison,indent=2))
    if not comparison["pass"]: raise RuntimeError("Dynamic original LayerCAM summary does not reproduce canonical metrics within tolerance")

    results,sim=summarize_results(df,out)
    validity={
        "n_attempted":int(len(df)),"eva_layercam_valid":int(df.eva_layercam_valid.sum()),"eva_ig_valid":int(df.eva_ig_valid.sum()),
        "eva_both_valid":int((df.eva_layercam_valid&df.eva_ig_valid).sum()),"hybrid_both_valid":int((df.hybrid_layercam_valid&df.hybrid_ig_valid).sum()),
        "token_structure":token_info,"original_reproduction":comparison,
        "xrv_checkpoint_sha256_after":sha256_file(Path(args.xrv_checkpoint)),"eva_checkpoint_sha256_after":sha256_file(Path(args.eva_checkpoint)),
        "checkpoint_unchanged":bool(sha256_file(Path(args.xrv_checkpoint))==EXPECTED_XRV_SHA and sha256_file(Path(args.eva_checkpoint))==EXPECTED_EVA_SHA)
    }
    (out/"comparator_validity.json").write_text(json.dumps(validity,indent=2))
    manifest={"command":" ".join(sys.argv),"completed_at":time.strftime('%Y-%m-%dT%H:%M:%S%z'),"seed":SEED,"ig_steps":args.ig_steps,"ig_internal_batch":args.ig_internal_batch,
              "difference_orientation":"COMPARATOR_MINUS_LAYERCAM","case_count":len(df),"device":str(device),"protocol_integrity":{"classifier_retrained":False,"xai_retuned_on_locked_test":False,"a8_modified":False}}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("STAGE7_DYNAMIC_RUN = COMPLETE",flush=True)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["preflight","full"],required=True)
    p.add_argument("--root",required=True,help="AURA_CXR project root")
    p.add_argument("--results-dir",required=True)
    p.add_argument("--dicom-dir",required=True)
    p.add_argument("--labels-csv",required=True)
    p.add_argument("--eva-repo",required=True)
    p.add_argument("--eva-upstream-checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--xrv-checkpoint",default="")
    p.add_argument("--eva-checkpoint",default="")
    p.add_argument("--xrv-script",default="")
    p.add_argument("--eva-script",default="")
    p.add_argument("--canonical-xai-script",default="")
    p.add_argument("--original-xai-per-sample",default="")
    p.add_argument("--device",default="auto")
    p.add_argument("--ig-steps",type=int,default=64)
    p.add_argument("--ig-internal-batch",type=int,default=16)
    p.add_argument("--progress-every",type=int,default=5)
    p.add_argument("--max-cases",type=int,default=0)
    p.add_argument("--save-maps",action="store_true")
    return p


def main():
    args=parser().parse_args(); root=Path(args.root); results=Path(args.results_dir)
    # Import hardening only: canonical historical scripts import sibling modules from ROOT.
    # This changes no scientific computation; it only makes the verified source tree importable.
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if not args.xrv_checkpoint: args.xrv_checkpoint=str(results/"models"/"xrv_development_refit.pt")
    if not args.eva_checkpoint: args.eva_checkpoint=str(results/"models"/"eva_x_development_refit.pt")
    if not args.xrv_script: args.xrv_script=str(root/"train_xrv_backbone.py")
    if not args.eva_script: args.eva_script=str(root/"train_eva_x_backbone.py")
    if not args.canonical_xai_script: args.canonical_xai_script=str(root/"eval_only_xai.py")
    if not args.original_xai_per_sample: args.original_xai_per_sample=str(results/"reports"/"xai_localization_per_sample.csv")
    run(args)

if __name__=="__main__": main()
