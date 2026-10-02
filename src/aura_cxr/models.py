"""Model preprocessing/build helpers with lazy heavy-framework imports.
Exact training implementations remain preserved in reproducibility/canonical_source_snapshot.
"""
import numpy as np

def eva_normalize(chw:np.ndarray,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))->np.ndarray:
    m=np.asarray(mean,dtype=np.float32)[:,None,None]; s=np.asarray(std,dtype=np.float32)[:,None,None]
    return ((chw.astype(np.float32)-m)/s).astype(np.float32)

def build_xrv_binary(dropout=0.3,weights='densenet121-res224-all'):
    import torch, torchxrayvision as xrv
    base=xrv.models.DenseNet(weights=weights); feat_dim=base.classifier.in_features if hasattr(base,'classifier') else 1024
    class XRVBinary(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=base; self.drop=torch.nn.Dropout(dropout); self.head=torch.nn.Linear(feat_dim,2)
        def forward(self,x):
            import torch.nn.functional as F
            f=self.backbone.features(x)
            if f.dim()==4: f=F.adaptive_avg_pool2d(F.relu(f,inplace=False),1).flatten(1)
            return self.head(self.drop(f))
    return XRVBinary()
