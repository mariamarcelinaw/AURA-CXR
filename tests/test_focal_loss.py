import math
def focal(logits,target,w,gamma=2.0):
 import numpy as np
 z=np.asarray(logits,float); p=np.exp(z-z.max()); p/=p.sum(); pt=p[target]; return ((1-pt)**gamma)*(-w[target]*math.log(pt))
def test_custom_focal_positive_and_negative():
 w=[0.7,1.4]; assert focal([0.2,1.1],1,w)>0; assert focal([1.2,-0.3],0,w)>0
def test_gamma_zero_reduces_to_weighted_ce():
 w=[0.7,1.4]; a=focal([0.2,1.1],1,w,0); import numpy as np; z=np.array([.2,1.1]);p=np.exp(z-z.max());p/=p.sum(); b=-w[1]*math.log(p[1]); assert abs(a-b)<1e-12
