import numpy as np
from aura_cxr.fusion import *
def test_frozen(): assert_frozen_a8()
def test_fusion_fixture():
 p1=np.array([.1,.8]);p2=np.array([.2,.7]);pr=np.array([.3,.6]); assert np.allclose(a4_probability(p1,p2),np.array([.15,.75]),rtol=0,atol=1e-15); out=a8_probability(p1,p2,pr); assert out.shape==(2,); assert np.all((out>0)&(out<1))
 # swapped columns must alter output
 assert not np.allclose(out,a8_probability(p1,pr,p2))
