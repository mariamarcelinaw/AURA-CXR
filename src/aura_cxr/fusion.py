"""A1-A10 composition helpers and frozen A8 stack.
A8 order is immutable: [p_xrv, p_eva_x, p_radiomics].
"""
import numpy as np
A8_ORDER=('p_xrv','p_eva_x','p_radiomics')
A8_INTERCEPT=-6.22883556957103
A8_COEFFICIENTS=np.array([7.964391214167865,4.559219751627475,-0.03641402783417562],dtype=np.float64)
A8_THRESHOLD=0.47000000000000003
A4_THRESHOLD=0.49
SOFT_WEIGHTS=np.array([0.3694867855722349,0.3571925056178238,0.2733207088099412],dtype=np.float64)
SOFT_THRESHOLD=0.435
A1_A10_DEFINITIONS={'A1':'XRV','A2':'EVA','A3':'LGBM','A4':'mean(XRV,EVA)','A5':'mean(XRV,LGBM)','A6':'mean(EVA,LGBM)','A7':'weighted_soft_vote(XRV,EVA,LGBM)','A8':'logistic_stack(XRV,EVA,LGBM)','A9':'exploratory four-deep stack','A10':'exploratory four-deep + LGBM stack'}

def sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x,dtype=np.float64)))
def a4_probability(p_xrv,p_eva): return (np.asarray(p_xrv,dtype=float)+np.asarray(p_eva,dtype=float))/2.0
def weighted_soft_vote(p_xrv,p_eva,p_rad):
    X=np.column_stack([p_xrv,p_eva,p_rad]).astype(float); return X@SOFT_WEIGHTS

def a8_probability(p_xrv,p_eva,p_rad):
    X=np.column_stack([p_xrv,p_eva,p_rad]).astype(np.float64); return sigmoid(A8_INTERCEPT+X@A8_COEFFICIENTS)
def threshold(prob,tau): return (np.asarray(prob)>=tau).astype(np.int64)
def assert_frozen_a8(order=A8_ORDER,intercept=A8_INTERCEPT,coef=A8_COEFFICIENTS,tau=A8_THRESHOLD):
    if tuple(order)!=A8_ORDER: raise AssertionError('A8 meta-feature order changed')
    if float(intercept)!=A8_INTERCEPT or not np.array_equal(np.asarray(coef,dtype=float),A8_COEFFICIENTS) or float(tau)!=A8_THRESHOLD: raise AssertionError('Frozen A8 values changed')
    return True
