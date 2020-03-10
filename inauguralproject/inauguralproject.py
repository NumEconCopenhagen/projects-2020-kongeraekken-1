import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def u_func(l,eps,kappa,nu,m,tau0,tau1,w):
    """
    CRRA utility function for an rational agent making a consumption and labour decision subject
    to constraints. The optimisation problem only considers labour as consumption is an implicit
    function hereof. 
    
    Variables:
    l: labour
    eps: Frisch elasticity of labour supply
    kappa: cut-off for top labour income bracket
    nu: a scalar for the weight put on disutility of labour
    m: cash-on-hand
    tau0: normal tax rate
    tau1: top labour income bracket tax rate
    w: wage rate
    
    """
    
    u = np.log(m + w*l - (tau0*w*l + tau1*np.max(w*l - kappa,0))) - nu*(l**(1+1/eps)/(1+1/eps))
    return u