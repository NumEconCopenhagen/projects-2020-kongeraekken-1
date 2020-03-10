import numpy as np
from scipy import optimize


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


def u_optimizer(eps,kappa,nu,m,tau0,tau1,w):
    """This function optimizes u_func wrt labor input. It prints out the optimal labor supply,
    the implied optimal consumption and the utility derived from this optimal combination"""

    def objective(l,eps,kappa,nu,m,tau0,tau1,w):
        return -u_func(l=l,eps=eps,kappa=kappa,nu=nu,m=m,tau0=tau0,tau1=tau1,w=w)

    sol = optimize.minimize_scalar(objective,method="bounded",
    bounds=(0,1),args=(eps,kappa,nu,m,tau0,tau1,w))

    l_star = sol.x
    c_star = m + w*l_star - (tau0*w*l_star + tau1*np.max(w*l_star - kappa,0))
    u_star = u_func(l=l_star,eps=eps,kappa=kappa,nu=nu,m=m,tau0=tau0,tau1=tau1,w=w)
    return l_star, c_star, u_star

