# import relevant packages
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# utility function
def u_func(l,eps,kappa,nu,m,tau0,tau1,w):
    """
    CRRA utility function for a rational agent making a consumption and labour decision subject
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
    
    u = np.log(m + w*l - (tau0*w*l + tau1*max(w*l - kappa,0))) - nu*(l**(1+1/eps)/(1+1/eps))
    return u

# scalar optimiser function
def u_optimiser(eps,kappa,nu,m,tau0,tau1,w):
    """
    Optimises u_func with respect to labour input. It prints out the optimal labour supply,
    the implied optimal consumption and the utility derived from this optimal combination.
    """

    def objective(l,eps,kappa,nu,m,tau0,tau1,w):
        return -u_func(l=l,eps=eps,kappa=kappa,nu=nu,m=m,tau0=tau0,tau1=tau1,w=w)

    sol = optimize.minimize_scalar(objective,method="bounded",
    bounds=(0,1),args=(eps,kappa,nu,m,tau0,tau1,w))

    l_star = sol.x
    c_star = m + w*l_star - (tau0*w*l_star + tau1*max(w*l_star - kappa,0))
    u_star = u_func(l=l_star,eps=eps,kappa=kappa,nu=nu,m=m,tau0=tau0,tau1=tau1,w=w)
    return l_star, c_star, u_star

# plot function
def two_figures(x_left, y_left, title_left, xlabel_left, ylabel_left, x_right, y_right, title_right, xlabel_right, ylabel_right):
    """ 
    Plots two aligned figures. 
    
    Inputs: should be self explanatory...
    Output: Two figures in 2D
    """
    # a. initialise figure
    fig = plt.figure(figsize=(10,4))# figsize is in inches...

    # b. left plot
    ax_left = fig.add_subplot(1,2,1)
    ax_left.plot(x_left,y_left)

    ax_left.set_title(title_left)
    ax_left.set_xlabel(xlabel_left)
    ax_left.set_ylabel(ylabel_left)

    # c. right plot
    ax_right = fig.add_subplot(1,2,2)

    ax_right.plot(x_right, y_right)

    ax_right.set_title(title_right)
    ax_right.set_xlabel(xlabel_right)
    ax_right.set_ylabel(ylabel_right)

# tax revenue function
def tax_revenue(seed,size,low,high,eps=0.3,tau0=0.4,tau1=0.1,kappa=0.4):
    """
    Calculates the total tax revenue for a given number of agents with utility defined as
    u_func and heterogeneous income (uniformly distributed).

    Inputs:
    seed: seed number
    size: number of random incomes (i.e. agents) drawn from a uniform  distribution
    low: lower bound of uniform distribution
    high: higher bound of uniform distribution
    eps: Frisch elasticity of labour supply

    Local variables:

    kappa: cut-off for top labour income bracket
    nu: a scalar for the weight put on disutility of labour
    m: cash-on-hand
    tau0: normal tax rate
    tau1: top labour income bracket tax rate
    wi: random numbers drawn from a uniform distribution

    Output:
    Total tax revenue
    """
    # a. set seed, draw random numbers
    np.random.seed(seed)
    wi = np.random.uniform(low=low,high=high,size=size)

    # b. define local parameter values
    nu = 10
    m = 1

    # c. solve each individual's optimisation problem
    tax_rev = 0

    for i, wi in enumerate (wi):        
        lc = u_optimiser(eps,kappa,nu,m,tau0,tau1,wi)
        tax_i = tau0*wi*lc[0] + tau1*max(wi*lc[0]-kappa,0)
        tax_rev += tax_i

    return tax_rev