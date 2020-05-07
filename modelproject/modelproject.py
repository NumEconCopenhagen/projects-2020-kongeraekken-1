import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import ipywidgets as widgets
import time
from scipy import optimize
from scipy import interpolate
import sympy as sm
from IPython.display import display

# numerical optimization of consumption in the simple model
def solve_for_consumption(m1,r,beta,rho):
    """ solve for optimal consumption in period 1 and 2

    Args:
        m1 (float): period 1 cash on hand
        r (float): real interest rate
        beta (float): subjective discount rate (patience)
        alpha (float): coefficient of relative risk aversion
            (the inverse is the elasticity of intertemporal 
             substitution)

    Returns:
        result (RootResults): the solution represented as a RootResults object

    """ 
    
    # a. define objective function
    c1_rootform = lambda c1: c1 - ((1+r)*(m1-c1)/(beta*(1+r))**(1/rho))

    # b. call root finder
    result = optimize.root_scalar(c1_rootform,bracket=[0.1,100],method='brentq')
    c1_s = result.root
                                                           
    # c. residual calculation of c2
    c2_s = (1+r)*(m1-c1_s)
                                                           
    return c1_s, c2_s




# basic functions to construct value functions
def utility(c,rho):
    return c**(1-rho)/(1-rho)

def bequest(m,c,nu,kappa,rho):
    return nu*(m-c+kappa)**(1-rho)/(1-rho)


# value functions for risk scenario without tails
def v2_risk(c2,m2,rho,nu,kappa):
    return utility(c2,rho) + bequest(m2,c2,nu,kappa,rho)

def v1_risk(c1,m1,rho,beta,r,Delta,v2_interp):
    
    # a. v2 value, if low income
    m2_low = (1+r)*(m1-c1) + 1-Delta
    v2_low = v2_interp([m2_low])[0]
    
    # b. v2 value, if high income
    m2_high = (1+r)*(m1-c1) + 1+Delta
    v2_high = v2_interp([m2_high])[0]
    
    # c. expected v2 value
    v2_risk = 0.5*v2_low + 0.5*v2_high
    
    # d. total value
    return utility(c1,rho) + beta*v2_risk

# solve functions risk scenario without tails
def solve_period_2(rho,nu,kappa,Delta):

    # a. grids
    m2_vec = np.linspace(1e-8,5,500)
    v2_vec = np.empty(500)
    c2_vec = np.empty(500)

    # b. solve for each m2 in grid
    for i,m2 in enumerate(m2_vec):

        # i. objective
        obj = lambda c2: -v2_risk(c2,m2,rho,nu,kappa)

        # ii. initial value (consume half)
        x0 = m2/2

        # iii. optimizer
        result = optimize.minimize_scalar(obj,x0,method='bounded',bounds=[1e-8,m2])

        # iv. save
        v2_vec[i] = -result.fun
        c2_vec[i] = result.x
        
    return m2_vec,v2_vec,c2_vec

def solve_period_1(rho,beta,r,Delta,v2_interp,v1): # try without v1 as input

    # a. grids
    m1_vec = np.linspace(1e-8,4,100)
    v1_vec = np.empty(100)
    c1_vec = np.empty(100)
    
    # b. solve for each m1 in grid
    for i,m1 in enumerate(m1_vec):
        
        # i. objective
        obj = lambda c1: -v1(c1,m1,rho,beta,r,Delta,v2_interp)
        
        # ii. initial guess (consume half)
        x0 = m1*1/2
        
        # iii. optimize
        result = optimize.minimize_scalar(obj,x0,method='bounded',bounds=[1e-8,m1])
        
        # iv. save
        v1_vec[i] = -result.fun
        c1_vec[i] = result.x
     
    return m1_vec,v1_vec,c1_vec

# joint solve function
def solve_risk(rho,beta,r,Delta,nu,kappa,v1_risk):
    
    # a. solve period 2
    m2_vec,v2_vec,c2_vec = solve_period_2(rho,nu,kappa,Delta)
    
    # b. construct interpolator
    v2_interp = interpolate.RegularGridInterpolator((m2_vec,), v2_vec,
                                                    bounds_error=False,fill_value=None)
    
    # b. solve period 1
    m1_vec,v1_vec,c1_vec = solve_period_1(rho,beta,r,Delta,v2_interp,v1_risk) #ændrede v1 til v1_risk
    
    return m1_vec,c1_vec,m2_vec,c2_vec




# solve for tail risk scenario
def v1_tailrisk(c1,m1,rho,beta,r,Delta,v2_interp):
    
    # a. expected v2 value
    Ra = (1+r)*(m1-c1)
    v2 = 0
    y2s = [1-np.sqrt(Delta),1-Delta,1+Delta,1+np.sqrt(Delta)]
    probs = [0.1,0.4,0.4,0.1]
    for y2,prob in zip(y2s,probs):
        m2 = Ra + y2
        v2 += prob*v2_interp([m2])[0]
        
    # b. total value
    return utility(c1,rho) + beta*v2