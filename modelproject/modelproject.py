from scipy import optimize

def solve_for_consumption(m1,r,beta,rho):
    """ solve for optimal consumption in period 1 and 2

    Args:
        m1 (float): period 1 cash on hand
        g (float): technological growth rate
        n (float): population growth rate
        alpha (float): cobb-douglas parameter
        delta (float): capital depreciation rate 

    Returns:
        result (RootResults): the solution represented as a RootResults object

    """ 
    
    # a. define objective function
    f = lambda k: k**alpha
    obj_kss = lambda kss: kss - (s*f(kss) + (1-delta)*kss)/((1+g)*(1+n))

    #. b. call root finder
    result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='bisect')
    
    return result