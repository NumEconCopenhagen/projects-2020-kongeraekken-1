from scipy import optimize

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