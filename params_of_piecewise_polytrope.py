import matplotlib.pyplot as plt
import numpy as np


"""
Following construction of EoS is based on the piecewise polytrope framework.
Ref. 

Constraints on a phenomenologically parameterized neutron-star equation of state
Jocelyn S. Read (Wisconsin U., Milwaukee), Benjamin D. Lackey (Wisconsin U., Milwaukee), Benjamin J. Owen (Granada U., Theor. Phys. Astrophys. and Penn State U.), John L. Friedman (Wisconsin U., Milwaukee)
e-Print: 0812.2163 [astro-ph]
DOI: 10.1103/PhysRevD.79.124032
Published in: Phys.Rev.D 79 (2009), 124032
"""

### constants ### 
c = 2.99792458e10
G = 6.67428e-8

# Parameters of "Crust EoS" are fixed.
# Ref. J. S. Read, B. D. Lackey, B. J. Owen, and J. L. Friedman, Phys. Rev. D 79, 124032 (2009), arXiv:0812.2163 [astro-ph].
# [Name, upper_lim_of_rho [g/cm3] , Gamma, K]
fixed_crust = [["crust1", 2.44034e+07, 1.58425, 6.80110e-09],
               ["crust2",  3.78358e+11,  1.28733, 1.06186e-06],
               ["crust3",  2.62780e+12,  0.62223, 5.32697e+01],
               ["crust4",  None,         1.35692, 3.99874e-08]]

def p_rho_def(rho, K, Gamma):  # return p [g/cm3]
    return K*np.power(rho, Gamma)

def eps_rho_def(rho, K, Gamma, a):
    return (1+a)*rho + K*np.power(rho, Gamma)/(Gamma - 1)

def a_def(pre_eps_lim, pre_rho_lim, K, Gamma):
    return pre_eps_lim/pre_rho_lim - 1 - K*np.power(pre_rho_lim, Gamma - 1)/(Gamma-1)

def next_K_def(p_lim, rho_lim, next_Gamma):
    return p_lim/np.power(rho_lim, next_Gamma)

def calc_K1_rhob(log_p1, Gamma1, K_crust=3.99874e-08, Gamma_crust=1.35692): 
    # p1 [dyn/cm2]
    # rho_b : density [g/cm3] at the boundary btw. "Crust EoS" and "Inner EoS."
    K1 = np.power(10, log_p1)/c**2/np.power(10**14.7, Gamma1) # p1's unit is changed [dyn/cm2] -> [g/cm3] by 1/c**2.
    rho_b = np.power(K_crust/K1, 1/(Gamma1-Gamma_crust))
    return K1, rho_b

def calc_a_crust(param_crust = fixed_crust):
    a_crust_list = []
    for i in range(len(param_crust)):
        if i == 0:
            rho_1 = np.logspace(1, np.log10(param_crust[i][1]), 3)
        elif (i != (len(param_crust)-1)):
            rho_1 = np.logspace(np.log10(param_crust[i-1][1]), np.log10(param_crust[i][1]), 3)
        else:
            rho_1 = np.logspace(np.log10(param_crust[i-1][1]), 14.7, 3) # 10**14.7 is temporal upper lim.
        
        if i==0: 
            a_temp = 0
            a_crust_list.append(a_temp)
        else: 
            a_temp = a_def(eps_crust1[-1], param_crust[i-1][1], param_crust[i][3], param_crust[i][2])
            a_crust_list.append(a_temp)
    
        eps_crust1 = eps_rho_def(rho_1, param_crust[i][3],  param_crust[i][2], a_crust_list[i])
    return a_crust_list

def param_of_innerEoS(log_p1, Gamma, param_c=fixed_crust):
    K1, rhob = calc_K1_rhob(log_p1, Gamma[0]) # K1 and rhob are determined.
    rho_lim_list = [10**14.7, 10**15, 10**19] # rho range is [rhob, 10**16. rhob is added at last.
    
    K_list = [K1]
    a_list = []

    for i in range(len(Gamma)):
        if i==0:
            rho_piece = np.logspace(np.log10(rhob), np.log10(rho_lim_list[i]), 10000)
            K_crust_end = param_c[3][3]
            Gamma_crust_end = param_c[3][2]
            epsb = eps_rho_def(rhob, K_crust_end, Gamma_crust_end, calc_a_crust(param_crust=param_c)[-1])
            a_temp = a_def(epsb, rhob, K_list[i], Gamma[i])
            a_list.append(a_temp)
        else:
            rho_piece = np.logspace(np.log10(rho_lim_list[i-1]), np.log10(rho_lim_list[i]), 10000) 
            # 10**rho_lim_list[i] is included
            a_temp = a_def(eps_temp[-1], rho_lim_list[i-1], K_list[i], Gamma[i])
            a_list.append(a_temp) 
        
        p_temp = p_rho_def(rho_piece, K_list[i], Gamma[i])
        eps_temp = eps_rho_def(rho_piece, K_list[i],  Gamma[i], a_list[i])
    
        if (i+1 < len(Gamma)):
            K_temp = next_K_def(p_temp[-1], rho_piece[-1], Gamma[i+1])
            K_list.append(K_temp)
    
    rho_lim_list.insert(0, rhob) 
    return rho_lim_list, a_list,  K_list # Note that rho_lim_list[0] = rhob.

def joint_params(rho_lim_l, a_l, K_l, par_crust=fixed_crust):
    # Joint lists of parameters.
    rho_lim_all = np.append([rho[1] for rho in par_crust[:-1]], rho_lim_l)
    a_all = np.append(calc_a_crust(param_crust=par_crust), a_l) 
    K_all = np.append([k[3] for k in par_crust], K_l)
    return rho_lim_all, a_all, K_all

def get_all_params(log_p1, Gamma, p_c=fixed_crust):
    # All parameters (including Gamma) are returned.
    rho, a, k = param_of_innerEoS(log_p1, Gamma)
    rho_lim_all, a_all, K_all = joint_params(rho, a, k)
    Gamma_all = np.append([g[2] for g in p_c], Gamma)
    return rho_lim_all, a_all, K_all, Gamma_all


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    print("The calculation of one test case.")
    R, A, K, G = get_all_params(34.384, [3.005, 2.988, 2.851])
    print(R)
    print(A)
    print(K)
    print(G)

    p_check = []
    eps_check = []
    
    for i in range(len(G)):
        if i==0:
            rho_piece =  np.logspace(1, np.log10(R[i]), 1000)
        else:
            rho_piece = np.logspace(np.log10(R[i-1]), np.log10(R[i]), 1000) 
        # 10**rho_lim_list[i] is included
        
        p_temp = p_rho_def(rho_piece, K[i], G[i])
        eps_temp = eps_rho_def(rho_piece, K[i],  G[i], A[i])
        
        for j in range(len(rho_piece)):
            p_check.append(p_temp[j])
            eps_check.append(eps_temp[j])
    
    plt.scatter(np.log10(eps_check), np.log10(np.array(p_check)*c**2), s=0.1)
    plt.xlabel("energy density [g/cm^3]")
    plt.ylabel("pressure [dyn/cm^2]")
    plt.grid()
    plt.show()
    


