import numpy as np
from TOVsolver.constant import c,G
from TOVsolver.solver_code import  pressure_adind

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import optimize

def speed_of_sound_calc(density, pressure):

    speed_of_sound = []
    #density = density*c**2/G
    #pressure = pressure*c**4/G
    
    for i in range(0,len(density)-1):
        speed_of_sound.append((pressure[i+1]-pressure[i])/(density[i+1]-density[i]))
    d2 = []
    C_s= []
    #eps2 = []
    for i in range(0,len(speed_of_sound)):
        if density[i]> 1.5e-14:
            d2.append(density[i])
            C_s.append(speed_of_sound[i])
    return C_s,d2

def f_calc(density, pressure): #gcm3 G/c2, press G/c4
    f_val1 = np.array([])
    for i in range(0, len(pressure)-1):
        if (pressure[i+1]-pressure[i]) !=0:
            f_val1 = np.append(f_val1, (density[i+1]-density[i])/(pressure[i+1]-pressure[i]))
        else:
            adind, eds = pressure_adind(pressure[i], density, pressure)
            f_val1 = np.append(f_val1, adind*pressure[i]/(pressure[i]+density[i])) 
    p2 = np.array([])
    f_val2 = np.array([])
    for i in range(0, len(f_val1)):
        if density[i] > 1.5e-14:
            p2 = np.append(p2, pressure[i])
            f_val2 = np.append(f_val2, f_val1[i])
    return f_val2, p2


def intersection_forMonoM(mono_RM_list1, mono_RM_list2):

    fig, ax = plt.subplots(1,1)
    ax.plot(mono_RM_list1[1], mono_RM_list1[0])
    ax.plot(mono_RM_list2[1], mono_RM_list2[0])
    ax.grid()
    ax.set_title("Check uniquness of R(M)")
    ax.set_xlabel("M")
    ax.set_ylabel("R")
    fig.tight_layout()
    fig.show()
    
    unique_M1_idx = np.unique(mono_RM_list1[1], return_index=True)[1]
    unique_M1 = mono_RM_list1[1][unique_M1_idx]
    unique_R1 = mono_RM_list1[0][unique_M1_idx]
    R_M_func1 = UnivariateSpline(unique_M1, unique_R1, s=0, k=3)
    
    unique_M2_idx = np.unique(mono_RM_list2[1], return_index=True)[1]
    unique_M2 = mono_RM_list2[1][unique_M2_idx]
    unique_R2 = mono_RM_list2[0][unique_M2_idx]
    R_M_func2 = UnivariateSpline(unique_M2, unique_R2, s=0, k=3)

    f = lambda m: R_M_func1(m) - R_M_func2(m)
    m_intersect = optimize.newton(f, 0.1)
    print(f"(R, M) : ({(R_M_func1(m_intersect)+R_M_func1(m_intersect))/2}, {m_intersect})")
    return m_intersect

def calc_dist_inLambda(M_intersect, mono_MT_list1, mono_MT_list2):
    fig, ax = plt.subplots(1,1)
    ax.plot(mono_MT_list1[0], mono_MT_list1[1])
    ax.plot(mono_MT_list2[0], mono_MT_list2[1])
    ax.plot([M_intersect, M_intersect],[1, 1e6])
    ax.grid()
    ax.set_ylim(1, 1e6)
    ax.set_title("Check uniquness of Lambda(M)")
    ax.set_xlabel("M")
    ax.set_ylabel("T")
    ax.semilogy()
    fig.tight_layout()
    fig.show()

    unique_M1_idx = np.unique(mono_MT_list1[0], return_index=True)[1]
    unique_M1 = mono_MT_list1[0][unique_M1_idx]
    unique_T1 = mono_MT_list1[1][unique_M1_idx]
    TofM_func1 = UnivariateSpline(unique_M1, unique_T1, s=0, k=3)
    
    unique_M2_idx = np.unique(mono_MT_list2[0], return_index=True)[1]
    unique_M2 = mono_MT_list2[0][unique_M2_idx]
    unique_T2 = mono_MT_list2[1][unique_M2_idx]
    TofM_func2 = UnivariateSpline(unique_M2, unique_T2, s=0, k=3)
    
    print(TofM_func1(M_intersect), TofM_func2(M_intersect))
    
    return abs(TofM_func1(M_intersect) - TofM_func2(M_intersect))
