# Python packages
import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from scipy.constants import pi
from scipy.integrate import odeint, ode
from matplotlib import pyplot
from scipy import optimize
from itertools import repeat
from scipy.interpolate import interp1d
import TOVsolver.constant as constant
import copy

c = constant.c
G = constant.G
Msun = constant.Msun

dyncm2_to_MeVfm3 = constant.dyncm2_to_MeVfm3
gcm3_to_MeVfm3 = constant.gcm3_to_MeVfm3
oneoverfm_MeV = constant.oneoverfm_MeV


import TOVsolver.constant as constant

def Debug_readout(array, file_path='/Users/mnmgchjsk/JupyterLab/EoS_inference-main/debug.txt'):
    with open(file_path, mode='a') as f:
        f.writelines([str(val) + " "  for val in array])
        f.write("\n")


def pressure_adind(P, epsgrid, presgrid):    # return は無次元
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] *  np.power(P / presgrid[0], 3. / 5.)
        adind = 5. / 3. * presgrid[0] * np.power(eds / epsgrid[0], 5. / 3.) * 1. / eds * (eds + P) / P
        # print(f"adind idx, P = {idx}, {P}")
    if idx == len(presgrid):
        eds = epsgrid[-1] * np.power(P / presgrid[-1], 3. / 5.)
        adind = 5. / 3. * presgrid[-1] * np.power(eds / epsgrid[-1], 5. / 3.) * 1. / eds * (eds + P) / P
        # print(f"adind idx, P = {idx}, {P}")
    else:
        ci = np.log(presgrid[idx]/presgrid[idx-1]) / np.log(epsgrid[idx]/epsgrid[idx-1])
        eds = epsgrid[idx-1] * np.power(P / presgrid[idx-1], 1. / ci)
        adind = ci * presgrid[idx-1] * np.power(eds / epsgrid[idx-1], ci) * 1. / eds * (eds + P) / P
    return adind, eds


def point_Cs(P, epsgrid, presgrid):
    idx_l = np.searchsorted(presgrid, P, side="left")
    idx_r = np.searchsorted(presgrid, P, side="right")
    if idx_l != idx_r or idx_l-1 <= 0 or idx_r+1 >= len(presgrid):
        ad_idx, E = pressure_adind(P, epsgrid, presgrid)
        Cs = ad_idx*P/(P+E)
        print(f"### weak derivative, (idx_l, idx_r, P) = ({idx_l}, {idx_r}, {P})")
    else:
        Cs = (presgrid[idx_r+1]-presgrid[idx_l])/(epsgrid[idx_r+1]-epsgrid[idx_l])
    return Cs


def TOV(r, y, inveos):
    pres, m = y
    
    #eps = 10**inveos(np.log10(pres))
    eps = inveos(pres)
    dpdr = -(eps + pres) * (m + 4.*pi*r**3. * pres)
    dpdr = dpdr/(r*(r - 2.*m))
    dmdr = 4.*pi*r**2.0 * eps
    
    return np.array([dpdr, dmdr])

def TOV_def(r, y,inveos, f_eps, adind, Debug=False, dhr=1000):

    pres, m,h,b = y
    
    #energy_density = 10**inveos(np.log10(pres))
    eps = inveos(pres)
    dpdr = -(eps + pres) * (m + 4.*pi*r**3. * pres)
    dpdr = dpdr/(r*(r - 2.*m))
    dmdr = 4.*pi*r**2.0 * eps
    dhdr = b

    # dfdr の計算を書き加える
    f_poly= (eps+pres)/(pres*adind)
    h_p = dpdr*dhr
    if f_eps==None:
        f =  (inveos(pres+h_p)-inveos(pres-h_p))/(2*h_p)
    else:
        f = f_eps(eps)
    
    #print(f"err: {f_poly}, {f}")
    # f = deps/dp

    dbdr = 2. * np.power(1. - 2. * m / r, -1) * h * \
        (-2. * np.pi * (5. * eps + 9. * pres + f*(eps + pres)) + 3. / np.power(r,2) + 2. *
            np.power(1. - 2. * m / r,-1) * np.power(m / np.power(r,2) +
         4. * np.pi * r * pres,2)) \
        + 2. * b / r * np.power(1. - 2. * y[1] / r, -1) * \
        (-1. + m / r + 2. * np.pi * np.power(r,2.) * (eps - pres))

    if Debug == True:
        Debug_readout([eps, pres , r, m, h, b, adind, f])

    return np.array([dpdr, dmdr, dhdr, dbdr])

def tidal_deformability(y2, Mns, Rns):
    C = Mns / Rns
    Eps = 4. * C**3. * (13. - 11. * y2 + C * (3. * y2 - 2.) + 2. * C**2. * (1. + y2)) + \
        3. * (1. - 2. * C)**2. * (2. - y2 + 2. * C * (y2 - 1.)) * \
        np.log(1. - 2. * C) + 2. * C * (6. - 3. * y2 + 3. * C * (5. * y2 - 8.))
    
    tidal_def = 16. / (15. * Eps) * (1. - 2. * C)**2. *(2. + 2. * C * (y2 - 1.) - y2)

    #print(f"(Eps, y, tidal) = ({Eps:.1e}, {y2:.1e}, {tidal_def:.1e})")

    return tidal_def

# Function solves the TOV equation, returning mass and radius
def solveTOV_tidal(center_rho, energy_density, pressure, flist=[-1], CutP=0., Debug=False, delta_r_max=1e3):
    c = constant.c
    G = constant.G
    Msun = constant.Msun
    unique_pressure_indices = np.unique(pressure, return_index=True)[1]
    unique_pressure = pressure[unique_pressure_indices]

# Interpolate pressure vs. energy density
    eos = interp1d(energy_density[unique_pressure_indices], unique_pressure, kind='cubic', fill_value='extrapolate')
    inveos = interp1d(unique_pressure, energy_density[unique_pressure_indices], kind='cubic', fill_value='extrapolate')
    if flist[0]==-1:
        f_func = None
    else:
        f_func = interp1d(energy_density[unique_pressure_indices], \
                    flist[unique_pressure_indices], kind='cubic', fill_value='extrapolate')
    

    r = 4.441e-16
    dr = 10
    rhocent = center_rho * G*np.power(c,-2.)
    pcent = eos(rhocent)
    P0 = pcent - (2.*pi/3.)*(pcent + rhocent) *(3.*pcent + rhocent)*np.power(r,2.)
    if (pcent-P0)/pcent != 0.0:
        print("P0 is different from Pcent")
    # Pmin = max([min([pressure[1],P0/1e12]), CutP])
    Pmin = max([P0/1e12, CutP])
    m0 = 4./3. *pi *rhocent*np.power(r,3.)
    h0 = np.power(r,2.)
    b0 = 2. * r
    stateTOV = np.array([P0, m0, h0, b0])
    #C_s = point_Cs(P0, energy_density[unique_pressure_indices], unique_pressure)
    ad_index, _ = pressure_adind(P0, energy_density[unique_pressure_indices], unique_pressure)
    dhr = dr

    sy = ode(TOV_def, None).set_integrator('dopri5')     # dop853 もともと dopri5
    #have been modified from Irida to this integrator
    sy.set_initial_value(stateTOV, r).set_f_params(inveos, f_func, ad_index, Debug, dhr)
    while sy.successful():   # ただしくは ad_indexも更新すべき
        stateTOV_temp = sy.integrate(sy.t+dr)
        if stateTOV_temp[0] >= Pmin:
            stateTOV = copy.copy(stateTOV_temp)
        else:
            break
        ad_index, _ = pressure_adind(stateTOV[0], energy_density[unique_pressure_indices], unique_pressure)
        dpdr, dmdr, _, __= TOV_def(sy.t, stateTOV, inveos, f_func, ad_index, Debug, dr)
        # 2024/11/26 sy.t +dr を sy.t に書き換えた. sy.t は sy.integrate(sy.t+dr) で更新されている
        # print(dpdr*c**2/G*gcm3_to_MeVfm3)
        dr = min(0.46 * np.power((1./stateTOV[1] * dmdr - 1./stateTOV[0]*dpdr), -1.), delta_r_max)
        # dr = delta_r_max
        sy.set_f_params(inveos, f_func, ad_index, Debug, dr)
        
    if sy.successful() == False:
        print(f"Pmin cond: {stateTOV_temp[0] < Pmin}")

            
                
    
    Mb = stateTOV[1]
    Rns = sy.t
    # 天体表面で非ゼロのエネルギーを持つ時に補正項を加える
    # B は [g/cm^3]の単位の値に G/c**2 をかけたものを用いる 
    y = Rns * stateTOV[3] /stateTOV[2] - 4.*pi*np.power(Rns,3.)*inveos(stateTOV[0])/Mb
    print(inveos(stateTOV[0])*c**2/G*gcm3_to_MeVfm3, stateTOV[0]*c**2/G*gcm3_to_MeVfm3)
    # p(R) = p_c * 1e-12 となるまで積分しないといけない. 確認すること!

    # Debug_readout([Mb/Rns, y, Rns*stateTOV[3]/stateTOV[2]],file_path="/Users/mnmgchjsk/JupyterLab/EoS_inference-main/C[]_y[].txt")

    tidal = tidal_deformability(y, Mb, Rns)
    return Mb*c**2./G/Msun, Rns/1e5, tidal


def solveTOV(center_rho, energy_density, pressure, Pmin_idx=20):
    #eos = UnivariateSpline(np.log10(energy_density), np.log10(pres), k=1, s=0)
    #inveos = UnivariateSpline(np.log10(pres), np.log10(energy_density), k=1, s=0)
    #We could change this to Double Log Interpolation。
    c = constant.c
    G = constant.G
    Msun = constant.Msun

    unique_pressure_indices = np.unique(pressure, return_index=True)[1]
    unique_pressure = pressure[np.sort(unique_pressure_indices)]

# Interpolate pressure vs. energy density
    eos = interp1d(energy_density, pressure, kind='cubic', fill_value='extrapolate')

# Interpolate energy density vs. pressure
    inveos = interp1d(unique_pressure, energy_density[unique_pressure_indices], kind='cubic', fill_value='extrapolate')

    Pmin = pressure[Pmin_idx]
    r = 4.441e-16
    dr = h
    center_rho = center_rho * G/c**2.
    
    #pcent = 10**eos(np.log10(rhocent))
    pcent = eos(center_rho)
    P0 = pcent - (2.*pi/3.)*(pcent + center_rho) *(3.*pcent + center_rho)*r**2.
    m0 = 4./3. *pi *center_rho*r**3.
    stateTOV = np.array([P0, m0])
    
    sy = ode(TOV, None).set_integrator('dopri5')
    
  
    #have been modified from Irida to this integrator
    sy.set_initial_value(stateTOV, r).set_f_params(inveos)
    
    while sy.successful() and stateTOV[0]>Pmin:
        stateTOV = sy.integrate(sy.t+dr)
        dpdr, dmdr = TOV(sy.t+dr, stateTOV, inveos)
        dr = 0.46 * (1./stateTOV[1] * dmdr - 1./stateTOV[0]*dpdr)**(-1.)  

    return stateTOV[1]*c**2./G/Msun, sy.t/1e5  
