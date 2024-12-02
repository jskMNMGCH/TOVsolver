# Python packages
import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from scipy.constants import pi
from scipy.integrate import odeint, ode
from matplotlib import pyplot
from scipy import optimize
from itertools import repeat
import csv


# Import files
import TOVsolver.solver_code as TOV_solver
import TOVsolver.EoS_import as EoS_import
import TOVsolver.speed_of_sound as speed_of_sound
import TOVsolver.constant as constant
import bisect

def OutputMRT(input_file='',density=[],pressure=[], f=[-1], CutPress=0.0, rho_min=14.2, rho_max=15.6, rho_num=50, dr_max=1e3, Flag=False):
    
    """Outputs the mass, radius, and tidal deformability
    Args:
        central_density (float): central density that we want to compute
        density (array, optional): numpy 1Darray. Density of EoS
        pressure (array, optional): numpy 1Darray. pressure of EoS

    Returns:
        MRT: 2D array (rho_nym x 3) with mass, radius and tidal.
    """

    c = constant.c
    G = constant.G
    Msun = constant.Msun

    dyncm2_to_MeVfm3 = constant.dyncm2_to_MeVfm3
    gcm3_to_MeVfm3 = constant.gcm3_to_MeVfm3
    oneoverfm_MeV = constant.oneoverfm_MeV
    #############This is something we need to change, like the input for this EOS import should
    ############# be one file contatining Whole EOS. that first column is density and second is pressure
    energy_density, pressure = EoS_import.EOS_import(input_file,density,pressure)
    ############# Lets the user only input the EOS file path, then this EOS_import should have file
    ############# as input. and the outputMR should have a file as input too?
    
    Radius = []
    Mass = []
    tidal = []
    density = np.logspace(rho_min, rho_max , rho_num)

    for i in range(len(density)):
        try:
            solver = TOV_solver.solveTOV_tidal(density[i], energy_density, pressure, flist=f, CutP=CutPress,Debug=Flag, delta_r_max=dr_max)
            Radius.append(solver[1])
            #g/cm^3, g/cm^3 G/c^2, dyn/cm^2 G/c^4
            Mass.append(solver[0])
            
            tidal.append(solver[2])
    #This is sentense is for avoiding the outflow of the result, like when solveTOV blow up because of ill EOS, we need to stop
        except OverflowError as e:
            print("This EOS is ill-defined to reach an infinity result, that is not phyiscal.")
            break
    MRT = np.vstack((Radius, Mass,tidal)).T
    return MRT

def OutputC_s(input_file='',density=[],pressure=[]):

    energy_density, pressure = EoS_import.EOS_import(input_file,density,pressure)
    C_s = speed_of_sound.speed_of_sound_calc(energy_density, pressure)
    return C_s


def OutputMRTpoint(central_density,energy_density,pressure):

    """Outputs the mass, radius, and tidal deformability (single point)
    Args:
        central_density (float): central density that we want to compute
        density (array, optional): numpy 1Darray. Density of EoS
        pressure (array, optional): numpy 1Darray. pressure of EoS

    Returns:
        MRT (tuple): tuple with mass, radius and tidal.
    """

    c = constant.c
    G = constant.G
    Msun = constant.Msun

    dyncm2_to_MeVfm3 = constant.dyncm2_to_MeVfm3
    gcm3_to_MeVfm3 = constant.gcm3_to_MeVfm3
    oneoverfm_MeV = constant.oneoverfm_MeV
    
    Radius = []
    Mass = []
    tidal = [] 
    Pmin_idx = np.max([20, bisect.bisect_right(pressure, 0)])
#This following step is to make a dicision whether the EOS ingredients is always increase. We can do that outsie of this main to the 
#EOS import.
#if   all(x<y for x, y in zip(eps_total_poly[:], eps_total_poly[[1:])) and all(x<y for x, y in zip(pres_total_poly[j][:], pres_total_poly[j][1:])):
    try:
        Radius.append(TOV_solver.solveTOV_tidal(central_density, energy_density, pressure, Pmin_idx)[1])
        Mass.append(TOV_solver.solveTOV_tidal(central_density, energy_density, pressure, Pmin_idx)[0])
        tidal.append(TOV_solver.solveTOV_tidal(central_density, energy_density, pressure, Pmin_idx)[2])
    #This is sentense is for avoiding the outflow of the result, like when solveTOV blow up because of ill EOS, we need to stop
    except OverflowError as e:
        print("This EOS is ill-defined to reach an infinity result, that is not phyiscal, No Mass radius will be generated.")
    MRT = np.vstack((Radius, Mass,tidal)).T
        
    return MRT

