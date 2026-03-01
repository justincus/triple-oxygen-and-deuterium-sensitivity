# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:52:52 2024

@author: mcustado

This script generates the sensitivity plots in "Contrasting controls on δ18O-δ2H and δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)

Variable names:
    d = small delta (δ)
    D = big delta (Δ)
    
    [sample type]_dX
    where sample type are:
        inlet = measurements from the inlet canal
        creek = measurements from the surrounding creeks
        gw = measurements from groundwater and springs
        prec = measurements from preciptiation samples from UVU and USU stations (Utah Valley University and Utah State University)
        lake = measurements from Bear Lake (Mud Lake data not included in calculations)
        influx = total volume inflow to the lake 
        atm = atmospheric moisture (estimated)
        
    Other abbreviations:
        hum = humidity
        temp = temperature
        x = Xe (evaporation-to-inflow ratio)
        mo = month
        evap = evaporation flux
        amt = amount
        dist = distribution
        unc = uncertainty
        d_exc = d-excess
        
"""

import numpy as np
import matplotlib.pyplot as plt
import lake_balance_functions_R1 as lbf
import scipy.optimize as opt

#%% Input initial parameters 

# Input initial humidity (h), Xe (x_1), and temperature (temp)
# Median values are from generated distributions from the initial Monte Carlo simulations (custado_et_al_2025_mc_output_R1)

hum_ = np.mean([0.639371718367312, 0.589719446183092]) # calculated median values from both isotope systems
x_1 = np.mean([0.280514283927974, 0.266944086701599]) # calculated median values from both isotope systems
temp = 11.1243939237244

influx_d18O = -15.6033324979121 
influx_D17O = 19.2495896779815 
influx_dD = -117.48115843174 

atm_d18O = -21.5468199522826
atm_d17O = -11.4103579229986
atm_D17O = 24.4846044309685
atm_dD = -163.229226267567
atm_d_exc = 9.29114922289076

lake_d18O = -9.09549314428604
lake_D17O = -2.52476395920369
lake_dD = -84.7676704647129

# Provide input uncertanties 
# IQR values from generated distributions

hum_unc = np.mean([0.333536999709198, 0.399172780793854]) # Calculated IQR values from both isotope systems
x_1_unc = np.mean([0.0803139246362657, 0.0772361231020387]) # Calculated IQR values from both isotope systems
temp_unc = 10.9833880354573

influx_d18O_unc = 1.24032076219686
influx_D17O_unc = 10.4173393339832
influx_dD_unc = 9.42233971742214

atm_d18O_unc = 4.59032012710661
atm_d17O_unc = 2.44759483703044
atm_D17O_unc = 29.7575813924333
atm_dD_unc = 46.8244890562962
atm_d_exc_unc = 31.5468345768355

lake_d18O_unc = 0.189001353922414
lake_D17O_unc = 12.3133273870608
lake_dD_unc = 0.781547955177273

#%% Generate input distributions

# Function to truncate generated distributions within 15.9 and 84.1 percentile range, removing outputs >2 standard deviations
def truncate_dist(a0):
    mask = (a0 >= np.percentile(a0, 15.9)) & (a0 <= np.percentile(a0, 84.1))

    a = a0[mask]
    return a

# Input # of simulations

sim = 100000

# Generate distributions

hum_dist_ = np.random.default_rng().normal(hum_, hum_unc, sim)
hum_dist = truncate_dist(hum_dist_)

temp_dist_ = np.random.default_rng().normal(temp, temp_unc, sim)
temp_dist = truncate_dist(temp_dist_)

influx_d18O_dist_ = np.random.default_rng().normal(influx_d18O, influx_d18O_unc, sim)
influx_d18O_dist = truncate_dist(influx_d18O_dist_)

influx_D17O_dist_ = np.random.default_rng().normal(influx_D17O, influx_D17O_unc, sim) 
influx_D17O_dist = truncate_dist(influx_D17O_dist_)

influx_dD_dist_ = np.random.default_rng().normal(influx_dD, influx_dD_unc, sim) 
influx_dD_dist = truncate_dist(influx_dD_dist_)

lake_d18O_dist_ = np.random.default_rng().normal(lake_d18O, lake_d18O_unc, sim)
lake_d18O_dist = truncate_dist(lake_d18O_dist_)

lake_D17O_dist_ = np.random.default_rng().normal(lake_D17O, lake_D17O_unc, sim) 
lake_D17O_dist = truncate_dist(lake_D17O_dist_)

lake_dD_dist_ = np.random.default_rng().normal(lake_dD, lake_dD_unc, sim) 
lake_dD_dist = truncate_dist(lake_dD_dist_)

atm_d18O_dist_ = np.random.default_rng().normal(atm_d18O, atm_d18O_unc, sim) 
atm_d18O_dist = truncate_dist(atm_d18O_dist_)

atm_D17O_dist_ = np.random.default_rng().normal(atm_D17O, atm_D17O_unc, sim) 
atm_D17O_dist = truncate_dist(atm_D17O_dist_)

atm_d17O_dist_lnrz = atm_D17O_dist/1000 + 0.528* lbf.lnrz_d(atm_d18O_dist) # Calculate d17O from D17O data
atm_d17O_dist = 1000*(np.exp(atm_d17O_dist_lnrz/1000)-1) # Delinearize calculated d17O

atm_d_exc_dist_ = np.random.default_rng().normal(atm_d_exc, atm_d_exc_unc, sim) 
atm_d_exc_dist = truncate_dist(atm_d_exc_dist_)
atm_dD_dist = atm_d_exc_dist + 8*atm_d18O_dist # Calculate d-excess 

# Set up color scheme
col = ['#8da0cb', '#084594', 'orange']

#%% Define functions

def calc_lake_17(vars, h_, x_, temp_, influx_d18O_, influx_D17O_, atm_d18O_, atm_d17O_): # Input variables, in order: humidity, Xe, temperature, d18O of inflow, D17O of inflow, d18O of atmosphere, d17O of atmosphere)
    lake_R18O, lake_R17O = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_17_ = lbf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
    a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

    # Calculate isotopic composition of atmosphere
    
    # Calculate d17 from measured D17 data
    influx_d17O_ = influx_D17O_/1000 + 0.528*lbf.lnrz_d(influx_d18O_) # get d17 influx from D17 and d18
    i_17 = 1000*(np.exp(influx_d17O_/1000)-1) # delinearize influx d17

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_R17O = lbf.convert_d_to_R(i_17) 
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18O = lbf.convert_d_to_R(atm_d18O_) 
    atm_R17O = lbf.convert_d_to_R(atm_d17O_)
    
    # atm_R18O = influx_R18O/a_18_ # assume ratio of inflow is in equilibrium with atmospheric
    # atm_R17O = influx_R17O/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin
    eq1 = ((a_18_*a_k_18_0*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18O))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_17_*a_k_17_0*(1-h_)*influx_R17O + a_17_*x_*h_*atm_R17O))/(a_17_*a_k_17_0*(1-h_)*(1-x_)+x_) - lake_R17O

    return np.array([eq1, eq2])

def calc_lake_18(vars, h_, x_, temp_, influx_d18O_, influx_D_, atm_d18O_, atm_dD_):
    lake_R18O, lake_RD = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_D_ = lbf.fractionation_factor_dD(temp_) 
    
    a_k_18_0 = 1.0285**0.5 # Kinetic fractionation at zero humidity # Merlivat 1978
    a_k_D_0 = 1.0251**0.5 # Kinetic fractionation at zero humidity # Merlivat 1978

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_RD = lbf.convert_d_to_R(influx_D_) 
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18O = lbf.convert_d_to_R(atm_d18O_) 
    atm_RD = lbf.convert_d_to_R(atm_dD_)
    
    # Ra_18 = Ri_18/a_18_ # assume ratio of inflow is in equilibrium with atmospheric
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin (2021)
    eq1 = ((a_18_*a_k_18_0*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18O))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_D_*a_k_D_0*(1-h_)*influx_RD + a_D_*x_*h_*atm_RD))/(a_D_*a_k_D_0*(1-h_)*(1-x_)+x_) - lake_RD
    
    return np.array([eq1, eq2])

def calc_hum_17(vars, lake_d18O_, lake_D17O_, temp_, influx_d18O_, influx_D17O_, atm_d18O_, atm_d17O_):
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_17_ = lbf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
    a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

    # Calculate isotopic composition of atmosphere
    
    # Calculate d17 from measured D17 data
    influx_d17O_ = influx_D17O_/1000 + 0.528*lbf.lnrz_d(influx_d18O_) # get d17 influx from D17 and d18
    i_17 = 1000*(np.exp(influx_d17O_/1000)-1) # delinearize influx d17
    
    lake_d17O_ = lake_D17O_/1000 + 0.528*lbf.lnrz_d(lake_d18O_) # get d17 lake from D17 and d18
    l_17 = 1000*(np.exp(lake_d17O_/1000)-1) # delinearize lake d17

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_R17O = lbf.convert_d_to_R(i_17) 
    
    lake_R18O = lbf.convert_d_to_R(lake_d18O_)
    lake_R17O = lbf.convert_d_to_R(l_17)
    
    # ep_18 = (a_18_ - 1)*1000
    # ep_17 = (a_17_ - 1)*1000
    
    # atm_d18O_ = lbf.isotope_atm(atm_d18O_, ep_18, 1) 
    # atm_d17O_ = lbf.isotope_atm(atm_d17O_, ep_17, 1) 
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18O = lbf.convert_d_to_R(atm_d18O_) 
    atm_R17O = lbf.convert_d_to_R(atm_d17O_)
    
    # Ra_18 = Ri_18/a_18_ # assume ratio of inflow is in equilibrium with atmospheric
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin
    eq1 = ((a_18_*a_k_18_0*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18O))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_17_*a_k_17_0*(1-h_)*influx_R17O + a_17_*x_*h_*atm_R17O))/(a_17_*a_k_17_0*(1-h_)*(1-x_)+x_) - lake_R17O

    return np.array([eq1, eq2])

def calc_hum_18(vars, lake_d18O_, lake_dD_, temp_, influx_d18O_, influx_D_, atm_d18O_, atm_dD_):
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_D_ = lbf.fractionation_factor_dD(temp_) 
    
    a_k_18_0 = 1.0285**0.5 # Kinetic fractionation at zero humidity # Merlivat 1978
    a_k_D_0 = 1.0251**0.5 # Kinetic fractionation at zero humidity # Merlivat 1978

    # Convert d values to R
    influx_R18O = lbf.convert_d_to_R(influx_d18O_) 
    influx_RD = lbf.convert_d_to_R(influx_D_) 
    
    lake_R18O = lbf.convert_d_to_R(lake_d18O_)
    lake_RD = lbf.convert_d_to_R(lake_dD_)
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    atm_R18O = lbf.convert_d_to_R(atm_d18O_) 
    atm_RD = lbf.convert_d_to_R(atm_dD_)
    
    # Calculate R of atmoshperic moisture; assume ratio of inflow is in equilibrium with atmospheric
    # Ra_18 = Ri_18/a_18_ 
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin (2021)
    eq1 = ((a_18_*a_k_18_0*(1-h_)*influx_R18O + a_18_*x_*h_*atm_R18O))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - lake_R18O
    eq2 = ((a_D_*a_k_D_0*(1-h_)*influx_RD + a_D_*x_*h_*atm_RD))/(a_D_*a_k_D_0*(1-h_)*(1-x_)+x_) - lake_RD
    
    return np.array([eq1, eq2])

#%% Test sensitivity of lake isotopic composition in D17-d18 space to all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)

x_array = [0, 0.3, 1] # Xe values to calculate for

# Test sensitivity for all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)

# Set up array of functions containing which parameters to test
# Ignore code analysis error; variables (x_ind) will be defined in the loop function

labeled_arg_sets_lake17 = [
    ("test_all",     lambda ind: (hum_dist[ind], x_ind, temp_dist[ind], influx_d18O_dist[ind], influx_D17O_dist[ind], atm_d18O_dist[ind], atm_d17O_dist[ind])),
    ("just_humidity",lambda ind: (hum_dist[ind], x_ind, temp, influx_d18O, influx_D17O, atm_d18O, atm_d17O)),
    ("just_temp",   lambda ind: (hum_, x_ind, temp_dist[ind], influx_d18O, influx_D17O, atm_d18O, atm_d17O)),
    ("just_influx",  lambda ind: (hum_, x_ind, temp, influx_d18O_dist[ind], influx_D17O_dist[ind], atm_d18O, atm_d17O)),
    ("just_atm",     lambda ind: (hum_, x_ind, temp, influx_d18O, influx_D17O, atm_d18O_dist[ind], atm_d17O_dist[ind]))
]

# Loop through array of functions containing which parameters to test for sensitivity, plot each
D17O_curve_all = []

for label, args_func in labeled_arg_sets_lake17:
    
    d180_lnrz_curve = []
    D17O_curve = []

    for x_ind, c in zip(x_array, col):
    
        # Calculate and store the central curve (no uncertainty)
        initial_guess_hum = [hum_, x_1]
        curve = opt.fsolve(calc_lake_17, [1,1], args=(hum_, x_ind, temp, influx_d18O, influx_D17O, atm_d18O, atm_d17O))
        Rl_18_, Rl_17_ = curve
        dL_18_ = lbf.lnrz_R(Rl_18_)
        dL_17_ = lbf.lnrz_R(Rl_17_)
        DL_17 = dL_17_ - 0.528 * dL_18_
        
        d180_lnrz_curve.append(dL_18_)
        D17O_curve.append(DL_17 * 1000)
        # D17O_curve.append(DL_17)

        d180_lnrz_out_x = []
        D17O_out_x = []
    
        for ind in range(len(temp_dist)):
            
            print(ind)
            
            args = args_func(ind)
            sol = opt.fsolve(calc_lake_17, initial_guess_hum, args=args)
            Rl_18_, Rl_17_ = sol
            dL_18_ = lbf.lnrz_R(Rl_18_)
            dL_17_ = lbf.lnrz_R(Rl_17_)
            DL_17 = dL_17_ - 0.528 * dL_18_

            d180_lnrz_out_x.append(dL_18_)
            D17O_out_x.append(DL_17 * 1000)
            # D17O_out_x.append(DL_17)
            
        # Plot this set of simulations
        plt.scatter(d180_lnrz_out_x, D17O_out_x, s=10, color=c, edgecolor = 'white', linewidth=0.1, alpha=0.04) #edgecolors = 'black', 
        
        print(DL_17, dL_17_, dL_18_)
        
    # Plot curves for D17-d18 space
    z = np.polyfit(d180_lnrz_curve, D17O_curve, 2)
    f = np.poly1d(z)
    x_new = np.linspace(d180_lnrz_curve[0], d180_lnrz_curve[-1], 50)
    y_new = f(x_new)
    
    plt.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    plt.axhline(y=32, zorder=0, color='black', linewidth=0.5)
    plt.xlim(-18, 7)
    plt.ylim(-140, 80)

    plt.xlabel("δ'¹⁸O (‰)")
    plt.ylabel("∆'¹⁷O (per meg)")
    # plt.ylabel("∆'¹⁷O (‰)")
    plt.scatter(d180_lnrz_curve, D17O_curve, s=4, color='black')
    # plt.savefig('{path}'+label+"_lake_d17O.png", bbox_inches="tight", dpi=600)
    plt.show()
    
    D17O_curve_all.append(D17O_curve[1])
    
    print(label)
    
# plt.scatter(atm_d17O_dist, D17O_out_x, s=10, color=c, linewidth=0.1, alpha=0.3)

    
del (x_ind, label, ind)

#%% Test sensitivity of lake isotopic composition in d18-dD space to all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)

x_array = [0, 0.3, 1] # Xe values to calculate for

# Initialize output arrays
d180_curve = []
d_exc_curve = []

# Set up array of functions containing which parameters to test
# Ignore error; variables (x_ind) will be defined in the loop function
labeled_arg_sets_lake18 = [
    ("test_all",     lambda ind: (hum_dist[ind], x_ind, temp_dist[ind], influx_d18O_dist[ind], influx_dD_dist[ind], atm_d18O_dist[ind], atm_dD_dist[ind])),
    ("just_humidity", lambda ind: (hum_dist[ind], x_ind, temp, influx_d18O, influx_dD, atm_d18O, atm_dD)),
    ("just_temp", lambda ind: (hum_, x_ind, temp_dist[ind], influx_d18O, influx_dD, atm_d18O, atm_dD)),
    ("just_influx",  lambda ind: (hum_, x_ind, temp, influx_d18O_dist[ind], influx_dD_dist[ind], atm_d18O, atm_dD)),
    ("just_atm",     lambda ind: (hum_, x_ind, temp, influx_d18O, influx_dD, atm_d18O_dist[ind], atm_dD_dist[ind]))
]

# Loop through array of functions containing which parameters to test for sensitivity, plot each
for label, args_func in labeled_arg_sets_lake18:
    
    print(label)
    
    d180_curve = []
    d_exc_curve = []

    for x_ind, c in zip(x_array, col):
    
        d180_out_x = []
        d_exc_out_x = []
        
        initial_guess_hum = [hum_, x_1]    
        curve = opt.fsolve(calc_lake_18, initial_guess_hum, args = (hum_, x_ind, temp, influx_d18O, influx_dD, atm_d18O, atm_dD))
        Rl_18_, Rl_D_ = curve
        dL_18_ = lbf.lnrz_R(Rl_18_)
        dL_D_ = lbf.lnrz_R(Rl_D_)

        lake_d_exc = dL_D_ - 8*dL_18_
    
        d180_curve.append(dL_18_)
        d_exc_curve.append(lake_d_exc)
    
        for ind in range(len(temp_dist)):
            print(ind)

            args = args_func(ind)
            sol = opt.fsolve(calc_lake_18, initial_guess_hum, args=args)
            Rl_18_, Rl_D_ = sol
            dL_18_ = lbf.lnrz_R(Rl_18_)
            dL_D_ = lbf.lnrz_R(Rl_D_)
            
            lake_d_exc = dL_D_ - 8*dL_18_
            
            d180_out_x.append(dL_18_)
            d_exc_out_x.append(lake_d_exc)
            
        plt.scatter(d180_out_x, d_exc_out_x, s=10, color=c, edgecolor = 'white', linewidth=0.1, alpha=0.04) 
        plt.xlim(-20, 9)
        plt.ylim(-120, 70)
    
    plt.axhline(y=10, zorder=0, color='black', linewidth=0.5)
    plt.xlabel("δ¹⁸O (‰)")
    plt.ylabel("d-excess (‰)")
    plt.plot(d180_curve, d_exc_curve, ls='--', linewidth=0.5, color='black')
    plt.scatter(d180_curve, d_exc_curve, s=4, color='black')
    # plt.savefig('{path}'+label+"_lake_d18O.png", bbox_inches="tight", dpi=600)
    plt.show()
    
del (x_ind, label, ind)

#%% Test sensitivity of h and Xe to all parameters or individual parameters (lake, influx, atmospheric moisture)

labeled_arg_sets_hum17 = [
    ("test_all",     lambda ind: (lake_d18O_dist[ind], lake_D17O_dist[ind], temp_dist[ind], influx_d18O_dist[ind], influx_D17O_dist[ind], atm_d18O_dist[ind], atm_d17O_dist[ind])),
    ("just_temp",  lambda ind: (lake_d18O, lake_D17O, temp_dist[ind], influx_d18O, influx_D17O, atm_d18O, atm_d17O)),
    ("just_lake",    lambda ind: (lake_d18O_dist[ind], lake_D17O_dist[ind], temp, influx_d18O, influx_D17O, atm_d18O, atm_d17O)),
    ("just_influx",  lambda ind: (lake_d18O, lake_D17O, temp, influx_d18O_dist[ind], influx_D17O_dist[ind], atm_d18O, atm_d17O)),
    ("just_atm",  lambda ind: (lake_d18O, lake_D17O, temp, influx_d18O, influx_D17O, atm_d18O_dist[ind], atm_d17O_dist[ind])),

]

labeled_arg_sets_hum18 = [
    ("test_all",     lambda ind: (lake_d18O_dist[ind], lake_dD_dist[ind], temp_dist[ind], influx_d18O_dist[ind], influx_dD_dist[ind], atm_d18O_dist[ind], atm_dD_dist[ind])),
    ("just_temp",    lambda ind: (lake_d18O, lake_dD, temp_dist[ind], influx_d18O, influx_dD, atm_d18O, atm_dD)),
    ("just_lake",    lambda ind: (lake_d18O_dist[ind], lake_dD_dist[ind], temp, influx_d18O, influx_dD, atm_d18O, atm_dD)),
    ("just_influx",  lambda ind: (lake_d18O, lake_dD, temp, influx_d18O_dist[ind], influx_dD_dist[ind], atm_d18O, atm_dD)),
    ("just_atm",     lambda ind: (lake_d18O, lake_dD, temp, influx_d18O, influx_dD, atm_d18O_dist[ind], atm_dD_dist[ind]))
]

initial_guess_hum = [hum_, x_1]

for (label, args_func17), (_, args_func18) in zip(labeled_arg_sets_hum17,labeled_arg_sets_hum18):    

    x_18_dist = []
    h_18_dist = []

    x_17_dist = []
    h_17_dist = []

    for ind in range(len(temp_dist)):
        print(ind)
        
        # h and Xe in triple oxygen
        args17 = args_func17(ind)
        roots = opt.fsolve(calc_hum_17, initial_guess_hum, args=args17)
        h_17_dist.append(roots[0])
        x_17_dist.append(roots[1])
        
        # h and Xe in d18-dD
        args18 = args_func18(ind)
        roots = opt.fsolve(calc_hum_18, initial_guess_hum, args=args18)
        h_18_dist.append(roots[0])
        x_18_dist.append(roots[1]) 
    
    plt.scatter(np.array(x_18_dist), np.array(h_18_dist), color = 'blue', s=2, edgecolor='white', linewidth = 0.1, alpha=0.15, label='δ¹⁸O-δ²H')
    plt.scatter(np.array(x_17_dist), np.array(h_17_dist), color = 'red', s=2, edgecolor='white', linewidth = 0.1, alpha=0.15, label='Triple oxygen')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("Xₑ (Evaporation/Inflow)")
    plt.ylabel("h (Relative humidity)")
    plt.legend()
    # plt.savefig('{path}'+label+'_h_Xe.png', bbox_inches="tight", dpi=600)

    plt.show()
    
#%% Fig. 3c (run once h and Xe output distributions are generated)

def remove_improbable(data):
    new_list = [i for i in data if (i>=0 and i<=1)]
    return new_list

plt.scatter(np.array(x_18_dist), np.array(h_18_dist), color = '#8080ff', s=2.5, linewidth = 0.1, alpha=0.3, label='δ¹⁸O-δ²H')
plt.scatter(np.array(x_17_dist), np.array(h_17_dist), color = '#ff7676', s=2.5,  linewidth = 0.1, alpha=0.3, label='Triple oxygen')

# Error bars for d18O-dD output

x_18_dist_ = remove_improbable(x_18_dist)
h_18_dist_ = remove_improbable(h_18_dist)

x_18_median = np.median(x_18_dist_)
h_18_median = np.median(h_18_dist_)

h_18_dist_q25 = np.percentile(h_18_dist_, 25)
h_18_dist_q75 = np.percentile(h_18_dist_, 75)

x_18_dist_q25 = np.percentile(x_18_dist_, 25)
x_18_dist_q75 = np.percentile(x_18_dist_, 75)

h_18_lower = h_18_median - h_18_dist_q25   # distance downward
h_18_upper = h_18_dist_q75 - h_18_median   # distance upward

x_18_lower = x_18_median - x_18_dist_q25   # distance leftward
x_18_upper = x_18_dist_q75 - x_18_median   # distance rightward

plt.errorbar(x_18_median, h_18_median, yerr=[[h_18_lower], [h_18_upper]], xerr=[[x_18_lower], [x_18_upper]], color = 'blue', alpha=1, capsize=4, marker = 'o', linewidth=0.5)

# Error bars for triple oxygen output

x_17_dist_ = remove_improbable(x_17_dist)
h_17_dist_ = remove_improbable(h_17_dist)

x_17_median = np.median(x_17_dist_)
h_17_median = np.median(h_17_dist_)

h_17_dist_q25 = np.percentile(h_17_dist_, 25)
h_17_dist_q75 = np.percentile(h_17_dist_, 75)

x_17_dist_q25 = np.percentile(x_17_dist_, 25)
x_17_dist_q75 = np.percentile(x_17_dist_, 75)

h_17_lower = h_17_median - h_17_dist_q25   # distance downward
h_17_upper = h_17_dist_q75 - h_17_median   # distance upward

x_17_lower = x_17_median - x_17_dist_q25   # distance leftward
x_17_upper = x_17_dist_q75 - x_17_median   # distance rightward

plt.errorbar(x_17_median, h_17_median, yerr=[[h_17_lower], [h_17_upper]], xerr=[[x_17_lower], [x_17_upper]], color = 'red', alpha=1, capsize=4, marker = 'o', linewidth=0.5)

plt.xlim(0,1)
plt.ylim(0,1)

plt.xlabel("Xₑ (Evaporation/Inflow)")
plt.ylabel("h (Relative humidity)")
plt.legend()
# plt.savefig('{path}'+'fig_3c.png', bbox_inches="tight", dpi=600)

plt.show()

