# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:52:52 2024

@author: mcustado

This script generates the sensitivity plots in "Differing sensitivity of δ18O-δ2H versus δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import bear_lake_functions_v3_d17O as blf
import scipy.optimize as opt

#%% Input initial parameters 

# Input initial humidity (h), Xe (x_1), and temperature (temp)

hum = 0.62
x_1 = 0.38
temp = 11.15

influx_d18 = -15.475595964802046
influx_D17 = 19.52368567986617
influx_dD = -116.21142980800944

prec_d18_evap = -11.12584615
prec_d17_evap = -5.857917671
prec_dD_evap = -82.89143812

atm_d18 = -21.515
atm_d17 = -11.396
atm_dD = -162.774

lake_d18 = -9.097714285714286
lake_D17 = -2.93585714285714
lake_dD = -84.7865714285714

# Provide input uncertanties

unc_0 = 0.1

hum_unc = hum*unc_0
x_1_unc = abs(x_1*unc_0)
temp_unc = abs(temp*unc_0) 

# influx_d18_unc = 1.204437618927288
# influx_D17_unc = 12.842321932222596
# influx_dD_unc = 9.302312293634499

# prec_d18_unc = 0.239171769
# prec_d17_unc = 0.126446549
# prec_dD_unc = 1.832156117

# atm_d18_unc = 0.140179039578772
# atm_d17_unc = 0.0745182986081685
# atm_dD_unc = 0.965207057919828

# lake_d18_unc = 0.040
# lake_D17_unc = 2.812
# lake_dD_unc = 0.178

influx_d18_unc = abs(influx_d18*unc_0)
influx_D17_unc = abs(influx_D17*unc_0)
influx_dD_unc = abs(influx_dD*unc_0)

atm_d18_unc = abs(atm_d18*unc_0)
atm_d17_unc = abs(atm_d17*unc_0)
atm_dD_unc = abs(atm_dD*unc_0)

lake_d18_unc = abs(lake_d18*unc_0)
lake_D17_unc = abs(lake_D17*unc_0)
lake_dD_unc = abs(lake_dD*unc_0)


#%% Calculate uncertainty for the triple oxygen composition of the lake 

# Inputs: hum, temp, dX_P (evap-flux weighted). dX_S, dX_I

# Input # of simulations

sim = 50000

# Initialize arrays of input distributions
# Using random distribution

hum_dist = np.random.default_rng().uniform(hum-hum_unc, hum+hum_unc, sim)
x_1_dist = np.random.default_rng().uniform(x_1-x_1_unc, x_1+x_1_unc, sim)
temp_dist = np.random.default_rng().uniform(temp-temp_unc, temp+temp_unc, sim)

influx_d18_dist_unsorted = np.random.default_rng().uniform(influx_d18-influx_d18_unc, influx_d18+influx_d18_unc, sim)
influx_D17_dist_unsorted = np.random.default_rng().uniform(influx_D17-influx_D17_unc, influx_D17+influx_D17_unc, sim) 
influx_dD_dist = np.random.default_rng().uniform(influx_dD-influx_dD_unc, influx_dD+influx_dD_unc, sim) 

lake_d18_dist = np.random.default_rng().uniform(lake_d18-lake_d18_unc, lake_d18+lake_d18_unc, sim)
lake_D17_dist = np.random.default_rng().uniform(lake_D17-lake_D17_unc, lake_D17+lake_D17_unc, sim) 
lake_dD_dist = np.random.default_rng().uniform(lake_dD-lake_dD_unc, lake_dD+lake_dD_unc, sim) 

atm_d18_dist_unsorted = np.random.default_rng().uniform(atm_d18-atm_d18_unc, atm_d18+atm_d18_unc, sim)
atm_d17_dist_unsorted = np.random.default_rng().uniform(atm_d17-atm_d17_unc, atm_d17+atm_d17_unc, sim)
atm_dD_dist_unsorted = np.random.default_rng().uniform(atm_dD-atm_dD_unc, atm_dD+atm_dD_unc, sim)

atm_d18_dist = np.array(sorted(atm_d18_dist_unsorted)) 
atm_d17_dist = np.array(sorted(atm_d17_dist_unsorted)) 
atm_dD_dist = atm_dD_dist_unsorted

influx_d18_dist = np.array(influx_d18_dist_unsorted)
influx_D17_dist = np.array(influx_D17_dist_unsorted)

#%% Define functions

def calc_lake_17(vars, h_, x_, temp_, influx_d18_, influx_D17_, atm_d18_, atm_d17_):
    Rl_18, Rl_17 = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = blf.fractionation_factor_d18O(temp_) 
    a_17_ = blf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
    a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

    # Calculate isotopic composition of atmosphere
    
    # Calculate d17 from measured D17 data
    influx_d17_ = influx_D17_/1000 + 0.528*blf.lnrz_d(influx_d18_) # get d17 influx from D17 and d18
    i_17 = 1000*(np.exp(influx_d17_/1000)-1) # delinearize influx d17

    # Convert d values to R
    Ri_18 = blf.convert_d_to_R(influx_d18_) 
    Ri_17 = blf.convert_d_to_R(i_17) 
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    Ra_18 = blf.convert_d_to_R(atm_d18_) 
    Ra_17 = blf.convert_d_to_R(atm_d17_)
    
    # Ra_18 = Ri_18/a_18_ # assume ratio of inflow is in equilibrium with atmospheric
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin
    eq1 = ((a_18_*a_k_18_0*(1-h_)*Ri_18 + a_18_*x_*h_*Ra_18))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - Rl_18
    eq2 = ((a_17_*a_k_17_0*(1-h_)*Ri_17 + a_17_*x_*h_*Ra_17))/(a_17_*a_k_17_0*(1-h_)*(1-x_)+x_) - Rl_17

    return np.array([eq1, eq2])

def calc_lake_18(vars,  h_, x_, temp_, influx_d18_, influx_D_, atm_d18_, atm_dD_): 
    lake_d18_, lake_dD_ = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = blf.fractionation_factor_d18O(temp_) 
    a_D_ = blf.fractionation_factor_dD(temp_) 
    
    ep_18_ = (a_18_ - 1)*1000
    ep_D_ = (a_D_ - 1)*1000
    
    ep_k_18_ = blf.kinetic_en_d18O(h_) # Used for d18O-dD equations
    ep_k_D_ = blf.kinetic_en_dD(h_)
    
    eq1 = (((lake_d18_ - influx_d18_)*(1-h_+(0.001*ep_k_18_))) / (h_*(atm_d18_ - lake_d18_) + (ep_k_18_ + (ep_18_/a_18_))*(0.001*lake_d18_ + 1))) - x_
    eq2 = (((lake_dD_ - influx_D_)*(1-h_+(0.001*ep_k_D_))) / (h_*(atm_dD_ - lake_dD_) + (ep_k_D_ + (ep_D_/a_D_))*(0.001*lake_dD_ + 1))) - x_

    return np.array([eq1, eq2])

def calc_hum_17(vars, lake_d18_, lake_D17_, temp_, influx_d18_, influx_D17_, atm_d18_, atm_d17_):
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = blf.fractionation_factor_d18O(temp_) 
    a_17_ = blf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
    a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

    # Calculate isotopic composition of atmosphere
    
    # Calculate d17 from measured D17 data
    influx_d17_ = influx_D17_/1000 + 0.528*blf.lnrz_d(influx_d18_) # get d17 influx from D17 and d18
    i_17 = 1000*(np.exp(influx_d17_/1000)-1) # delinearize influx d17
    
    lake_d17_ = lake_D17_/1000 + 0.528*blf.lnrz_d(lake_d18_) # get d17 lake from D17 and d18
    l_17 = 1000*(np.exp(lake_d17_/1000)-1) # delinearize lake d17

    # Convert d values to R
    Ri_18 = blf.convert_d_to_R(influx_d18_) 
    Ri_17 = blf.convert_d_to_R(i_17) 
    
    Rl_18 = blf.convert_d_to_R(lake_d18_)
    Rl_17 = blf.convert_d_to_R(l_17)
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    Ra_18 = blf.convert_d_to_R(atm_d18_) 
    Ra_17 = blf.convert_d_to_R(atm_d17_)
    
    # Ra_18 = Ri_18/a_18_ # assume ratio of inflow is in equilibrium with atmospheric
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin
    eq1 = ((a_18_*a_k_18_0*(1-h_)*Ri_18 + a_18_*x_*h_*Ra_18))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - Rl_18
    eq2 = ((a_17_*a_k_17_0*(1-h_)*Ri_17 + a_17_*x_*h_*Ra_17))/(a_17_*a_k_17_0*(1-h_)*(1-x_)+x_) - Rl_17

    return np.array([eq1, eq2])

def calc_hum_18(vars, lake_d18_, lake_dD_, temp_, influx_d18_, influx_D_, atm_d18_, atm_dD_): 
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = blf.fractionation_factor_d18O(temp_) 
    a_D_ = blf.fractionation_factor_dD(temp_) 
    
    ep_18_ = (a_18_ - 1)*1000
    ep_D_ = (a_D_ - 1)*1000
    
    ep_k_18_ = blf.kinetic_en_d18O(h_) # Used for d18O-dD equations
    ep_k_D_ = blf.kinetic_en_dD(h_)
    
    eq1 = (((lake_d18_ - influx_d18_)*(1-h_+(0.001*ep_k_18_))) / (h_*(atm_d18_ - lake_d18_) + (ep_k_18_ + (ep_18_/a_18_))*(0.001*lake_d18_ + 1))) - x_
    eq2 = (((lake_dD_ - influx_D_)*(1-h_+(0.001*ep_k_D_))) / (h_*(atm_dD_ - lake_dD_) + (ep_k_D_ + (ep_D_/a_D_))*(0.001*lake_dD_ + 1))) - x_

    return np.array([eq1, eq2])

#%% Test sensitivity of lake isotopic composition in D17-d18 space to all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)

x_array = [0, 0.3, 1] # Xe values to calculate for

# Test sensitivity for all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)
# Ignore error; variables (x_ind) will be defined in the loop function

# Set up array of functions containing which parameters to test
labeled_arg_sets_17 = [
    ("test_all",     lambda ind: (hum_dist[ind], x_ind_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_D17_dist[ind], atm_d18_dist[ind], atm_d17_dist[ind])),
    ("just_humidity",lambda ind: (hum_dist[ind], x_ind, temp, influx_d18, influx_D17, atm_d18, atm_d17)),
    ("just_x",       lambda ind: (hum, x_ind_dist[ind], temp, influx_d18, influx_D17, atm_d18, atm_d17)),
    ("just_influx",  lambda ind: (hum, x_ind, temp, influx_d18_dist[ind], influx_D17_dist[ind], atm_d18, atm_d17)),
    ("just_atm",     lambda ind: (hum, x_ind, temp, influx_d18, influx_D17, atm_d18_dist[ind], atm_d17_dist[ind]))
]

# Loop through array of functions containing which parameters to test for sensitivity, plot each
for label, args_func in labeled_arg_sets_17:
    
    print(label)
    
    d180_lnrz_curve = []
    D17O_curve = []

    for x_ind in x_array:
    
        # Calculate and store the central curve (no uncertainty)
        initial_guess_hum = [0.9, 0.9]
        curve = opt.fsolve(calc_lake_17, initial_guess_hum, args=(hum, x_ind, temp, influx_d18, influx_D17, atm_d18, atm_d17))
        Rl_18_, Rl_17_ = curve
        dL_18_ = blf.lnrz_R(Rl_18_)
        dL_17_ = blf.lnrz_R(Rl_17_)
        DL_17 = dL_17_ - 0.528 * dL_18_
        
        d180_lnrz_curve.append(dL_18_)
        D17O_curve.append(DL_17 * 1000)
    
        d180_lnrz_out_x = []
        D17O_out_x = []
    
        for ind in range(sim):
            
            print(ind)
            
            x_ind_unc = abs(x_ind*unc_0)
            x_ind_dist = np.random.default_rng().uniform(x_ind-x_ind_unc, x_ind+x_ind_unc, sim)
            args = args_func(ind)
            sol = opt.fsolve(calc_lake_17, initial_guess_hum, args=args)
            Rl_18_, Rl_17_ = sol
            dL_18_ = blf.lnrz_R(Rl_18_)
            dL_17_ = blf.lnrz_R(Rl_17_)
            DL_17 = dL_17_ - 0.528 * dL_18_

            d180_lnrz_out_x.append(dL_18_)
            D17O_out_x.append(DL_17 * 1000)
    
        # Plot this set of simulations
        plt.scatter(d180_lnrz_out_x, D17O_out_x, s=10, color='gray', linewidth=0.1, alpha=0.3)
        plt.xlim(-18, 0)
        plt.ylim(-80,50)
    # Plot curves for D17-d18 space
    z = np.polyfit(d180_lnrz_curve, D17O_curve, 2)
    f = np.poly1d(z)
    x_new = np.linspace(d180_lnrz_curve[0], d180_lnrz_curve[-1], 50)
    y_new = f(x_new)
    
    plt.plot(x_new, y_new, ls='--', linewidth=0.5, color='black')
    plt.axhline(y=33, zorder=0)

    plt.xlabel("δ'¹⁸O (‰)")
    plt.ylabel("∆'¹⁷O (per meg)")
    plt.scatter(d180_lnrz_curve, D17O_curve, s=4, color='black')
    # plt.savefig(f'...\{label}_lake_d17_50k.png', bbox_inches="tight", dpi=600)
    plt.show()
    
del (x_ind, label, ind)

#%% Test sensitivity of lake isotopic composition in d18-dD space to all parameters or individual parameters (humidity, Xe, influx, atmospheric moisture)


x_array = [0, 0.3, 1] # Xe values to calculate for

# Initialize output arrays
d180_curve = []
d_exc_curve = []

# Set up array of functions containing which parameters to test
# Ignore error; variables (x_ind) will be defined in the loop function
labeled_arg_sets_18 = [
    ("test_all",     lambda ind: (hum_dist[ind], x_ind_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_dD_dist[ind], atm_d18_dist[ind], atm_dD_dist[ind])),
    ("just_humidity", lambda ind: (hum_dist[ind], x_ind, temp, influx_d18, influx_dD, atm_d18, atm_dD)),
    ("just_x",       lambda ind: (hum, x_ind_dist[ind], temp, influx_d18, influx_dD, atm_d18, atm_dD)),
    ("just_influx",  lambda ind: (hum, x_ind, temp, influx_d18_dist[ind], influx_dD_dist[ind], atm_d18, atm_dD)),
    ("just_atm",     lambda ind: (hum, x_ind, temp, influx_d18, influx_dD, atm_d18_dist[ind], atm_dD_dist[ind]))
]

# Loop through array of functions containing which parameters to test for sensitivity, plot each
for label, args_func in labeled_arg_sets_18:
    
    print(label)
    
    d180_curve = []
    d_exc_curve = []

    for x_ind in x_array:
    
        d180_out_x = []
        d_exc_out_x = []
        
        initial_guess_hum = [-9.280, -84.787]
    
        curve = opt.fsolve(calc_lake_18, initial_guess_hum, args = (hum, x_ind, temp, influx_d18, influx_dD, atm_d18, atm_dD))
    
        lake_d18 = curve[0]
        lake_dD = curve[1]
        
        lake_d_exc = lake_dD - 8*lake_d18
    
        d180_curve.append(lake_d18)
        d_exc_curve.append(lake_d_exc)
    
        for ind in np.arange(0,sim,1):
            print(ind)
                
            x_ind_unc = abs(x_ind*unc_0)
            x_ind_dist = np.random.default_rng().uniform(x_ind-x_ind_unc, x_ind+x_ind_unc, sim)
        
            args = args_func(ind)
            sol = opt.fsolve(calc_lake_18, initial_guess_hum, args=args)
            
            lake_d18 = sol[0]
            lake_dD = sol[1]
            
            lake_d_exc = lake_dD - 8*lake_d18
            
            d180_out_x.append(lake_d18)
            d_exc_out_x.append(lake_d_exc)
            
        plt.scatter(d180_out_x, d_exc_out_x, s=10, color='gray', linewidth=0.1, alpha=0.3)
        plt.xlim(-18, 0)
        plt.ylim(-80,50)
    
    plt.axhline(y=10, zorder=0)
    plt.xlabel("δ'¹⁸O (‰)")
    plt.ylabel("d-excess (‰)")
    plt.plot(d180_curve, d_exc_curve, ls='--', linewidth=0.5, color='black')
    plt.scatter(d180_curve, d_exc_curve, s=4, color='black')
    # plt.savefig(f'...\{label}_lake_d18_50k.png', bbox_inches="tight", dpi=600)

    plt.show()
    
del (x_ind, label, ind)

#%% Test sensitivity of h and Xe to all parameters or individual parameters (lake, influx, atmospheric moisture)

labeled_arg_sets_17 = [
    ("test_all",     lambda ind: (lake_d18_dist[ind], lake_D17_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_D17_dist[ind], atm_d18_dist[ind], atm_d17_dist[ind])),
    ("just_lake",    lambda ind: (lake_d18_dist[ind], lake_D17_dist[ind], temp, influx_d18, influx_D17, atm_d18, atm_d17)),
    ("just_influx",  lambda ind: (lake_d18, lake_D17, temp, influx_d18_dist[ind], influx_D17_dist[ind], atm_d18, atm_d17)),
    ("just_atm",     lambda ind: (lake_d18, lake_D17, temp, influx_d18, influx_D17, atm_d18_dist[ind], atm_d17_dist[ind]))
]


labeled_arg_sets_18 = [
    ("test_all",     lambda ind: (lake_d18_dist[ind], lake_dD_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_dD_dist[ind], atm_d18_dist_unsorted[ind], atm_dD_dist[ind])),
    ("just_lake",    lambda ind: (lake_d18_dist[ind], lake_dD_dist[ind], temp, influx_d18, influx_dD, atm_d18, atm_dD)),
    ("just_influx",  lambda ind: (lake_d18, lake_dD, temp, influx_d18_dist[ind], influx_dD_dist[ind], atm_d18, atm_dD)),
    ("just_atm",     lambda ind: (lake_d18, lake_dD, temp, influx_d18, influx_dD, atm_d18_dist_unsorted[ind], atm_dD_dist[ind]))
]


for args_func17, args_func18 in zip(labeled_arg_sets_17, labeled_arg_sets_18):
    
    x_18_dist = []
    h_18_dist = []

    x_17_dist = []
    h_17_dist = []

    for ind in np.arange(0,sim,1):
        print(ind)
        
        # h and Xe in triple oxygen
        args17 = args_func17[1](ind)
        roots = opt.fsolve(calc_hum_17, [0.66, 0.29], args=args17)
        h_17_dist.append(roots[0])
        x_17_dist.append(roots[1])
        
        # h and Xe in d18-dD
            
        args18 = args_func18[1](ind)
        roots = opt.fsolve(calc_hum_18, [0.74, 0.33], args=args18)
        h_18_dist.append(roots[0])
        x_18_dist.append(roots[1])
    
    
    mask = (np.array(h_18_dist) > 0) & (np.array(h_18_dist) < 1) & (np.array(x_18_dist) > 0) & (np.array(x_18_dist) < 1)
    mask_17 = (np.array(h_17_dist) > 0) & (np.array(h_17_dist) < 1) & (np.array(x_17_dist) > 0) & (np.array(x_17_dist) < 1)
    
    plt.scatter(np.array(x_18_dist)[mask], np.array(h_18_dist)[mask], color = 'blue', s=2, edgecolor='none', alpha=0.6, label='δ¹⁸O-δ²H')
    plt.scatter(np.array(x_17_dist)[mask_17], np.array(h_17_dist)[mask_17], color = 'red', s=2, edgecolor='none', alpha=0.6, label='Triple oxygen')
    plt.xlabel("Xₑ (Evaporation/Inflow)")
    plt.ylabel("h (Relative humidity)")
    plt.legend()
    # plt.savefig(f'...\{args_func17[0]}_h_xe_comparison_20_perc_50k_t2.png', bbox_inches="tight", dpi=600)

    
    plt.show()