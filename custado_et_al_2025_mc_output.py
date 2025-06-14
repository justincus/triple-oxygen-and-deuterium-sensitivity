# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:52:52 2024

@author: mcustado

This script runs the Monte Carlo analysis in "Differing sensitivity of δ18O-δ2H versus δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
The parameters calculated in this file are the isotopic composition of atmospheric moisture, humidity, Xe, theta lake, and 18Orucp using Eqs. 1, 2, S-16, S-17
"""

import numpy as np
import matplotlib.pyplot as plt
import lake_balance_functions as lbf
import scipy.optimize as opt

#%% Input initial parameters 

# Input initial humidity (h), Xe (x_1), and temperature (temp)

hum = 0.62 
x_1 = 0.38
temp = 11.15

# Input isotopic parameters for: 
    # Influx (influx_dX)
    # Evaporation flux-weightedcipitation (prec_dX_evap)
    # Atmospheric moisture (atm_dX),
    # Lake (lake_dX)
    # dX represents the isotopes: d18 = d18O; d17 = d17O; dD = d2H; D17 = O-17 excess

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

# Provide input uncertanties (_unc)

unc_0 = 0.05 # Set general uncertainty (for humidity and temperatue)

hum_unc = hum*unc_0
x_1_unc = abs(x_1*0.30)
temp_unc = abs(temp*unc_0) 

influx_d18_unc = 0.05656421990436
influx_D17_unc = 1.72772541859255
influx_dD_unc = 0.544442582386388

prec_d18_unc = 0.239171769
prec_d17_unc = 0.126446549
prec_dD_unc = 1.832156117

lake_d18_unc = 0.040
lake_D17_unc = 2.812
lake_dD_unc = 0.178

#%% Initialize distribution arrays for Monte Carlo analysis  

# Input # of simulations

sim = 100000

# Initialize arrays of input distributions. These are random uniform distributions.

hum_dist = np.random.default_rng().uniform(hum-hum_unc, hum+hum_unc, sim)
x_1_dist = np.random.default_rng().uniform(x_1-x_1_unc, x_1+x_1_unc, sim)
temp_dist = np.random.default_rng().uniform(temp-temp_unc, temp+temp_unc, sim)

influx_d18_dist = np.random.default_rng().uniform(influx_d18-influx_d18_unc, influx_d18+influx_d18_unc, sim)
influx_D17_dist = np.random.default_rng().uniform(influx_D17-influx_D17_unc, influx_D17+influx_D17_unc, sim) 
influx_dD_dist = np.random.default_rng().uniform(influx_dD-influx_dD_unc, influx_dD+influx_dD_unc, sim) 

prec_d18_dist = np.random.default_rng().uniform(prec_d18_evap-prec_d18_unc, prec_d18_evap+prec_d18_unc, sim)
prec_d17_dist = np.random.default_rng().uniform(prec_d17_evap-prec_d17_unc, prec_d17_evap+prec_d17_unc, sim)
prec_dD_dist = np.random.default_rng().uniform(prec_dD_evap-prec_dD_unc, prec_dD_evap+prec_dD_unc, sim)

lake_d18_dist = np.random.default_rng().uniform(lake_d18-lake_d18_unc, lake_d18+lake_d18_unc, sim)
lake_D17_dist = np.random.default_rng().uniform(lake_D17-lake_D17_unc, lake_D17+lake_D17_unc, sim) 
lake_dD_dist = np.random.default_rng().uniform(lake_dD-lake_dD_unc, lake_dD+lake_dD_unc, sim) 

#%% Define functions

# Mass balance equations in the triple oxygen space to calculate for humidity and Xe

def calc_hum_17(vars, lake_d18_, lake_D17_, temp_, influx_d18_, influx_D17_, atm_d18_, atm_d17_):
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_17_ = lbf.fractionation_factor_d17O(a_18_)
    
    a_k_18_0 = 1.0285**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Merlivat 1978
    a_k_17_0 = 1.0146**0.5 # Used for d17O-d18O equations, kinetic fractionation at zero humidity # Voigt 2021, Barkan and Luz 2007

    # Calculate isotopic composition of atmosphere
    
    # Calculate d17 from measured D17 data
    influx_d17_ = influx_D17_/1000 + 0.528*lbf.lnrz_d(influx_d18_) # get d17 influx from D17 and d18
    i_17 = 1000*(np.exp(influx_d17_/1000)-1) # delinearize influx d17
    
    lake_d17_ = lake_D17_/1000 + 0.528*lbf.lnrz_d(lake_d18_) # get d17 lake from D17 and d18
    l_17 = 1000*(np.exp(lake_d17_/1000)-1) # delinearize lake d17

    # Convert d values to R
    Ri_18 = lbf.convert_d_to_R(influx_d18_) 
    Ri_17 = lbf.convert_d_to_R(i_17) 
    
    Rl_18 = lbf.convert_d_to_R(lake_d18_)
    Rl_17 = lbf.convert_d_to_R(l_17)
    
    # Calculate R of atmoshperic moisture; assume ratio of evaporation flux-weighted precipitation is in equilibrium with atmospheric moisture (Gibson et al. 2016)
    Ra_18 = lbf.convert_d_to_R(atm_d18_) 
    Ra_17 = lbf.convert_d_to_R(atm_d17_)
    
    # Calculate R of atmoshperic moisture; assume ratio of inflow is in equilibrium with atmospheric
    # Ra_18 = Ri_18/a_18_ 
    # Ra_17 = Ri_17/a_17_ 

    # Equation from Surma (2018), Voigt (2021), Passey and Levin (2021)
    eq1 = ((a_18_*a_k_18_0*(1-h_)*Ri_18 + a_18_*x_*h_*Ra_18))/(a_18_*a_k_18_0*(1-h_)*(1-x_)+x_) - Rl_18
    eq2 = ((a_17_*a_k_17_0*(1-h_)*Ri_17 + a_17_*x_*h_*Ra_17))/(a_17_*a_k_17_0*(1-h_)*(1-x_)+x_) - Rl_17

    return np.array([eq1, eq2])

# Mass balance equations in the d18O-dD space to calculate for humidity and Xe

def calc_hum_18(vars, lake_d18_, lake_dD_, temp_, influx_d18_, influx_D_, atm_d18_, atm_dD_): 
    h_, x_,  = vars
    
    # Calculate fractionation factors from given humidity and temp
    a_18_ = lbf.fractionation_factor_d18O(temp_) 
    a_D_ = lbf.fractionation_factor_dD(temp_) 
    
    ep_18_ = (a_18_ - 1)*1000
    ep_D_ = (a_D_ - 1)*1000
    
    ep_k_18_ = lbf.kinetic_en_d18O(h_) # Used for d18O-dD equations
    ep_k_D_ = lbf.kinetic_en_dD(h_)
    
    eq1 = (((lake_d18_ - influx_d18_)*(1-h_+(0.001*ep_k_18_))) / (h_*(atm_d18_ - lake_d18_) + (ep_k_18_ + (ep_18_/a_18_))*(0.001*lake_d18_ + 1))) - x_
    eq2 = (((lake_dD_ - influx_D_)*(1-h_+(0.001*ep_k_D_))) / (h_*(atm_dD_ - lake_dD_) + (ep_k_D_ + (ep_D_/a_D_))*(0.001*lake_dD_ + 1))) - x_

    return np.array([eq1, eq2])

# Equation to reconstruct the d18O of unevaporated catchment precipitation (rucp). Equation from Passey and Ji (2019)

def rucp(D17O_ref, D17O_lake, m_ref, d18O_lake, m_lake):
    rucp = (D17O_ref/1000-D17O_lake/1000+((m_lake-m_ref)*d18O_lake))/(m_lake-m_ref)
    
    return rucp

# Function to remove improbable data for h and Xe (less than zero or more than one)

def remove_improbable(data):
    new_list = [i for i in data if (i>=0 and i<=1)]
    return new_list

#%% Monte Carlo analysis to calculate for the isotopic composition of atmospheric moisture, humidity, Xe

# Initialize output arrays

atm_d18_dist_unsorted = []
atm_d17_dist_unsorted = []
atm_dD_dist_unsorted = []

x_dist = []
h_dist = []

x_17_dist = []
h_17_dist = []

initial_guess = [hum, x_1]

## Run simulations

for ind in np.arange(0,sim,1):
    print(ind)
    
    # Calculate fractionation factors
    
    a_18 = lbf.fractionation_factor_d18O(temp_dist[ind]) 
    a_17 = lbf.fractionation_factor_d17O(a_18)
    a_D = lbf.fractionation_factor_dD(temp)

    ep_18 = (a_18 - 1)*1000
    ep_17 = (a_17 - 1)*1000
    ep_D = (a_D - 1)*1000
    
    # Calculate isotopic composition of atmospheric moisture

    x = lbf.isotope_atm(prec_d18_dist[ind], ep_18, 1) 
    atm_d18_dist_unsorted.append(x)
    
    x = lbf.isotope_atm(prec_d17_dist[ind], ep_17, 1) 
    atm_d17_dist_unsorted.append(x)

    x = lbf.isotope_atm(prec_dD_dist[ind], ep_D, 1) 
    atm_dD_dist_unsorted.append(x)

# Sort generated distribution of isotopic values for atmospheric moisture to ensure reasonable pairs of values for d17 and d18

atm_d18_dist = sorted(atm_d18_dist_unsorted)
atm_d17_dist = sorted(atm_d17_dist_unsorted)
atm_dD_dist = sorted(atm_dD_dist_unsorted)

for ind in np.arange(0,sim,1):
    print(ind)
    
    # Calculate h and Xe in triple oxygen space

    roots = opt.fsolve(calc_hum_17, [0.54, 0.4], args=(lake_d18_dist[ind], lake_D17_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_D17_dist[ind], atm_d18_dist[ind], atm_d17_dist[ind]))
    h_17_dist.append(roots[0])
    x_17_dist.append(roots[1])
    
    # Calculate h and Xe in d18-dDspace

    roots = opt.fsolve(calc_hum_18, initial_guess, args=(lake_d18_dist[ind], lake_dD_dist[ind], temp_dist[ind], influx_d18_dist[ind], influx_dD_dist[ind], atm_d18_dist_unsorted[ind], atm_dD_dist_unsorted[ind]))
    h_dist.append(roots[0])
    x_dist.append(roots[1])

# Compile calculated distributions

dist = [atm_d18_dist, atm_d17_dist, atm_dD_dist, remove_improbable(x_dist), remove_improbable(h_dist), remove_improbable(x_17_dist), remove_improbable(h_17_dist)]
variable = ["δ¹⁸O, atmosphere (‰)", 'δ¹⁷O, atmosphere (‰)', 'δ²H, atmosphere (‰)', 'Xₑ, δ¹⁸O-δ²H', 'h, δ¹⁸O-δ²H', 'Xₑ, Triple oxygen', 'h, Triple oxygen']

for d, v in zip(dist, variable):

    # Print summary statistics

    print("\nvariable: \t", v)
    print("\nmean output: \t", np.nanmean(d))
    print("median output: \t", np.median(d))
    print("std output: \t", np.nanstd(d))
    print("15.9 perc output: \t", np.percentile(d, 15.9))
    print("84.1 perc output: \t", np.percentile(d, 84.1))
    print("minimum output: \t", np.min(d))
    print("maximum output: \t", np.max(d))
    print("number of simulations: \t", sim)
    
    # Plot output histograms
    
    plt.hist(d, color='gray', edgecolor = "black")
    plt.ylabel('Count')
    plt.xlabel(v)
    # plt.savefig('{path}'+v+"_hist.png", bbox_inches="tight", dpi=600)
    plt.show()

#%% Monte Carlo analysis to calculate theta lake and rucp

# Input calculated values and uncertainties (unc) for humidity (h) and Xe (x) to generate evaporation trajectory 
# mwl refers to Global Meteoric Water Line 

h_calc = 0.539509371
x_calc = 0.29281996168543
mwl_D17 = 32

h_calc_unc = 0.116520879
x_calc_unc = 0.015156871
mwl_D17_unc = 15

# Initialize distribution arrays for analysis (sim = number of simulations similar to previous section)

h_calc_dist = np.random.default_rng().uniform(h_calc-h_calc_unc, h_calc+h_calc_unc, sim)
x_calc_dist = np.random.default_rng().uniform(x_calc-x_calc_unc, x_calc+x_calc_unc, sim)
x_calc_dist = np.random.default_rng().uniform(x_calc-x_calc_unc, x_calc+x_calc_unc, sim)
mwl_D17_dist = np.random.default_rng().uniform(mwl_D17-mwl_D17_unc, mwl_D17+mwl_D17_unc, sim)

# Initialize output arrays

lake_theta_array = []
rucp_array = []

influx_d17_array = []
lake_d17_array = []

for ind in np.arange(0,sim,1):
    
    # Calculate theta_lake by using (1) inflow and lake isotopic data or (2) equation from Passey and Ji (2019)
    # Comment out which method to not use
    
    #### Method (1): using inflow and lake isotopic data ####
    # influx_d17_ = influx_D17_dist[ind]/1000 + 0.528*blf.lnrz_d(influx_d18_dist[ind]) # get d17 influx from D17 and d18
    # influx_d17_array.append(influx_d17_)
    
    # lake_d17_ = lake_D17_dist[ind]/1000 + 0.528*blf.lnrz_d(lake_d18_dist[ind]) # get d17 lake from D17 and d18
    # lake_d17_array.append(lake_d17_)

    # y = [influx_d17_, lake_d17_]
    # x = [blf.lnrz_d(influx_d18_dist[ind]), blf.lnrz_d(lake_d18_dist[ind])]
    
    # m_lake = np.polyfit(x,y,1)
    # m_lake1 = m_lake[0]
    # lake_theta_array.append(m_lake1)
    
    #### Method (2): using equation from Passey and Ji (2019) ####
    m_lake1 = 1.9856*(lake_D17_dist[ind]/1000**3) + 0.5730*(lake_D17_dist[ind]/1000**2) + 0.0601*lake_D17_dist[ind]/1000 + 0.5236 # probably a different curve for this
    lake_theta_array.append(m_lake1)

    r = rucp(32, lake_D17, 0.528, lbf.lnrz_d(lake_d18), m_lake1)
    
    # Delinearize output rucp
    r = 1000*(np.exp(r/1000)-1)
    rucp_array.append(r)

# Print summary statistics

dist = [lake_theta_array, rucp_array, influx_d17_array, lake_d17_array]
variable = ["θ lake", "δ'¹⁸O rucp (‰)", "δ'17O, influx (‰)", "δ'17O, lake (‰)"]

for d, v in zip(dist[0:2], variable[0:2]):

    
    print("\nvariable: \t", v)
    print("\nmean output: \t", np.nanmean(d))
    print("median output: \t", np.median(d))
    print("std output: \t", np.nanstd(d))
    print("15.9 perc output: \t", np.percentile(d, 15.9))
    print("84.1 perc output: \t", np.percentile(d, 84.1))
    print("minimum output: \t", np.min(d))
    print("maximum output: \t", np.max(d))
    print("number of simulations: \t", sim)
    
    plt.hist(d, color='gray', edgecolor = "black")
    plt.ylabel('Count')
    plt.xlabel(v)
    # plt.savefig('{path}'+v+"_hist.png", bbox_inches="tight", dpi=600)
    plt.show()

