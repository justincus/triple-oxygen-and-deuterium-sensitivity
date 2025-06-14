# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:04:54 2024

@author: mcustado

This script contains functions used in "Differing sensitivity of δ18O-δ2H versus δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)
The functions here are called in 'custado_et_al_2025_mc_output.py', 'custado_et_al_2025_plots.py', and 'custado_et_al_2025_sensitivity_plot.py'
"""

import numpy as np

#%% Set up mass balance equations (Gonfiantini, 1981)

# Let:
# h = Humidity
# F = Flux data (discharge/volume change):
    # I = Inflow
    # O = Outflow
# V = Volume of lake

# dX/R = Isotope values (in per mil (d) or ratio (R)) of:
    # L = Lake
    # I = Inflow
    # O = Outlet
    # E = Evaporation
    # P = Precipitation
    # A = Atmosphere (turbulent atmospheric region, based on Craig-Gordon model)
    # LT = Isotopic composition of the lake at specific time
    # LS = Isotopic composition of the lake at steady-state
    
# ep_k = kinetic enrichment factor
# ep_eq = equilibrium enrichment factor
# alfa = isotopic fractionation factor

# Assumptions: 
    # 1) Lake volume does not change significantly over time
    # 2) Lake is well-mixed

# Estimate isotopic composition of atm moisture
# Atmospheric moisture upwind of the lake is assumed to be in equilibrium with mean annual precipitation or if site is seasonal, precipitation during the evaporation season. Ideally, evaporation flux-weighted precipitation isotope data (see also Gibson et al., 2008, 2015)
# k parameter = seasonality factor. 0.5 (highly seasonal) to 1 (non-seasonal). Gibston, et al., 2015

def isotope_atm (precip, ep_eq, k): 
    dX_A = (precip-(k*ep_eq))/(1+(0.001*k*ep_eq))
    return dX_A

# Estimate isotopic composition of lake at steady state (dX_LS)

# using d18O-dD system (Craig and Gordon, 1965; Gonfiantini, 1981; Gibson, et al., 2016)
def mass_balance_ssx (h, ep_k, ep_eq, alfa, dX_A, dX_I, x):
    a = ((h*dX_A)+ep_k+(ep_eq/alfa))/(1-h+(0.001*ep_k))
    b = (h-0.001*(ep_k+(ep_eq/alfa)))/(1-h+(0.001*ep_k))
    dX_LS = ((x*a)+dX_I)/(1+(b*x))
    limit = a/b
    return dX_LS, limit

# using d17O-d18O system (Form of equation following Passey and Levin, 2021)
def passey_and_levin(h, alfa_eq, alfa_k, Ra, Ri, x):
    Rl = (alfa_eq*alfa_k*(1-h)*Ri + alfa_eq*x*h*Ra)/(alfa_eq*alfa_k*(1-h)*(1-x)+x)
    return Rl

#%% Set up equations for fractionation and enrichment factors

def fractionation_factor_d18O (temp): # Temperature input in deg_C
    alpha = np.exp((-7.685/(10**3)) + (6.7123/(273.15 + temp)) - (1666.4/((273.15 + temp)**2)) + (350410/((273.15 + temp)**3)))
    return alpha

def fractionation_factor_dD (temp): # Temperature input in deg_C
    alpha = np.exp((1158.8*(((273.15 + temp)**3)/(10**12))) - (1620.1*(((273.15 + temp)**2)/(10**9))) + (794.84*((273.15 + temp)/(10**6))) - (161.04/(10**3)) + (2999200/((273.15 + temp)**3)))
    return alpha

def fractionation_factor_d17O (alpha):
    alpha_17 = alpha**0.529
    return alpha_17

def fractionation_factor_k_d17O (alpha):
    alpha_17 = alpha**0.5185
    return alpha_17

def kinetic_en_d18O(humidity):
    ep_k = 0.5*28.4*(1-humidity) 
    return ep_k

def kinetic_en_dD(humidity):
    ep_k = 12.5*(1-humidity)
    return ep_k

def kinetic_en_d17O(humidity):
    ep_k = 0.5*14.71*(1-humidity)   #14.64 value from  from Pierchala et al. 2022; Barkan and Luz (2007) is 14.71 # Removed 0.5 multiplier
    return ep_k

#%% Set up functions to volume-weighted isotopic average of the creeks and total influx

def creek_wt_isotope (paris, scharles, bloom, big, neden, seden, swan):
    paris_disch = 8422054.769 # ave vol/month
    scharles_disch = 50838702.09
    bloom_disch = 27569586.17
    big_disch = 15493626.95
    neden_disch = 2504866.992
    seden_disch = 2504866.992
    swan_disch = 33209268.49

    wt_isotope_sum = paris_disch*paris + scharles_disch*scharles + bloom_disch*bloom + big_disch*big + neden_disch*neden + seden_disch*seden + swan_disch*swan
    disch_sum = paris_disch + scharles_disch + bloom_disch + big_disch + neden_disch + seden_disch + swan_disch
    wt_isotope = wt_isotope_sum/disch_sum
    
    return wt_isotope

def calc_wt_influx (inlet, creek, prec, gw):
    
    inlet_disch = 317867056.27 # ave vol/yr
    creek_disch = 145049044.91 # ave vol/yr
    prec_disch = 107856450.33 # ave vol/yr
    gw_disch = 76217.6184230801 # ave vol/yr
    sum_disch = inlet_disch + creek_disch + prec_disch + gw_disch

    wt_ave = (inlet*inlet_disch + creek*creek_disch + prec*prec_disch + gw*gw_disch)/(sum_disch)
    
    return wt_ave

#%% Functions to convert between ratio and delta notation and linearize values

def convert_d_to_R(dX):
    R = (dX/1000)+1
    return R

def convert_R_to_d(R):
    d = 1000*(R-1)
    return d

def lnrz_d (d):
    x = 1000*np.log((d/1000)+1)
    return x

def lnrz_R (R):
    x = 1000*np.log(R)
    return x

#%% Function to calculate the standard error of the mean (SEM)

def uncertainty_mean(data):

    n = np.count_nonzero(~np.isnan(data))
    if n < 2:
       raise ValueError("Data set must contain at least two values")
    std_dev = np.nanstd(data, ddof=1)  # ddof=1 for sample standard deviation
    sem = std_dev / np.sqrt(n)
    return sem
