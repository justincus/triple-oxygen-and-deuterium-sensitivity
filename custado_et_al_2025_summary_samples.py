# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:46:06 2025

@author: mcustado

This script calculates summary mean and uncertainty from the measured samples used in "Contrasting controls on δ18O-δ2H and δ18O-δ17O systematics in a balance-filled lake" by Custado et al. (2025)

Variable names:
    d = small delta (δ)
    D = big delta (Δ)
    
    [sample type]_dX
    where sample type are:
        inlet = measurements from the inlet canal
        creek = measurements from the surrounding creeks
        gw = measurements from groundwater and springs
        lake = measurements from Bear Lake (Mud Lake data not included in calculations)
        influx = total volume inflow to the lake 
        
    Other abbreviations:
        mo = month
        evap = evaporation flux
        amt = amount
        dist = distribution
        unc = uncertainty
        d_exc = d-excess
"""

import numpy as np
import pandas as pd
import lake_balance_functions_R1 as lbf

# -------------------------------------------------------------------------------------
# Load measured isotope data
# -------------------------------------------------------------------------------------

# Load sample mastersheet

df_ = pd.read_csv(r'...\\bl_d17o_datasheet.csv') # Insert file directory path
df_['Sample_Collection_Date'] = pd.to_datetime(df_['Sample_Collection_Date'])
mask = ~np.isnan(df_['dD']) # Remove samples with unpaired dD
df = df_[mask]

# Load environmental/climate parameters 
# (extracted from North American Regional Reanalysis dataset)

cl = pd.read_csv(r'...\\monthly_clim_parameters.csv') # Insert file directory path

# Separate parameters

evap_rate = cl['evap_kg_m2']
prec_amt = cl['prec_amt_kg_m2']

# -------------------------------------------------------------------------------------
# Calculate isotopic composition of inputs to mass balance equations from the dataset
# -------------------------------------------------------------------------------------
# 1. Precipitation (Amount-weighted and evaporation flux-weighted)
# 2. Influx (volume-weighted from each component)
# 3. Lake
# Let:
# dX be the isotopic parameters: d18 = d18O; d17 = d17O; dD = d2H; D17 = O-17 excess; d_exc = d-excess;
# Other symbols: _mean = mean; _unc = uncertainty
# -------------------------------------------------------------------------------------

### 1. Precipitation ###

precip = df.loc[(df['Sample_type']=="Precipitation")]
precip['Sample_Collection_Date'] = pd.to_datetime(precip['Sample_Collection_Date'])
precip = precip.set_index('Sample_Collection_Date')

prec_d18O_mo = precip.groupby(precip.index.month)['d18O'].mean()
prec_d17O_mo = precip.groupby(precip.index.month)['d17O'].mean()
prec_D17O_mo = precip.groupby(precip.index.month)['D17O'].mean()
prec_dD_mo = precip.groupby(precip.index.month)['dD'].mean()
prec_d_exc_mo = precip.groupby(precip.index.month)['d_exc'].mean()

# 1a. Annual volume-weighted precipitation

prec_d18O_amt = lbf.get_weighted_ave(prec_d18O_mo.values, prec_amt.values)
prec_d17O_amt = lbf.get_weighted_ave(prec_d17O_mo.values, prec_amt.values)
prec_D17O_amt = lbf.get_weighted_ave(prec_D17O_mo.values, prec_amt.values)
prec_dD_amt = lbf.get_weighted_ave(prec_dD_mo.values, prec_amt.values)
prec_d_exc_amt = lbf.get_weighted_ave(prec_d_exc_mo.values, prec_amt.values)

# 1b. Evap flux-weighted precipitation

prec_d18O_evap = lbf.get_weighted_ave(prec_d18O_mo.values, evap_rate.values)
prec_d17O_evap = lbf.get_weighted_ave(prec_d17O_mo.values, evap_rate.values)
prec_D17O_evap = lbf.get_weighted_ave(prec_D17O_mo.values, evap_rate.values)
prec_dD_evap = lbf.get_weighted_ave(prec_dD_mo.values, evap_rate.values)
prec_d_exc_evap = lbf.get_weighted_ave(prec_d_exc_mo.values, evap_rate.values)

prec_d18O_unc = lbf.uncertainty_mean(precip['d18O'])
prec_d17O_unc = lbf.uncertainty_mean(precip['d17O'])
prec_D17O_unc = lbf.uncertainty_mean(precip['D17O'])
prec_dD_unc = lbf.uncertainty_mean(precip['dD'])
prec_d_exc_unc = lbf.uncertainty_mean(precip['d_exc'])

### 2. Influx ###

# 2a. Inlet / Bear River upstream 

inlet_d17 = df.loc[df['Sample_type']=="Inlet"]['d17O'].mean()
inlet_d17O_unc = df.loc[df['Sample_type']=="Inlet"]['D17O_unc']

inlet_d18 = df.loc[df['Sample_type']=="Inlet"]['d18O'].mean()
inlet_d18O_unc = df.loc[df['Sample_type']=="Inlet"]['d18O_unc']

inlet_D17 = df.loc[df['Sample_type']=="Inlet"]['D17O'].mean()
inlet_D17O_unc = df.loc[df['Sample_type']=="Inlet"]['D17O_unc']

inlet_D17O_paired = df.loc[df['Sample_type']=="Inlet"]['D17O'][1]

inlet_dD = df.loc[df['Sample_type']=="Inlet"]['dD'].mean()
inlet_dD_unc = df.loc[df['Sample_type']=="Inlet"]['dD_unc'][1]

inlet_d_exc = df.loc[df['Sample_type']=="Inlet"]['d_exc'].mean()
inlet_d_exc_unc = df.loc[df['Sample_type']=="Inlet"]['d_exc_unc'][1]

# 2b. Creeks

paris_iso = df.loc[df['Site_name'].astype(str).str.contains('Paris')]
scharles_iso = df.loc[df['Site_name'].astype(str).str.contains('St Charles Creek')]
bloom_iso = df.loc[df['Site_name'].astype(str).str.contains('Bloomington Creek')]
big_iso = df.loc[df['Site_name'].astype(str).str.contains('Big Creek')]
neden_iso = df.loc[df['Site_name'].astype(str).str.contains('North Eden Creek')]
seden_iso = df.loc[df['Site_name'].astype(str).str.contains('South Eden Creek')]
swan_iso = df.loc[df['Site_name'].astype(str).str.contains('Swan Creek')]

creeks_used_iso = pd.concat([paris_iso, scharles_iso, bloom_iso, big_iso, neden_iso, seden_iso,swan_iso])

creek_d17 = lbf.creek_wt_isotope(paris_iso['d17O'].mean(), scharles_iso['d17O'].mean(),bloom_iso['d17O'].mean(),big_iso['d17O'].mean(),neden_iso['d17O'].mean(),seden_iso['d17O'].mean(),swan_iso['d17O'])
creek_D17 = lbf.creek_wt_isotope(paris_iso['D17O'].mean(),scharles_iso['D17O'].mean(),bloom_iso['D17O'].mean(),big_iso['D17O'].mean(),neden_iso['D17O'].mean(),seden_iso['D17O'].mean(),swan_iso['D17O'])
creek_d18 = lbf.creek_wt_isotope(paris_iso['d18O'].mean(),scharles_iso['d18O'].mean(),bloom_iso['d18O'].mean(),big_iso['d18O'].mean(),neden_iso['d18O'].mean(),seden_iso['d18O'].mean(),swan_iso['d18O'])
creek_dD = lbf.creek_wt_isotope(paris_iso['dD'].mean(),scharles_iso['dD'].mean(),bloom_iso['dD'].mean(),big_iso['dD'].mean(),neden_iso['dD'].mean(),seden_iso['dD'].mean(),swan_iso['dD'])
creek_d_exc = lbf.creek_wt_isotope(paris_iso['d_exc'].mean(),scharles_iso['d_exc'].mean(),bloom_iso['d_exc'].mean(),big_iso['d_exc'].mean(),neden_iso['d_exc'].mean(),seden_iso['d_exc'].mean(),swan_iso['d_exc'])

creek_d17O_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d17O'].values, scharles_iso['d17O'].values,bloom_iso['d17O'].values,big_iso['d17O'].values,neden_iso['d17O'],seden_iso['d17O'].values,swan_iso['d17O'].values]).ravel())
creek_D17O_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['D17O'].values,scharles_iso['D17O'].values,bloom_iso['D17O'].values,big_iso['D17O'].values,neden_iso['D17O'].values,seden_iso['D17O'].values,swan_iso['D17O'].values]).ravel())
creek_d18O_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d18O'].values,scharles_iso['d18O'].values,bloom_iso['d18O'].values,big_iso['d18O'].values,neden_iso['d18O'].values,seden_iso['d18O'].values,swan_iso['d18O'].values]).ravel())
creek_dD_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['dD'].values,scharles_iso['dD'].values,bloom_iso['dD'].values,big_iso['dD'].values,neden_iso['dD'].values,seden_iso['dD'].values,swan_iso['dD'].values]).ravel())
creek_d_exc_unc = lbf.uncertainty_mean(np.concatenate([paris_iso['d_exc'].values,scharles_iso['d_exc'].values,bloom_iso['d_exc'].values,big_iso['d_exc'].values,neden_iso['d_exc'].values,seden_iso['d_exc'].values,swan_iso['d_exc'].values]).ravel())

# 2c. Groundwater + Springs

gw_d17 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d17O'].mean()
gw_d18 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d18O'].mean()
gw_D17 = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['D17O'].mean()
gw_dD = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['dD'].mean()
gw_d_exc = df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d_exc'].mean()

gw_d17O_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d17O'])
gw_d18O_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d18O'])
gw_D17O_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['D17O'])
gw_dD_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['dD'])
gw_d_exc_unc = lbf.uncertainty_mean(df.loc[(df['Sample_type']=="Ground") | (df['Sample_type']=='Spring')]['d_exc'])

# 2d Get volume-weighted isotopic composition of inflow

influx_d17 = lbf.calc_wt_influx(inlet_d17, creek_d17.values[0], prec_d17O_amt, gw_d17)
influx_D17 = lbf.calc_wt_influx(inlet_D17, creek_D17.values[0], prec_D17O_amt, gw_D17)
influx_d18 = lbf.calc_wt_influx(inlet_d18, creek_d18.values[0], prec_d18O_amt, gw_d18) 
influx_dD =  lbf.calc_wt_influx(inlet_dD, creek_dD.values[0], prec_dD_amt, gw_dD) 
influx_d_exc = lbf.calc_wt_influx(inlet_d_exc, creek_d_exc.values[0], prec_d_exc_amt, gw_d_exc) 

influx_d18O_unc = lbf.uncertainty_mean([inlet_d18O_unc[1],creek_d18O_unc,gw_d18O_unc,prec_d18O_unc])
influx_d17O_unc = lbf.uncertainty_mean([inlet_d17O_unc[1],creek_d17O_unc,gw_d17O_unc,prec_d17O_unc])
influx_D17O_unc = lbf.uncertainty_mean([inlet_D17O_unc[1],creek_D17O_unc,gw_D17O_unc,prec_D17O_unc])
influx_dD_unc =  lbf.uncertainty_mean([inlet_dD_unc,creek_dD_unc,gw_dD_unc,prec_dD_unc]) 
influx_d_exc_unc = lbf.uncertainty_mean([inlet_d_exc_unc,creek_d_exc_unc,gw_d_exc_unc,prec_d_exc_unc]) 

### 3. Lake ###

lake_d17 = df.loc[df['Sample_type']=="Lake"]['d17O'].drop([3]).mean() # drop mud lake data
lake_d18 = df.loc[df['Sample_type']=="Lake"]['d18O'].drop([3]).mean() # drop mud lake data
lake_D17 = df.loc[df['Sample_type']=="Lake"]['D17O'].drop([3]).mean() # drop mud lake data
lake_dD = df.loc[df['Sample_type']=="Lake"].drop([3])['dD'].mean() # drop mud lake data
lake_d_exc = df.loc[df['Sample_type']=="Lake"].drop([3])['d_exc'].mean() # drop mud lake data

lake_d17O_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['d17O'].drop([3])) # drop mud lake data
lake_d18O_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['d18O'].drop([3])) # drop mud lake data
lake_D17O_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"]['D17O'].drop([3])) # drop mud lake data
lake_dD_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"].drop([3])['dD']) # drop mud lake data
lake_d_exc_unc = lbf.uncertainty_mean(df.loc[df['Sample_type']=="Lake"].drop([3])['d_exc']) # drop mud lake data

### Print and save results ###
print("Isotopic inputs to mass balance",
      "\nIsotope\t", "Value\t", "Uncertainty",    
     "\nd18O precip (evap flux-weighted): \t", prec_d18O_evap, "\t", prec_d18O_unc,
     "\nd17O precip (evap flux-weighted): \t", prec_d17O_evap, "\t", prec_d17O_unc,
     "\nD17O precip (evap flux-weighted): \t", prec_D17O_evap, "\t", prec_D17O_unc,
     "\ndD precip (evap flux-weighted): \t", prec_dD_evap, "\t", prec_dD_unc,
     "\nd-excess precip (evap flux-weighted): \t", prec_d_exc_evap, "\t", prec_d_exc_unc,
     
     "\nd18O influx: \t", influx_d18, "\t", influx_d18O_unc,
     "\nd17O influx: \t", influx_d17, "\t", influx_d17O_unc,
     "\nD17O influx: \t", influx_D17, "\t", influx_D17O_unc,
     "\ndD influx: \t", influx_dD, "\t", influx_dD_unc,
     "\nd-excess influx: \t", influx_d_exc, "\t", influx_d_exc_unc,     
     
     "\nd18O lake: \t", lake_d18, "\t", lake_d18O_unc,
     "\nd17O lake: \t", lake_d17, "\t", lake_d17O_unc,
     "\nD17O lake: \t", lake_D17, "\t", lake_D17O_unc,
     "\ndD lake: \t", lake_dD, "\t", lake_dD_unc,
     "\nd-excess lake: \t", lake_d_exc, "\t", lake_d_exc_unc,   
     
     "\nInflux components",
     "\nIsotope\t", "Value\t", "Uncertainty",
     "\nd18O precip (amount-weighted): \t", prec_d18O_amt, "\t", prec_d18O_unc,
     "\nd17O precip (amount-weighted): \t", prec_d17O_amt, "\t", prec_d17O_unc,
     "\nD17O precip (amount-weighted): \t", prec_D17O_amt, "\t", prec_D17O_unc,
     "\ndD precip (amount-weighted): \t", prec_dD_amt, "\t", prec_dD_unc,
     "\nd-excess precip (amount-weighted): \t", prec_d_exc_amt, "\t", prec_d_exc_unc,
     
     "\nd18O inlet: \t", inlet_d18, "\t", inlet_d18O_unc[1],
     "\nd17O inlet: \t", inlet_d17, "\t", inlet_d17O_unc[1],
     "\nD17O inlet: \t", inlet_D17, "\t", inlet_D17O_unc[1],
     "\ndD inlet: \t", inlet_dD, "\t", inlet_dD_unc,
     "\nd-excess inlet: \t", inlet_d_exc, "\t", inlet_d_exc_unc,    
     
     "\nd18O creek: \t", creek_d18.values[0], "\t", creek_d18O_unc,
     "\nd17O creek: \t", creek_d17.values[0], "\t", creek_d17O_unc,
     "\nD17O creek: \t", creek_D17.values[0], "\t", creek_D17O_unc,
     "\ndD creek: \t", creek_dD.values[0], "\t", creek_dD_unc,
     "\nd-excess creek: \t", creek_d_exc.values[0], "\t", creek_d_exc_unc,    
     
     "\nd18O gw: \t", gw_d18, "\t", gw_d18O_unc,
     "\nd17O gw: \t", gw_d17, "\t", gw_d17O_unc,
     "\nD17O gw: \t", gw_D17, "\t", gw_D17O_unc,
     "\ndD gw: \t", gw_dD, "\t", gw_dD_unc,
     "\nd-excess gw: \t", gw_d_exc, "\t", gw_d_exc_unc,    
     
     "\nd18O gw: \t", gw_d18, "\t", gw_d18O_unc,
     "\nd17O gw: \t", gw_d17, "\t", gw_d17O_unc,
     "\nD17O gw: \t", gw_D17, "\t", gw_D17O_unc,
     "\ndD gw: \t", gw_dD, "\t", gw_dD_unc,
     "\nd-excess gw: \t", gw_d_exc, "\t", gw_d_exc_unc
     )
      
all_data = {
    "prec_d18O_evap": [prec_d18O_evap, prec_d18O_unc],
    "prec_d17O_evap": [prec_d17O_evap, prec_d17O_unc],
    "prec_D17O_evap": [prec_D17O_evap, prec_D17O_unc],
    "prec_dD_evap": [prec_dD_evap, prec_dD_unc],
    "prec_d_exc_evap": [prec_d_exc_evap, prec_d_exc_unc],

    "influx_d18O": [influx_d18, influx_d18O_unc],
    "influx_d17O": [influx_d17, influx_d17O_unc],
    "influx_D17O": [influx_D17, influx_D17O_unc],
    "influx_dD": [influx_dD, influx_dD_unc],
    "influx_d_exc": [influx_d_exc, influx_d_exc_unc],

    "lake_d18O": [lake_d18, lake_d18O_unc],
    "lake_d17O": [lake_d17, lake_d17O_unc],
    "lake_D17O": [lake_D17, lake_D17O_unc],
    "lake_dD": [lake_dD, lake_dD_unc],
    "lake_d_exc": [lake_d_exc, lake_d_exc_unc],

    "prec_d18O_amt": [prec_d18O_amt, prec_d18O_unc],
    "prec_d17O_amt": [prec_d17O_amt, prec_d17O_unc],
    "prec_D17O_amt": [prec_D17O_amt, prec_D17O_unc],
    "prec_dD_amt": [prec_dD_amt, prec_dD_unc],
    "prec_d_exc_amt": [prec_d_exc_amt, prec_d_exc_unc],

    "inlet_d18O": [inlet_d18, inlet_d18O_unc[1]],
    "inlet_d17O": [inlet_d17, inlet_d17O_unc[1]],
    "inlet_D17O": [inlet_D17, inlet_D17O_unc[1]],
    "inlet_dD": [inlet_dD, inlet_dD_unc],
    "inlet_d_exc": [inlet_d_exc, inlet_d_exc_unc],
    
    "creek_d18O": [creek_d18.values[0], creek_d18O_unc],
    "creek_d17O": [creek_d17.values[0], creek_d17O_unc],
    "creek_D17O": [creek_D17.values[0], creek_D17O_unc],
    "creek_dD": [creek_dD.values[0], creek_dD_unc],
    "creek_d_exc": [creek_d_exc.values[0], creek_d_exc_unc],

    "gw_d18O": [gw_d18, gw_d18O_unc],
    "gw_d17O": [gw_d17, gw_d17O_unc],
    "gw_D17O": [gw_D17, gw_D17O_unc],
    "gw_dD": [gw_dD, gw_dD_unc],
    "gw_d_exc": [gw_d_exc, gw_d_exc_unc],
}

# convert to DataFrame (transpose so each variable is a row)
all_data_df = pd.DataFrame(all_data, index=["Value", "Uncertainty"]).T.reset_index()
all_data_df.columns = ["variable", "value", "uncertainty"]

# save to CSV
all_data_df.to_csv(r'...\isotopic_inputs.csv') # Insert file directory path





      

