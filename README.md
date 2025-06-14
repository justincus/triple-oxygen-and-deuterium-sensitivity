The Python scripts and CSV files uploaded in this repository are used in the isotope mass balance calculations described in Custado, et al. (2025): Differing sensitivity of δ18O-δ2H versus δ18O-δ17O systematics in a balance-filled lake 

The description of each file is provided below:

- "lake_balance_functions": Contains functions related to some of the isotope calculations performed and the derivation of input parameters (fractionation and enrichment factors, isotopic composition of the atmosphere, etc.). These functions are called in the other scripts.
- "custado_et_al_2025_mc_output": Executes the Monte Carlo analysis to derive values for the isotopic composition of atmospheric moisture, humidity, Xe, theta lake, and 18Orucp using Eqs. 1, 2, S-16, S-17
- "custado_et_al_2025_sensitivty_plot": Executes the sensitivity analysis as illustrated in Fig. 3
- "custado_et_al_2025_plots": Gemerates Fig. 2
- "bl_d17o_datasheet.csv": Master data spreadsheet, including sample locations and isotopic values
