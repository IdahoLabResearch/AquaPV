# Copyright 2024, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pandas as pd
import numpy as np
import sys


#######################################################
# Get the CAPEX cost
# This is the main function call that will call all
# the functions and return the CAPEX
#######################################################

def get_CAPEX(
        heuristics_file:str = "src/CAPEX_HEURISTICS.xlsx",
        project_size_MW_DC:float = 1,
        DC_AC_conversion:float = 1.3,
        tilt:int = 12,
        module_wattage_DC:int = 600,
        module_price_watt_DC:float = 0.31,
        inverter_price_watt_DC:float = 0.04,
        average_distance_to_shore_ft:float = 100,
        sales_tax_percent:float = 6,
        substation_upgrade_YES_or_NO:str = "NO",
        grid_connection_voltage_kV:float = 167,
        commercial_operation_date:int = 2024, 
        ) -> pd.DataFrame:

    
    # Get the user input system parameters
    user_input = import_user_input(
        project_size_MW_DC,
        DC_AC_conversion,
        tilt,
        module_wattage_DC,
        module_price_watt_DC,
        inverter_price_watt_DC,
        average_distance_to_shore_ft,
        sales_tax_percent,
        substation_upgrade_YES_or_NO,
        grid_connection_voltage_kV)

    heuristics = import_heuristics(heuristics_file)
    
    # Initialize dataframe and add all the CAPEX categories
    CAPEX = pd.DataFrame(columns=["Cost $/W", "Total Cost", "Hard/Soft Cost"])
    CAPEX.loc["Asset Management"] = asset_management_cost(heuristics, user_input)
    CAPEX.loc["Gen-Tie"] = gen_tie_cost(heuristics, user_input)
    CAPEX.loc["General Construction"] = general_construction_cost(heuristics, user_input)
    CAPEX.loc["AC EBOS"] = AC_EBOS_cost(heuristics, user_input)
    CAPEX.loc["Inverter"] = inverter_cost(user_input)
    CAPEX.loc["Anchoring and Mooring"] = anchoring_mooring_cost(heuristics, user_input)
    CAPEX.loc["DC EBOS"] = DC_EBOS_cost(heuristics, user_input)
    CAPEX.loc["Substation"] = substation_cost(heuristics, user_input)
    CAPEX.loc["Engineering"] = development_engineering_cost(heuristics, user_input)
    CAPEX.loc["Racking"]= racking_cost(heuristics, user_input)
    CAPEX.loc["Install Labor"] = install_labor_cost(heuristics, user_input)
    CAPEX.loc["Solar Panels"] = solar_panel_cost(user_input)

    # Now we can adjust based of COD, commercial operation date based on NORIA data
    if commercial_operation_date > 2024:
        
        # Use this number to adjust our values
        number_of_years = commercial_operation_date - 2024

        # Labor increase of 4% per year but decrese 2% per year from industry advancement for a total of 2% per year
        CAPEX.loc["Install Labor", ["Cost $/W", "Total Cost"]] = compound_annual_growth(CAPEX.loc["Install Labor"], rate=2, years=number_of_years)

        # Inflation increase of 2.5% per year. Racking and A&M will reduce by 4% per year through industry advancement for -1.5% per year
        CAPEX.loc["DC EBOS", ["Cost $/W", "Total Cost"]] = compound_annual_growth(CAPEX.loc["DC EBOS"], rate=2.5, years=number_of_years)
        CAPEX.loc["Racking", ["Cost $/W", "Total Cost"]] = compound_annual_growth(CAPEX.loc["Racking"], rate=-1.5, years=number_of_years)
        CAPEX.loc["Anchoring and Mooring", ["Cost $/W", "Total Cost"]] = compound_annual_growth(CAPEX.loc["Anchoring and Mooring"], rate=-1.5, years=number_of_years)
        
    
    # We don't want general construction here. Create list to insert into CAPEX [$/W, TotalCost, "Soft Cost"]
    # Update to metric: op = system_subtotal * [-0.03329 ln(Size_ac) + 0.24022]
    system_subtotal = float(CAPEX.drop("General Construction").iloc[:, 1].sum())
    total_overhead_profit = system_subtotal * ((-0.03329 * np.log(user_input["Project Size (MW_AC)"]) ) + 0.24022)
    overhead_profit_per_W_ac = total_overhead_profit / user_input["Project Size (W_AC)"]
    CAPEX.loc["Overhead and Profit"] = [overhead_profit_per_W_ac, total_overhead_profit, "Soft Cost"]
    
    CAPEX = CAPEX.sort_values(by="Total Cost", ascending=False)

    # Currently the $/W is in ac, we want it in dc
    CAPEX["Cost $/W"] = CAPEX["Cost $/W"] / DC_AC_conversion
    CAPEX = CAPEX.rename(columns={"Cost $/W": "Cost $/W_dc"})
    
    project_total = list(CAPEX.iloc[:, 0:2].sum().values)
    project_total.append("Total Cost")
    CAPEX.loc["Project Cost"] = project_total
    
    CAPEX.index.rename("CAPEX", inplace=True)

    return CAPEX


# New function for compunding increase over years
def compound_annual_growth(amount, rate, years):
    amount_per_watt = amount["Cost $/W"]
    total_amount = amount["Total Cost"]
    for _ in range(years):
        amount_per_watt *= (1 + rate/100)
        total_amount *= (1 + rate/100)
    return amount_per_watt, total_amount

#######################################################
# Import User Input and Heuristics
#######################################################

def import_user_input(
    project_size_MW_DC,
    DC_AC_conversion,
    tilt,
    module_wattage_DC,
    module_price_watt_DC,
    inverter_price_watt_DC,
    average_distance_to_shore_ft,
    sales_tax_percent,
    substation_upgrade_YES_or_NO,
    grid_connection_voltage_kV
    ):

    df = pd.Series()
    df["Project Size (MW_DC)"] = project_size_MW_DC
    df["DC/AC Conversion"] = DC_AC_conversion
    df["Tilt"] = tilt
    df["Module Wattage (W_DC)"] = module_wattage_DC
    df["Module Price ($/W_DC)"] = module_price_watt_DC
    df["Inverter Price ($/W_DC)"] = inverter_price_watt_DC
    df["Average Distance to Shore (ft)"] = average_distance_to_shore_ft
    df["Sales Tax (%)"] = sales_tax_percent / 100
    df["Substation Upgrade (YES/NO)"] = substation_upgrade_YES_or_NO
    df["Grid Connection Voltage (kV)"] = grid_connection_voltage_kV
    
    #TODO: Add a bunch of check here to make sure value from sheet or passed in are valid

    # For example check the tilt angle
    if df["Tilt"] not in [5, 12, 15]:
        print("Tilt angle must be 5, 12, 0r 15")
        sys.exit(1) # Not sure this is the best way, will check once connected to application
    
    # Add additional values to the series
    df["Project Size (MW_AC)"] = df["Project Size (MW_DC)"] / df["DC/AC Conversion"]
    df["Project Size (kW_AC)"] = df["Project Size (MW_AC)"] * 1_000
    df["Project Size (W_AC)"] = df["Project Size (kW_AC)"] * 1_000
    df["Project Size (W_DC)"] = df["Project Size (W_AC)"] * df["DC/AC Conversion"]
    df["Project Size (kW_DC)"] = df["Project Size (kW_AC)"] * df["DC/AC Conversion"]
    
    return df


def import_heuristics(excel_file):

    # Read the data for each one of the Excel tabs ["Substation", "Anchoring", "MW_Run", "MW_Wattage"]
    substation = pd.read_excel(excel_file, sheet_name="Substation")

    anchoring = pd.read_excel(excel_file, index_col=0, sheet_name="Anchoring", header=None).T
    anchoring.dropna(how="any", axis=1, inplace=True)

    MW_Run = pd.read_excel(excel_file, index_col=0, sheet_name="MW_Run", header=None).T

    MW_Wattage = pd.read_excel(excel_file, sheet_name="MW_Wattage", skiprows=[0])
    MW_Wattage.dropna(how="all", inplace=True)
    
    return {"Substation": substation, 
            "Anchoring": anchoring, 
            "MW Run": MW_Run, 
            "MW Wattage": MW_Wattage}

#######################################################
# Get all the Cost: per/W, Total, "Hard/Soft Cost"
#######################################################

def substation_cost(heuristics, user_input):
    
    # The heuristic is a dict, and we want a dataframe of the "Excel tab" we want
    df = heuristics["Substation"].copy()
    
    if user_input["Substation Upgrade (YES/NO)"] == "YES":

        # This is for the Utility grid
        if user_input["Grid Connection Voltage (kV)"] >= 115:
            df = df[df["Type"] == "Utility Grid"]
            total_cost = np.interp(user_input["Project Size (kW_AC)"], df["kW_AC"], df["Cost"])
        elif user_input["Grid Connection Voltage (kV)"] < 115:
            df = df[df["Type"] == "Distribution Grid"]
            total_cost = np.interp(user_input["Project Size (kW_AC)"], df["kW_AC"], df["Cost"])
        else:
            total_cost = np.interp(user_input["Project Size (kW_AC)"], df["kW_AC"], df["Cost"])

    else:
        total_cost = 0

    return total_cost / user_input["Project Size (W_DC)"], total_cost, "Hard Cost"


def anchoring_mooring_cost(heuristics, user_input):

    df = heuristics["Anchoring"].copy()
    
    # Use the rows with correct module and tilt
    df = df[(df["Module Wattage"] == user_input["Module Wattage (W_DC)"]) & (df["Module Tilt"] == user_input["Tilt"])]

    # Interp to get the cost 
    cost =  np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["Cost: Piers/Anchoring ($/W)"])

    # Get additional cost factor based on tilt
    distance = user_input["Average Distance to Shore (ft)"]
    if user_input["Tilt"] == 5:
        adder = distance * 0.00002
    elif user_input["Tilt"] == 12:
        adder = distance * 0.00004
    elif user_input["Tilt"] == 15:
        adder = distance * 0.00005
    
    cost += adder
    
    return cost, cost * user_input["Project Size (W_AC)"], "Hard Cost"


def asset_management_cost(heuristics, user_input):
    
    df = heuristics["MW Run"].copy()
   
    cost = np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["Asset Management"])
 
    cost = cost + (cost * user_input["Sales Tax (%)"] * 0.2935)
    
    return cost, cost * user_input["Project Size (W_AC)"], "Soft Cost"


def general_construction_cost(heuristics, user_input):
    
    df = heuristics["MW Run"].copy()

    cost = np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["General Construction"])

    return cost, cost * user_input["Project Size (W_AC)"], "Soft Cost"


def development_engineering_cost(heuristics, user_input):
    
    df = heuristics["MW Run"].copy()
    
    cost = np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["Development & Engineering"])

    return cost, cost * user_input["Project Size (W_AC)"], "Soft Cost"


def racking_cost(heuristics, user_input):
    
    df = heuristics["MW Run"].copy()

    cost = np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["Racking"])
    cost = cost + (cost * user_input["Sales Tax (%)"] * 0.8026)
    
    return cost, cost * user_input["Project Size (W_AC)"], "Hard Cost"


def gen_tie_cost(heuristics, user_input):

    df = heuristics["MW Run"].copy()
    
    cost = np.interp(user_input["Project Size (kW_AC)"], df["System kW_AC"], df["Gen-Tie"])

    # Gen-Tie cable cost
    cable_cost = {"250 kcmil": 8.05,
                  "300 kcmil": 9.40,
                  "500 kcmil": 15.81,
                  "600 kcmil": 20.21,
                  "750 kcmil": 23.65}
    
    size = user_input["Project Size (MW_AC)"]
    distance = user_input["Average Distance to Shore (ft)"]
    
    if size < 5:
        adder = distance * cable_cost["250 kcmil"] * 3
    elif size < 10:
        adder = distance * cable_cost["250 kcmil"] * 2 * 3
    elif size < 15:
        adder = distance * cable_cost["250 kcmil"] * 3 * 3
    elif size < 20:
        adder = distance * cable_cost["250 kcmil"] * 4 * 3
    elif size < 25:
        adder = distance * cable_cost["250 kcmil"] * 5 * 3
    elif size <= 50:
        adder = distance * cable_cost["500 kcmil"] * 5 * 3
    elif size <= 75:
        adder = distance * cable_cost["600 kcmil"] * 5 * 3
    else:
        adder = distance * cable_cost["750 kcmil"] * 5 * 3
    
    adder /= (size * 1_000_000)
    
    cost += adder
    
    return cost, cost * user_input["Project Size (W_AC)"], "Hard Cost"


def DC_EBOS_cost(heuristics, user_input):
    
    df = heuristics["MW Wattage"].copy()
    
    # Get the data from correct panel size
    df = df[df["Panel Size (W_DC)"] == user_input["Module Wattage (W_DC)"]]
    df.drop("Panel Size (W_DC)", axis=1, inplace=True)
    df.set_index("Sector", inplace=True)
    df = df.T
    df.index.name = "System (kW_AC)"
    
    cost = np.interp(user_input["Project Size (kW_AC)"], df.index, df["DC EBOS"])
    cost = cost + (cost * user_input["Sales Tax (%)"] * 0.9946)

    return cost, cost * user_input["Project Size (W_AC)"], "Hard Cost"


def AC_EBOS_cost(heuristics, user_input):
    
    df = heuristics["MW Wattage"].copy()
    
    # Get the data from correct panel size
    df = df[df["Panel Size (W_DC)"] == user_input["Module Wattage (W_DC)"]]
    df.drop("Panel Size (W_DC)", axis=1, inplace=True)
    df.set_index("Sector", inplace=True)
    df = df.T
    df.index.name = "System (kW_AC)"
    
    cost = np.interp(user_input["Project Size (kW_AC)"], df.index, df["AC EBOS"])
    cost = cost + (cost * user_input["Sales Tax (%)"] * 1.0)
    
    return cost, cost * user_input["Project Size (W_AC)"], "Hard Cost"


def install_labor_cost(heuristics, user_input):
    
    df = heuristics["MW Wattage"].copy()
    
    # Get the data from correct panel size
    df = df[df["Panel Size (W_DC)"] == user_input["Module Wattage (W_DC)"]]
    df.drop("Panel Size (W_DC)", axis=1, inplace=True)
    df.set_index("Sector", inplace=True)
    df = df.T
    df.index.name = "System (kW_AC)"
    
    cost = np.interp(user_input["Project Size (kW_AC)"], df.index, df["Install Labor"])

    return cost, cost * user_input["Project Size (W_AC)"], "Soft Cost"


def inverter_cost(user_input):
    cost = user_input["Inverter Price ($/W_DC)"]
    cost = cost + (cost * user_input["Sales Tax (%)"])
    
    return cost, cost * user_input["Project Size (W_DC)"], "Hard Cost"


def solar_panel_cost(user_input):
    
    cost = user_input["Module Price ($/W_DC)"]
    cost = cost + (cost * user_input["Sales Tax (%)"])
    
    return cost, cost * user_input["Project Size (W_DC)"], "Hard Cost"


