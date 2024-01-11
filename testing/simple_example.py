import AquaPV as apv

# Create an AquaPV object and select the hydro, solar, and price data files
obj = apv.AquaPV(name="Simple AquaPV example",
                 hydro_file="example_data/Hydro_gen_1MW.csv",
                 solar_file="example_data/CAISO_PV_gen.csv",
                 price_file="example_data/CAISO_price.csv")

# Look at the object information
print(obj)

# Inspect the data we imported
obj.plot_imported_data().show()

# Inspect how it is merged together and extrapolated
obj.plot_merged_data().show()

# Look at financial metrics
obj.payback_period_table().show()

# Update the capex and lifetime
obj.set_capex(low=900_000, baseline=1_000_000, high=1_100_000)
obj.set_life_expectancy(25)

# Look at the object information
print(obj)

# Look at financial metrics
obj.payback_period_table().show()

