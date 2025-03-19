# Copyright 2024, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pgeocode

from datetime import datetime

from pvlib.iotools import get_pvgis_hourly
import plotly.express as px
import plotly

import json
import pandas as pd
import numpy as np


def getSolarGen(
        zipcode, 
        size=1, 
        dc_ac_ratio=1.3, 
        efficiency=90, 
        tilt=10, 
        num_years=1
        ):

    # Get info from zipcode
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(zipcode)

    city = location["place_name"]
    state = location["state_name"]
    county = location['county_name']
    latitude = location['latitude']
    longitude = location['longitude'] 

    # Now get the solar output for the location
    end_year = 2020
    start_year = 2020 - num_years +1
    start_date = datetime.strptime(f'01-01-{start_year} 0', '%m-%d-%Y %H').date()
    end_date = datetime.strptime(f'12-31-{end_year} 23', '%m-%d-%Y %H').date()

    df = get_pvgis_hourly(latitude=latitude,
                            longitude=longitude,
                            start=start_date,
                            end=end_date,
                            raddatabase="PVGIS-ERA5",
                            components=False,
                            surface_tilt=tilt,
                            surface_azimuth=180,
                            outputformat='json',
                            usehorizon=True,
                            userhorizon=None,
                            pvcalculation=True,
                            peakpower=size,
                            pvtechchoice='crystSi',  # panel_type,
                            mountingplace='free',
                            loss=100 - efficiency,
                            trackingtype=0,
                            optimal_surface_tilt=False,
                            optimalangles=False,
                            url='https://re.jrc.ec.europa.eu/api/v5_2/',
                            map_variables=True,
                            timeout=30)[0]
    

    # This is in UTC time and we want data on the hour
    df.index = df.index.tz_localize(None)
    df = df.resample('60min').mean()

    df.rename(columns={"P": "Solar Generation (MW)"}, inplace=True)
    df.index.rename("Date", inplace=True)
    df["Solar Generation (MW)"] = df["Solar Generation (MW)"] / 1000

    max_output = size / dc_ac_ratio
    df.loc[df["Solar Generation (MW)"] > max_output , "Solar Generation (MW)"] = max_output

    return df["Solar Generation (MW)"]