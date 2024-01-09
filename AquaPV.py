__author__ = "Tyler Phillips"
__copyright__ = ""
__credits__ = ["Juan Gallego Calderon, Tanveer Hussain, Mucun Sun"]
__license__ = ""
__version__ = "1.0.1"
__maintainer__ = "Tyler Phillips"
__email__ = "tyler.phillips@inl.gov"
__status__ = "Development"

# Import the needed packages
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.figure_factory as ff


def import_hydro_data(file):
    """
    Hydro generation data file import from csv to pandas dataframe.
    Needs to be a years worth of data and starting on Jan 1st.
    @param file: csv file with two columns [datetime, generation (MW)]
    @return: pandas Dataframe
    """

    # Try to read the file
    try:
        df = pd.read_csv(file)

        # Rename the columns for future use and format datetime as index
        df.columns = ["Date", "Hydro Gen (MW)"]
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df.set_index("Date", inplace=True)

        # Resample to 1 hour timesteps
        df = df.resample("60T").mean()

        # Remove any local datetime attached
        df.index = df.index.tz_localize(None)

    except Exception as e:
        print(f"Could not read data/ {file} \n{e}")
        return None

    return df


def import_solar_data(file):
    """
    Solar generation data file import from csv to pandas dataframe.
    Needs to be a years worth of data and starting on Jan 1st.
    @param file: csv file with two columns [datetime, generation (MW)]
    @return: pandas Dataframe
    """

    # Try to read the file
    try:
        df = pd.read_csv(file)

        # Rename the columns for future use and format datetime as index
        df.columns = ["Date", "Solar Gen (MW)"]
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df.set_index("Date", inplace=True)

        # Resample to 1 hour timesteps
        df = df.resample("60T").mean()

        # Remove any local datetime attached
        df.index = df.index.tz_localize(None)

    except Exception as e:
        print(f"Could not read data/ {file} \n{e}")
        return None

    return df


def import_price_data(file):
    """
    Price data file import from csv to pandas dataframe.
    Needs to be a years worth of data and starting on Jan 1st.
    @param file: csv file with two columns [datetime, electricity price ($)]
    @return: pandas Dataframe
    """

    # Try to read the file
    try:
        df = pd.read_csv(file)

        # Rename the columns for future use and format datetime as index
        df.columns = ["Date", "Price ($/MW)"]
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df.set_index("Date", inplace=True)

        # Resample to 1 hour timesteps
        df = df.resample("60T").mean()

        # Remove any local datetime attached
        df.index = df.index.tz_localize(None)

    except Exception as e:
        print(f"Could not read data/ {file} \n{e}")
        return None

    return df


class AquaPV(object):
    """
    AquaPV object contains all the hydro, solar, and pricing data.
    Methods available for techno-economic analysis and plotting
    """

    def __init__(self,
                 name: str = "Project Name",
                 hydro_file: str = None,
                 solar_file: str = None,
                 price_file: str = None,
                 pv_size_MW: float = 1.0,
                 capex_low: float = 1_120_000,
                 capex_baseline: float = 1_180_000,
                 capex_high: float = 1_270_000,
                 opex_low: float = 7_505,
                 opex_baseline: float = 7_900,
                 opex_high: float = 8_295,
                 incentive_ITC_percent: float = 30,
                 incentive_PTC_cents_per_kWh: float = 2.75,
                 PTC_num_years: float = 10,
                 annual_discount_rate: float = 0.05,
                 life_expectancy: float = 30,
                 ):
        """
        Constructor for the AquaPV object
        @param name: Name of installation
        @param hydro_file: Hydro generation csv file
        @param solar_file: Solar generation csv file
        @param price_file: Electricity price csv file
        @param pv_size_MW: Size of solar PV installation
        @param capex_low: Lower estimate of solar installation [$/MW]
        @param capex_baseline: Estimate of solar installation [$/MW]
        @param capex_high: Upper estimate of solar installation [$/MW]
        @param opex_low: Lower estimate of solar yearly cost [$*MW/yr]
        @param opex_baseline: Estimate of solar yearly cost [$*MW/yr]
        @param opex_high: High estimate of solar yearly cost [$*MW/yr]
        @param incentive_ITC_percent: Investment Tax Credit % (0-100)
        @param incentive_PTC_cents_per_kWh: Production Tax Credit [cents/kWh]
        @param PTC_num_years: Number of years for PTC
        @param annual_discount_rate: Annual discount rate (0-1)
        @param life_expectancy: Number of years solar PV in operation
        """

        # Initialize the object attributes
        self.name = name
        self.pv_size_MW = pv_size_MW

        self.capex_high = capex_high
        self.capex_baseline = capex_baseline
        self.capex_low = capex_low

        self.opex_high = opex_high
        self.opex_baseline = opex_baseline
        self.opex_low = opex_low

        self.incentive_ITC_percent = incentive_ITC_percent
        self.incentive_PTC_cents_per_kWh = incentive_PTC_cents_per_kWh
        self.PTC_num_years = PTC_num_years

        self.annual_discount_rate = annual_discount_rate
        self.life_expectancy = life_expectancy

        # self.data is the main dataframe with hydro, solar, and price data over the life_expectancy
        # This dataframe can only be constructed after the hydro, solar, and price have been input
        self.data = None

        # Setting the hydro if file is given in initialization
        if hydro_file is not None:
            self.hydro = import_hydro_data(hydro_file)
            self.hydro_file = hydro_file
        else:
            self.hydro = None
            self.hydro_file = None

        # Setting the solar if file is given in initialization
        if solar_file is not None:
            self.solar = import_solar_data(solar_file)
            self.solar_file = solar_file
        else:
            self.solar = None
            self.solar_file = None

        # Setting the price if file is given in initialization
        if price_file is not None:
            self.price = import_price_data(price_file)
            self.price_file = price_file
        else:
            self.price = None
            self.price_file = None

        # if hydro, solar, and price are given we can construct the main dataframe
        if hydro_file is not None and solar_file is not None and price_file is not None:
            self.data = self.get_AquaPV_dataframe()

    def get_AquaPV_dataframe(self):
        """
        This is the main method that merges the input hydro, solar, and price data into a single DataFrame

        @return: pandas Dataframe
        """
        hydro = self.hydro.copy()
        solar = self.solar.copy()
        price = self.price.copy()

        # Let's drop the leap day, so we can merge data regardless of year
        hydro = hydro[~((hydro.index.month == 2) & (hydro.index.day == 29))]
        solar = solar[~((solar.index.month == 2) & (solar.index.day == 29))]
        price = price[~((price.index.month == 2) & (price.index.day == 29))]

        def get_index_year_info(df):
            start_year = df.index[0].year
            end_year = df.index[-1].year
            num_years = len(df.index.year.unique())
            return start_year, end_year, num_years

        def get_index_day_info(df):
            start_day = df.index[0].dayofyear
            end_day = df.index[-1].dayofyear
            return start_day, end_day

        # Get needed info about the dataframes
        hydro_start_year, hydro_end_year, hydro_num_years = get_index_year_info(hydro)
        hydro_start_day, hydro_end_day = get_index_day_info(hydro)
        hydro_num_hours = len(hydro.index)

        solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)
        solar_start_day, solar_end_day = get_index_day_info(solar)
        solar_num_hours = len(solar.index)

        price_start_year, price_end_year, price_num_years = get_index_year_info(price)
        price_start_day, price_end_day = get_index_day_info(price)
        price_num_hours = len(solar.index)

        # If data starts on Dec 31st, delete that day and start on Jan 1st
        if hydro_start_day == 365:
            hydro = hydro[~((hydro.index.year == hydro_start_year) & (hydro.index.day == 365))]
            hydro_start_year, hydro_end_year, hydro_num_years = get_index_year_info(hydro)
            hydro_start_day, hydro_end_day = get_index_day_info(hydro)
            hydro_num_hours = len(hydro.index)

        if solar_start_day == 365:
            solar = solar[~((solar.index.year == solar_start_year) & (solar.index.day == 365))]
            solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)
            solar_start_day, solar_end_day = get_index_day_info(solar)
            solar_num_hours = len(solar.index)

        if price_start_day == 365:
            price = price[~((price.index.year == price_start_year) & (price.index.day == 365))]
            price_start_year, price_end_year, price_num_years = get_index_year_info(price)
            price_start_day, price_end_day = get_index_day_info(price)
            price_num_hours = len(price.index)

        # If they all start on day 1 of the year
        if hydro_start_day == 1 and solar_start_day == 1 and price_start_day == 1:

            # Let's align the years
            start_year = max([hydro_start_year, solar_start_year, price_start_year])

            hydro.index = hydro.index + pd.DateOffset(years=(start_year - hydro_start_year))
            solar.index = solar.index + pd.DateOffset(years=(start_year - solar_start_year))
            price.index = price.index + pd.DateOffset(years=(start_year - price_start_year))

            hydro_start_year, hydro_end_year, hydro_num_years = get_index_year_info(hydro)
            solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)
            price_start_year, price_end_year, price_num_years = get_index_year_info(price)

            # First make sure we have at least 98% of a years worth of data
            # This won't check if there is more than one  year of data correct
            must_have_data_percent = 0.98
            if hydro_num_hours < must_have_data_percent * 8760:
                print("Too much hydro generation missing data")
                return None

            if solar_num_hours < must_have_data_percent * 8760:
                print("Too much solar generation missing data")
                return None

            if price_num_hours < must_have_data_percent * 8760:
                print("Too much price missing data")
                return None

            # If we have under a year, lets fill in the few missing points
            if hydro_num_hours < 8760:
                date_index = pd.date_range(start=f"1/1/{start_year}", end=f"12/31/{start_year} 23:00", freq="H")
                hydro = hydro.reindex(date_index)
                hydro.fillna(hydro.mean(), inplace=True)

            if solar_num_hours < 8760:
                date_index = pd.date_range(start=f"1/1/{start_year}", end=f"12/31/{start_year} 23:00", freq="H")
                solar = solar.reindex(date_index)
                solar.fillna(0, inplace=True)

            if price_num_hours < 8760:
                date_index = pd.date_range(start=f"1/1/{start_year}", end=f"12/31/{start_year} 23:00", freq="H")
                price = price.reindex(date_index)
                price.fillna(price.mean(), inplace=True)

            # Now they should be at least a years worth of data
            # Trim them down to fill exactly year(s) worth of data so we can duplicate them
            # If the last day is not 365, we will remove the last partial year of data
            if hydro_end_day != 365:
                hydro = hydro[~(hydro.index.year == hydro_end_year)]
                hydro_start_year, hydro_end_year, hydro_num_years = get_index_year_info(hydro)

            if solar_end_day != 365:
                solar = solar[~(solar.index.year == solar_end_year)]
                solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)

            if price_end_day != 365:
                price = price[~(price.index.year == price_end_year)]
                price_start_year, price_end_year, price_num_years = get_index_year_info(price)

            # Now they should all start at same year and be a "perfect" year(s) worth of data
            # Let's make all of them the same number of years
            max_years = max([self.life_expectancy, hydro_num_years, solar_num_years, price_num_years])

            if hydro_num_years < max_years:

                copy = hydro.copy()
                # Let's add append to the end as long as its not long enough
                while len(hydro.index.year.unique()) < max_years:
                    copy_first_year = copy.index[0].year
                    old_last_year = hydro.index[-1].year
                    temp = copy.copy()
                    add_years = old_last_year - copy_first_year + 1
                    temp.index = temp.index + pd.DateOffset(years=add_years)
                    hydro = pd.concat([hydro, temp])

            if solar_num_years < max_years:

                copy = solar.copy()
                # Let's add append to the end as long as its not long enough
                while len(solar.index.year.unique()) < max_years:
                    copy_first_year = copy.index[0].year
                    old_last_year = solar.index[-1].year
                    temp = copy.copy()
                    add_years = old_last_year - copy_first_year + 1
                    temp.index = temp.index + pd.DateOffset(years=add_years)
                    solar = pd.concat([solar, temp])

            if price_num_years < max_years:

                copy = price.copy()
                # Let's add append to the end as long as its not long enough
                while len(price.index.year.unique()) < max_years:
                    copy_first_year = copy.index[0].year
                    old_last_year = price.index[-1].year
                    temp = copy.copy()
                    add_years = old_last_year - copy_first_year + 1
                    temp.index = temp.index + pd.DateOffset(years=add_years)
                    price = pd.concat([price, temp])

            # Now they should all be the same length so lets merge them into a dataframe
            df = hydro.copy()
            df = df.join(solar, how="left")
            df = df.join(price, how="left")
            df.fillna(0, inplace=True)

            # Trim dataframe based on life expectancy
            if len(df.index.year.unique()) > self.life_expectancy:
                last_year = self.life_expectancy + df.index[0].year
                df = df[df.index < pd.to_datetime(f"{last_year}")]

            # Add all columns we will use here
            df["Hydro Revenue ($)"] = df["Hydro Gen (MW)"] * df["Price ($/MW)"]
            df["Solar Revenue ($)"] = df["Solar Gen (MW)"] * df["Price ($/MW)"]

            df["PTC Incentives ($)"] = df["Solar Gen (MW)"] * self.incentive_PTC_cents_per_kWh * 1000 / 100
            df["PTC Incentives ($)"].iloc[self.PTC_num_years * 8760 + 1:] = 0

            df["Capex High ($)"] = 0
            df.loc[df.index[0], "Capex High ($)"] = self.capex_high * self.pv_size_MW

            df["Capex Baseline ($)"] = 0
            df.loc[df.index[0], "Capex Baseline ($)"] = self.capex_baseline * self.pv_size_MW

            df["ITC Incentives ($)"] = 0
            df.loc[df.index[0], "ITC Incentives ($)"] = self.capex_baseline * self.incentive_ITC_percent / 100

            df["Capex Low ($)"] = 0
            df.loc[df.index[0], "Capex Low ($)"] = self.capex_low * self.pv_size_MW

            df["Opex High ($)"] = self.opex_high / (365 * 24) * self.pv_size_MW
            df["Opex Baseline ($)"] = self.opex_baseline / (365 * 24) * self.pv_size_MW
            df["Opex Low ($)"] = self.opex_low / (365 * 24) * self.pv_size_MW

        else:
            print("Make sure data starts Jan, 1st")

        return df

    #################################
    # Setter methods
    #################################
    def set_name(self, name):
        """
        Set the name of the FPV project
        @param name:
        @return:
        """
        self.name = name

    def set_hydro_data(self, file):

        self.hydro = import_hydro_data(file)
        self.hydro_file = file

        # If we have all the data, construct the dataframe
        if self.solar is not None and self.price is not None:
            self.get_AquaPV_dataframe()

    def set_solar_data(self, file):

        self.solar = import_solar_data(file)
        self.solar_file = file

        # If we have all the data, construct the dataframe
        if self.hydro is not None and self.price is not None:
            self.get_AquaPV_dataframe()

    def set_price_data(self, file):

        self.price = import_price_data(file)
        self.price_file = file

        # If we have all the data, construct the dataframe
        if self.hydro is not None and self.solar is not None:
            self.get_AquaPV_dataframe()

    def set_PV_size_MW(self, PV_size):

        self.pv_size_MW = PV_size

        if self.data is not None:
            self.get_AquaPV_dataframe()

    def set_capex(self, low=None, baseline=None, high=None):

        if low is not None:
            self.capex_low = low

            if self.data is not None:
                self.data["Capex Low ($)"] = 0
                self.data.loc[self.data.index[0], "Capex Low ($)"] = low

        if baseline is not None:
            self.capex_baseline = baseline

            if self.data is not None:
                self.data["Capex Baseline ($)"] = 0
                self.data.loc[self.data.index[0], "Capex Baseline ($)"] = baseline

        if high is not None:
            self.capex_high = high

            if self.data is not None:
                self.data["Capex High ($)"] = 0
                self.data.loc[self.data.index[0], "Capex High ($)"] = high

    def set_opex(self, low=None, baseline=None, high=None):

        if low is not None:
            self.opex_low = low

            if self.data is not None:
                self.data["Opex Low ($)"] = low / (365 * 24)

        if baseline is not None:
            self.opex_baseline = baseline

            if self.data is not None:
                self.data["Opex Baseline ($)"] = baseline / (365 * 24)

        if high is not None:
            self.opex_high = high

            if self.data is not None:
                self.data["Opex High ($)"] = high / (365 * 24)

    def set_ITC_incentive(self, ITC_percent):

        self.incentive_ITC_percent = ITC_percent

        if self.data is not None:
            self.data["ITC Incentives ($)"] = 0
            self.data.loc[self.data.index[0], "ITC Incentives ($)"] = self.capex_baseline * self.incentive_ITC_percent / 100

    def set_PTC_incentive(self, PTC_cents_per_kWh):

        self.incentive_PTC_cents_per_kWh = PTC_cents_per_kWh

        if self.data is not None:
            self.data["PTC Incentives ($)"] = self.data["Solar Gen (MW)"] * self.incentive_PTC_cents_per_kWh * 1000 / 100
            self.data["PTC Incentives ($)"].iloc[self.PTC_num_years * 8760 + 1:] = 0

    def set_PTC_incentive_num_years(self, num_years):

        self.PTC_num_years = num_years

        if self.data is not None:
            self.data["PTC Incentives ($)"] = self.data["Solar Gen (MW)"] * self.incentive_PTC_cents_per_kWh * 1000 / 100
            self.data["PTC Incentives ($)"].iloc[self.PTC_num_years * 8760 + 1:] = 0

    def set_life_expectancy(self, number_of_years):

        self.life_expectancy = number_of_years

        # Only update if we have all needed data
        if self.data is not None:
            self.get_AquaPV_dataframe()

    #################################
    # Getter methods
    #################################

    def get_data_statistics(self):

        stats = self.data.describe()
        return stats

    #################################
    # Financial Metrics
    #################################
    def get_payback_period(self):

        capex_list = ["Capex Low", "Capex Baseline", "Capex High"]
        opex_list = ["Opex Low", "Opex Baseline", "Opex High"]

        # Construct needed dataframe, already have the incentives
        df = self.data.copy()

        # Start dict for results
        results = dict()

        for capex in capex_list:
            for opex in opex_list:

                # Need to add capex and opex
                df["Capex + Opex ($)"] = df[f"{capex} ($)"] + df[f"{opex} ($)"]

                # Get total cost (capex + opex) - incentives
                df["Total Cost ($)"] = df["Capex + Opex ($)"]
                df["Total Cost ITC ($)"] = df["Capex + Opex ($)"] - df[f"ITC Incentives ($)"]
                df["Total Cost PTC ($)"] = df["Capex + Opex ($)"] - df[f"PTC Incentives ($)"]

                payback_date = (df["Solar Revenue ($)"].cumsum() > df["Total Cost ($)"].cumsum()).idxmax()
                payback_num_days = int(str(payback_date - df.index[0]).split(" ")[0])

                ITC_payback_date = (df["Solar Revenue ($)"].cumsum() > df["Total Cost ITC ($)"].cumsum()).idxmax()
                ITC_payback_num_days = int(str(ITC_payback_date - df.index[0]).split(" ")[0])

                PTC_payback_date = (df["Solar Revenue ($)"].cumsum() > df["Total Cost PTC ($)"].cumsum()).idxmax()
                PTC_payback_num_days = int(str(PTC_payback_date - df.index[0]).split(" ")[0])

                if payback_num_days == 0:
                    payback_num_days = np.nan
                    years_frac = np.nan
                    years = np.nan
                    days = np.nan
                else:
                    years_frac = payback_num_days / 365
                    years = int(np.floor(years_frac))
                    days = payback_num_days % 365

                if ITC_payback_num_days == 0:
                    ITC_payback_num_days = np.nan
                    ITC_years_frac = np.nan
                    ITC_years = np.nan
                    ITC_days = np.nan

                else:
                    ITC_years_frac = ITC_payback_num_days / 365
                    ITC_years = int(np.floor(ITC_years_frac))
                    ITC_days = ITC_payback_num_days % 365

                if PTC_payback_num_days == 0:
                    PTC_payback_num_days = np.nan
                    PTC_years_frac = np.nan
                    PTC_years = np.nan
                    PTC_days = np.nan

                else:
                    PTC_years_frac = PTC_payback_num_days / 365
                    PTC_years = int(np.floor(PTC_years_frac))
                    PTC_days = PTC_payback_num_days % 365

                results.update({f"{capex}, {opex}": {"ITC": ITC_years_frac,
                                                     "PTC": PTC_years_frac,
                                                     "None": years_frac}})

        return results

    def get_LCOE(self):

        # Let's now do that for energy, it's the NPV equation with energy instead of cost
        yearly_energy = self.data["Solar Gen (MW)"].groupby(self.data.index.year).sum()
        yearly_energy = npf.npv(self.annual_discount_rate, yearly_energy)

        LCOE = dict()

        # Now lets go over each cost scenario
        for capex in ["Capex Low", "Capex Baseline", "Capex High"]:
            for opex in ["Opex Low", "Opex Baseline", "Opex High"]:
                # Need to find the cost
                yearly_cost = self.data[f"{capex} ($)"] + self.data[f"{opex} ($)"]

                ITC_yearly_cost = yearly_cost - self.data["ITC Incentives ($)"]
                PTC_yearly_cost = yearly_cost - self.data["PTC Incentives ($)"]

                ITC_yearly_cost = ITC_yearly_cost.groupby(self.data.index.year).sum()
                PTC_yearly_cost = PTC_yearly_cost.groupby(self.data.index.year).sum()
                yearly_cost = yearly_cost.groupby(self.data.index.year).sum()

                ITC_cost = npf.npv(self.annual_discount_rate, ITC_yearly_cost)
                PTC_cost = npf.npv(self.annual_discount_rate, PTC_yearly_cost)
                cost = npf.npv(self.annual_discount_rate, yearly_cost)

                ITC_lcoe = ITC_cost / yearly_energy
                PTC_lcoe = PTC_cost / yearly_energy
                lcoe = cost / yearly_energy

                LCOE.update({f"{capex}, {opex}": {"ITC": ITC_lcoe,
                                                  "PTC": PTC_lcoe,
                                                  "None": lcoe}})

        return LCOE

    def get_NPV(self):

        NPV = dict()

        for capex in ["Capex Low", "Capex Baseline", "Capex High"]:
            for opex in ["Opex Low", "Opex Baseline", "Opex High"]:
                # Revenue - Cost + Incentives
                yearly_cash_flow = self.data["Solar Revenue ($)"] - \
                                   self.data[f"{capex} ($)"] - \
                                   self.data[f"{opex} ($)"]

                ITC_yearly_cash_flow = yearly_cash_flow + self.data["ITC Incentives ($)"]
                PTC_yearly_cash_flow = yearly_cash_flow + self.data["PTC Incentives ($)"]

                yearly_cash_flow = yearly_cash_flow.groupby(yearly_cash_flow.index.year).sum()
                ITC_yearly_cash_flow = ITC_yearly_cash_flow.groupby(ITC_yearly_cash_flow.index.year).sum()
                PTC_yearly_cash_flow = PTC_yearly_cash_flow.groupby(PTC_yearly_cash_flow.index.year).sum()

                npv = npf.npv(self.annual_discount_rate, yearly_cash_flow)
                ITC_npv = npf.npv(self.annual_discount_rate, ITC_yearly_cash_flow)
                PTC_npv = npf.npv(self.annual_discount_rate, PTC_yearly_cash_flow)

                NPV.update({f"{capex}, {opex}": {"ITC": ITC_npv,
                                                 "PTC": PTC_npv,
                                                 "None": npv}})

        return NPV

    def get_ROI(self):

        ROI = dict()

        for capex in ["Capex Low", "Capex Baseline", "Capex High"]:
            for opex in ["Opex Low", "Opex Baseline", "Opex High"]:

                # We will consider incentives revenue
                revenue = self.data["Solar Revenue ($)"]
                ITC_revenue = self.data["Solar Revenue ($)"] + self.data["ITC Incentives ($)"]
                PTC_revenue =  self.data["Solar Revenue ($)"] + self.data["PTC Incentives ($)"]

                cost = self.data[f"{capex} ($)"] + self.data[f"{opex} ($)"]
                cost = cost.sum()

                roi = (revenue.sum() - cost) / cost * 100
                ITC_roi = (ITC_revenue.sum() - cost) / cost * 100
                PTC_roi = (PTC_revenue.sum() - cost) / cost * 100


                ROI.update({f"{capex}, {opex}": {"ITC": ITC_roi,
                                                 "PTC": PTC_roi,
                                                 "None": roi}})

        return ROI

    def get_IRR(self):

        IRR = dict()

        for capex in ["Capex Low", "Capex Baseline", "Capex High"]:
            for opex in ["Opex Low", "Opex Baseline", "Opex High"]:

                revenue = self.data["Solar Revenue ($)"] - self.data[f"{capex} ($)"] - self.data[f"{opex} ($)"]

                ITC_revenue = revenue + self.data["ITC Incentives ($)"]
                PTC_revenue = revenue + self.data["PTC Incentives ($)"]

                irr = npf.irr(revenue.groupby(revenue.index.year).sum())
                ITC_irr = npf.irr(ITC_revenue.groupby(ITC_revenue.index.year).sum())
                PTC_irr = npf.irr(PTC_revenue.groupby(PTC_revenue.index.year).sum())

                IRR.update({f"{capex}, {opex}": {"ITC": ITC_irr,
                                                 "PTC": PTC_irr,
                                                 "None": irr}})

        return IRR

    #################################
    # Plotting functions
    #################################

    def plot_settings(self):
        right_legend = dict(font_size=14, yanchor="top", y=0.99, xanchor="right", x=0.99, title=None,
                            bordercolor="black",
                            borderwidth=1)  # , bgcolor='rgba(0,0,0,0.05)'
        left_legend = dict(font_size=14, yanchor="top", y=0.99, xanchor="left", x=0.01, title=None, bordercolor="black",
                           borderwidth=1)  # , bgcolor='rgba(0,0,0,0.05)'

        margin = dict(l=70, r=50, b=60, t=70)
        axis_font_size = 18

        return right_legend, left_legend, margin, axis_font_size

    def plot_hydro(self, high_resolution=False, height=450, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        if high_resolution is True:
            fig = px.line(self.hydro)
        else:
            fig = px.line(self.hydro[::skip_freq])

        fig.update_layout(title=dict(text='Hydro Data', x=0.5, font=dict(size=24)))
        fig.update_layout(height=height, legend=right_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(title="Generation (MW)")
        return fig

    def plot_solar(self, high_resolution=False, height=450, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        if high_resolution is True:
            fig = px.line(self.solar)
        else:
            fig = px.line(self.solar[::skip_freq])

        fig.update_layout(title=dict(text='Solar PV Data', x=0.5, font=dict(size=24)))
        fig.update_layout(height=height, legend=right_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(title="Generation (MW)")
        return fig

    def plot_price(self, high_resolution=False, height=450, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        if high_resolution is True:
            fig = px.line(self.price)
        else:
            fig = px.line(self.price[::skip_freq])

        fig.update_layout(title=dict(text='Price Data', x=0.5, font=dict(size=24)))
        fig.update_layout(height=height, legend=right_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(title="Cost ($/MW)")
        return fig

    def plot_imported_data(self, high_resolution=False, height=450, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        df = self.hydro.copy()
        df = pd.concat([df, self.solar])
        df = pd.concat([df, self.price])

        if high_resolution is True:
            fig = px.line(df)
        else:
            fig = px.line(df[::skip_freq])

        fig.update_layout(title=dict(text='Imported Hydro, Solar, and Price Data', x=0.5, font=dict(size=24)))
        fig.update_layout(height=height, legend=right_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(title="")
        return fig

    def plot_merged_data(self, high_resolution=False, height=450, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        if high_resolution is True:
            fig = px.line(self.data[["Hydro Gen (MW)", "Solar Gen (MW)", "Price ($/MW)"]])
        else:
            fig = px.line(self.data[["Hydro Gen (MW)", "Solar Gen (MW)", "Price ($/MW)"]][::skip_freq])

        fig.update_layout(title=dict(text='Merged & Extrapolated Hydro, Solar, and Price Data', x=0.5, font=dict(size=24)))
        fig.update_layout(height=height, legend=right_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(title="")
        return fig

    def plot_revenue(self, high_resolution=False, skip_freq=10):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        if high_resolution is True:
            fig = px.line(self.data[['Hydro Revenue ($)', 'Solar Revenue ($)']].cumsum())
        else:
            fig = px.line(self.data[['Hydro Revenue ($)',
                                     'Solar Revenue ($)',
                                     'ITC Incentives ($)',
                                     'PTC Incentives ($)']].cumsum()[0::skip_freq])

        fig.update_layout(title=dict(text='Hydro, Solar, and Incentive Revenue', x=0.5, font=dict(size=25)))
        fig.update_layout(height=300, legend=left_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(yaxis_title="Revenue ($)")
        return fig

    def plot_payback_period(self):
        right_legend, left_legend, margin, axis_font_size = self.plot_settings()

        df = self.data.copy()

        df["Solar + ITC ($)"] = df["Solar Revenue ($)"] + df["ITC Incentives ($)"]
        df["Solar + PTC ($)"] = df["Solar Revenue ($)"] + df["PTC Incentives ($)"]

        df["High Capex/Opex ($)"] = (df["Capex High ($)"] + df["Opex High ($)"])
        df["Baseline Capex/Opex ($)"] = df["Capex Baseline ($)"] + df["Opex Baseline ($)"]
        df["Low Capex/Opex ($)"] = df["Capex Low ($)"] + df["Opex Low ($)"]

        df.reset_index(inplace=True)
        df.index = df.index / (365 * 24)

        fig = px.line(df[["Solar Revenue ($)",
                          "Solar + ITC ($)",
                          "Solar + PTC ($)",
                          "High Capex/Opex ($)",
                          "Baseline Capex/Opex ($)",
                          "Low Capex/Opex ($)"
                          ]].cumsum()[::24 * 7])
        fig.update_layout(title=dict(text='Payback Period', x=0.5, font=dict(size=25)))
        fig.update_layout(height=300, legend=left_legend, margin=margin, font=dict(size=axis_font_size))
        fig.update_layout(yaxis_title="Revenue & Cost ($)", xaxis_title='Years')
        fig.update_yaxes(rangemode='tozero')
        fig.update_xaxes(zeroline=False)
        return fig

    def payback_period_table(self):

        payback_period = self.get_payback_period()

        df = pd.DataFrame({'Capex': ['High', 'High', 'High', 'Baseline', 'Baseline', 'Baseline', 'Low', 'Low', 'Low'],
                           'Opex': ['High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', ],
                           'None': [None, None, None, None, None, None, None, None, None],
                           'ITC': [None, None, None, None, None, None, None, None, None],
                           'PTC': [None, None, None, None, None, None, None, None, None], })

        for row in range(0, len(df.index)):
            capex = df.iloc[row, 0]
            opex = df.iloc[row, 1]

            values = payback_period[f"Capex {capex}, Opex {opex}"]

            ITC = values["ITC"]
            PTC = values["PTC"]
            no_incentive = values["None"]

            df.iloc[row, 2] = no_incentive
            df.iloc[row, 3] = ITC
            df.iloc[row, 4] = PTC

        df[["None", "ITC", "PTC"]] = df[["None", "ITC", "PTC"]].applymap("{0:,.1f}".format)

        table = ff.create_table(df)
        table.update_layout(title=dict(text='Payback Period (years)', x=0.5, font=dict(size=25)), margin=dict(t=40))

        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 15

        return table

    def LCOE_table(self):

        payback_period = self.get_LCOE()

        df = pd.DataFrame({'Capex': ['High', 'High', 'High', 'Baseline', 'Baseline', 'Baseline', 'Low', 'Low', 'Low'],
                           'Opex': ['High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', ],
                           'None': [None, None, None, None, None, None, None, None, None],
                           'ITC': [None, None, None, None, None, None, None, None, None],
                           'PTC': [None, None, None, None, None, None, None, None, None], })

        for row in range(0, len(df.index)):
            capex = df.iloc[row, 0]
            opex = df.iloc[row, 1]

            values = payback_period[f"Capex {capex}, Opex {opex}"]

            ITC = values["ITC"]
            PTC = values["PTC"]
            no_incentive = values["None"]

            df.iloc[row, 2] = no_incentive
            df.iloc[row, 3] = ITC
            df.iloc[row, 4] = PTC

        df[["None", "ITC", "PTC"]] = df[["None", "ITC", "PTC"]].applymap("{0:,.1f}".format)

        table = ff.create_table(df)
        table.update_layout(title=dict(text='Levelized Cost of Energy ($/MW)', x=0.5, font=dict(size=25)), margin=dict(t=40))

        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 16

        return table

    def NPV_table(self):

        payback_period = self.get_NPV()

        df = pd.DataFrame({'Capex': ['High', 'High', 'High', 'Baseline', 'Baseline', 'Baseline', 'Low', 'Low', 'Low'],
                           'Opex': ['High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', ],
                           'None': [None, None, None, None, None, None, None, None, None],
                           'ITC': [None, None, None, None, None, None, None, None, None],
                           'PTC': [None, None, None, None, None, None, None, None, None], })

        for row in range(0, len(df.index)):
            capex = df.iloc[row, 0]
            opex = df.iloc[row, 1]

            values = payback_period[f"Capex {capex}, Opex {opex}"]

            ITC = values["ITC"]
            PTC = values["PTC"]
            no_incentive = values["None"]

            df.iloc[row, 2] = no_incentive
            df.iloc[row, 3] = ITC
            df.iloc[row, 4] = PTC

        df[["None", "ITC", "PTC"]] = df[["None", "ITC", "PTC"]].applymap("{0:,.0f}".format)


        table = ff.create_table(df)
        table.update_layout(title=dict(text='Net Present Value ($)', x=0.5, font=dict(size=25)), margin=dict(t=40))

        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 16

        return table

    def ROI_table(self):

        payback_period = self.get_ROI()

        df = pd.DataFrame({'Capex': ['High', 'High', 'High', 'Baseline', 'Baseline', 'Baseline', 'Low', 'Low', 'Low'],
                           'Opex': ['High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', ],
                           'None': [None, None, None, None, None, None, None, None, None],
                           'ITC': [None, None, None, None, None, None, None, None, None],
                           'PTC': [None, None, None, None, None, None, None, None, None], })

        for row in range(0, len(df.index)):
            capex = df.iloc[row, 0]
            opex = df.iloc[row, 1]

            values = payback_period[f"Capex {capex}, Opex {opex}"]

            ITC = values["ITC"]
            PTC = values["PTC"]
            no_incentive = values["None"]

            df.iloc[row, 2] = no_incentive
            df.iloc[row, 3] = ITC
            df.iloc[row, 4] = PTC

        df[["None", "ITC", "PTC"]] = df[["None", "ITC", "PTC"]].applymap("{0:,.1f}".format)

        table = ff.create_table(df)
        table.update_layout(title=dict(text='Return on Investment (%)', x=0.5, font=dict(size=25)), margin=dict(t=40))

        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 16

        return table

    def IRR_table(self):

        payback_period = self.get_IRR()

        df = pd.DataFrame({'Capex': ['High', 'High', 'High', 'Baseline', 'Baseline', 'Baseline', 'Low', 'Low', 'Low'],
                           'Opex': ['High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', 'High', 'Baseline', 'Low', ],
                           'None': [None, None, None, None, None, None, None, None, None],
                           'ITC': [None, None, None, None, None, None, None, None, None],
                           'PTC': [None, None, None, None, None, None, None, None, None], })

        for row in range(0, len(df.index)):
            capex = df.iloc[row, 0]
            opex = df.iloc[row, 1]

            values = payback_period[f"Capex {capex}, Opex {opex}"]

            ITC = values["ITC"]
            PTC = values["PTC"]
            no_incentive = values["None"]

            df.iloc[row, 2] = no_incentive
            df.iloc[row, 3] = ITC
            df.iloc[row, 4] = PTC

        df[["None", "ITC", "PTC"]] = df[["None", "ITC", "PTC"]].applymap("{0:,.1f}".format)

        table = ff.create_table(df)

        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 16

        return table

    def __str__(self):
        # TODO finish this string to print
        out = "*" * 50 + "\n"
        out += self.name + ":\n"
        out += "*" * 50 + "\n"

        out += f"Data Files:\n"
        out += f"\tHydro Gen: {self.hydro_file}\n"
        out += f"\tSolar Gen: {self.solar_file}\n"
        out += f"\tPrice: {self.price_file}\n\n"

        out += f"Capital expense:\n"
        out += f"\tHigh: ${self.capex_high:,.0f}\n"
        out += f"\tBaseline: ${self.capex_baseline:,.0f}\n"
        out += f"\tLow: ${self.capex_low:,.0f}\n\n"

        out += f"Operational expense:\n"
        out += f"\tHigh: ${self.opex_high:,.0f}\n"
        out += f"\tBaseline: ${self.opex_baseline:,.0f}\n"
        out += f"\tLow: ${self.opex_low:,.0f}\n\n"

        out += f"Tax incentives:\n"
        out += f"\tITC: {self.incentive_ITC_percent}%\n"
        out += f"\tPTC: {self.incentive_PTC_cents_per_kWh} cents/kW\n\n"

        out += f"Other:\n"
        out += f"\tAnnual discount rate: {self.annual_discount_rate}\n"
        out += f"\tProject lifetime: {self.life_expectancy} years\n"


        out += "*" * 50 + "\n\n"
        return out
