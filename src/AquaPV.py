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
        df.columns = ["Date", "Solar Generation (MW)"]
        df["Date"] = pd.to_datetime(df["Date"])#, infer_datetime_format=True)
        df.set_index("Date", inplace=True)

        # Resample to 1 hour timesteps
        df = df.resample("60min").mean()

        # Remove any local datetime attached
        df.index = df.index.tz_localize(None)

    except Exception as e:
        print(f"Could not read data from: {file} \n{e}")
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
        df["Date"] = pd.to_datetime(df["Date"])#, infer_datetime_format=True)
        df.set_index("Date", inplace=True)

        # Resample to 1 hour timesteps
        df = df.resample("60min").mean()

        # Remove any local datetime attached
        df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"Could not read data/ {file} \n{e}")
        return None



class AquaPV(object):
    """
    AquaPV object contains all the hydro, solar, and pricing data.
    Methods available for techno-economic analysis and plotting
    """

    def __init__(self,
                 name: str = "Project Name",
                 solar_file: str = None,
                 price_file: str = None,
                 CAPEX_estimate: float = 1_000_000,
                 CAPEX_CI_percent: float = 5,
                 OPEX_estimate: float = 1_000,
                 OPEX_CI_percent: float = 5,
                 ITC_percent: float = 30,
                 PTC_cents_per_kWh: float = 2.75,
                 PTC_number_years: float = 10,
                 annual_discount_rate: float = 5,
                 life_expectancy: float = 30,
                 ):

        # Initialize the object attributes
        self.name = name

        self.CAPEX_baseline = CAPEX_estimate
        self.CAPEX_high = CAPEX_estimate * (1 + (CAPEX_CI_percent / 100))
        self.CAPEX_low = CAPEX_estimate * (1 - (CAPEX_CI_percent / 100))
        
        self.OPEX_baseline = OPEX_estimate
        self.OPEX_high = OPEX_estimate * (1 + (OPEX_CI_percent / 100))
        self.OPEX_low = OPEX_estimate * (1 - (OPEX_CI_percent / 100))
        
        self.ITC_percent = ITC_percent
        self.PTC_cents_per_kWh = PTC_cents_per_kWh
        self.PTC_number_years = PTC_number_years

        self.annual_discount_rate = annual_discount_rate / 100
        self.life_expectancy = life_expectancy

        # self.data is the main dataframe with hydro, solar, and price data over the life_expectancy
        # This dataframe can only be constructed after the solar and price have been input
        self.data = None

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

        # if solar and price are given we can construct the main dataframe
        if solar_file is not None and price_file is not None:
            self.data = self.get_AquaPV_dataframe()


    def get_AquaPV_dataframe(self):
        """
        This is the main method that merges the input solar and price data into a single DataFrame

        @return: pandas Dataframe
        """

        # Try to import the data, if not return string
        try:
            solar = self.solar.copy()
            price = self.price.copy()

            # Let's drop the leap day, so we can merge the data regardless of year
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
            

            solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)
            solar_start_day, solar_end_day = get_index_day_info(solar)
            solar_num_hours = len(solar.index)

            price_start_year, price_end_year, price_num_years = get_index_year_info(price)
            price_start_day, price_end_day = get_index_day_info(price)
            price_num_hours = len(solar.index)

            # If data starts on Dec 31st, delete that day and start on Jan 1st
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
            if solar_start_day == 1 and price_start_day == 1:

                # Let's align the years
                start_year = max([solar_start_year, price_start_year])

                solar.index = solar.index + pd.DateOffset(years=(start_year - solar_start_year))
                price.index = price.index + pd.DateOffset(years=(start_year - price_start_year))

                solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)
                price_start_year, price_end_year, price_num_years = get_index_year_info(price)


                #######################################################################################
                # None of this works because we fill in missing data on the import, move it there later
                #######################################################################################
                # # First make sure we have at least 98% of a years worth of data
                # # This won't check if there is more than one year of data correct
                # must_have_data_percent = 0.98
                # if solar_num_hours < must_have_data_percent * 8760:
                #     print("Too much solar generation missing data")
                #     return None

                # if price_num_hours < must_have_data_percent * 8760:
                #     print("Too much price missing data")
                #     return None


                # If we have under a year, lets fill in the few missing points
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
                if solar_end_day < 365:
                    solar = solar[~(solar.index.year == solar_end_year)]
                    solar_start_year, solar_end_year, solar_num_years = get_index_year_info(solar)

                if price_end_day < 365:
                    price = price[~(price.index.year == price_end_year)]
                    price_start_year, price_end_year, price_num_years = get_index_year_info(price)


                # Now they should all start at same year and be a "perfect" year(s) worth of data
                # Let's make all of them the same number of years
                max_years = self.life_expectancy

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
                df = solar.copy()
                df = df.join(price, how="left")
                df.fillna(0, inplace=True)

                # Trim dataframe based on life expectancy
                if len(df.index.year.unique()) > self.life_expectancy:
                    last_year = self.life_expectancy + df.index[0].year
                    df = df[df.index < pd.to_datetime(f"{last_year}")]


                # Add all columns we will use here
                df["Solar Revenue ($)"] = df["Solar Generation (MW)"] * df["Price ($/MW)"]

                df["PTC Incentives ($)"] = df["Solar Generation (MW)"] * self.PTC_cents_per_kWh * 1000 / 100
                # df["PTC Incentives ($)"].iloc[self.PTC_number_years * 8760 + 1:] = 0
                df.iloc[self.PTC_number_years * 8760 + 1:, len(df.columns)-1] = 0

                df["CAPEX High ($)"] = 0
                df.loc[df.index[0], "CAPEX High ($)"] = self.CAPEX_high

                df["CAPEX Baseline ($)"] = 0
                df.loc[df.index[0], "CAPEX Baseline ($)"] = self.CAPEX_baseline

                df["CAPEX Low ($)"] = 0
                df.loc[df.index[0], "CAPEX Low ($)"] = self.CAPEX_low

                df["ITC Incentives ($)"] = 0
                df.loc[df.index[0], "ITC Incentives ($)"] = self.CAPEX_baseline * self.ITC_percent / 100

                df["OPEX High ($)"] = self.OPEX_high / (365 * 24) 
                df["OPEX Baseline ($)"] = self.OPEX_baseline / (365 * 24) 
                df["OPEX Low ($)"] = self.OPEX_low / (365 * 24) 

            else:
                print("Make sure data starts Jan, 1st")

            return df
        except:
            return "Error loading data, check format against example provided"


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

    def set_solar_data(self, file):

        self.solar = import_solar_data(file)
        self.solar_file = file

        # If we have all the data, construct the dataframe
        if self.price is not None:
            self.data = self.get_AquaPV_dataframe()

    def set_price_data(self, file):

        self.price = import_price_data(file)
        self.price_file = file

        # If we have all the data, construct the dataframe
        if self.solar is not None:
            self.data = self.get_AquaPV_dataframe()


    def set_(self, low=None, baseline=None, high=None):

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
            self.data = self.get_AquaPV_dataframe()


    def set_annual_discount_rate(self, annual_discount_rate):

        # Input will be value 0 to 100 but needs to be decimal value
        self.annual_discount_rate = annual_discount_rate / 100

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

        # Construct needed dataframe, already have the incentives
        df = self.data.copy()

        # Dataframe for the results
        results = pd.DataFrame(index=[x for x in range(1, 10)], 
                               columns=["CAPEX", "OPEX", "ITC", "PTC", "None"])

        i = 0;
        for capex in ["CAPEX Low", "CAPEX Baseline", "CAPEX High"]:
            for opex in ["OPEX Low", "OPEX Baseline", "OPEX High"]:

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

                # If no payback reached
                if payback_num_days == 0:
                    years_frac = np.nan
                    years = np.nan
                    days = np.nan
                else:
                    years_frac = payback_num_days / 365
                    years = int(np.floor(years_frac))
                    days = payback_num_days % 365

                # If no payback reached
                if ITC_payback_num_days == 0:
                    ITC_years_frac = np.nan
                    ITC_years = np.nan
                    ITC_days = np.nan

                else:
                    ITC_years_frac = ITC_payback_num_days / 365
                    ITC_years = int(np.floor(ITC_years_frac))
                    ITC_days = ITC_payback_num_days % 365
                
                # If no payback reached
                if PTC_payback_num_days == 0:
                    PTC_years_frac = np.nan
                    PTC_years = np.nan
                    PTC_days = np.nan

                else:
                    PTC_years_frac = PTC_payback_num_days / 365
                    PTC_years = int(np.floor(PTC_years_frac))
                    PTC_days = PTC_payback_num_days % 365

                results.iloc[i, :] = [capex.split(" ")[1], 
                                      opex.split(" ")[1], 
                                      ITC_years_frac, 
                                      PTC_years_frac, 
                                      years_frac]
                i += 1

        return results

    def get_LCOE(self):

        # Let's now do that for energy, it's the NPV equation with energy instead of cost
        yearly_energy = self.data["Solar Generation (MW)"].groupby(self.data.index.year).sum()
        yearly_energy = npf.npv(self.annual_discount_rate, yearly_energy)

        # Dataframe for the results
        results = pd.DataFrame(index=[x for x in range(1, 10)], 
                               columns=["CAPEX", "OPEX", "ITC", "PTC", "None"])

        i = 0;
        # Now lets go over each cost scenario
        for capex in ["CAPEX Low", "CAPEX Baseline", "CAPEX High"]:
            for opex in ["OPEX Low", "OPEX Baseline", "OPEX High"]:
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

                results.iloc[i, :] = [capex.split(" ")[1], 
                                      opex.split(" ")[1], 
                                      ITC_lcoe, 
                                      PTC_lcoe, 
                                      lcoe]
                i += 1

        return results

    def get_NPV(self):

        # Dataframe for the results
        results = pd.DataFrame(index=[x for x in range(1, 10)], 
                               columns=["CAPEX", "OPEX", "ITC", "PTC", "None"])

        i = 0;

        for capex in ["CAPEX Low", "CAPEX Baseline", "CAPEX High"]:
            for opex in ["OPEX Low", "OPEX Baseline", "OPEX High"]:
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

                results.iloc[i, :] = [capex.split(" ")[1], 
                                      opex.split(" ")[1], 
                                      ITC_npv, 
                                      PTC_npv, 
                                      npv]
                i += 1

        return results

    def get_ROI(self):

        # Dataframe for the results
        results = pd.DataFrame(index=[x for x in range(1, 10)], 
                               columns=["CAPEX", "OPEX", "ITC", "PTC", "None"])

        i = 0;
        for capex in ["CAPEX Low", "CAPEX Baseline", "CAPEX High"]:
            for opex in ["OPEX Low", "OPEX Baseline", "OPEX High"]:

                # We will consider incentives revenue
                revenue = self.data["Solar Revenue ($)"]
                ITC_revenue = self.data["Solar Revenue ($)"] + self.data["ITC Incentives ($)"]
                PTC_revenue =  self.data["Solar Revenue ($)"] + self.data["PTC Incentives ($)"]

                cost = self.data[f"{capex} ($)"] + self.data[f"{opex} ($)"]
                cost = cost.sum()

                roi = (revenue.sum() - cost) / cost * 100
                ITC_roi = (ITC_revenue.sum() - cost) / cost * 100
                PTC_roi = (PTC_revenue.sum() - cost) / cost * 100

                results.iloc[i, :] = [capex.split(" ")[1], 
                                      opex.split(" ")[1], 
                                      ITC_roi, 
                                      PTC_roi, 
                                      roi]
                i += 1

        return results

    def get_IRR(self):

        # Dataframe for the results
        results = pd.DataFrame(index=[x for x in range(1, 10)], 
                               columns=["CAPEX", "OPEX", "ITC", "PTC", "None"])

        i = 0;
        for capex in ["CAPEX Low", "CAPEX Baseline", "CAPEX High"]:
            for opex in ["OPEX Low", "OPEX Baseline", "OPEX High"]:
                revenue = self.data["Solar Revenue ($)"] - self.data[f"{capex} ($)"] - self.data[f"{opex} ($)"]

                ITC_revenue = revenue + self.data["ITC Incentives ($)"]
                PTC_revenue = revenue + self.data["PTC Incentives ($)"]

                irr = npf.irr(revenue.groupby(revenue.index.year).sum())
                ITC_irr = npf.irr(ITC_revenue.groupby(ITC_revenue.index.year).sum())
                PTC_irr = npf.irr(PTC_revenue.groupby(PTC_revenue.index.year).sum())

                results.iloc[i, :] = [capex.split(" ")[1], 
                                      opex.split(" ")[1], 
                                      ITC_irr * 100, 
                                      PTC_irr * 100, 
                                      irr * 100]
                i += 1

        return results

    #################################
    # Plotting functions
    #################################

    def format_plot(self, fig, main_title='', x_title='', y_title='', height=450, margin=None, legend_location="top-right", legend_title=""):

        if margin is None:
            margin = dict(t=75, b=75, l=75, r=25)

        fig.update_xaxes(title=x_title, showgrid=True,
                         title_font=dict(color="black", size=20),
                         tickfont=dict(size=16, color='black')
                         )
        fig.update_yaxes(title=y_title, showgrid=True, title_font=dict(color="black", size=20),
                         tickfont=dict(size=16, color='black'))
        fig.update_layout(title=dict(text=main_title,
                                     font=dict(size=24, color='black'),
                                     x=0.5,
                                     xanchor='center'
                                     ),
                          height=height,
                          margin=margin,
                          paper_bgcolor='rgba(251, 252, 252, 1)',
                          plot_bgcolor='rgba(251, 252, 252, 1)',
                          legend_title=legend_title,
                          legend=dict(title_font_color="black",
                                      title_font_size=14,
                                      font=dict(color="black", size=12),
                                      bgcolor='rgba(255, 255, 255, 0.8)',
                                      bordercolor='rgba(50, 50, 50, 0.8)',
                                      borderwidth=1,
                                      ),
                          )
        if legend_location == "top-right":
            fig.update_layout(legend=dict(xanchor='right',
                                          x=0.99,
                                          y=0.99))
        elif legend_location == 'top-left':
            fig.update_layout(legend=dict(x=0.01,
                                          y=0.99))

        return fig


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

    def plot_imported_data(self, high_resolution=False, height=400, skip_freq=10):

        df = self.hydro.copy()
        
        df = pd.concat([df, self.solar])
        df = pd.concat([df, self.price])

        if high_resolution is True:
            fig = px.line(df, template='plotly_white')
        else:
            fig = px.line(df[::skip_freq])

        fig = self.format_plot(fig,
                               main_title="Imported Data",
                               x_title='Date',
                               y_title='Values',
                               height=350,
                               margin=None,
                               legend_location="top-left")

        return fig

    def plot_merged_data(self, high_resolution=False, height=350, skip_freq=30):
        

        if high_resolution is True:
            fig = px.line(self.data[["Solar Generation (MW)", "Price ($/MW)"]])
        else:
            fig = px.line(self.data[["Solar Generation (MW)", "Price ($/MW)"]][::skip_freq])

        # fig = self.format_plot(fig, main_title='Merged & Extrapolated Hydro, Solar, and Price Data', x_title='Date', y_title='Values', height=350, margin=None, legend_location="top-right")

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

        fig = self.format_plot(fig, main_title='Revenue ($)', x_title='Date',
                               y_title='Revenue ($)', height=350, margin=None, legend_location="top-left")


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

        fig = self.format_plot(fig, main_title='Revenue, Cost, Payback Period', x_title='Years',
                               y_title='Revenue & Cost ($)', height=350, margin=None, legend_location="top-right")

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
        table.update_layout(title=dict(text='Payback Period (years)', x=0.5, xanchor='center', font=dict(size=25)), margin=dict(t=40))

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
        table.update_layout(title=dict(text='Levelized Cost of Energy ($/MW)', x=0.5, xanchor='center', font=dict(size=25)), margin=dict(t=40))

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
        table.update_layout(title=dict(text='Net Present Value ($)', x=0.5, xanchor='center', font=dict(size=25)), margin=dict(t=40))

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
        table.update_layout(title=dict(text='Return on Investment (%)', x=0.5, xanchor='center', font=dict(size=25)), margin=dict(t=40))

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

        out = "*" * 50 + "\n"
        out += self.name + ":\n"
        out += "*" * 50 + "\n"

        out += f"Capital expense ($):\n"
        out += f"\tHigh: ${self.CAPEX_high:,.0f}\n"
        out += f"\tBaseline: ${self.CAPEX_baseline:,.0f}\n"
        out += f"\tLow: ${self.CAPEX_low:,.0f}\n\n"

        out += f"Operational expense ($/year):\n"
        out += f"\tHigh: {self.OPEX_high:,.0f}\n"
        out += f"\tBaseline: {self.OPEX_baseline:,.0f}\n"
        out += f"\tLow: {self.OPEX_low:,.0f}\n\n"

        out += f"Tax incentives:\n"
        out += f"\tITC: {self.ITC_percent}%\n"
        out += f"\tPTC: {self.PTC_cents_per_kWh} cents/kW\n\n"

        out += f"Other:\n"
        out += f"\tAnnual discount rate: {self.annual_discount_rate}\n"
        out += f"\tProject lifetime: {self.life_expectancy} years\n"

        out += "*" * 50 + "\n\n"
        return out
