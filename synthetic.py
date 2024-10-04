"""
This file containts functions and classes to generate synthetic 
equilibriums (e.g., market clearing demand, supply and price using a wind instrument). 

It builds on top of model.py and database.py to retrieve and transform the data
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.optimize as opt

from enum import Enum, auto
from functools import lru_cache, cached_property
from dataclasses import dataclass, field
from model import interval, HourlySeries, Dummy, average
from database import Areas, PriceArea
from estimation import ar_model_fit, ar_model_analysis, ar_manual, ArParams, exog_list
from itertools import product


from typing import Any, Callable
from datetime import datetime, timedelta, timezone, date


@dataclass(frozen=True)
class Consumption:
    """the purpose of this class is to be able to hash a pd.Series to make all processes faster while ensuring that
    both the synthetic and real data can share the functions in estimation.py

    start: pd.Timestamp
    end: pd.Timestamp
    price_area: PriceArea
    """

    start: pd.Timestamp
    end: pd.Timestamp
    price_area: PriceArea

    @cached_property
    def interval(self):
        return interval(start=self.start, end=self.end)

    @lru_cache
    def __call__(self) -> pd.Series:
        return HourlySeries(self.interval).consumption(price_area=self.price_area)


class Country:
    """Use this class to manage the cut-offs and inconsistencies of empirical data.

    This class queries the database and addresses the known inconsistencies of the data,
    such as the change in the German bidding zone in 2018 or the adoption of EUR as the
    currency in Poland.
    
    It also manages the removal of national holidays from the series for the application
    to real German data"""

    def __init__(self, area: PriceArea, series: HourlySeries, remove_holidays:bool) -> None:
        self.area: PriceArea = area
        self.series: HourlySeries = series

        # Filter INTERVAL based on holidays where value is 1
        if remove_holidays:
            print(len(self.series.timestamps))
            holidays_data = Dummy(series.timestamps).public_holiday(area)
            interval = series.timestamps[~(holidays_data == 1)]
            self.series = HourlySeries(interval)
            print(len(self.series.timestamps))

    def clip(self, series:pd.Series) -> pd.Series:
        return series.loc[series.index.isin(self.series.timestamps)]
    
    def handle_german_bidding_zone_split(self, attribute: str):
        """Handle the German bidding zone split after 2018."""

        _attribute = attribute if attribute != "wind_generation" else "total_wind_generation"

        if self.area in [Areas.DE1, Areas.DE2]:
            if (
                self.series.timestamps.min() < pd.Timestamp("201809302300", tz="Europe/Berlin")
                and pd.Timestamp("201810010000", tz="Europe/Berlin") < self.series.timestamps.max()
            ):
                series1 = HourlySeries(
                    interval(
                        start=self.series.timestamps.min(),
                        end=pd.Timestamp("201809302300", tz="Europe/Berlin"),
                    )
                )
                series2 = HourlySeries(
                    interval(
                        start=pd.Timestamp("201810010000", tz="Europe/Berlin"),
                        end=self.series.timestamps.max(),
                    )
                )
                return self.clip(
                    pd.concat(
                        [
                            getattr(series1, _attribute)(price_area=Areas.DE1),
                            getattr(series2, _attribute)(price_area=Areas.DE2),
                        ]
                    )
                    .ffill()
                    .rename(attribute)  # type: ignore
                )
        return None

    @cached_property
    def consumption(self):
        result = self.handle_german_bidding_zone_split('consumption')
        if result is not None:
            return result
        return self.clip(
            self.series.consumption(price_area=self.area).ffill().rename("consumption")
        )

    @cached_property
    def price(self):
        result = self.handle_german_bidding_zone_split('price')
        if result is not None:
            return result
        
        elif self.area == Areas.PL:
            """WARNING for Day-Ahead Prices in Poland:

            Please note that due to distinct market arrangements for certain periods Day-ahead Prices are expressed in different currencies:
            From 5 January 2015 Day-ahead Prices for the Polish Area are expressed in EUR/MWh.
            Between 2 March 2017 and 19 November 2019 Day-ahead Prices are published in PLN/MWh.
            Starting from 20 November 2019 Day-ahead Prices are equal to fixed auction prices from the Single Day-ahead Coupling in EUR/MWh.

            We set a reference exchange rate PLN/EUR to be 0.23 for the whole period."""

            series = self.series.price(price_area=self.area).rename("price").ffill()
            mask = (
                series.index >= pd.Timestamp("201703020000", tz="Europe/Berlin")
            ) & (series.index <= pd.Timestamp("201911192300", tz="Europe/Berlin"))
            series[mask] *= 0.23
            return series

        return self.series.price(price_area=self.area).rename("price").ffill()

    @cached_property
    def wind_speed(self):
        result = self.handle_german_bidding_zone_split('wind_speed')
        if result is not None:
            return result
        else:
            return self.clip(
                self.series.wind_speed(price_area=self.area)
                .ffill()
                .rename("wind_speed")
            )

    @cached_property
    def wind_generation(self):
        result = self.handle_german_bidding_zone_split('wind_generation')
        if result is not None:
            return result
        elif self.area == Areas.PL or self.area == Areas.FR:
            return self.clip(
                self.series.onshore_wind_generation(price_area=self.area)
                .ffill()
                .rename("wind_generation")
            )
        else:
            return self.clip(
                self.series.total_wind_generation(price_area=self.area)
                .ffill()
                .rename("wind_generation")
            )


@dataclass
class RandomCached:
    """Class to generate a random normal error e.g. white noise for Demand and Supply
    It stores the normally distributed error by timestamp so that it can always be retrieved for the optimization process

    Create class instance calling RandomCached.normal(mu, sigma)"""

    random_generator: Callable[[], float]
    cache: dict[Any, float] = field(default_factory=dict)

    def __call__(self, value) -> float:
        try:
            return self.cache[value]
        except KeyError:
            self.cache[value] = self.random_generator()
            return self.cache[value]

    @staticmethod
    def normal(mu, sigma) -> "RandomCached":
        return RandomCached(
            random_generator=lambda: np.random.normal(loc=mu, scale=sigma)
        )

    def __repr__(self) -> str:
        return "error[...]"


@dataclass
class AutocorrelatedCached:
    """Class to generate an autocorrelated error or OVB for Demand or Supply
    It stores the error in a dictionary with key:timestamp so that it can always be retrieved for the optimization process

    Create class instance calling RandomCached.normal(mu, sigma, lags_dict)

    Currently the lags_dict is set by default to 1 lag with alpha=0.2 and there is no implementation to modify it through the function "generate_equilibrium" and derivates
    """

    random_generator: Callable[[], float]
    lags_dict: dict[int, float] = field(default_factory=dict)
    cache: dict[Any, float] = field(default_factory=dict)

    def __call__(self, value) -> float:
        try:
            return self.cache[value]
        except KeyError:
            previous_errors = {}
            for lag in self.lags_dict.keys():
                if lag > len(self.cache):
                    pass
                else:
                    previous_errors[lag] = self.cache[value - timedelta(hours=lag)]

            ar = (
                sum((previous_errors[l] * coef for l, coef in self.lags_dict.items()))
                if previous_errors
                else 0
            )
            self.cache[value] = self.random_generator() + ar
            return self.cache[value]

    @staticmethod
    def normal(mu, sigma, lags_dict={1: 0.2}) -> "AutocorrelatedCached":
        """note that method name is a inaccurate"""
        return AutocorrelatedCached(
            random_generator=lambda: np.random.normal(loc=mu, scale=sigma),
            lags_dict=lags_dict,
        )

    def __repr__(self) -> str:
        return "error[...]"


class WindOptions(Enum):
    """Pass an option of this class to function get_wind_series(wind_type=WindOptions)

    Options:
    --------
    - ACTUAL: the actual series of wind generation
    - RESIDUAL: a time series of the errors of regressing actual wind on its controls
    - SHUFFLED: a wind realistic in scale but which follows an AR(0) process
    - SYNTHETIC_AR_N: specify number of lags for AR(p)
    """

    ACTUAL = auto()
    RESIDUAL = auto()
    SHUFFLED = auto()
    SYNTHETIC_AR_N = auto()

    
def get_wind_series(
    wind_type: WindOptions,
    series: HourlySeries,
    price_area: PriceArea,

    residual_lags: int = 26,

    synthetic_nlags: int | None = None,
    synthetic_analysis_series: HourlySeries = HourlySeries(
        interval("201901290000", "202001282300")
    ),
    synthetic_analysis_dummy: Dummy = Dummy(interval("201901290000", "202001282300")),
    synthetic_n_observations: int = 87600,
    use_controls_for_synthetic_wind: bool = False,
    manual_lags_sum: float | None = None,
) -> pd.Series:
    """Retrieves a "wind" pd.Series to be used as instrument

    Parameters:
    ----------
    wind_type: WindOptions,
    series: HourlySeries,
    price_area: PriceArea
    synthetic_nlags: int | None = None
        Needs to be set as a positive integer if wind_type is WindOptions.SYNTHETIC_AR_N,
    residual_lags: int = 26 -> residual wind seasonality"""

    match wind_type:
        case WindOptions.ACTUAL:
            if (price_area == Areas.DE1 or price_area == Areas.DE2) and (
                series.timestamps.min()
                < pd.Timestamp("201809302300", tz="Europe/Berlin")
                and pd.Timestamp("201810010000", tz="Europe/Berlin")
                < series.timestamps.max()
            ):
                series1 = HourlySeries(
                    interval(
                        start=series.timestamps.min(),
                        end=pd.Timestamp("201809302300", tz="Europe/Berlin"),
                    )
                )
                series2 = HourlySeries(
                    interval(
                        start=pd.Timestamp("201810010000", tz="Europe/Berlin"),
                        end=series.timestamps.max(),
                    )
                )
                return (
                    pd.concat(
                        [
                            series1.total_wind_generation(price_area=Areas.DE1).div(
                                1000
                            ),
                            series2.total_wind_generation(price_area=Areas.DE2).div(
                                1000
                            ),
                        ]
                    )
                    .ffill()
                    .rename("wind_generation")
                )
            elif price_area == Areas.PL or price_area == Areas.FR:
                return (
                    series.onshore_wind_generation(price_area=price_area)
                    .ffill()
                    .rename("wind_generation")
                )
            else:
                return (
                    series.total_wind_generation(price_area=price_area)
                    .ffill()
                    .rename("wind_generation")
                )

        case WindOptions.RESIDUAL:
            return (
                sm.OLS(
                    endog=get_wind_series(
                        wind_type=WindOptions.ACTUAL,
                        series=series,
                        price_area=price_area,
                    ),
                    exog=exogenous_variables(
                        start=series.timestamps.min(),
                        end=series.timestamps.max(),
                        price_area=price_area,
                    ),
                )
                .fit()
                .resid.rename("residual wind")
            )

        case WindOptions.SHUFFLED:
            actual_wind = get_wind_series(
                wind_type=WindOptions.ACTUAL, series=series, price_area=price_area
            )
            return pd.Series(
                np.random.permutation(actual_wind),
                index=actual_wind.index,
                name="shuffled wind",
            )

        case WindOptions.SYNTHETIC_AR_N:
            assert isinstance(synthetic_nlags, int) and synthetic_nlags > 0, "Specify a valid number of lags"
            actual_wind = get_wind_series(
                wind_type=WindOptions.ACTUAL,
                series=synthetic_analysis_series,
                price_area=price_area,
            )

            residuals = (
                sm.OLS(
                    actual_wind,
                    pd.concat(
                        exog_list(
                            series=synthetic_analysis_series,
                            dummy=synthetic_analysis_dummy,
                            price_area=price_area,
                        ),
                        axis=1,
                    ),
                )
                .fit()
                .resid
            )
            dep_variable = residuals if use_controls_for_synthetic_wind else actual_wind

            params = ar_model_analysis(
                ar_model_fit(series=dep_variable, lags=synthetic_nlags, exog=None)  # type: ignore
            )

            if manual_lags_sum is not None:
                actual_lags_sum: float = sum(params.lags_dict.values())
                lags_dict: dict[int, float] = {
                    k: v / actual_lags_sum * manual_lags_sum
                    for k, v in params.lags_dict.items()
                }
                params = ar_manual(dep_variable, lags=lags_dict)

            values = []

            for i in range(synthetic_n_observations + 672):
                if i < synthetic_nlags:
                    values.append(
                        actual_wind.mean()
                        + np.random.normal(loc=0.0, scale=params.error_sd * 10)
                    )
                else:
                    values.append(
                        params.constant
                        + sum(
                            values[i - lag] * params.lags_dict[-lag]
                            for lag in range(1, synthetic_nlags + 1)
                        )
                        + np.random.normal(loc=0.0, scale=params.error_sd)
                    )

            return pd.Series(
                values,
                index=interval(
                    start=pd.Timestamp("201901010000"),
                    end=pd.Timestamp("201901290000")
                    + pd.Timedelta(hours=synthetic_n_observations - 1),
                ),
                name=f"synthetic wind AR{synthetic_nlags}",
            )

        case _:
            raise Exception("Use one of the predefined options")


"""Config parameters"""
SUPPLY_SLOPE = 500
SUPPLY_INTERCEPT = 24000
SUPPLY_SIGMA = 1


@dataclass
class Supply:
    """Supply class to store supply curve, uses a wind to return a quantity at time t

    FORMULA: 24000 + 500*price + self.wind.at[time] + error

    price_area: PriceArea
    wind: pd.Series
    error: RandomCached"""

    price_area: PriceArea
    wind: pd.Series
    error: RandomCached
    intercept: float = SUPPLY_INTERCEPT
    slope: float = SUPPLY_SLOPE

    def __call__(self, price: float, time: pd.Timestamp) -> float:
        # IMPORTANT : functional form of supply curve
        #           : linear dependency on price and wind speed
        error = self.error(time)
        return SUPPLY_INTERCEPT + SUPPLY_SLOPE * price + self.wind.at[time] + error

    def copy(self) -> "Supply":
        return Supply(
            price_area=self.price_area,
            wind=self.wind,
            error=self.error,
            intercept=self.intercept,
            slope=self.slope,
        )


@lru_cache
def average_supply_price(
    quantity: float, series: HourlySeries, price_area: PriceArea
) -> float:
    return (
        quantity
        - SUPPLY_INTERCEPT
        - get_wind_series(
            wind_type=WindOptions.ACTUAL, series=series, price_area=price_area
        ).mean()
    ) / SUPPLY_SLOPE


# @dataclass(frozen=True)
# class Demand:
#     """Demand class to store demand curve, when called with timestamp t returns a quantity

#     FORMULA: self.constant + self.elasticity * price + ar + error

#     base_constant: float
#     lags: dict[int, float]
#     elasticity: float
#     error: RandomCached | AutocorrelatedCached
#     average_supply_price: float"""

#     base_constant: float
#     lags: dict[int, float]
#     elasticity: float
#     error: RandomCached | AutocorrelatedCached
#     average_supply_price: float

#     @cached_property
#     def constant(self) -> float:
#         # intercept of demand equation/function
#         return (self.base_constant) * (
#             1 - sum(self.lags.values())
#         ) - self.elasticity * self.average_supply_price

#     # @property
#     # def formula(self) -> str:
#     #     lags = " + ".join([f"{v}y<sub>t{k}</sub>" for k, v in self.lags.items()])
#     #     return f"y<sub>t</sub> = {self.constant} + {lags} + {self.elasticity}&#8727;price + &#949;<sub>t</sub> + {self.error.normal}"

#     @property
#     def formula(self) -> str:
#         lags = " + ".join([f"{v}*D_{f't-{k}'}" for k, v in self.lags.items()])
#         return f"$D_t = {self.constant} + {lags} + {self.elasticity}*P_t + {self.error.normal}"

#     def __call__(
#         self, price: float, time: pd.Timestamp, lagged_demand: dict[int, float] = dict()
#     ) -> float:
#         for k in self.lags.keys():
#             assert (
#                 k in lagged_demand
#             ), f"A demand of lag {k} should be included in the input"

#         ar = sum((lagged_demand[l] * coef for l, coef in self.lags.items()))
#         error = self.error(
#             time
#         )  # + (0.9*self.error(time - pd.Timedelta(hours=1)) if time - pd.Timedelta(hours=1) in INTERVAL else 0)

#         # TODO: check that the elasticity has to be summed with the constant
#         # IMPORTANT: functional form of demand function
#         #            assuming linear dependency on price
#         #            alternative: price ** self.elasticity for an exponential form

#         return self.constant + self.elasticity * price + ar + error

#     def fallback(self, time) -> float:
#         average_price = (
#             57.0  # Just an assumption to let the series reach stationarity quicker
#         )
#         return (self.constant + average_price * self.elasticity) / (
#             1 - sum(self.lags.values())
#         )


@dataclass(frozen=True)
class IndividualDemand:
    """Demand class to store demand curve, when called with timestamp t returns a quantity

    FORMULA: self.constant + self.elasticity * price + ar + error

    base_constant: float
    lags: dict[int, float]
    elasticity: float
    error: RandomCached | AutocorrelatedCached
    average_supply_price: float
    cross_price_elasticity: dict[int, float]"""

    base_constant: float
    lags: dict[int, float]
    elasticity: float
    error: RandomCached | AutocorrelatedCached
    average_supply_price: float
    cross_price_elasticity: dict[int, float]

    @cached_property
    def constant(self) -> float:
        return (self.base_constant) * (1 - sum(self.lags.values())) - (
            self.elasticity + sum(self.cross_price_elasticity.values())
        ) * self.average_supply_price

    def __call__(
        self,
        price: float,
        time: pd.Timestamp,
        lagged_price: dict[int, float],
        lagged_demand: dict[int, float] = dict(),
    ) -> float:
        for k in self.lags.keys():
            assert (
                k in lagged_demand
            ), f"A demand of lag {k} should be included in the input"

        ar = sum((lagged_demand[l] * coef for l, coef in self.lags.items()))

        for k in self.cross_price_elasticity.keys():
            assert (
                k in lagged_price
            ), f"A price of lag {k} should be included in the input"

        cross_price_elasticity = sum(
            (lagged_price[l] * coef for l, coef in self.cross_price_elasticity.items())
        )

        error = self.error(time)

        return (
            self.constant
            + self.elasticity * price
            + cross_price_elasticity
            + ar
            + error
        )

    def fallback(self, time) -> float:
        average_price = (
            57.0  # Just an assumption to let the series reach stationarity quicker
        )
        return (self.constant + average_price * self.elasticity) / (
            1 - sum(self.lags.values())
        )

    def copy(self) -> "IndividualDemand":
        return IndividualDemand(
            base_constant=self.base_constant,
            elasticity=self.elasticity,
            error=self.error,
            lags=self.lags,
            average_supply_price=self.average_supply_price,
            cross_price_elasticity=self.cross_price_elasticity,
        )


@dataclass(frozen=True)
class AggregatedDemand:
    """
    Class to manage several IndividualDemands put together. 
    Each demand has it's own dynamics, but the market clearing takes
    place at the aggregated level.

    Uses a list of individualDemand as components
    """
    components: list[IndividualDemand]
    dataframe_demands: pd.DataFrame = field(default_factory=pd.DataFrame)

    def clear(self):
        self.dataframe_demands = pd.DataFrame()  # type:ignore

    @property
    def max_lag(self):
        return max([len(d.lags) for d in self.components])

    @property
    def lags(self) -> list[int]:
        return [-(lag + 1) for lag in range(0, self.max_lag)]

    @property
    def formula(self) -> str:
        return ""

    def store_demand(self, time, col, value):
        self.dataframe_demands.at[time, col] = value

    def __call__(
        self, price: float, time: pd.Timestamp, lagged_price: dict[int, float]
    ) -> float:  # Designed only for one term of cross price elasticity
        for i, demand in enumerate(self.components):
            # Recover past values of individual demand
            lagged_individual_demand = {}
            for lag in demand.lags.keys():
                lag_time: pd.Timestamp = time - pd.Timedelta(hours=1)
                if (
                    lag + len(self.dataframe_demands) > 0
                    and len(self.dataframe_demands.iloc[lag]) - i >= 1
                ):
                    lagged_individual_demand[lag] = self.dataframe_demands.loc[
                        lag_time
                    ][i]
                else:
                    lagged_individual_demand[lag] = demand.fallback(time)
            # Compute current individual demand
            individual_demand = demand(
                price=price,
                time=time,
                lagged_price=lagged_price,
                lagged_demand=lagged_individual_demand,
            )
            # Store current individual demand
            self.dataframe_demands.at[time, i] = individual_demand
        # Return current sum of individual demands
        return self.dataframe_demands.loc[time].sum()   # type: ignore

    def fallback(self, time) -> float:
        for i, demand in enumerate(self.components):
            # compute current individual demand (as fallback)
            individual_demand = demand.fallback(time)
            # Store current individual demand
            self.dataframe_demands.at[time, i] = individual_demand
        # Return current sum of individual demands
        return self.dataframe_demands.loc[time].sum()

    def copy(self) -> "AggregatedDemand":
        return AggregatedDemand(components=[d.copy() for d in self.components])


@dataclass
class EquilibriumResults:
    """Class to store the three series of a market equilibrium

    demand: pd.Series   =   supply: pd.Series
    price: pd.Series"""

    demand: pd.Series
    supply: pd.Series
    price: pd.Series

    @staticmethod
    def from_dict(
        demand: dict[int, float],
        supply: dict[int, float],
        price: dict[int, float],
        times: pd.DatetimeIndex,
        discard: int = 168 * 4,
    ) -> "EquilibriumResults":
        
        def to_series(
            d: dict[int, float], name: str, index: pd.DatetimeIndex
        ) -> pd.Series:
            ds = pd.Series(d).sort_index()
            ds.name = name
            ds.index = index
            return ds[discard:]

        return EquilibriumResults(
            demand=to_series(demand, name="demand", index=times),
            supply=to_series(supply, name="supply", index=times),
            price=to_series(price, name="price", index=times),
        )


@dataclass(frozen=True)
class IndEquilibrium:
    """Class to manage a Demand curve and a Supply curve to be cleared over a timeperiod

    supply: Supply
    demand: Demand
    times: pd.DatetimeIndex

    method .clearing conducts the optimization and returns an
    'EquilibriumResults' instance with the demand, supply and price series
    """

    supply: Supply
    demand: IndividualDemand

    times: pd.DatetimeIndex
    analytical_solution: bool = True

    @cached_property
    def index(self) -> pd.DatetimeIndex:
        assert isinstance(self.clearing.price.index, pd.DatetimeIndex)
        return pd.DatetimeIndex(data=self.clearing.price.index)

    @cached_property
    def nlags(self) -> int:
        return len(self.demand.lags)

    def calculate_price(
        self,
        time: pd.Timestamp,
        lagged_price: dict[int, float],
        lagged_demand: dict[int, float],
    ):
        ar_demand_term: float = sum(
            (lagged_demand[l] * coef for l, coef in self.demand.lags.items())
        )
        cross_price_term = sum(
            (
                lagged_price[l] * coef
                for l, coef in self.demand.cross_price_elasticity.items()
            )
        )

        try:
            w = self.supply.wind.at[time]
        except KeyError as e:
            print(f"error {e} with wind")
            print(f"Length of wind series = {len(self.supply.wind)}")
            print(f"lenght of times = {len(self.times)}")
        try:
            w = self.supply.error(time)
        except KeyError as e:
            print(f"error {e} with error")
        try:
            d = self.demand.error(time)
        except KeyError as e:
            print(f"error {e} with demand_error")

        return (
            self.demand.constant
            + cross_price_term
            + ar_demand_term
            + self.demand.error(time)
            - self.supply.intercept
            - self.supply.wind.at[time]
            - self.supply.error(time)
        ) / (
            self.supply.slope
            - self.demand.elasticity
            + sum(self.demand.cross_price_elasticity)
        )

    @cached_property
    def clearing(self) -> EquilibriumResults:
        pri: dict[int, float] = {}
        dem: dict[int, float] = {}
        sup: dict[int, float] = {}

        # using enumerate appears to change the hash around the light saving time
        # not sure why this is the case. Would be interesting to investigate!
        for i in range(len(self.times)):
            lagged_demand: dict[int, float] = {}
            for lag in self.demand.lags.keys():
                lagged_i = i + lag

                if lagged_i >= 0:
                    lagged_demand[lag] = dem[lagged_i]
                else:
                    lagged_demand[lag] = self.demand.fallback(lagged_i)

            lagged_price: dict[int, float] = {}
            for lag in self.demand.cross_price_elasticity.keys():
                lagged_i = i + lag

                if lagged_i >= 0:
                    lagged_price[lag] = pri[lagged_i]
                else:
                    lagged_price[lag] = self.demand.average_supply_price

            if self.analytical_solution:
                price = self.calculate_price(
                    time=self.times[i],
                    lagged_demand=lagged_demand,
                    lagged_price=lagged_price,
                )
            else:
                # function to be minimized
                def objective_function(price, time=self.times[i]):
                    return self.demand(
                        price=price,
                        time=time,
                        lagged_demand=lagged_demand,
                        lagged_price=lagged_price,
                    ) - self.supply(price=price, time=time)

                result = opt.root_scalar(  # type: ignore
                    objective_function,
                    # args=(time,),
                    bracket=[-100.0, 1000.0],
                    method="brentq",
                    xtol=1e-12,
                )

                if not result.converged:
                    raise ValueError("Failed to find clearing price")
                price = result.root

            pri[i] = price
            dem[i] = self.demand(
                time=self.times[i],
                price=price,
                lagged_demand=lagged_demand,
                lagged_price=lagged_price,
            )
            sup[i] = self.supply(time=self.times[i], price=price)

        return EquilibriumResults.from_dict(
            demand=dem,
            supply=sup,
            price=pri,
            times=self.times,
        )

    def copy(self) -> "IndEquilibrium":
        return IndEquilibrium(
            demand=self.demand, supply=self.supply, times=self.times.copy()
        )


@dataclass(frozen=True)
class AggEquilibrium:
    """Class to manage an Aggregated Demand curve and a Supply curve to be cleared over a timeperiod

    supply: Supply
    demand: Demand
    times: pd.DatetimeIndex

    method .clearing conducts the optimization and returns an
    'EquilibriumResults' instance with the demand, supply and price series
    """

    supply: Supply
    demand: AggregatedDemand

    times: pd.DatetimeIndex

    analytical_solution = True

    @cached_property
    def index(self) -> pd.DatetimeIndex:
        assert isinstance(self.clearing.price.index, pd.DatetimeIndex)
        return pd.DatetimeIndex(data=self.clearing.price.index)

    def demand_n(self, n: int, common_index: bool = True):
        assert (
            0 <= n <= len(self.demand.components) - 1
        ), f"The specified individual demand D{n} does not exist"
        series = self.demand.dataframe_demands[n]
        if common_index:
            series = series.loc[self.index]
        return series

    @cached_property
    def nlags(self) -> int:
        return len(self.demand.lags)

    def calculate_price(
        self, time: pd.Timestamp, lagged_price: dict[int, float]
    ) -> float:
        d1 = self.demand.components[0]

        lagged_d1 = {}
        for lag in d1.lags.keys():
            lag_time: pd.Timestamp = time - pd.Timedelta(hours=1)  # MARK
            if (
                lag + len(self.demand.dataframe_demands) > 0
                and len(self.demand.dataframe_demands.iloc[lag]) - 0 >= 1
            ):
                lagged_d1[lag] = self.demand.dataframe_demands.loc[lag_time][0]
            else:
                lagged_d1[lag] = d1.fallback(time)

        d1_ar_term: float = sum((lagged_d1[l] * coef for l, coef in d1.lags.items()))
        d1_cross_price_term: float = sum(
            (lagged_price[l] * coef for l, coef in d1.cross_price_elasticity.items())
        )

        d1_sum = d1.constant + d1_ar_term + d1_cross_price_term + d1.error(time)

        d2 = self.demand.components[1]
        lagged_d2 = {}
        for lag in d2.lags.keys():
            lag_time: pd.Timestamp = time - pd.Timedelta(hours=1)  # MARK
            if (
                lag + len(self.demand.dataframe_demands) > 0
                and len(self.demand.dataframe_demands.iloc[lag]) - 1 >= 1
            ):
                lagged_d2[lag] = self.demand.dataframe_demands.loc[lag_time][1]
            else:
                lagged_d2[lag] = d2.fallback(time)

        d2_ar_term: float = sum((lagged_d2[l] * coef for l, coef in d2.lags.items()))
        d2_cross_price_term: float = sum(
            (lagged_price[l] * coef for l, coef in d2.cross_price_elasticity.items())
        )

        d2_sum = d2.constant + d2_ar_term + d2_cross_price_term + d2.error(time)

        return (
            d1_sum
            + d2_sum
            - self.supply.intercept
            - self.supply.wind.at[time]
            - self.supply.error(time)
        ) / (
            self.supply.slope
            - d1.elasticity
            + sum(d1.cross_price_elasticity.values())
            - d2.elasticity
            + sum(d2.cross_price_elasticity.values())
        )

    @cached_property
    def clearing(self) -> EquilibriumResults:
        pri = {}
        dem = {}
        sup = {}

        for i in range(len(self.times)):

            cross_price_lags: list[int] = list(self.demand.components[0].cross_price_elasticity.keys()) + list(self.demand.components[1].cross_price_elasticity.keys())  # type: ignore

            lagged_price: dict[int, float] = {}
            for lag in cross_price_lags:
                lagged_i = i + lag

                if lagged_i >= 0:
                    lagged_price[lag] = pri[lagged_i]
                else:
                    lagged_price[lag] = self.demand.components[0].average_supply_price

            if self.analytical_solution:
                price = self.calculate_price(
                    time=self.times[i], lagged_price=lagged_price
                )
            else:

                def objective_function(price, time=self.times[i]):
                    return self.demand(
                        price=price, time=time, lagged_price=lagged_price
                    ) - self.supply(price=price, time=time)

                result = opt.root_scalar(  # type: ignore
                    objective_function,
                    # args=(time,),
                    bracket=[-100.0, 1000.0],
                    method="brentq",
                    xtol=1e-12,
                )

                if not result.converged:
                    raise ValueError("Failed to find clearing price")
                price = result.root
            pri[i] = price
            dem[i] = self.demand(
                time=self.times[i], price=price, lagged_price=lagged_price
            )
            sup[i] = self.supply(time=self.times[i], price=price)

        return EquilibriumResults.from_dict(
            demand=dem,
            supply=sup,
            price=pri,
            times=self.times,
        )

    def copy(self) -> "AggEquilibrium":
        return AggEquilibrium(
            demand=self.demand, supply=self.supply, times=self.times.copy()
        )


@lru_cache
def exogenous_variables(
    start: pd.Timestamp, end: pd.Timestamp, price_area: PriceArea, minimal: bool = False
) -> pd.DataFrame:
    """Returns the dataframe of variables to use as exogenous controls.

    start:pd.Timestamp,
    end:pd.Timestamp,
    """
    series = HourlySeries(interval(start, end))
    dummy = Dummy(interval(start, end))
    variables = exog_list(
        series=series, dummy=dummy, price_area=price_area, minimal=minimal
    )
    # variables.append(series.price(price_area))
    return pd.concat(variables, axis=1)


if __name__ == "__main__":
    pass
