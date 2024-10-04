"""
This file containts functions and classes to 
run analysis on demand data, be it Synthetic or Actual data. 

It builds on top of model.py to run the analysis
and is applied in plots_synthetic.ipynb and plots_application.ipynb

The core components of this module are the classes IVOptions and ModelInputParams,
which manage the settings for running an IV model, 
and the function IV_results, which runs the model.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.ar_model as ar_model
from statsmodels.tsa.arima.model import ARIMA
from linearmodels import OLS
from model import interval, HourlySeries, Dummy, IVModel, ModelData, IVResults
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from database import PriceArea, Areas
from math import log
from typing import Any


@dataclass
class ArParams:
    """holds the results of an AR Analysis

    ar_model_analysis(ar_model_fit()) -> ArParams

    constant: float
    lags_dict: dict[int, float]
    error_sd: float"""

    constant: float
    lags_dict: dict[int, float]
    error_sd: float


def ar_model_fit(
    series: pd.Series, lags: int, exog: pd.DataFrame | None
) -> ar_model.AutoRegResultsWrapper:
    """returns the default AR model results from any given time series with input n lags and input exogenous variables dataframe"""
    model = ar_model.AutoReg(series, lags=lags, trend="c", exog=exog, missing="drop")
    results: ar_model.AutoRegResultsWrapper = model.fit()
    return results


def ar_manual(series: pd.Series, lags: dict[int, float]) -> ArParams:
    dss = [series.shift(lag) * coef for lag, coef in lags.items()]
    resid = series - pd.concat(dss, axis=1).sum(axis=1)
    return ArParams(
        constant=float(series.mean()),
        lags_dict=lags,
        error_sd=resid.std(),
    )


def ar_model_analysis(ar_results: ar_model.AutoRegResultsWrapper) -> ArParams:
    """extracts the custom ArParams from any AR model results object, usually called on top of ar_model_fit()"""
    constant = ar_results.params["const"]  # type: ignore
    lags_dict = (
        {-i: ar_results.params.iloc[i] for i in ar_results.ar_lags}
        if ar_results.ar_lags
        else {}
    )  # To avoid TypeError if lags = 0
    sd_error_term = ar_results.resid.std()

    return ArParams(constant=constant, lags_dict=lags_dict, error_sd=sd_error_term)


@dataclass
class ModelResults:
    """This class is a result wrapper for OLS and IV analysis to be homogeneous

    estimate: float             -> point estimate
    ci: tuple[float, float]     -> confidence interval
    model: str                  -> name of the model used
    f_statistic: float          -> f-statistic of the first stage (if 2SLS)
    mse: float                  -> mean squared error"""

    estimate: float
    ci: tuple[float, float]
    model: str
    latex_name: str
    f_statistic: float = np.NaN
    mse: float = np.NaN


@dataclass
class MultipleModelResults(ModelResults):
    """This class is a variation of the wrapper 'ModelResults' designed to contain more than one estimate"""

    estimate: dict[str, float]
    ci: dict[str, tuple[float, float]]
    model: str
    latex_name: str
    f_statistic: float = np.NaN
    mse: float = np.NaN


def ex_post_bias(demand: pd.Series, wind: pd.Series, nlags: int) -> float:
    """return 1/(1 - sum(ac_d(i) for i in I)) * sum(ac_w(i) for i in W)\n
    based on the PAC parameters (ex-post calculation)"""

    wind_lags: dict[int, float] = ar_model_analysis(
        ar_model_fit(wind, lags=nlags, exog=None)  # type: ignore
    ).lags_dict
    demand_lags: dict[int, float] = ar_model_analysis(
        ar_model_fit(demand, lags=nlags, exog=None)  # type: ignore
    ).lags_dict

    bias: float = 1 / (1 - sum(wind_lags.values()) * sum(demand_lags.values()))

    return bias


def get_solar_generation(
        series:HourlySeries,
        price_area:PriceArea
) -> pd.Series:
    if (price_area == Areas.DE1 or price_area == Areas.DE2) and (
        series.timestamps.min() < pd.Timestamp("201809302300", tz="Europe/Berlin")
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

        solar_generation = pd.concat(
            [
                series1.solar_generation(price_area=Areas.DE1).div(1000),
                series2.solar_generation(price_area=Areas.DE2).div(1000),
            ]
        ).rename("generation.DE.Solar")
    else:
        solar_generation = series.solar_generation(price_area)
    return solar_generation


@lru_cache
def exog_list(
    series: HourlySeries,
    dummy: Dummy,
    price_area: PriceArea,
    minimal: bool = False,
    constant: bool = True,
    wind: bool = False,
    holidays: bool = False,
) -> list[pd.Series]:
    """
    This function returns the list of exogenous or control variables.

    Flags:
    ----------
    minimal : bool = false
        If true, returns the minimal necessary set to remove confounding between wind, price and demand: temperature, solar generation, month and hour
    
    constant : bool = true
        If true, adds a constant variable to the return

    wind : bool = false
        If true, returns the set of variables relevant to control wind generation: sunlight, temperature, solar generation, month, hour and year

    holidays : bool = true
        If true, adds the weighted index variables of public and school holidays to the return


    !!!! Since this function is cached, always copy the return before modifying it e.g. before appending other columns !!!!"""
    solar_generation = get_solar_generation(series=series, price_area=price_area)

    if minimal:
        return (
            [dummy.constant()]
            + dummy.month()
            + dummy.hour()
            + [solar_generation, series.hdd(price_area), series.cdd(price_area)]
        )

    base = [dummy.constant()] if constant else []

    if wind:
        return (
            base
            + [
                series.sunlight(price_area),
                solar_generation,
                series.hdd(price_area),
                series.cdd(price_area),
            ]
            + dummy.hour()
            + dummy.month()
            + dummy.year()
        )

    return (
        base
        + [
            series.gas_price(price_area),
            series.coal_price(),
            series.eua_price(),
            series.sunlight(price_area),
            solar_generation,
            series.hdd(price_area),
            series.cdd(price_area),
        ]
        + dummy.weekhour()
        + dummy.month()
        + dummy.year()
        + [dummy.public_holiday(price_area)]
        + [dummy.school_holiday()]
        + [dummy.week_53()]

        + ([
            holiday
            for holiday in [dummy.public_holiday(), dummy.school_holiday()]
            if holiday.std() > 0
        ] if holidays else [])
    )


def OLS_results(
    demand: pd.Series,
    price: pd.Series,
    series: HourlySeries,
    dummy: Dummy,
    price_area: PriceArea,
    controls: bool = False,
    nuisance: bool = False,
    log_log: bool = False,
    **kwargs,
) -> ModelResults:
    """Returns the ModelResults of an OLS analysis

    demand: pd.Series,
    price:pd.Series,
    series:HourlySeries,
    dummy:Dummy,
    controls:bool = False,
    diff: int | None = None"""
    assert isinstance(
        demand.index, pd.DatetimeIndex
    ), "Demand does not have a DatetimeIndex"

    if log_log:
        demand = demand.apply(lambda x: x if x <= 0 or x >= 1 else log(x))
        price = price.apply(lambda x: x if x <= 0 or x >= 1 else log(x))

    if controls:
        if "minimal_controls" in kwargs and kwargs["minimal_controls"]:
            minimal = kwargs["minimal_controls"]
        else:
            minimal = False
        exog: list[pd.Series] = exog_list(
            series=series, dummy=dummy, price_area=price_area, minimal=minimal
        ).copy()
        if log_log:
            nexog = []
            for ser in exog:
                if 0 <= ser.min() and ser.max() <= 1:
                    nexog.append(ser)
                else:
                    try:
                        if ser.value_counts().loc[0] / len(ser) >= 0.1:
                            nexog.append(ser)
                        else:
                            nexog.append(
                                pd.Series(
                                    np.where(ser <= 0, np.nan, np.log(ser)),
                                    index=ser.index,
                                    name=ser.name,
                                )
                            )
                    except KeyError:
                        nexog.append(
                            pd.Series(
                                np.where(ser <= 0, np.nan, np.log(ser)),
                                index=ser.index,
                                name=ser.name,
                            )
                        )

    else:
        exog = [dummy.constant()]

    if nuisance:
        name = "OLS Nuisance"
        if "order" in kwargs:
            order = kwargs["order"]
        else:
            order = 2
        for i in range(order):
            shifted_demand = demand.copy().shift(i + 1)
            shifted_demand.name = f"demand_t-{i+1}"
            exog.append(shifted_demand)
    else:
        name = "OLS"

    exog_df: pd.DataFrame = pd.concat([price] + exog, axis=1)

    fitted = sm.OLS(demand, exog_df, missing="drop").fit()

    price_coeff = fitted.params["price"]
    price_ci: tuple[float, float] = tuple(fitted.conf_int(alpha=0.05).loc["price"] - price_coeff)  # type: ignore

    mse = 0

    # Returns price estimate, confidence interval, model
    return ModelResults(
        estimate=price_coeff, ci=price_ci, latex_name=name, model=name, mse=mse
    )


class IVOptions(Enum):
    """Options for the different IV designs. 
    
    Affects the sets that will be used as instrumental, exogenous, and endogenous variables.

    Options:
    ---------
    - REGULAR: 
        Wind lags used as instruments and P_t as endogenous.

    ** CIV Estimators **
        - CONDITIONAL_WIND:
            Only W_t as instrument, wind lags used as exogenous.
        - CONDITIONAL_DEMAND:
            Wind lags used as instruments, demand lags used as exogenous.
        - CONDITIONAL_DW:
            Only W_t as instrument, demand and wind lags used as exogenous.
        - CONDITIONAL_H0:
            Only W_t as instrument, demand, price and wind lags used as exogenous.
 
    ** NIV Estimators **
        - TRUNCATED_NUISANCE_ORDER
            Wind lags used as instruments, demand lags as endogenous
        - CLEAN_2dim_ORDER
            Wind lags used as instruments, price lags as endogenous
        - TRUNCATED_IV_2dim_ORDER
            Wind lags used as instruments, price lags as endogenous, and demand lags as exogeneous
        - IV_2dim_ORDER
            Wind lags used as instruments, price lags and demand lags as endogenous

    
    ** Other options ** 
    (implemented but not in use)
        - NUISANCE_ORDER,
        - IV_3dim_ORDER,
        - TRUNCATED_IV_3dim_ORDER,
        - RESIDUAL_WIND,
        - ARIMA_FIT_IV,
        - CONDITIONAL_TRUNCATED_H0
    """

    # Regular Estimator
    REGULAR = auto()
    
    # CIV Estimators
    CONDITIONAL_WIND = auto()
    CONDITIONAL_DEMAND = auto()
    CONDITIONAL_DW = auto()
    CONDITIONAL_H0 = auto()
    CONDITIONAL_TRUNCATED_H0 = auto()
    
    # NIV Estimators
    NUISANCE_ORDER = auto()
    TRUNCATED_NUISANCE_ORDER = auto()
    CLEAN_2dim_ORDER = auto()
    IV_2dim_ORDER = auto()
    TRUNCATED_IV_2dim_ORDER = auto()
    IV_3dim_ORDER = auto()
    TRUNCATED_IV_3dim_ORDER = auto()
    
    # Special Cases
    RESIDUAL_WIND = auto()
    ARIMA_FIT_IV = auto()



@dataclass
class ModelInputParams:
    """
    Class to specify the IV model to run.

    Parameters:
    ---------
    strategy : IVOptions
        Sets the estimator, and decides how the different orders will be interpreted.
        See usage section below.
    order : int = 1
    order_d : int = 1
    order_w : int = 1
    order_price : int = 1
    
    Usage:
    ---------
    - REGULAR: 
        {order} wind lags used as instruments.

    ** CIV Estimators **
        - CONDITIONAL_WIND:
            Only W_t as instrument, {order} wind lags used as exogenous.
        - CONDITIONAL_DEMAND:
            {order_w} wind lags used as instruments, {order} demand lags used as exogenous.
        - CONDITIONAL_DW:
            Only W_t as instrument, {order} demand and wind lags used as exogenous.
        - CONDITIONAL_H0:
            Only W_t as instrument, {order} demand, price and wind lags used as exogenous.
 
    ** NIV Estimators **
        - TRUNCATED_NUISANCE_ORDER
            {order} wind lags used as instruments, {order_d} demand lags as endogenous
        - CLEAN_2dim_ORDER
            {order} wind lags used as instruments, {order_price} price lags as endogenous
        - TRUNCATED_IV_2dim_ORDER
            {order} wind lags used as instruments, {order_price} price lags as endogenous, and {order_d} demand lags as exogeneous
        - IV_2dim_ORDER
            {order} wind lags used as instruments, {order_price} price lags and {order_d} demand lags as endogenous

    
    ** Other options ** 
    (implemented but not in use)
        - NUISANCE_ORDER,
        - IV_3dim_ORDER,
        - TRUNCATED_IV_3dim_ORDER,
        - RESIDUAL_WIND,
        - ARIMA_FIT_IV,
        - CONDITIONAL_TRUNCATED_H0
    """

    strategy: IVOptions
    order: int = 1
    order_d: int = 1
    order_w: int = 1
    order_price: int = 1

    def get_name_component(self, component:str, order:int, start:int=0, prefix=""):
        "Helper function to create names"
        if order > 0:
            return f"{component}{start}{prefix}:{component}{order}"
        return f"{component}{start}"

    @property
    def name(self) -> str:
        w_component = self.get_name_component("W", self.order)
        p_component = self.get_name_component("P", self.order_price)
        d_component = self.get_name_component("D", self.order_d, 1)
        
        match self.strategy:
            case IVOptions.REGULAR:
                return f"I({w_component})-E(P0)-C()"

            case IVOptions.CONDITIONAL_DEMAND:
                return f"I({self.get_name_component('W', self.order_w)})-E(P0)-C({self.get_name_component('D', self.order, 1)})"
            case IVOptions.CONDITIONAL_WIND:
                return f"I(W0)-E(P0)-C({self.get_name_component('W', self.order, 1)})"
            case IVOptions.CONDITIONAL_DW:
                return f"I(W0)-E(P0)-C({d_component}, {self.get_name_component('W', self.order_w, 1)})"
            
            case IVOptions.CONDITIONAL_H0:
                return f"I(W0)-E(P0)-C({self.get_name_component('W', self.order, 1), self.get_name_component('P', self.order, 1), self.get_name_component('D', self.order, 1)})"

            case IVOptions.NUISANCE_ORDER:
                return f"I({w_component})-E(P0, D1:D{self.order})-C()"
            case IVOptions.TRUNCATED_NUISANCE_ORDER:
                return f"I({w_component})-E(P0, D1{f':D{self.order_d}' if self.order_d else ''})-C()"

            case IVOptions.CLEAN_2dim_ORDER:
                return f"I({w_component})-E({p_component})-C()"
            case IVOptions.IV_2dim_ORDER:
                return f"I({w_component})-E({p_component}, D{self.order_d})-C()"
            case IVOptions.TRUNCATED_IV_2dim_ORDER:
                return (
                    f"I({w_component})-E({p_component})-C(D{self.order_d})"
                )
            
            case IVOptions.IV_3dim_ORDER:
                return "IVOptions.IV_3dim_ORDER"
            
            case IVOptions.TRUNCATED_IV_3dim_ORDER:
                return "IVOptions.TRUNCATED_IV_3dim_ORDER"
            
            case IVOptions.ARIMA_FIT_IV:
                return "ARIMA"
            
            case IVOptions.RESIDUAL_WIND:
                return "RESIDUAL WIND"

            case IVOptions.CONDITIONAL_TRUNCATED_H0:
                return f"CONDITIONAL TRUNCATED H0"

            case _:
                raise Exception(f"Implement a name for the model {self.strategy}")
            

    def var_set(self, var:str|list[str], order:int, start:int) -> str:
        "Auxiliary method for producing the latex names of the estimators."

        if isinstance(var, str):
            if order == 0:
                return f"{var}_t"
            elif order == 1 and start == 1:
                return f"{var}_{{t-1}}"
            else:
                return f"\\{{{var}_{{t-s}}\\}}_{{s={start}}}^{{{order}}}"
        else:
            return f"\\{{{', '.join([v+'_{{t-s}}' for v in var])}\\}}_{{s={start}}}^{{{order}}}"


    @property
    def latex_name(self) -> str:
        match self.strategy:
            case "OLS":
                return "OLS"
            case IVOptions.REGULAR:
                return f"$CIV({self.var_set('W', self.order, 0)}|P_t\\rightarrow D_t|\\emptyset)$"

            case IVOptions.CONDITIONAL_DEMAND:
                return f"$CIV({self.var_set('W', self.order_w, 0)}|P_t\\rightarrow D_t|{self.var_set('D', self.order, 1)})$"
            case IVOptions.CONDITIONAL_WIND:
                return f"$CIV(W_t|P_t\\rightarrow D_t|{self.var_set('W', self.order, 1)})$"
            case IVOptions.CONDITIONAL_DW:
                return f"$CIV(W_t|P_t\\rightarrow D_t|{self.var_set('W', self.order_w, 1)}, {self.var_set('W', self.order_d, 1)})$"

            case IVOptions.NUISANCE_ORDER:
                return f"$CIV({self.var_set('W', self.order, 0)}|(P_t, D_{{t-1}}{f', ..., D_{self.order}' if self.order > 1 else ''})\\rightarrow D_t|\\emptyset)$"
            case IVOptions.TRUNCATED_NUISANCE_ORDER:
                return f"$CIV({self.var_set('W', self.order, 0)}|(P_t, D_{{t-1}}{f':D_{self.order_d}' if self.order_d > 1 else ''})\\rightarrow D_t|\\emptyset)$"

            case IVOptions.CLEAN_2dim_ORDER:
                return f"$CIV({self.var_set('W', self.order, 0)}|({self.var_set('P', self.order_price, 0)})\\rightarrow D_t|\\emptyset)$"

            case IVOptions.IV_2dim_ORDER:
                dx = f", D_{{t-{self.order_d}}}"
                endog = f"({self.var_set('P', self.order_price, 0)}{dx})"
                return f"$CIV({self.var_set('W', self.order, 0)}|{endog}\\rightarrow D_t|\\emptyset)$"

            case IVOptions.TRUNCATED_IV_2dim_ORDER:
                return f"$CIV({self.var_set('W', self.order, 0)}|({self.var_set('P', self.order_price, 0)})\\rightarrow D_t|D_{{t-{self.order_d}}})$"

            case IVOptions.IV_3dim_ORDER:
                endog = f"({self.var_set('P', self.order_price, 0)}{self.var_set('D', self.order_d, 1)})"
                return f"$CIV({self.var_set('W', self.order, 0)}|{endog}\\rightarrow D_t|\\emptyset)$"

            case IVOptions.TRUNCATED_IV_3dim_ORDER:

                return f"$CIV({self.var_set('W', self.order, 0)}|({self.var_set('P', self.order_price, 0)})\\rightarrow D_t|{self.var_set('D', self.order_d, 1)})$"

            case IVOptions.RESIDUAL_WIND:
                return f"$CIV(\\epsilon_t^W|P_t\\rightarrow D_t)|\\emptyset)$"
            
            case IVOptions.ARIMA_FIT_IV:
                return f"$CIV(W_t|P_t\\rightarrow D_t)|ARIMA)$"
            
            case IVOptions.CONDITIONAL_H0:
                return f"$CIV(W_t|P_t \\rightarrow D_t| {self.var_set(['W', 'P', 'D'], self.order, 1)})$"
            
            case IVOptions.CONDITIONAL_TRUNCATED_H0:
                return f"$CIV(W_t|P_t\\rightarrow D_t|W_{{t-1}},...,W_{{t-{self.order}}}; D_{{t-1}},...,D_{{t-{self.order}}})$"


            case _:
                return self.name


def find_best_arima_model(time_series:pd.Series, p_range:range, d_range:range, q_range:range) -> tuple[tuple[int, int, int], Any]:
    """
    Find the best ARIMA model for a given time series.

    Parameters:
    -----
    time_series (pd.Series): The time series data.
    p_range (range): Range of p values to try.
    d_range (range): Range of d values to try.
    q_range (range): Range of q values to try.

    Returns:
    -----
    tuple: A tuple with the best p, d, q values, 
    and the corresponding model.
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        print(f"New best order = {best_order}")
                        best_model = model_fit
                except:
                    continue
    
    if best_order is None:
        raise RuntimeError("No arima model could be found")

    return best_order, best_model


def analyze_column_rank(df, verbose:bool=False) -> list[str]:
    # Determine if the DataFrame has full column rank
    rank = np.linalg.matrix_rank(df.values)
    n_rows, n_cols = df.shape
    if rank == n_cols:
        if verbose:
            print("The DataFrame has full column rank.")
        return []
    
    if verbose:
        print(f"The DataFrame does not have full column rank. Rank: {rank}, Columns: {n_cols}")
    
    # Identify linearly dependent columns
    col_indices = np.arange(n_cols)
    independent_columns = []
    dependent_columns = []
    
    for col in col_indices:
        # Check the rank without the current column
        cols_to_check = np.delete(col_indices, col)
        submatrix = df.iloc[:, cols_to_check].values
        submatrix_rank = np.linalg.matrix_rank(submatrix)
        
        if submatrix_rank == rank:
            dependent_columns.append(df.columns[col])
        else:
            independent_columns.append(df.columns[col])
    
    if verbose:
        print(f"Independent Columns ({len(independent_columns)}): {independent_columns}")
        print(f"Dependent Columns ({len(dependent_columns)}): {dependent_columns}")

    return dependent_columns

def IV_results(
    demand: pd.Series,
    price: pd.Series,
    wind: pd.Series,
    series: HourlySeries,
    dummy: Dummy,
    price_area: PriceArea,
    model: ModelInputParams,
    controls: bool = False,
    cov_type: str = "kernel",
    log_log: bool = False,
    **kwargs,
) -> ModelResults:
    """
    Returns the ModelResults of an IV analysis, following the different IV model options

    Parameters:
    ------
    demand: pd.Series,
    price: pd.Series,
    wind: pd.Series,
    series: HourlySeries,
    dummy: Dummy,
    price_area: PriceArea,
    model: ModelInputParams,
    controls: bool = False,
    cov_type: str = "kernel",
    log_log: bool = False,
    **kwargs,"""

    # Defaults for REGULAR model
    instrument = [wind]
    endog = [price]

    # Get exogenous
    if controls:
        if "minimal_controls" in kwargs and kwargs["minimal_controls"]:
            minimal = kwargs["minimal_controls"]
        else:
            minimal = False
        exog = exog_list(
            series=series, dummy=dummy, price_area=price_area, minimal=minimal
        ).copy()
    else:
        exog = [dummy.constant()]

    match model.strategy:

        case IVOptions.REGULAR:
            instrument = [wind] + [
                wind.shift(i).rename(f"wind_t-{i}") for i in range(1, model.order + 1)
            ]

        case (
            IVOptions.CONDITIONAL_DEMAND
        ):  # Add as exogenous one shifted dependend variable (demand) for each 'order' specified

            for i in range(model.order):
                shift_instrument = wind.copy().shift(i + 1)
                shift_instrument.name = f"instrument shifted {i+1}"
                instrument.append(shift_instrument)

            for i in range(model.order):
                shifted_demand = demand.copy().shift(i + 1)
                shifted_demand.name = f"industrial_demand_t-{i+1}"
                exog.append(shifted_demand)

        case (
            IVOptions.CONDITIONAL_WIND
        ):  # Add as exogenous one shifted instrument (wind) for each 'order' specified

            for i in range(model.order):
                shift_instrument = wind.copy().shift(i + 1)
                shift_instrument.name = f"instrument shifted {i+1}"
                exog.append(shift_instrument)

        case (
            IVOptions.CONDITIONAL_DW
        ):  # Add as exogenous both shifted demand (dependent) and wind (instrument)

            for i in range(model.order_d):
                shifted_demand = demand.copy().shift(i + 1)
                shifted_demand.name = f"industrial_demand_t-{i+1}"
                exog.append(shifted_demand)
            for i in range(model.order_w):
                shift_instrument = wind.copy().shift(i + 1)
                shift_instrument.name = f"instrument shifted {i+1}"
                exog.append(shift_instrument)

        case IVOptions.NUISANCE_ORDER:
            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [price] + [
                demand.copy().shift(i).rename(f"demand_t-{i}")
                for i in range(1, model.order + 1)
            ]

        case IVOptions.TRUNCATED_NUISANCE_ORDER:
            if model.order == 1:
                print(f"Careful! T-NIV 1 is identical to NIV 1")

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [price] + [
                demand.copy().shift(i).rename(f"demand_t-{i}")
                for i in range(1, model.order_d + 1)
            ]

        case IVOptions.CLEAN_2dim_ORDER:
            assert model.order >= 3, "Order must be at least 3"

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [price] + [
                price.copy().shift(i).rename(f"price_t-{i}")
                for i in range(1, model.order_price + 1)
            ]

        case IVOptions.IV_2dim_ORDER:
            assert model.order >= 3, "Order must be at least 3"

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [
                price,
                demand.copy().shift(model.order_d).rename(f"demand_t-{model.order_d}"),
            ] + [
                price.copy().shift(i).rename(f"price_t-{i}")
                for i in range(1, model.order_price + 1)
            ]

        case IVOptions.IV_3dim_ORDER:
            assert model.order >= 3, "Order must be at least 3"

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [
                price
            ] + [            
                demand.copy().shift(i).rename(f"demand_t-{i}")
                for i in range(1, model.order_d + 1)
            ] + [
                price.copy().shift(i).rename(f"price_t-{i}")
                for i in range(1, model.order_price + 1)
            ]

        case IVOptions.TRUNCATED_IV_2dim_ORDER:
            assert model.order >= 3, "Order must be at least 3"

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [price] + [
                price.copy().shift(i).rename(f"price_t-{i}")
                for i in range(1, model.order_price + 1)
            ]
            exog.append(
                demand.copy().shift(model.order_d).rename(f"demand_t-{model.order_d}")
            )

        case IVOptions.TRUNCATED_IV_3dim_ORDER:
            assert model.order >= 3, "Order must be at least 3"

            instrument = [wind] + [
                wind.copy().shift(i).rename(f"Wind_t-{i}")
                for i in range(1, model.order + 1)
            ]
            endog = [price] + [
                price.copy().shift(i).rename(f"price_t-{i}")
                for i in range(1, model.order_price + 1)
            ]
            for i in range(1, model.order_d + 1):
                exog.append(
                    demand.copy().shift(i).rename(f"demand_t-{i}")
                )

        case IVOptions.RESIDUAL_WIND:
            from typing import TypeVar
            P = TypeVar("P", pd.Series, pd.DataFrame)

            def _log_log_transform(data: P) -> P:
                def _log_log_series(ser: pd.Series) -> pd.Series:
                    if 0 <= ser.min() and ser.max() <= 1:
                        series = ser
                    else:
                        try:
                            if ser.value_counts().loc[0] / len(ser) >= 0.1:
                                series = ser
                            else:
                                series = pd.Series(
                                    np.where(ser < 1, np.nan, np.log(ser)),
                                    index=ser.index,
                                    name=ser.name,
                                )
                        except KeyError:
                            series = pd.Series(
                                np.where(ser < 1, np.nan, np.log(ser)),
                                index=ser.index,
                                name=ser.name,
                            )
                    return series

                def _log_log_data(data: P) -> P:
                    if isinstance(data, pd.DataFrame):
                        for col in data.columns:
                            data[col] = _log_log_series(data[col])
                    elif isinstance(data, pd.Series):
                        data = _log_log_series(data)

                    return data

                def _deconcatenate_dataframe(data: pd.DataFrame) -> list[pd.Series]:
                    series_list = [series for _, series in data.items()]
                    return series_list

                return data
            
            exog_df = pd.concat(
                (exog_list(series=series, dummy=dummy, price_area=price_area, minimal=False, wind=True).copy() + 
                 [wind.copy().shift(i).rename(f"wind_t-{i}") for i in range(1, model.order + 1)]), axis=1
                )
            exog_df = exog_df.loc[exog_df.index.isin(wind.index)]


            if log_log:
                residuals = sm.OLS(_log_log_transform(wind), exog=_log_log_transform(exog_df), missing="drop").fit().resid.rename("price")
            else:
                residuals = sm.OLS(wind, exog=exog_df, missing="drop").fit().resid.rename("wind_generation_residuals")
            
            instrument = [residuals]

        case IVOptions.ARIMA_FIT_IV:
            (p, d, q), best_model = find_best_arima_model(wind, range(0, 50), range(0, 50), range(0, 50))
            print("ARIMA FIT")
            print(f"(p, d, q) = ({p}, {d}, {q})")
            residuals = best_model.resid

            for i in range(1, p + 1):
                exog.append(wind.copy().shift(i).rename(f"wind_t-{i}"))
            for i in range(1, d + 1):
                exog.append(wind.copy().diff(i).rename(f"wind_diff_{i}"))
            for i in range(1, q + 1):
                exog.append(residuals.copy().diff(i).rename(f"residuals_t-{i}"))

        case IVOptions.CONDITIONAL_H0:
            if len(demand) > 13*8760 and not controls:
                # For some reason, this estimator won't run with a constant with samples of 15 years or more
                exog = []
            for i in range(model.order):
                shifted_wind = wind.copy().shift(i + 1)
                shifted_wind.name = f"wind_t-{i+1}"
                exog.append(shifted_wind)

                shifted_price = price.copy().shift(i + 1)
                shifted_price.name = f"price_t-{i+1}"
                exog.append(shifted_price)

                shifted_demand = demand.copy().shift(i + 1)
                shifted_demand.name = f"demand_t-{i+1}"
                exog.append(shifted_demand)

        case IVOptions.CONDITIONAL_TRUNCATED_H0:
            for i in range(model.order):
                shifted_wind = wind.copy().shift(i + 1)
                shifted_wind.name = f"wind_t-{i+1}"
                exog.append(shifted_wind)

                shifted_demand = demand.copy().shift(i + 1)
                shifted_demand.name = f"demand_t-{i+1}"
                exog.append(shifted_demand)

        case _:
            raise Exception("Use one of the predefined options")


    data = ModelData.construct(
        dependent=demand,
        exogenous=exog,
        endogenous=endog,
        instruments=instrument,
        scaling=False,
    )

    if "subset_hour" in kwargs:
        if kwargs["subset_hour"] is not None:
            data = data.subset_hour(kwargs["subset_hour"])
            data = data.drop_empty_series()

    elif (
        "start_hour" in kwargs
        and kwargs["start_hour"] is not None
        and "end_hour" in kwargs
        and kwargs["end_hour"] is not None
    ):
        data = data.subset_hours_range(kwargs["start_hour"], kwargs["end_hour"])
        data = data.drop_empty_series()
        print("SUBSET HOUR RANGE")

    elif "subset_months" in kwargs and kwargs["subset_months"]:
        data = data.subset_months(kwargs["subset_months"])
        data = data.drop_empty_series()

    if log_log:
        data = data.log_log_transform(instrument=True if model.strategy != IVOptions.RESIDUAL_WIND else False)

    if "split_gas" in kwargs and kwargs["split_gas"] == "low":
        data = data.subset(
            idx=split_series(series.gas_price(price_area))[0],
            exclude=("d.year2021", "d.year2020"),
        )

    elif "split_gas" in kwargs and kwargs["split_gas"] == "high":
        data = data.subset(idx=split_series(series.gas_price(price_area))[1])

    if "drop_holidays" in kwargs and kwargs["drop_holidays"] == True:
        assert 'd.public_holiday' in data.exogenous, "Trying to remove holidays but they are not included as controls. This is possible but not under the current implementation"
        holidays_data = data.exogenous['d.public_holiday']
        non_holiday_mask = series.timestamps[~(holidays_data == 1)]
        data.subset(non_holiday_mask, exclude=())   # TODO: decide through kwargs whether to then exclude the holidays control or not

    iv_model = IVModel(data=data, cov_type=cov_type)
    try:
        fitted: IVResults = iv_model.fitted_model  # type: ignore
    except ValueError as e:
        # Drop columns that prevent full column rank and try again
        print("Exog")
        ex = analyze_column_rank(iv_model.data.exogenous)
        print("Endog")
        en = analyze_column_rank(iv_model.data.endogenous)
        if not ex and not en:
            print("Exog and Endog")
            ex_en = analyze_column_rank(pd.concat([iv_model.data.exogenous, iv_model.data.endogenous], axis=1))
        else:
            ex_en = []

        full = list(set(ex+en+ex_en))

        prev = len(iv_model.data.exogenous.columns) + len(iv_model.data.endogenous.columns)
        print("\n\nWARNING!\nDropping columns", full, "to ensure full column rank!\n")
        data = data.drop(full)
        iv_model = IVModel(data=data, cov_type=cov_type)
        post = len(iv_model.data.exogenous.columns) + len(iv_model.data.endogenous.columns)
        print(f"Columns reduced from {prev} to {post}")

        fitted: IVResults = iv_model.fitted_model  # type: ignore


    mse = 0

    if model.strategy in [
        IVOptions.IV_2dim_ORDER,
        IVOptions.TRUNCATED_IV_2dim_ORDER,
        IVOptions.CLEAN_2dim_ORDER,
        IVOptions.NUISANCE_ORDER,
        IVOptions.TRUNCATED_NUISANCE_ORDER,
        IVOptions.IV_3dim_ORDER,
        IVOptions.TRUNCATED_IV_3dim_ORDER,
    ]:

        estimate = {}
        ci = {}

        variable_names: list[str] = fitted.params.index  # type: ignore
        for var in variable_names:
            if var == "price":
                k = "_P0"
            elif var.startswith("price"):
                n = var.split("-")[1]
                k = "_P" + n
            elif var.startswith("demand"):
                n = var.split("-")[1]
                k = "_D" + n
            else:
                continue
            estimate[k] = fitted.params[var]
            ci[k] = fitted.conf_int(level=0.95).loc[var] - fitted.params[var]

        f_statistic = fitted.first_stage.diagnostics["f.stat"]["price"]
        mse = 0
        return MultipleModelResults(
            estimate=estimate,
            ci=ci,
            model=model.name,
            latex_name=model.latex_name,
            f_statistic=f_statistic,
            mse=mse,
        )

    else:
        price_coeff = fitted.params["price"]
        price_ci: tuple[float, float] = tuple(fitted.conf_int(level=0.95).loc["price"] - price_coeff)  # type: ignore
        return ModelResults(
            estimate=price_coeff,
            ci=price_ci,
            model=model.name,
            latex_name=model.latex_name if not model.strategy == IVOptions.ARIMA_FIT_IV else f"$CIV(W_t|(P_t->D_t)|ARIMA({p}, {q}, {d}))$",
            f_statistic=fitted.first_stage.diagnostics["f.stat"]["price"],
            mse=mse,
        )


def autocorrelation_analysis(
    series: pd.Series,
    nlags: int,
    exog: pd.DataFrame | None = None,
    controlled: bool = False,
) -> dict[int, float]:
    if controlled:
        series = sm.OLS(endog=series, exog=exog).fit().resid

    lags_dict: dict[int, float] = {}
    acf_coefs = sm.tsa.stattools.acf(x=series, nlags=nlags)

    assert isinstance(
        acf_coefs, np.ndarray
    ), "Something is wrong with the acf function results"

    for i in range(nlags):
        lags_dict[i + 1] = acf_coefs[i + 1]

    return lags_dict


def partial_autocorrelation_analysis(
    series: pd.Series,
    nlags: int,
    exog: pd.DataFrame | None = None,
    controlled: bool = False,
) -> dict[int, float]:
    if controlled:
        series = sm.OLS(endog=series, exog=exog).fit().resid

    lags_dict: dict[int, float] = {}
    pacf_coefs = sm.tsa.stattools.pacf(x=series, nlags=nlags)

    assert isinstance(
        pacf_coefs, np.ndarray
    ), "Something is wrong with the pacf function results"

    for i in range(nlags):
        lags_dict[-(i + 1)] = pacf_coefs[i + 1]

    return lags_dict


def split_series(
    series: pd.Series, buffer: int = 26
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    series = series.iloc[buffer:]
    median_value = series.median()

    low_half = series[series <= median_value]
    high_half = series[series > median_value]

    low_index = low_half.index
    high_index = high_half.index

    return low_index, high_index  # type: ignore
