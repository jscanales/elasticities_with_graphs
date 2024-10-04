"""Model definition

This module defines four core classes of this analysis used to define model specifications with a certain degree of flexibility:
- HourlySeries manages a collection of variables that can be gathered and incorporated in the model which are continuous
- Dummy manages a collection of variables that can be gathered and incorporated in the model which are binary
- IVModel is a custom wrapper of the underlying model that we use, linearmodels.iv.model.IV2SLS, designed to work with ModelData
- ModelData is the manager of the input data for the model, performing all the transformations of the input data necessary for 
    running the model, such as treating the NA, differencing, taking logs, subsetting hours, etc. 

Besides that, the module defines some other helper functions to perfom these tasks
"""

from __future__ import annotations
from typing import Iterable, TypeVar, Sequence
from functools import cached_property, lru_cache, _lru_cache_wrapper
import numpy as np
import pandas as pd
from math import sin, cos, pi, log
from datetime import datetime
from database import PriceArea, Query
from linearmodels.iv.model import IV2SLS
from linearmodels.iv.results import IVResults
from dataclasses import dataclass
from suntime import Sun

from data.entosoe import PsrTypes

from data.public_holidays import is_public_holiday, is_school_holiday
from data.gas_coal import _gas_price, _coal_price, _eua_price

DataCollection = dict[str, pd.DataFrame]


def treat_nan(
    dfs: list[pd.Series | pd.DataFrame | None], warning_threshold: float = 0.03
) -> tuple[pd.Series | pd.DataFrame | None, ...]:
    result = []

    for df in dfs:
        if df is None:
            result.append(None)
            continue

        idx_common = set.intersection(*[set(df.index) for df in dfs if df is not None])

        def nan_locator(data):
            if isinstance(data, pd.DataFrame):
                nans = set(data.index[data.isna().any(axis=1)])
            elif isinstance(data, pd.Series):
                nans = set(data.index[data.isna()])
            else:
                raise Exception("You are passing an unexpected type!")

            return nans

        idx_common_clean = idx_common.difference(
            *[nan_locator(df) for df in dfs if df is not None]
        )

        if (len(idx_common_clean) / len(idx_common)) < 1 - warning_threshold:
            print(
                f"WARNING: There appears to be more NaN observations than expected: {len(idx_common_clean)} remaining observations compared to initially {len(idx_common)}"
            )

        idx = list(idx_common_clean)
        idx.sort()

        result.append(df.loc[idx] if df is not None else None)

    return tuple(result)


"""
def concatenate(dfs: DataCollection, separator: str = ".") -> pd.DataFrame:
    df = pd.concat(dfs, axis=1)
    df.columns = [f"{level0}{separator}{level1}" for level0, level1 in df.columns]
    return df

def preprocess_inputs(*dfs: tuple[pd.DataFrame, ...]) -> tuple[pd.DataFrame, ...]:
    return treat_nan([concatenate(df) for df in dfs])
"""


# https://stackoverflow.com/questions/31093261/routine-to-extract-linear-independent-rows-from-a-rank-deficient-matrix
def columns_independent(df: pd.DataFrame, tol: float = 1e-10) -> bool:
    _, r = np.linalg.qr(df.T)
    return (np.abs(np.diag(r)) >= tol).all()  # type: ignore


def hours_of_sunlight(
    timestamps: Iterable[datetime], latitude: float, longitude: float
) -> list[float]:

    sun = Sun(lat=latitude, lon=longitude)

    return [
        (
            1
            if sun.get_sunrise_time(t).replace(tzinfo=t.tzinfo) < t
            and t + pd.Timedelta(hours=1)
            < sun.get_sunset_time(t).replace(tzinfo=t.tzinfo)
            else 0
        )
        for t in timestamps
    ]


def within_dates(timestamp, start_month, start_day, end_month, end_day) -> bool:
    year = timestamp.year

    start_date = datetime(year, start_month, start_day, tzinfo=timestamp.tzinfo)
    end_date = datetime(year, end_month, end_day, tzinfo=timestamp.tzinfo)

    return start_date <= timestamp <= end_date


P = TypeVar("P", pd.Series, pd.DataFrame)


@dataclass(frozen=True)
class ModelData:
    dependent: pd.Series
    exogenous: pd.DataFrame
    endogenous: pd.DataFrame
    instruments: pd.DataFrame

    @classmethod
    def construct(
        cls,
        dependent: pd.Series,
        exogenous: list[pd.Series] | None,
        endogenous: list[pd.Series],
        instruments: list[pd.Series],
        scaling: bool = False,
    ):

        _depe = dependent.copy()
        _endo = pd.concat(endogenous, axis=1)
        if exogenous is not None:
            _exog = pd.concat(exogenous, axis=1)  # tpye: ignore
        else:
            print("No objects to concatenate for _exog")
            _exog = None
        _inst = pd.concat(instruments, axis=1)

        _depe, _endo, _exog, _inst = treat_nan(dfs=[_depe, _endo, _exog, _inst])

        def delete_zero_only(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[:, (df != 0).any(axis=0)].copy()

        if scaling:
            assert _depe is not None
            _depe = _depe / _depe.mean()

        return ModelData(
            dependent=_depe, endogenous=_endo, exogenous=delete_zero_only(_exog), instruments=_inst  # type: ignore
        )

    def normalize(self) -> ModelData:
        def _norm(data: P) -> P:
            return (data - data.mean()) / data.std()

        return ModelData(
            dependent=_norm(data=self.dependent),
            exogenous=_norm(data=self.exogenous),
            endogenous=_norm(data=self.endogenous),
            instruments=_norm(data=self.instruments),
        )

    def subset(
        self, idx: pd.DatetimeIndex, exclude: tuple[str, ...] = tuple()
    ) -> ModelData:
        def get_slice(data: P) -> P:
            d = data.loc[idx].copy()
            return d

        def delete_zero_only(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[:, (df != 0).any(axis=0)].copy()

        def delete_one_only(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[
                :, ~((df == 1).all(axis=0) & (df.columns != "constant"))
            ].copy()

        def exclude_names(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[:, [c for c in df.columns if c not in exclude]].copy()

        return ModelData(
            dependent=get_slice(data=self.dependent),
            exogenous=exclude_names(
                delete_one_only(delete_zero_only(get_slice(data=self.exogenous)))
            ),
            endogenous=exclude_names(
                delete_one_only(delete_zero_only(get_slice(data=self.endogenous)))
            ),
            instruments=get_slice(data=self.instruments),
        )

    def mean(self, freq) -> ModelData:
        return ModelData(
            dependent=self.dependent.groupby(pd.Grouper(freq=freq)).mean(),
            exogenous=self.exogenous.groupby(pd.Grouper(freq=freq)).mean(),
            endogenous=self.endogenous.groupby(pd.Grouper(freq=freq)).mean(),
            instruments=self.instruments.groupby(pd.Grouper(freq=freq)).mean(),
        )

    def subset_hour(self, hour: int) -> ModelData:
        def get_slice(data: P) -> P:
            idx = data.index
            assert isinstance(
                idx, pd.DatetimeIndex
            ), "Subset only works if index is a DatetimeIndex"
            d = data[idx.hour == hour].copy()
            return d

        def _clean_weekhour(data: P) -> P:
            columns_to_remove = [f"d.weekhour{i}" for i in range(1, 24)]
            filtered_df = data.drop(columns=columns_to_remove)
            return filtered_df

        return ModelData(
            dependent=get_slice(data=self.dependent),
            exogenous=_clean_weekhour(get_slice(data=self.exogenous)),
            endogenous=get_slice(data=self.endogenous),
            instruments=get_slice(data=self.instruments),
        )

    def diff(self, order: int, skip_instrument=False):

        if skip_instrument:
            instruments = self.instruments.iloc[order:].copy()
        else:
            instruments = self.instruments.diff(order).iloc[order:].copy()

        if self.exogenous is not None:
            exogenous = self.exogenous.diff(order).iloc[order:].copy()
        else:
            exogenous = None

        return ModelData(
            dependent=self.dependent.diff(order).iloc[order:].copy(),
            exogenous=exogenous,
            endogenous=self.endogenous.diff(order).iloc[order:].copy(),
            instruments=instruments,
        )

    def drop_empty_series(self):
        columns_to_drop = []
        for col in self.exogenous.columns:
            is_always_0 = (self.exogenous[col] == 0).all()
            is_always_1 = (self.exogenous[col] == 1).all()

            if (is_always_0 or is_always_1) and not col == "constant":
                columns_to_drop.append(col)

        self.exogenous.drop(labels=columns_to_drop, axis=1, inplace=True)

        return ModelData(
            dependent=self.dependent,
            exogenous=self.exogenous,
            endogenous=self.endogenous,
            instruments=self.instruments,
        )

    def log_log_transform(self, instrument:bool) -> "ModelData":
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

        return ModelData.construct(
            dependent=_log_log_data(self.dependent),
            exogenous=_deconcatenate_dataframe(_log_log_data(self.exogenous)),
            endogenous=_deconcatenate_dataframe(_log_log_data(self.endogenous)),
            instruments=_deconcatenate_dataframe(_log_log_data(self.instruments)) if instrument else _deconcatenate_dataframe(self.instruments),
        )

    def subset_hours_range(self, start_hour: int, end_hour: int) -> ModelData:
        def get_slice(data: P, start: int, end: int) -> P:
            idx = data.index
            assert isinstance(
                idx, pd.DatetimeIndex
            ), "Subset only works if the index is a DatetimeIndex"
            if end > start:
                d = data[(idx.hour >= start) & (idx.hour <= end)].copy()
            else:
                d = data[(idx.hour >= start) | (idx.hour <= end)].copy()
            return d

        def _clean_weekhour(data: P) -> P:
            columns_to_remove = [f"d.weekhour{i}" for i in range(1, 24)]
            filtered_df = data.drop(columns=columns_to_remove)
            return filtered_df

        return ModelData(
            dependent=get_slice(data=self.dependent, start=start_hour, end=end_hour),
            exogenous=_clean_weekhour(
                get_slice(data=self.exogenous, start=start_hour, end=end_hour)
            ),
            endogenous=get_slice(data=self.endogenous, start=start_hour, end=end_hour),
            instruments=get_slice(
                data=self.instruments, start=start_hour, end=end_hour
            ),
        )

    def subset_months(self, months: list[int]) -> ModelData:
        def get_slice(data: P) -> P:
            idx = data.index
            assert isinstance(
                idx, pd.DatetimeIndex
            ), "Subset only works if index is a DatetimeIndex"
            d = data[idx.month.isin(months)].copy()
            return d

        def _clean_month(data: P) -> P:
            columns_to_remove = [f"d.month{i}" for i in range(2, 12)]
            for col in data.columns:
                if "year" in col:
                    columns_to_remove.append(col)
            filtered_df = data.drop(columns=columns_to_remove)
            return filtered_df

        return ModelData(
            dependent=get_slice(data=self.dependent),
            exogenous=_clean_month(get_slice(data=self.exogenous)),
            endogenous=get_slice(data=self.endogenous),
            instruments=get_slice(data=self.instruments),
        )
    
    def drop(self, vars:list[str]):
        for df in self.exogenous, self.endogenous, self.instruments:
            for var in vars:
                if var in df.columns:
                    print(f"{var} should be dropped")

        ex = self.exogenous.drop(columns=vars, inplace=False, errors="ignore")
        en = self.endogenous.drop(columns=vars, inplace=False, errors="ignore")
        ins = self.instruments.drop(columns=vars, inplace=False, errors="ignore")

        for df in ex, en, ins:
            for var in vars:
                if var in df.columns:
                    print(f"{var} has not been dropped")

        return ModelData(self.dependent, ex, en, ins)



@dataclass(frozen=True)
class IVModel:
    """
    cov_type:
        "robust" / "heteroskedastic" -> Accounts for heteroskedasticity no for auto-correlation adjustment (default)
        "kernel" -> Heteroskedasticity and autocorrelation robust inference;
            "bartlett" / "newey-west" -> kernel_weight_bartlett (default)
            "quadratic-spectral" / "qs" / "andrews" -> kernel_weight_quadratic_spectral
            "gallant" / "parzen" -> kernel_weight_parzen

    background on the kernel options:
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation
        Author(s): Donald W. K. Andrews
        https://www.jstor.org/stable/pdf/2938229.pdf
    """

    data: ModelData
    cov_type: str = "kernel"  # "kernel"

    @cached_property
    def fitted_model(self):

        return IV2SLS(
            dependent=self.data.dependent,  # <- consumption
            exog=self.data.exogenous,  # <- dummy + control
            endog=self.data.endogenous,  # <- price
            instruments=self.data.instruments,  # <- wind
        ).fit(cov_type=self.cov_type)


def to_series(index, values: Sequence[float | bool | None], name: str) -> pd.Series:
    return pd.Series(values, index=index, name=name, dtype=float)


@dataclass(frozen=True, eq=False)
class Dummy:
    timestamps: pd.DatetimeIndex

    @lru_cache
    def constant(self) -> pd.Series:
        return to_series(index=self.timestamps, values=1.0, name="constant")  # type: ignore

    @lru_cache
    def trend(self) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=list(np.linspace(0.0, 1.0, num=len(self.timestamps))),
            name="trend",
        )

    @lru_cache
    def hour(self, skip_first=True) -> list[pd.Series]:
        _hours = self.timestamps.hour.unique().sort_values()
        if skip_first:
            _hours = _hours[1:]

        return [
            to_series(
                index=self.timestamps,
                values=(self.timestamps.hour == h),
                name=f"d.hour{h}",
            )
            for h in _hours
        ]

    @lru_cache
    def month(self, skip_first=True) -> list[pd.Series]:
        _months = self.timestamps.month.unique().sort_values()
        if skip_first:
            _months = _months[1:]

        return [
            to_series(
                index=self.timestamps,
                values=(self.timestamps.month == m),
                name=f"d.month{m}",
            )
            for m in _months
        ]

    @lru_cache
    def year(self, skip_first=True) -> list[pd.Series]:
        _years = self.timestamps.year.unique().sort_values()
        if skip_first:
            _years = _years[1:]

        return [
            to_series(
                index=self.timestamps,
                values=(self.timestamps.year == y),
                name=f"d.year{y}",
            )
            for y in _years
        ]

    @lru_cache
    def weekday(self, skip_first=True) -> list[pd.Series]:
        _weekdays = self.timestamps.weekday.unique().sort_values()
        if skip_first:
            _weekdays = _weekdays[1:]
        return [
            to_series(
                index=self.timestamps,
                values=(self.timestamps.weekday == wd),
                name=f"d.weekday{wd}",
            )
            for wd in _weekdays
        ]

    @lru_cache
    def weekhour(self, skip_first=True) -> list[pd.Series]:
        def get_weekhour(idx):
            return (idx.dayofweek) * 24 + (idx.hour)

        _weekhour = get_weekhour(self.timestamps).unique().sort_values()
        if skip_first:
            _weekhour = _weekhour[1:]

        return [
            to_series(
                index=self.timestamps,
                values=(get_weekhour(self.timestamps) == wh),
                name=f"d.weekhour{wh}",
            )
            for wh in _weekhour
        ]

    @lru_cache
    def school_holiday(self) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                is_school_holiday(time.replace(tzinfo=None)) for time in self.timestamps
            ],
            name="d.school_holidays",
        )

    @lru_cache
    def public_holiday(self, area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                is_public_holiday(time.replace(tzinfo=None), area=area.entsoe)
                for time in self.timestamps
            ],
            name="d.public_holiday",
        )

    @lru_cache
    def week_53(self):
        return to_series(
            index=self.timestamps,
            values=[
                within_dates(
                    timestamp=time,
                    start_day=24,
                    start_month=12,
                    end_day=31,
                    end_month=12,
                )
                for time in self.timestamps
            ],
            name="d.week_53",
        )

    @lru_cache
    def level_break(self, b: pd.Timestamp) -> pd.Series:
        as_string = b.strftime("%Y-%m-%d")
        values = list(self.timestamps >= b.replace(tzinfo=None))
        return to_series(
            index=self.timestamps,
            values=values,
            name=f"break-{as_string}",
        )

    @lru_cache
    def sinusoid(self, period: float) -> tuple[pd.Series, pd.Series]:
        angle = (2.0 * pi) / period

        ds_sin = to_series(
            index=self.timestamps,
            values=[sin(angle * d.toordinal()) for d in self.timestamps],
            name=f"sin{period}",
        )
        ds_cos = to_series(
            index=self.timestamps,
            values=[cos(angle * d.toordinal()) for d in self.timestamps],
            name=f"cos{period}",
        )

        return ds_sin, ds_cos


@dataclass(frozen=True, eq=False)
class HourlySeries:
    timestamps: pd.DatetimeIndex
    query: Query = Query()

    @lru_cache
    def consumption(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.consumption(time=timestamp, price_area=price_area)
                for timestamp in self.timestamps
            ],
            name=f"consumption.{price_area.entsoe}",
        )

    # def consumptions(self, codes: list[int], price_area:PriceArea) -> pd.Series:
    #     dss = [self.consumption(code=code, price_area=price_area) for code in codes]
    #     ds = reduce(lambda ds1, ds2: ds1.add(ds2), dss)
    #     ds.name = f"consumption.{price_area.energie}.agg.H"
    #     return ds

    @lru_cache
    def price(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.day_ahead_price(time=time, price_area=price_area)
                for time in self.timestamps
            ],
            name=f"price.{price_area.entsoe}.H",
        )

    @lru_cache
    def wind_speed(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.weather(time=time, price_area=price_area).wind_speed
                for time in self.timestamps
            ],
            name="wind_speed.H",
        )

    @lru_cache
    def temperature(self, price_area: PriceArea, celsius: bool = False) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                (
                    self.query.weather(time=time, price_area=price_area).temperature
                    if not celsius
                    else self.query.weather(
                        time=time, price_area=price_area
                    ).temperature
                    - 273.15
                )
                for time in self.timestamps
            ],
            name="temperature.H",
        )

    """For the methodology used to calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD), see Eurostat
    https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm"""

    @lru_cache
    def hdd(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                np.nan if pd.isna(t) else 18 - (t - 273.15) if (t - 273.15) <= 15 else 0
                for t in self.temperature(price_area).copy()
            ],
            name="heating_degree_days.H",
        )

    @lru_cache
    def cdd(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                np.nan if pd.isna(t) else (t - 273.15) - 21 if (t - 273.15) >= 24 else 0
                for t in self.temperature(price_area).copy()
            ],
            name="cooling_degree_days.H",
        )

    @lru_cache
    def gas_price(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[_gas_price(time, price_area.entsoe) for time in self.timestamps],
            name="gas_price.D",
        )

    @lru_cache
    def coal_price(self) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[_coal_price(time) for time in self.timestamps],
            name="coal_price.D",
        )

    @lru_cache
    def eua_price(self) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[_eua_price(time) for time in self.timestamps],
            name="eua_price.D",
        )

    @lru_cache
    def sunlight(self, price_area: PriceArea) -> pd.Series:
        # TODO: check that we this is how we intend implementing "sunlight"
        # Coordinates refer to Denmark geographic center

        coordinates: dict[str, list[float]] = {
            "ES": [40.183100, 3.410400],
            "DE_LU": [51.094815, 10.265166],  # Not True
            "DE_AT_LU": [51.094815, 10.265166],  # Not True: Germany's central point
            "PL": [52.112795, 19.211936],
            "DK_1": [56.263900, 9.501800],
            "DK_2": [56.263900, 9.501800],
            "FR": [46.453400, 2.240400],
        }

        key: str = price_area.entsoe
        latitude, longitude = coordinates[key]

        return to_series(
            index=self.timestamps,
            values=hours_of_sunlight(
                self.timestamps, latitude=latitude, longitude=longitude
            ),
            name="sunlight.H",
        )

    @lru_cache
    def solar_generation(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.generation(
                    time=timestamp, price_area=price_area, psr_type=PsrTypes.solar
                )
                for timestamp in self.timestamps
            ],
            name=f"generation.{price_area.entsoe}.solar",
        )

    @lru_cache
    def onshore_wind_generation(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.generation(
                    time=timestamp,
                    price_area=price_area,
                    psr_type=PsrTypes.wind_onshore,
                )
                for timestamp in self.timestamps
            ],
            name=f"generation.{price_area.entsoe}.onshore",
        )

    @lru_cache
    def offshore_wind_generation(self, price_area: PriceArea) -> pd.Series:
        return to_series(
            index=self.timestamps,
            values=[
                self.query.generation(
                    time=timestamp,
                    price_area=price_area,
                    psr_type=PsrTypes.wind_offshore,
                )
                for timestamp in self.timestamps
            ],
            name=f"generation.{price_area.entsoe}.offshore",
        )

    @lru_cache
    def total_wind_generation(self, price_area: PriceArea) -> pd.Series:
        return self.onshore_wind_generation(
            price_area=price_area
        ) + self.offshore_wind_generation(price_area=price_area)


def calender_mean(ds: pd.Series, freq: str) -> pd.Series:
    # label='left' and closed='left' are used to ensure the same behaviour between daily and weekly resampling
    assert freq in [
        "H",
        "D",
        "W-MON",
        "MS",
    ], f"{freq=} has not been tested, use either 'D' for daily, 'W-MON' for weekly' or 'MS' for monthly"

    return ds.resample(rule=freq, label="left", closed="left").mean()


def average(ds: pd.Series, freq: str, delay: int | None = None) -> pd.Series:
    grouped_series = calender_mean(ds=ds, freq=freq)

    levels = str(ds.name).split(".")
    # H can be omitted when aggregating
    if levels[-1] == "H":
        del levels[-1]

    levels = levels + [f"{freq}"]

    if delay is not None:
        grouped_series = grouped_series.shift(delay)
        levels = levels + [f"L{delay}"]

    grouped_series.name = ".".join(levels)

    return grouped_series.reindex(ds.index, method="ffill")


def lag(ds: pd.Series, freq: str, delay: int) -> pd.Series:
    assert freq in ["H", "D", "M"]

    new_ds = ds.shift(periods=delay, freq=freq)
    new_ds.name = str(ds.name) + f".L({str(delay) + freq})"
    return new_ds


def interval(start, end) -> pd.DatetimeIndex:
    return pd.date_range(
        start=start,
        end=end,
        freq="H",
        tz="Europe/Berlin",
    )


if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt

    # idx = pd.date_range(start=pd.Timestamp("2020 01 01 00"), freq="H", periods=100000)

    # iv = IVModel(
    #     dependent=pd.Series(np.random.rand(len(idx)), index=idx),
    #     endogenous=pd.DataFrame(
    #         np.random.rand(len(idx), 2), index=idx, columns=["en1", "en2"]
    #     ),
    #     exogenous=pd.DataFrame(
    #         np.random.rand(len(idx), 2), index=idx, columns=["ex1", "ex2"]
    #     ),
    #     instruments=pd.DataFrame(
    #         np.random.rand(len(idx), 2), index=idx, columns=["in1", "in2"]
    #     ),
    # )

    # m = iv.fitted_model()
    # m.params
    # m.resids.plot()
    # plt.show()

    # area = Areas.DK1
    # code = 381
    # days = pd.date_range(
    #     start=pd.Timestamp("201901010000"),
    #     end=pd.Timestamp("202208302300"),
    #     freq="D",
    #     tz="Europe/Berlin",
    # )

    # hs = HourlySeries(days=days)
    # ds = hs.consumption(price_area=area, code=code)

    # ds.plot()
    # plt.show()
    pass
