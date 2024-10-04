"""
This file containts functions and classes to manage the process of creating and running simulations.

It builds on top of synthetic.py and estimation.py to generate synthetic data and apply the 
estimation strategies.

The core class of this module is Simulation, which uses a sample size, a list of ModelInputParameters and
a StructuralModel to produce IV estimates from synthetic data. 
Results are handled by the class ModelMetrics.
"""

import numpy as np
import pandas as pd
import random

from dataclasses import dataclass, field
from functools import cached_property
from enum import Enum, auto

from model import Dummy, HourlySeries, interval
from database import PriceArea, Areas
from estimation import (
    ModelInputParams,
    ModelResults,
    MultipleModelResults,
    IVOptions,
    IV_results,
)
from synthetic import (
    AggEquilibrium,
    IndEquilibrium,
    Supply,
    IndividualDemand,
    AggregatedDemand,
    WindOptions,
    get_wind_series,
    RandomCached,
    average_supply_price,
    Consumption,
    SUPPLY_INTERCEPT,
    SUPPLY_SIGMA,
    SUPPLY_SLOPE,
)


@dataclass
class ModelMetrics:
    """Stores several relevant metrics of an estimator over different time series.
    Each estimator has a corresponding ModelMetrics within the results of a Simulation.
    A new ModelResult is added with the funcion ModelMetrics.add_result()
    The metrics then available are:
        n: int                          -   number of results stored
        estimates: list                 -   the list of point estimates
        mean: int                       -   average estimate across all model results
        ci_upper_bound: list            -   the upper bound of the confidence interval (95% confidence) with mean 0
        ci_lower_bound: list            -   the lower bound of the confidence interval (95% confidence) with mean 0
        average_percentage_error: float -   average percentage error across all models
        average_coverage: float         -   average coverage error across all models
        average_f_statistic: float      -   average F-statistic of the second stage across all models
        average_distance: float         -   average distance between the point estimate and the upper bound of the confidence interval across all models

    Make sure the true elasticity value is the correct one for the simulation that you are running
    You can always update the true value afterwards with ModelMetrics.update_true_elasticity()
    """

    name: str
    results: list[ModelResults] = field(default_factory=list)
    elasticity: float = -100

    def add_result(self, result: ModelResults | MultipleModelResults) -> None:
        self.results.append(result)

    def update_true_elasticity(self, new_value: float, warning: bool = True):
        self.elasticity = new_value
        if warning and self.results:
            print(
                f"WARNING! Attempting to change the elasticity value of the {self.name} ModelMetrics when there are already some results (n = {self.n})"
            )
            print(
                f"Make sure this is consistent since it can lead to problems with the percentage error calculations"
            )

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def estimates(self) -> list[float]:
        return [result.estimate for result in self.results]

    @property
    def ci_upper_bound(self) -> list[float]:
        return [result.ci[1] for result in self.results]

    @property
    def ci_lower_bound(self) -> list[float]:
        return [result.ci[0] for result in self.results]

    @property
    def true_ci(self) -> list[tuple[float, float]]:
        return [
            (
                self.ci_lower_bound[i] + self.estimates[i],
                self.ci_upper_bound[i] + self.estimates[i],
            )
            for i in range(self.n)
        ]

    @property
    def mean(self) -> float:
        try:
            return sum(self.estimates) / self.n
        except ZeroDivisionError:
            return np.NaN

    @property
    def average_distance(self) -> float:
        return round(sum(result.ci[1] for result in self.results) / self.n, 2)

    @property
    def percentage_errors(self) -> list[float]:
        return [
            (self.elasticity - result.estimate) / (self.elasticity) * 100
            for result in self.results
        ]
    
    @property
    def absolute_percentage_errors(self) -> list[float]:
        return [abs(error) for error in self.percentage_errors]

    @property
    def average_percentage_error(self) -> float:
        return round(sum(self.percentage_errors) / self.n, 2)

    @property
    def average_absolute_percentage_error(self) -> float:
        return round(sum([abs(err) for err in self.percentage_errors]) / self.n, 2)

    @property
    def coverage_count(self) -> int:
        return sum(
            result.ci[0] + result.estimate
            <= self.elasticity
            <= result.ci[1] + result.estimate
            for result in self.results
        )

    @property
    def average_coverage(self) -> float:
        return round(self.coverage_count / self.n, 2)

    @property
    def average_f_statistic(self) -> float | None:
        return round(sum(result.f_statistic for result in self.results) / self.n, 2)

    def print(self):
        print(
            f"{self.name}:\n\tAverage CI length:\t{self.average_distance}\n\tAbsolute percentage error:\t{self.average_absolute_percentage_error}\n\tCoverage ratio:\t{self.average_coverage}\n\tAverage F-Statistic:\t{self.average_f_statistic}"
        )

    def to_dict(self):
        return {
            "model": self.name,
            "average_distance": self.average_distance,
            "average_percentage_error": self.average_percentage_error,
            "average_coverage": self.average_coverage,
            "average_f_statistic": self.average_f_statistic,
        }


class StructuralModel(Enum):
    Model_I = auto()  # simple, autocorrelated demand
    Model_II = auto()  # heterogeneous demand
    Model_III = auto()  # cross price elasticity
    Model_IV = auto()  # complex model


@dataclass
class Simulation:
    """
    The Simulation class handles running and analyzing simulation models related to 
    demand, supply, and equilibrium in various structural models.

    Attributes:
        sample (int): The number of observations (time points) in the simulation.
        models (dict[str, ModelMetrics]): Dictionary that holds the simulation results for 
                                          different models. Each model's results are stored 
                                          in a ModelMetrics object.
        consumption_analysis_interval (pd.DatetimeIndex): Time interval for analyzing consumption 
                                                          data. Defaults to 2019-2020.
        price_area (PriceArea): The price area for analyzing consumption data. 
                                Defaults to Areas.DE2.

    Properties:
        interval (pd.DatetimeIndex): Cached property that provides the time interval used in 
                                     the simulation.
        dummy (Dummy): Cached property that provides a dummy manager for simulations.
        series (HourlySeries): Cached property that provides the hourly time series manager 
                               for simulations.
        average_demand (float): Cached property that computes the average demand during the 
                                consumption analysis interval in the price area.
        average_price (float): Cached property that computes the average price for supply during
                                the consumption analysis interval in the price area.

    Methods:
        add_result(result): Adds a ModelResults or MultipleModelResults object to the 
                            corresponding model in the simulation.
        clean_empty(): Removes models from the `models` attribute that do not have any results.
        report_as_text(): Prints a summary of the results of each model.
        report_as_dataframe(): Returns the results of each model as a pandas DataFrame.
        analyze_equilibrium(eq, models, controls, print_results): Runs the equilibrium analysis 
                                                                  on the specified models.
        restart(): Clears all model results and resets the simulation.
        get_equilibrium(demand, supply): Generates an equilibrium object based on demand and 
                                         supply inputs.
        run_single_simulation(i, seed, estimators, model, controls, wind_lags, demand_arg, 
                              wind_manual_ar_sum, progress): Runs a single instance of the 
                                                              simulation with the specified 
                                                              parameters.
        get_supply(wind_manual_ar_sum, wind_lags): Returns a Supply object with specified wind 
                                                   settings and supply parameters.
        get_demand(model, demand_param): Returns a demand object (either IndividualDemand or 
                                        AggregatedDemand) based on the structural model.
        run_simulations_by_model(runs, estimators, model, demand_arg, simulation_count, 
                                 preset_seeds, wind_manual_ar_sum, wind_lags): Runs multiple 
                                 simulations for a given model with specified parameters.
    """
    sample: int
    models: dict[str, ModelMetrics] = field(default_factory=dict)

    # Time and place to use as reference for the demand parameters
    consumption_analysis_interval: pd.DatetimeIndex = interval(
        start=pd.Timestamp("201901010000"), end=pd.Timestamp("202001282300")
    )
    price_area: PriceArea = Areas.DE2

    @cached_property
    def interval(self):
        return interval(
            start=pd.Timestamp("201901010000"),
            end=pd.Timestamp("201901290000") + pd.Timedelta(hours=self.sample - 1),
        )

    @cached_property
    def dummy(self):
        return Dummy(self.interval)

    @cached_property
    def series(self):
        return HourlySeries(self.interval)

    @cached_property
    def average_demand(self):
        return Consumption(
            start=self.consumption_analysis_interval.min(),
            end=self.consumption_analysis_interval.max(),
            price_area=self.price_area,
        )().mean()

    @cached_property
    def average_price(self):
        return average_supply_price(
            quantity=self.average_demand,
            series=HourlySeries(self.consumption_analysis_interval),
            price_area=self.price_area,
        )

    def add_result(self, result: ModelResults | MultipleModelResults):
        """Adds a result to the corresponding model.
        If the model doesn't have a ModelMetrics result holder yet, a new one is created.

        If the result argument is a MultipleModelResults object (if the IV estimator has more than one dimension),
        a different ModelMetrics is created for each estimate
        (i.e., one for P_t -> D_t and one for D_{t-1} - D_t)"""

        if isinstance(result, MultipleModelResults):
            model_results_list: list[ModelResults] = []
            for k in result.estimate.keys():
                model_results = ModelResults(
                    estimate=result.estimate[k],
                    ci=result.ci[k],
                    model=result.model + k,
                    f_statistic=result.f_statistic,
                    mse=result.mse,
                    latex_name=result.latex_name,
                )
                model_results_list.append(model_results)
            for model in model_results_list:
                try:
                    self.models[model.model].add_result(model)
                except KeyError:
                    self.models[model.model] = ModelMetrics(name=model.model)
                    self.models[model.model].add_result(model)

        else:
            try:
                self.models[result.model].add_result(result)
            except KeyError:
                self.models[result.model] = ModelMetrics(name=result.model)
                self.models[result.model].add_result(result)

    def clean_empty(self):
        """Cleans up models without any results,
        in case models were initialized but never used"""
        empty_models = []
        for name, model in self.models.items():
            if model.n == 0:
                empty_models.append(name)
        for name in empty_models:
            self.models.pop(name)

    def report_as_text(self):
        """Prints the results of each model"""
        for results in self.models.values():
            results.print()

    def report_as_dataframe(self):
        """Returns the results of each model as single pandas.DataFrame"""
        results_dict = [results.to_dict() for results in self.models.values()]
        return pd.DataFrame(results_dict)

    def analyze_equilibrium(
        self,
        eq: IndEquilibrium | AggEquilibrium,
        models: list[ModelInputParams],
        controls: bool,
        print_results:bool
    ):
        """Analyzes equilibrium using specified models
        Part of running a simulation, essentially producing IV_results"""
        if len(eq.clearing.price) != len(eq.clearing.demand) != len(eq.supply.wind):
            print(len(eq.clearing.price)) 
            print(len(eq.clearing.demand)) 
            print(len(eq.supply.wind)) 

        for index, mod in enumerate(models):
            print(f"starting with {mod.strategy}")
            try:
                result: ModelResults = IV_results(
                    price_area=self.price_area,
                    demand=eq.clearing.demand,
                    price=eq.clearing.price,
                    wind=eq.supply.wind,
                    dummy=self.dummy,
                    series=self.series,
                    model=mod,
                    n_lags=eq.nlags,
                    controls=controls,
                )
                got_result = True
            except np.linalg.LinAlgError or ValueError:
                got_result = False
            # else:
            #     raise Exception("One of the models is not specified correctly", mod.strategy)

            if got_result:
                if print_results:
                    print(result.estimate)
                self.add_result(result)

    def restart(self):
        """Restarts the simulation by clearing all model results"""
        self.models.clear()

    def get_equilibrium(
        self, demand: AggregatedDemand | IndividualDemand, supply: Supply
    ) -> IndEquilibrium | AggEquilibrium:
        """Runs the simulation with specified parameters and
        runs the analysis of the time series for the specified estimators"""
        if isinstance(demand, AggregatedDemand):
            eq = AggEquilibrium(
                supply=supply.copy(), demand=demand.copy(), times=self.interval
            )
        else:
            eq = IndEquilibrium(
                supply=supply.copy(), demand=demand.copy(), times=self.interval
            )

        return eq

    def run_single_simulation(
        self,
        i: int,
        seed: int,
        estimators: list[ModelInputParams],
        model: StructuralModel,
        controls: bool,
        wind_lags: int,
        demand_arg: None | float = None,
        wind_manual_ar_sum: float | None = None,
        progress: bool = True,
    ):
        """Runs the simulation with specified parameters and
        runs the analysis of the time series for the specified estimators"""

        demand = (
            self.get_demand(model, demand_arg)
            if demand_arg is not None
            else self.get_demand(model)
        )
        supply = self.get_supply(wind_manual_ar_sum, wind_lags)

        eq = self.get_equilibrium(demand, supply)

        self.analyze_equilibrium(eq=eq, models=estimators, controls=controls, print_results=progress)

        if progress:
            print(f"Simulation {i+1} concluded. \t Seed No. {seed}\n")

    def get_supply(self, wind_manual_ar_sum: float | None, wind_lags:int) -> Supply:
        """Gets the standard supply equation. Supply parameters are imported from Synthetic.
        The function returns an instance of the Supply class"""
        return Supply(
            price_area=self.price_area,
            wind=get_wind_series(
                wind_type=WindOptions.SYNTHETIC_AR_N,
                series=HourlySeries(self.consumption_analysis_interval),
                price_area=self.price_area,
                synthetic_nlags=wind_lags,
                synthetic_n_observations=self.sample,
                manual_lags_sum=wind_manual_ar_sum,
            ),
            error=RandomCached.normal(0, SUPPLY_SIGMA),
            slope=SUPPLY_SLOPE,
            intercept=SUPPLY_INTERCEPT,
        )

    def get_demand(
        self, model: StructuralModel, demand_param: float|None = None
    ) -> IndividualDemand | AggregatedDemand:
        """Gets the demand based on the structural equation for Models I, II and III.

    Args:
        model (StructuralModel): The model to use (Model_I, Model_II, Model_III).
        demand_param (float, optional): Optional parameter to adjust demand elasticity. 
                                        For Model I and II, it adjusts the autocorrelation coefficient. 
                                        It defaults to 0.9.
                                        For Model III, it adjusts the cross-price elasticity.
                                        It defaults to 50.

    Returns:
        IndividualDemand or AggregatedDemand: Demand object based on the structural model.
    """
        demand_param = demand_param if demand_param is not None else {
            StructuralModel.Model_I: 0.9,
            StructuralModel.Model_II: 0.9,
            StructuralModel.Model_III: 50,
        }.get(model, None)

        match model:
            case StructuralModel.Model_I:
                return IndividualDemand(
                    base_constant=self.average_demand,
                    lags={-1: demand_param},
                    elasticity=-100,
                    error=RandomCached.normal(0, 2000),
                    average_supply_price=self.average_price,
                    cross_price_elasticity={},
                )

            case StructuralModel.Model_II:
                return AggregatedDemand(
                    components=[
                        IndividualDemand(
                            base_constant=self.average_demand * 0.5,
                            lags={},
                            elasticity=-100,
                            error=RandomCached.normal(0, 2000),
                            average_supply_price=self.average_price,
                            cross_price_elasticity={},
                        ),
                        IndividualDemand(
                            base_constant=self.average_demand * 0.5,
                            lags={-1: demand_param},
                            elasticity=0,
                            error=RandomCached.normal(0, 2000),
                            average_supply_price=self.average_price,
                            cross_price_elasticity={},
                        ),
                    ]
                )

            case StructuralModel.Model_III:
                return IndividualDemand(
                    base_constant=self.average_demand,
                    lags={},
                    elasticity=-100,
                    error=RandomCached.normal(0, 2000),
                    average_supply_price=self.average_price,
                    cross_price_elasticity={-1: demand_param},
                )

            case StructuralModel.Model_IV:
                return AggregatedDemand(
                    components=[
                        IndividualDemand(
                            base_constant=self.average_demand * 0.5,
                            lags={-1: 0.2},
                            elasticity=-100,
                            error=RandomCached.normal(0, 2000),
                            average_supply_price=self.average_price,
                            cross_price_elasticity={-1: 50},
                        ),
                        IndividualDemand(
                            base_constant=self.average_demand * 0.5,
                            lags={-1: 0.9},
                            elasticity=0,
                            error=RandomCached.normal(0, 2000),
                            average_supply_price=self.average_price,
                            cross_price_elasticity={},
                        ),
                    ]
                )

            case _:
                raise NotImplementedError(f"Unrecognized model {model}")

    def run_simulations_by_model(
        self,
        runs: int,
        estimators: list[ModelInputParams],
        model: StructuralModel,
        demand_arg: float | None,
        simulation_count: int = 0,
        preset_seeds: list[int] = [],
        wind_manual_ar_sum: float | None = None,
        wind_lags:int = 26, 
    ):
        """Runs multiple simulations of a StructuralModel with specified parameters and
        runs the analysis of the synthetic time series for the specified estimators.
        The results are directly stored in the Simulation isntance."""

        if preset_seeds:
            seeds: list[int] = preset_seeds
        else:
            seeds: list[int] = random.sample(range(1, 1000001), runs)

        for n, seed in enumerate(seeds):
            np.random.seed(seed)

            self.run_single_simulation(
                i=simulation_count + n,
                seed=seed,
                model=model,
                demand_arg=demand_arg,
                estimators=estimators,
                wind_manual_ar_sum=wind_manual_ar_sum,
                wind_lags=wind_lags,
                controls=False,
            )
