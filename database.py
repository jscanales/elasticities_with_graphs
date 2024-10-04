"""Database interaction

This module is responsible for the interaction with the database. This includes:
- populating the database by using the individual submodule in the data subfolder
- calling value from the database
- providing an API that can be imported, hence, used in other modules
"""

from functools import cached_property, lru_cache, reduce
from typing import Iterable

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date

import sqlite3
import pandas as pd

import entsoe.exceptions

import data.entosoe
import data.merra2

from data.country_coordinates import mask_from_codes

DATABASE = "database.db"
START_TIME = "20170101"
END_TIME = "202212312359"
MERRA2_FILE = "data/MERRA2/MERRA2_400.tavg1_2d_flx_Nx.20170101.SUB.nc"

def create_schema(cursor: sqlite3.Cursor):

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
            Consumption (
                Time TEXT,
                PriceArea TEXT,
                Value REAL,
                UNIQUE(Time, PriceArea)
            );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
            DayAheadPrice (
                Time TEXT,
                PriceArea TEXT,
                EUR REAL,
                UNIQUE(Time, PriceArea)
            );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
            Generation (
                Time TEXT,
                PriceArea TEXT,
                PsrType TEXT,
                Value REAL,
                UNIQUE(Time, PriceArea, PsrType)
            );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
            Weather (
                Time TEXT,
                PriceArea TEXT,
                WindSpeed REAL,
                Temperature REAL,
                UNIQUE(Time, PriceArea)
            );
        """
    )


def to_nones(x: Iterable, length=1):
    if x is None:
        return [None] * length

    return x


class DatetimeConverter:
    @classmethod
    def to_string(cls, x: datetime) -> str:
        return x.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    @classmethod
    def to_datetime(cls, x: str) -> datetime:
        d, t = x.split("T")
        args = [*d.split("-"), *t.split(":")[:2]]
        args = map(int, args)
        return datetime(*args, tzinfo=timezone.utc)


@dataclass(frozen=True)
class PriceArea:
    entsoe: str = ""
    coord: str|tuple[str,...] = ""


class Areas:
    DK1 = PriceArea(entsoe=data.entosoe.PriceArea.DK1, coord="DNK")
    DK2 = PriceArea( entsoe=data.entosoe.PriceArea.DK2, coord="DNK")
    ES = PriceArea(entsoe=data.entosoe.PriceArea.ES, coord="ESP")
    DE1 = PriceArea(entsoe=data.entosoe.PriceArea.DE1, coord=("DEU", "AUT", "LUX"))
    DE2 = PriceArea(entsoe=data.entosoe.PriceArea.DE2, coord=("DEU", "LUX"))
    FR = PriceArea(entsoe=data.entosoe.PriceArea.FR, coord="FRA")
    UK = PriceArea(entsoe=data.entosoe.PriceArea.UK, coord="GBR")
    PL = PriceArea(entsoe=data.entosoe.PriceArea.PL, coord="POL")


@dataclass
class DayAheadPrice:
    EUR: float
    DKK: float


@dataclass
class Weather:
    temperature: float
    wind_speed: float


@dataclass(frozen=True)
class Query:
    cursor: sqlite3.Cursor = sqlite3.connect(database=DATABASE).cursor()        

    @lru_cache
    def generation(
        self,
        time: datetime,
        price_area: PriceArea,
        psr_type: str,
    ) -> float | None:

        t = DatetimeConverter.to_string(time)
        result = self.cursor.execute(
            f"""SELECT Value FROM Generation WHERE
                    Time = ? AND
                    PriceArea = ? AND
                    PsrType = ? ;""",
            (t, price_area.entsoe, psr_type),
        ).fetchone()

        [result] = to_nones(result, 1)

        return result

    def generation_save(self, time: datetime, price_area: PriceArea, psr_type: str, value):
            t = DatetimeConverter.to_string(time)
            self.cursor.execute(
                f"""INSERT INTO Generation(Time, PriceArea, PsrType, Value) VALUES(?, ?, ?, ?);""",
                (t, price_area.entsoe, psr_type, value),
            )
        

    @lru_cache
    def day_ahead_price(
        self, time, price_area: PriceArea
    ) -> float | None:

        t = DatetimeConverter.to_string(time)
        result = self.cursor.execute(
            """ SELECT EUR FROM DayAheadPrice WHERE
                Time = ? AND
                PriceArea = ?;""",
            (t, price_area.entsoe)
        ).fetchone()

        [result] = to_nones(result, 1)

        return result


    def day_ahead_price_save(self, time, price_area: PriceArea, eur: float):
        t = DatetimeConverter.to_string(time)
        self.cursor.execute(
            f"""INSERT INTO DayAheadPrice(Time, PriceArea, EUR) VALUES(?, ?, ?);""",
            (t, price_area.entsoe, eur),
        )

        
    @lru_cache
    def consumption(
        self,
        time: datetime,
        price_area: PriceArea,
        value=None,
    ) -> float | None:
        t = DatetimeConverter.to_string(time)

        if value is None:
            result = self.cursor.execute(
                """ SELECT Value FROM Consumption WHERE
                    Time=? AND 
                    PriceArea=?;""",
                (t, price_area.entsoe),
            ).fetchone()

            [result] = to_nones(result, 1)
            return result


    def entsoe_load_save(self, time: datetime, price_area: PriceArea, value):
        t = DatetimeConverter.to_string(time)
        self.cursor.execute(
            f"""INSERT INTO Consumption(Time, PriceArea, Value) VALUES(?, ?, ?);""",
            (t, price_area.entsoe, value),
        )


    @lru_cache
    def weather(self, time, price_area:PriceArea) -> Weather:
        t = DatetimeConverter.to_string(time)
        result = self.cursor.execute(
            """ SELECT WindSpeed, Temperature FROM Weather WHERE Time = ? AND PriceArea = ?; """, (t,price_area.entsoe)
        ).fetchone()

        wind_speed, temperature = to_nones(result, 2)
        return Weather(temperature=temperature, wind_speed=wind_speed)

    def weather_save(self, time, area, weather: Weather):
        self.cursor.execute(
            f"""INSERT INTO Weather(Time, PriceArea, WindSpeed, Temperature)
                VALUES(?, ?, ?, ?);""",
            (time, area, weather.wind_speed, weather.temperature),
        )


@dataclass
class BatchUpload:
    query: Query = Query()

    def generation(self, psr_type: str, price_area: PriceArea):
        start = pd.Timestamp(START_TIME, tzinfo=timezone.utc)
        end = pd.Timestamp.now(tz=timezone.utc)
        end = pd.Timestamp(END_TIME, tzinfo=timezone.utc)

        print(f"Generation for {price_area} from {start} to {end} ({psr_type})")
        records = data.entosoe.get_generation(
            start=start, end=end, zone=price_area.entsoe, psr_type=psr_type
        )

        for time, value in records.items():
            self.query.generation_save(
                time=time,
                price_area=price_area,
                psr_type=psr_type,
                value=value,
            )

    def day_ahead_price(self, price_area: PriceArea):
        start = pd.Timestamp(START_TIME, tzinfo=timezone.utc)
        end = pd.Timestamp.now(tz=timezone.utc)
        end = pd.Timestamp(END_TIME, tzinfo=timezone.utc)

        print(f"Price for {price_area} from {start} to {end}")
        records: dict = data.entosoe.get_prices(start=start, end=end, zone=price_area.entsoe)

        for time, value in records.items():
            assert isinstance(time, datetime), f"Time is not a datetime object: {type(time)}"
            self.query.day_ahead_price_save(
                time=time,
                price_area=price_area,
                eur=value,
            )
    
    def entsoe_load(self, price_area:PriceArea):
        start = pd.Timestamp(START_TIME, tzinfo=timezone.utc)
        end = pd.Timestamp.now(tz=timezone.utc)
        end = pd.Timestamp(END_TIME, tzinfo=timezone.utc)
        print(f"Load for {price_area} from {start} to {end}")
        
        records = data.entosoe.get_load(start=start, end=end, zone=price_area.entsoe)
        for time, value in records.items():
            assert isinstance(time, datetime), f"Time is not a datetime object: {type(time)}"
            self.query.entsoe_load_save(
                time=time,
                price_area=price_area,
                value=value,
            )

    # Downloading is not automated
    def weather(self, area:PriceArea):
        country_mask = mask_from_codes(grid_file=r"data\europe_country_grids.csv", 
                                       code=area.coord, 
                                       merra2_file=MERRA2_FILE)

        start = pd.Timestamp(START_TIME, tzinfo=timezone.utc)
        end = pd.Timestamp.now(tz=timezone.utc)
        end = pd.Timestamp(END_TIME, tzinfo=timezone.utc)

        print(f"Weather for {area} from {start} to {end}")
        days = pd.date_range(start=start, end=end, freq="D")

        def upload_file(day: date):
            file = data.merra2.merra2_file(date=day)

            wwind = data.merra2.average_weather(
                file=file,
                attribute=data.merra2.WeatherAttribute.wind_speed,
                country_mask=country_mask
            )
            ttemp = data.merra2.average_weather(
                file=file,
                attribute=data.merra2.WeatherAttribute.temperature,
                country_mask=country_mask
            )

            for i in range(24):
                hour = timedelta(hours=i)
                time: str = DatetimeConverter.to_string(day + hour)

                self.query.weather_save(
                    time=time,
                    area=area.entsoe,
                    weather=Weather(temperature=ttemp[hour], wind_speed=wwind[hour]),
                )

        for day in days:
            try:
                upload_file(day)
                print(f"Day {day} in area {area.coord}")
            except FileNotFoundError as e:
                print("WARNING", e)


def build_database():
    query = Query()
    create_schema(cursor=query.cursor)

    for area in Areas.DE1, Areas.DE2:
        print(f"Starting with {area.entsoe}")
        BatchUpload(query).entsoe_load(area)
        BatchUpload(query).day_ahead_price(area)
        BatchUpload(query).weather(area)
        for gen_t in data.entosoe.PsrTypes.solar, data.entosoe.PsrTypes.wind_offshore, data.entosoe.PsrTypes.wind_onshore:
            print(f"Starting with {gen_t}")
            try:
                BatchUpload(query).generation(psr_type=gen_t, price_area=area)
            except entsoe.exceptions.NoMatchingDataError:
                print(f"WARNING: No data for type {gen_t} in {area.entsoe}")
    query.cursor.connection.commit()


if __name__ == "__main__":
    days = tuple(
        pd.date_range(
            start=pd.Timestamp("201601010000"),
            end=pd.Timestamp("202312312300"),
            freq="D",
            tz="Europe/Berlin",
        )
    )

    build_database()

