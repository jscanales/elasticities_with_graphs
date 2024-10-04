import os
import pathlib
import re
import netCDF4
import requests
import csv
import numpy as np
from pathlib import Path
from datetime import date, timedelta

try: 
    from data.config import Nasa
    from data.country_coordinates import read_grid_csv
except ModuleNotFoundError as e:
    print(e, "Are you executing the file directly?")
    from config import Nasa
    from country_coordinates import read_grid_csv


class WeatherAttribute:
    temperature: str = "TLML"
    wind_speed: str = "SPEED"


def download_file(url: str, save_in: str | Path):
    
    (file_name,) = re.findall(r"(?<=LABEL=)(.+?)(?=\&SHORTNAME)", url)
    file_path = save_in / file_name

    if os.path.exists(file_path):  # Check if file exists
        print(f"File already exists: {file_name}")
        return
    
    r = requests.get(
        url=url,
        headers=Nasa.header,
        auth=(Nasa.user, Nasa.password),
    )

    if r.status_code == 200:  # 200
        (file_name,) = re.findall('filename="(.+)"', r.headers["content-disposition"])

        with open(save_in / file_name, "wb") as file:
            file.write(r.content)

    elif r.status_code == 404:  # Non existing, try alternative label
        alt_url = re.sub(r"(?<=MERRA2_)(.+?)(?=\.tavg1)", "401", url)

        (file_name,) = re.findall(r"(?<=LABEL=)(.+?)(?=\&SHORTNAME)", alt_url)
        file_path = save_in / file_name

        if os.path.exists(file_path):  # Check if file exists
            print(f"File already exists: {file_name}")
            return

        alt_r = requests.get(
            url=alt_url,
            headers=Nasa.header,
            auth=(Nasa.user, Nasa.password),
        )
        if alt_r.status_code == 200:  # 200
            with open(file_path, "wb") as file:
                file.write(alt_r.content)
        else:
            raise Exception(f"Both requests (400 and 401) failed for URL: {url}")
    else:
        print("Status code: ", r.status_code)
        raise Exception(
            f"Something went wrong with: {url}. Maybe you need to update the request header stored in the config file?",
            "More instructions can be found within the MERRA2 folder readme file."
        )


FOLDER = Path('data', 'MERRA2')


def merra2_file(date: date, folder=FOLDER):
    regex = f"MERRA2_40.\\.tavg1_2d_flx_Nx\\.{date.strftime('%Y%m%d')}\\.SUB\\.nc"
    file_list = [f for f in os.listdir(folder) if re.match(regex, f)]
    if len(file_list) == 0:
        raise FileNotFoundError(f"No MERRA2 file seems to exist for this {date}")
    (f,) = file_list
    return folder / f

def average_weather(file: str|Path, attribute: str, country_mask) -> dict[timedelta, float]:
    with netCDF4.Dataset(file, "r") as f:   # type: ignore
        f.variables["time"]
        tt = [timedelta(minutes=float(i)) for i in f.variables["time"][:]]
        xx = f.variables[attribute]  # 3D matrix, (time, lat, lon)

        latitudes = f.variables['lat'][:]
        longitudes = f.variables['lon'][:]
        variable_data = f.variables[attribute][:]

        hourly_averages = {}

        for i, t in enumerate(tt):
            data_hour = variable_data[i, :, :]   
            data_hour_country = data_hour[country_mask]
            average_data_hour_country = np.mean(data_hour_country)

            # Store the average wind speed in the dictionary with the hour as the key
            hourly_averages[t] = float(average_data_hour_country)

        return hourly_averages



if __name__ == "__main__":

    # Download weather raster file
    
    urls = open(r"data\MERRA2\subset_M2T1NXFLX_5.12.4_Europe_20160101_20231231.txt").read().split("\n")

    save_in = Path(".", "data\\MERRA2")
    for url in urls:
        print(f"working on file {urls.index(url)}/{len(urls)} \t-\t{round(urls.index(url)/len(urls)*100, 2)}%")
        download_file(url=url, save_in=save_in)
    
    # coordinates = read_grid_csv("data\europe_country_grids.csv")

    # f = merra2_file(date=date(2020, 12, 31))
    # n = average_weather(f, 'SPEED', coordinates=coordinates.get('ESP'))[timedelta(seconds=64800)]

    # (""" ? """, (1, ))
