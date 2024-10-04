import netCDF4
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from data.country_coordinates import mask_from_codes

def average_weather(file: str, attribute: str, country_mask) -> dict[timedelta, float]:
    with netCDF4.Dataset(file, "r") as f:
        f.variables["time"]
        tt = [timedelta(minutes=float(i)) for i in f.variables["time"][:]]
        xx = f.variables[attribute]  # 3D matrix, (time, lat, lon)

        latitudes = f.variables['lat'][:]
        longitudes = f.variables['lon'][:]
        variable_data = f.variables[attribute][:]

        hourly_averages = {}

        for i, t in enumerate(tt):
            data_hour = variable_data[i, :, :] 
            print(len(data_hour))  
            data_hour_country = data_hour[country_mask]
            print(len(data_hour_country))  
            average_data_hour_country = np.mean(data_hour_country)

            # Store the average wind speed in the dictionary with the hour as the key
            hourly_averages[t] = float(average_data_hour_country)


    # Create an empty list to store hourly average data
    hourly_averages = []

    for i, t in enumerate(tt):
        data_hour = variable_data[i, :, :]
        data_hour_country = data_hour[country_mask]
        average_data_hour_country = np.mean(data_hour_country)

        # Append the average temperature to the list
        hourly_averages.append(average_data_hour_country)

    return longitudes, latitudes, hourly_averages



# # Call the function to get the data
# longitudes, latitudes, hourly_averages = average_weather('data/MERRA2/MERRA2_400.tavg1_2d_flx_Nx.20230522.SUB.nc', 'TLML', mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='ESP'))

# # Create a geospatial plot
# plt.figure(figsize=(12, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()

# # Create a scatter plot of the average temperatures on the map
# sc = ax.scatter(longitudes, latitudes, c=hourly_averages, cmap='coolwarm', transform=ccrs.PlateCarree())

# # Add colorbar
# plt.colorbar(sc, label='Average Temperature')

# plt.title('Average Temperature Map')
# plt.show()


# spain = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='ESP')
# germany = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='DEU')
# france = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='FRA')

# merged_mask = np.logical_or(spain, germany)
# merged_mask = np.logical_or(merged_mask, france)

# rotated_mask = np.rot90(merged_mask, k=1)


# plt.figure(figsize=(8, 10))  # Adjust the figure size as needed
# plt.imshow(rotated_mask, cmap='viridis', origin='upper', extent=[0, rotated_mask.shape[1], 0, 100])
# plt.title('Country Mask')
# plt.colorbar(label='Mask Value')
# plt.show()

with netCDF4.Dataset('data/MERRA2/MERRA2_400.tavg1_2d_flx_Nx.20150110.SUB.nc', "r") as f:
    f.variables["time"]
    tt = [timedelta(minutes=float(i)) for i in f.variables["time"][:]]
    xx = f.variables["TLML"]  # 3D matrix, (time, lat, lon)

    latitudes = f.variables['lat'][:]
    longitudes = f.variables['lon'][:]
    variable_data = f.variables["TLML"][:]

    hourly_averages = {}

    for i, t in enumerate(tt):
        data_hour = variable_data[i, :, :] 
    
    data_hour = variable_data[15, :, :] 

    data_hour = data_hour -273.15

    spain = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='ESP')
    germany = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='DEU')
    france = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='FRA')
    italy = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='ITA')
    poland = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='POL')
    greatbritain = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code='GBR')

    eu = mask_from_codes(grid_file=r"data\europe_country_grids.csv", code=["ESP", "ITA", "DEU", "LUX", "AUT", "FRA", "POL", "DNK", "GBR", "IRL", "PRT", "GRC", "CZE"])

    merged_mask = np.logical_or(spain, germany)
    merged_mask = np.logical_or(merged_mask, france)
    merged_mask = np.logical_or(merged_mask, italy)
    merged_mask = np.logical_or(merged_mask, poland)
    merged_mask = np.logical_or(merged_mask, greatbritain)

    print(spain.shape)
    print(data_hour.shape)

    data_hour = data_hour * eu

    rotated_mask = np.rot90(data_hour, k=1)



    plt.figure(figsize=(8, 10))  # Adjust the figure size as needed
    plt.imshow(data_hour, cmap='viridis', origin='lower', extent=[0, data_hour.shape[1], 0, data_hour.shape[0]])
    plt.title('Country Mask')
    plt.colorbar(label='Mask Value')
    plt.show()