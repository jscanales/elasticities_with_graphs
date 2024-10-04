import csv
import sys
import numpy as np
import re
import netCDF4
from typing import List, Sequence
from pathlib import Path
import json


# Set a larger field size limit if necessary
csv.field_size_limit(100000000)

def round_coordinates_to_decimals(coordinates, round_int:int|None):
    if isinstance(coordinates, list):
        return [round_coordinates_to_decimals(coord, round_int=round_int) for coord in coordinates]
    elif isinstance(coordinates, dict):
        return {key: round_coordinates_to_decimals(value, round_int=round_int) for key, value in coordinates.items()}
    elif isinstance(coordinates, (float, int)):
        return round(coordinates, round_int)
    else:
        return coordinates


def extract_coordinates_from_csv(csv_file, round_int:int|None=None, filter:str|None=None):
    coordinates_dict = {}

    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            coordinates_str = row[1].strip('"')
            coordinates = eval(coordinates_str)
            rounded_coordinates = round_coordinates_to_decimals(coordinates, round_int=round_int)
            if filter is None or row[6] == filter:
                coordinates_dict[row[2]] = rounded_coordinates

    return coordinates_dict


def is_inside(polygon:list[list[float]], xp:float, yp:float) -> bool:
    count = 0
    for i, coord in enumerate(polygon):
        if coord == None:
            return False
        x1, y1 = coord
        x2, y2 = polygon[i-1]
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-71)*(x2-x1)):
            count += 1
    return count%2 == 1


def fill_polygon(merra2_grid_array, country_border:list):
    if isinstance(country_border[0][0], list):
        return [fill_polygon(merra2_grid_array, polygon) for polygon in country_border]
    elif isinstance(country_border[0][0], float):
        points_to_add = []
        for (x, y) in merra2_grid_array:
            if is_inside(country_border, x, y):
                points_to_add.append([x, y])
        country_border.extend(points_to_add)
        return country_border
    else:
        raise Exception("The function fill_polygon is not working as expected, review the input data")

def unique_coordinates(flattened_coordinates):
    if len(flattened_coordinates) > 0:
        unique_coordinates_set = set(tuple(coord) for coord in flattened_coordinates)
        unique_coordinates_set.discard((np.nan, np.nan))
        return list(unique_coordinates_set)
    else:
        return flattened_coordinates

def flatten_coordinates(nested_list):
    flattened_list = []
    nones = 0
    for item in nested_list:
        try:
            if isinstance(item[0], list):
                flattened_list.extend(flatten_coordinates(item))
            else:
                flattened_list.append(item)
        except TypeError:
            if item == None:
                nones += 1
            else:
                raise Exception(f"TypeError not due to item == None")
    if nones > 0:
        print(f"{nones} items were nones. Maybe this should be investigated!!!")

    return flattened_list

def round_coordinates_to_merra2_grid(coordinates, array) -> list[float]|list[list[float]]:
    if isinstance(coordinates[0], float):
        lon, lat = coordinates

        # MERRA-2 grid resolution
        lat_res = 0.5
        lon_res = 0.625

        # Calculate the nearest rounded value by finding the element in the array that is closest to the provided lat/lon 
        rounded_lat = array[:, 0][np.abs(array[:, 0] - lat).argmin()]
        rounded_lon = array[:, 1][np.abs(array[:, 1] - lon).argmin()]

        # Calculate the distance between the original and rounded coordinates
        lat_distance = abs(rounded_lat - lat)
        lon_distance = abs(rounded_lon - lon)

        # Check if the distance exceeds the MERRA-2 grid unit resolution
        if lat_distance > lat_res or lon_distance > lon_res:
            return [np.nan, np.nan]

        return [rounded_lat, rounded_lon]
    elif isinstance(coordinates[0], list):
        return [round_coordinates_to_merra2_grid(coord, array) for coord in coordinates]    # type: ignore
    else:
        print(type(coordinates[0]))
        raise TypeError("Wrong type")

def borders_to_merra2_grid(coordinates, grid_array) -> dict[str, list]:
    merra2_grids_dict = {}
    n = 1
    for country_code, item in coordinates.items():
        print(f"working on territory {n} ({country_code})")

        coordinate_set = item['coordinates']
        
        rounded_coordinates = round_coordinates_to_merra2_grid(coordinate_set, grid_array)

        full_grid_polygons = fill_polygon(merra2_grid_array=grid_array, country_border=rounded_coordinates)

        unique_grid_points_list = unique_coordinates(flatten_coordinates(full_grid_polygons))

        merra2_grids_dict[country_code] = unique_grid_points_list

        n+=1
    return merra2_grids_dict

def get_grid_array(file) -> np.ndarray[float, float]:
    with netCDF4.Dataset(file, "r") as f:   # type: ignore
        # Get the latitude and longitude variables from the dataset
        latitudes = f.variables['lat'][:]
        longitudes = f.variables['lon'][:]

        # Create a meshgrid of latitudes and longitudes
        lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

        # Flatten the meshgrid arrays and combine them into a single array of unique combinations
        unique_combinations = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        return unique_combinations

def write_coordinates_to_csv(coordinates, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Country Code', 'Rounded Coordinates'])
        for country_code, coords in coordinates.items():
            writer.writerow([country_code, coords])


def generate_country_grid_csv(input_boundaries_file:str|Path, input_merra2_file:str|Path, output_file:str|Path=Path(".", "data", "rounded_coordinates.csv"), filter:str|None=None):
    coordinates = extract_coordinates_from_csv(input_boundaries_file, round_int=3, filter=filter)

    grid_array = get_grid_array(input_merra2_file)

    rounded_country_coordinates = borders_to_merra2_grid(coordinates, grid_array)

    write_coordinates_to_csv(rounded_country_coordinates, output_file)


def create_geojson_feature(coord) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "coordinates": coord,
            "type": "Point"
        }
    }


def country_coord_geojson(country_coordinates) -> str:
    # Create a list of GeoJSON Features from the coordinates
    features_list = [create_geojson_feature(coord) for coord in country_coordinates]

    # Create the GeoJSON FeatureCollection
    geojson_data = {
        "type": "FeatureCollection",
        "features": features_list
    }

    # Output the GeoJSON data

    geojson_string = json.dumps(geojson_data)
    return geojson_string


def string_to_list_of_tuples(coord_string) -> list[tuple[float]]:
    current_e = None
    error_count:int = 0

    coordinates = coord_string.split('), (')

    clean_coordinates = []
    for coord in coordinates: 
        clean_coordinates.append(coord.replace('(', '').replace(')', '').replace(' ', ''))
    
    tuple_list = []
    
    for coordinate in clean_coordinates:
        try:
            lat, lon = map(float, coordinate.split(','))
            tuple_list.append((lat, lon))
        except ValueError as e:
            error_count += 1
            if not current_e:
                print(e)
                print(f"Skipping point ({error_count})")                
            current_e = e

    if error_count:
        print(f"{error_count} points were skipped")    
    return tuple_list

def read_grid_csv(csv_file) -> dict[str, list[tuple[float]]]:
    coordinates_dict: dict[str, list[tuple[float]]] = {}

    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            country_code = row[0].strip()
            coordinates_str = row[1].strip().strip('[]')
            coordinates_dict[country_code] = string_to_list_of_tuples(coordinates_str)

    return coordinates_dict

def add_unique_elements(original_list:list, new_elements:list) -> list:
        unique_elements_set = set(original_list)
        unique_new_elements = [element for element in new_elements if element not in unique_elements_set]
        return original_list + unique_new_elements

def grid_from_codes(grid_file:str|Path, code:str|Sequence[str]) -> list[tuple[float]]:
    if isinstance(code, str):
        return read_grid_csv(grid_file).get(code, [])
    elif isinstance(code, list) or isinstance(code, Sequence):
        grid_dict = read_grid_csv(grid_file)
        combined_grid = []
        for c in code:
            grid = grid_dict.get(c)
            if grid:
                combined_grid: list[tuple[float]] = add_unique_elements(combined_grid, grid)
        return combined_grid
    else:
        raise TypeError
    
def create_masks_from_grid_dict(grid_dict:dict[str, list], merra2_file:str|Path) -> dict[str, np.ndarray]:
    mask_dict:dict[str, np.ndarray] = {}

    nc_file = netCDF4.Dataset(merra2_file, 'r')     # type: ignore
    latitudes = nc_file.variables['lat'][:]
    longitudes = nc_file.variables['lon'][:]

    for code, surface in grid_dict.items():
        # Get a matrix
        matrix = nc_file.variables['SPEED'][:][0, :, :]
        in_country_mask = np.zeros_like(matrix, dtype=bool)
        
        # Iterate through each point on the grid
        for lat_index in range(latitudes.shape[0]):
            for lon_index in range(longitudes.shape[0]):
                # Create a Point object for the current latitude and longitude
                point = tuple([latitudes[lat_index], longitudes[lon_index]])
                
                # Check if the point is within the country's shape
                if point in surface:
                    in_country_mask[lat_index, lon_index] = True
        mask_dict[code] = in_country_mask
    return mask_dict


def create_masks_from_grid_list(grid_list:list[tuple[float]], merra2_file:str|Path) -> np.ndarray:
    nc_file = netCDF4.Dataset(merra2_file, 'r')     # type: ignore
    latitudes = nc_file.variables['lat'][:]
    longitudes = nc_file.variables['lon'][:]

    # Get a matrix
    matrix = nc_file.variables['SPEED'][:][0, :, :]
    in_country_mask = np.zeros_like(matrix, dtype=bool)
    
    # Iterate through each point on the grid
    for lat_index in range(latitudes.shape[0]):
        for lon_index in range(longitudes.shape[0]):
            # Create a Point object for the current latitude and longitude
            point = tuple([latitudes[lat_index], longitudes[lon_index]])
            
            # Check if the point is within the country's shape
            if point in grid_list:
                in_country_mask[lat_index, lon_index] = True

    return in_country_mask

def mask_from_codes(grid_file:str|Path, code:str|Sequence[str], merra2_file:str|Path) -> np.ndarray:
    return create_masks_from_grid_list(grid_list=grid_from_codes(grid_file=grid_file, code=code), merra2_file=merra2_file)


if __name__ == "__main__":
    csv_file = Path(".", "data", "world-administrative-boundaries.csv")
    grid_file = Path(".", "data", "europe_country_grids.csv")
    merra2_file = Path(".", "data", "MERRA2", "MERRA2_400.tavg1_2d_flx_Nx.20150101.SUB.nc")
    mask_file = Path(".", "data", "europe_country_masks.csv")

    generate_country_grid_csv(input_boundaries_file=csv_file, input_merra2_file=merra2_file, output_file=grid_file, filter="Europe")

    grid_dict = read_grid_csv(grid_file)
    mask_dict = create_masks_from_grid_dict(grid_dict, merra2_file)
    write_coordinates_to_csv(mask_dict, mask_file)

    print(country_coord_geojson(grid_dict.get('ESP')))

    
    print(True in mask_from_codes(grid_file=grid_file, code='ESP', merra2_file=merra2_file))

    for point in grid_from_codes(grid_file=grid_file, code='ESP'):
        print(point in get_grid_array(merra2_file))
        

    