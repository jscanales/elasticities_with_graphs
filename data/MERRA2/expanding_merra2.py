import datetime

# Dictionary containing the data on coordinates for different countries
COUNTRY_COORDINATES: dict[str, str] = {
    "Denmark": "52.753,5.01,59.96,15.029",
    "Europe": "34.0,-26.0,72.0,43.0"
    }

def create_merra2_subset(start_date, end_date, country):
    # Retrieve the coordinates for the specified country
    country_coordinates = COUNTRY_COORDINATES.get(country)
    if country_coordinates is None:
        raise ValueError(f"Coordinates for {country} are not defined.")

    # Make a list with all the links
    date_format = "%Y.%m.%d"
    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date-start_date).days + 1)]
    all_links = []
    for date in date_range:
        link = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F{date.strftime('%Y')}/{date.strftime('%m')}/MERRA2_400.tavg1_2d_flx_Nx.{date.strftime('%Y%m%d')}.nc4&FORMAT=bmM0Lw&BBOX={country_coordinates}&LABEL=MERRA2_400.tavg1_2d_flx_Nx.{date.strftime('%Y%m%d')}.SUB.nc&SHORTNAME=M2T1NXFLX&SERVICE=L34RS_MERRA2&VERSION=1.02&DATASET_VERSION=5.12.4&VARIABLES=SPEED%2CTLML"
        all_links.append(link)

    # Add the new links
    file_name = f"data/MERRA2/subset_M2T1NXFLX_5.12.4_{country}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.txt"
    with open(file_name, "w") as f:
        for link in all_links:
            f.write(link + "\n")

if __name__ == "__main__":
    create_merra2_subset("2016.01.01", "2023.12.31", "Europe")
