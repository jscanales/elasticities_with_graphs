import pandas as pd
from datetime import timezone
from entsoe import EntsoePandasClient

from data.config import Entsoe


class PsrTypes:
    # https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    solar: str = "B16"
    wind_offshore: str = "B18"
    wind_onshore: str = "B19"


class PriceArea:
    # https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    DK1 = "DK_1"
    DK2 = "DK_2"
    ES = "ES"
    DE1 = "DE_AT_LU"
    DE2 = "DE_LU"       # Switch at 01.10.2018
    UK = "UK"
    PL = "PL"
    FR = "FR"


def get_generation(
    start: pd.Timestamp, end: pd.Timestamp, zone: str, psr_type: str
) -> dict[pd.Timestamp, float]:

    client = EntsoePandasClient(api_key=Entsoe.token)
    df = client.query_generation(
        country_code=zone, start=start, end=end, psr_type=psr_type
    )
    
    # Depending on the queried period, the 'Actual Aggregated' level is missing causing
    # This is why columns are summed rather than selecting the one containing the data
    # Although this is error prone, there appears to be no overlap
    return df.sum(axis=1).to_dict()

def get_load(start: pd.Timestamp, end: pd.Timestamp, zone: str
) -> dict[pd.Timestamp, float]:

    client = EntsoePandasClient(api_key=Entsoe.token)
    df = client.query_load(
        country_code=zone, start=start, end=end
    )
    return df.sum(axis=1).to_dict()

def get_prices(start: pd.Timestamp, end: pd.Timestamp, zone: str) -> dict[pd.Timestamp, float]:
    client = EntsoePandasClient(api_key=Entsoe.token)
    ser = client.query_day_ahead_prices(
        country_code=zone, start=start, end=end, resolution='60T' # type: ignore
    )
    return ser.to_dict()

if __name__ == "__main__":

    client = EntsoePandasClient(api_key=Entsoe.token)

    start = pd.Timestamp("20150101T00", tzinfo=timezone.utc) # TODO
    end =   pd.Timestamp("20171231T23", tzinfo=timezone.utc) # TODO

    # ds = client.query_generation(
    #     country_code="ES", start=start, end=end, psr_type="B19"
    # )
    df = get_prices(start=start, end=end, zone= 'DE_AT_LU')

