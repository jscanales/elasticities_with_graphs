import holidays
import pandas as pd
from data.entosoe import PriceArea
from datetime import date, datetime
import os

_school_holidays = {
    2019: [
        (pd.Timestamp("20190101"), pd.Timestamp("20190102")),
        (pd.Timestamp("20190211"), pd.Timestamp("20190215")),
        (pd.Timestamp("20190414"), pd.Timestamp("20190422")),
        (pd.Timestamp("20190501"), pd.Timestamp("20190501")),
        (pd.Timestamp("20190517"), pd.Timestamp("20190517")),
        (pd.Timestamp("20190530"), pd.Timestamp("20190531")),
        (pd.Timestamp("20190605"), pd.Timestamp("20190605")),
        (pd.Timestamp("20190609"), pd.Timestamp("20190610")),
        (pd.Timestamp("20190701"), pd.Timestamp("20190809")),
        (pd.Timestamp("20191014"), pd.Timestamp("20191018")),
        (pd.Timestamp("20191223"), pd.Timestamp("20191231")),
    ],
    2020: [
        (pd.Timestamp("20200101"), pd.Timestamp("20200101")),
        (pd.Timestamp("20200210"), pd.Timestamp("20200214")),
        (pd.Timestamp("20200405"), pd.Timestamp("20200413")),
        (pd.Timestamp("20200501"), pd.Timestamp("20200501")),
        (pd.Timestamp("20200508"), pd.Timestamp("20200508")),
        (pd.Timestamp("20200521"), pd.Timestamp("20200522")),
        (pd.Timestamp("20200601"), pd.Timestamp("20200601")),
        (pd.Timestamp("20200605"), pd.Timestamp("20200605")),
        (pd.Timestamp("20200629"), pd.Timestamp("20200630")),
        (pd.Timestamp("20200701"), pd.Timestamp("20200807")),
        (pd.Timestamp("20201012"), pd.Timestamp("20201016")),
        (pd.Timestamp("20201221"), pd.Timestamp("20201231")),
    ],
    2021: [
        (pd.Timestamp("20210101"), pd.Timestamp("20210101")),
        (pd.Timestamp("20210215"), pd.Timestamp("20210219")),
        (pd.Timestamp("20210328"), pd.Timestamp("20210405")),
        (pd.Timestamp("20210430"), pd.Timestamp("20210430")),
        (pd.Timestamp("20210513"), pd.Timestamp("20210514")),
        (pd.Timestamp("20210523"), pd.Timestamp("20210524")),
        (pd.Timestamp("20210628"), pd.Timestamp("20210806")),
        (pd.Timestamp("20211018"), pd.Timestamp("20211022")),
        (pd.Timestamp("20211220"), pd.Timestamp("20211231")),
    ],
    2022: [
        (pd.Timestamp("20220101"), pd.Timestamp("20220101")),
        (pd.Timestamp("20220214"), pd.Timestamp("20220218")),
        (pd.Timestamp("20220410"), pd.Timestamp("20220418")),
        (pd.Timestamp("20220513"), pd.Timestamp("20220513")),
        (pd.Timestamp("20220526"), pd.Timestamp("20220527")),
        (pd.Timestamp("20220605"), pd.Timestamp("20220606")),
        (pd.Timestamp("20220627"), pd.Timestamp("20220805")),
        (pd.Timestamp("20221017"), pd.Timestamp("20221021")),
        (pd.Timestamp("20221222"), pd.Timestamp("20221231")),
    ],
    2023: [
        (pd.Timestamp("20230101"), pd.Timestamp("20230102")),
        (pd.Timestamp("20230213"), pd.Timestamp("20230217")),
        (pd.Timestamp("20230402"), pd.Timestamp("20230410")),
        (pd.Timestamp("20230501"), pd.Timestamp("20230501")),
        (pd.Timestamp("20230505"), pd.Timestamp("20230505")),
        (pd.Timestamp("20230518"), pd.Timestamp("20230519")),
        (pd.Timestamp("20230528"), pd.Timestamp("20230529")),
        (pd.Timestamp("20230605"), pd.Timestamp("20230605")),
        (pd.Timestamp("20230626"), pd.Timestamp("20230804")),
        (pd.Timestamp("20231016"), pd.Timestamp("20231020")),
        (pd.Timestamp("20231225"), pd.Timestamp("20231226")),
    ],
}
# _public_holidays = holidays.DK()

_public_holidays_DK = holidays.DK()
_public_holidays_DE =  pd.read_csv("data//germany_public_holiday_index.csv", index_col=0, parse_dates=True)
_public_holidays_ES = holidays.ES()
_public_holidays_FR = holidays.FR()
_public_holidays_PL = holidays.PL()


'''

Germany

Federal States (Bundesländer) population taken from  Statistisches Bundesamt Deutschland (web):
Population as of 31.12.2022 by nationaly and federal states
https://www.destatis.de/EN/Themes/Society-Environment/Population/Current-Population/Tables/population-by-laender.html

Land                            Total       %               Germans         Foreigners  %       from EU     %

Germany	                (DE)    84358845    100	            72,034,650	    12,324,195	14.6	4,598,602	37.3	
Baden-Württemberg       (BW)   	11280257	13,37175373     9,268,020	    2,012,237	17.8	885,873	    44.0	
Bayern	                (BY)    13369393	15,84824093     11,295,899	    2,073,494	15.5	928,664	    44.8	
Berlin	                (BE)    3755251	    4,45152017      2,920,902	    834,349	    22.2	243,458	    29.2	
Brandenburg	            (BB)    2573135	    3,050225498     2,393,263	    179,872	    7.0	    55,315	    30.8	
Bremen	                (HB)    684864	    0,811846108     540,716	        144,148	    21.0	38,678	    26.8	
Hamburg	                (HH)    1892122	    2,242944412     1,528,839	    363,283	    19.2	106,652	    29.4	
Hessen	                (HE)    6391360	    7,576395812     5,195,585	    1,195,775	18.7	449,617	    37.6	
Mecklenburg-Vorpommern	(MV)    1628378	    1,930299069     1,522,941	    105,437	    6.5	    33,194	    31.5	
Niedersachsen	        (NI)    8140242	    9,649541788     7,180,456	    959,786	    11.8	340,357	    35.5	
Nordrhein-Westfalen	    (NW)    18139116	21,50232853     15,310,749	    2,828,367	15.6	954,643	    33.8	
Rheinland-Pfalz	        (RP)    4159150	    4,930306952     3,593,124	    566,026	    13.6	231,827	    41.0	
Saarland	            (SL)    992666	    1,176718339     853,477	        139,189	    14.0	53,934	    38.7	
Sachsen	                (SN)    4086152	    4,843774236     3,785,988	    300,164	    7.3	    87,232	    29.1	
Sachsen-Anhalt	        (ST)    2186643	    2,592073185     2,025,653	    160,99	    7.4	    40,913	    25.4	
Schleswig-Holstein	    (SH)    2953270	    3,500842146     2,653,483	    299,787	    10.2	98,065	    32.7	
Thüringen	            (TH)    2126846	    2,521189094     1,965,555	    161,291	    7.6	    50,18	    31.1	

Public holiday dates come from the package holidays

School holidate dates come from the website https://www.feiertagskalender.ch/

'''

BUNDESLANDER_POP_DICT: dict[str, float] =   {'BW':13.37175373, "BY":15.84824093, "BE":4.45152017, "BB":3.050225498, "HB":0.811846108 , "HH": 2.242944412,
                                             "HE":7.576395812 , "MV":1.930299069, "NI":9.649541788, "NW":21.50232853, "RP":4.930306952, "SL":1.176718339, 
                                             "SN":4.843774236, "ST":2.592073185, "SH":3.500842146, "TH":2.521189094}


def is_public_holiday(day: pd.Timestamp, area:str) -> bool | float:
    if area == PriceArea.DK1 or area == PriceArea.DK2:
        return day.floor('D') in _public_holidays_DK
    
    elif area == PriceArea.DE1 or area == PriceArea.DE2:
        day = day.floor("D")
        if day in _german_school_holidays.index:
            holiday_index:float = _public_holidays_DE.loc[day]['Holiday_Index']
            return holiday_index
        else:
            print(f"Day {day} not included in holiday range (2015-01-01 - 2023-12-31), holiday index assigned to be 0 by default")
            return 0.0
    
    elif area == 'ES':
        return day.floor('D') in _public_holidays_ES
    elif area == 'FR':
        return day.floor('D') in _public_holidays_FR
    elif area == 'PL':
        return day.floor('D') in _public_holidays_PL
    else:
        raise Exception(f"area {area} not recognized")

_german_school_holidays = pd.read_csv("data//germany_school_holiday_index.csv", index_col=0, parse_dates=True)

def is_school_holiday(day: pd.Timestamp) -> float:
    day = day.floor("D")
    if day in _german_school_holidays.index:
        holiday_index:float = _german_school_holidays.loc[day]['Holiday_Index']
        return holiday_index
    else:
        print(f"Day {day} not included in holiday range (2015-01-01 - 2023-12-31), holiday index assigned to be 0 by default")
        return 0.0


if __name__ == '__main__':
    GERMAN_SCHOOL_HOLIDAYS:dict[int, dict] = {}
    PATH = 'data//holidays'


    def parse_date(date_string:str, year:str|int):
        
        date_list = date_string.replace(',', '').split()
        if len(date_list) == 2:
            month_str, day_str = date_list
            complete_date_str = f"{year}-{month_str}-{day_str}"

        elif len(date_list) == 3:
            month_str, day_str, year_str = date_list
            day_str = day_str
            complete_date_str = f"{year_str}-{month_str}-{day_str}"

        else:
            raise Exception(f"This date has different info than 'month + day' or 'month + day + year: {date_string} in {year}")


        timestamp = pd.Timestamp(complete_date_str)
        return timestamp

    def parse_range(date_range:str, year:str|int):
        start_str, end_str = date_range.split(' - ')
        return parse_date(start_str, year), parse_date(end_str, year)

    def parse_day(day:str, year:str|int):
        return parse_date(day, year), parse_date(day, year)


    for filename in os.listdir(PATH):
        if filename.endswith('.csv') and filename.startswith('public_school_holidays_'):
            year = int(filename.split('_')[3].split('.')[0])
            df = pd.read_csv(PATH+'//'+filename, sep=';')


            GERMAN_SCHOOL_HOLIDAYS[year] = {}

            year_school_holidays = {}

            for index, row in df.iterrows():
                state = row['Bundeslander']  # Extract the state name
                holidays_list = []  # Initialize a dictionary to store holidays for the state

                # Iterate over the holiday columns and add them to the holidays dictionary
                for holiday_type, date_range in row.items():
                    if holiday_type != 'Bundeslander':
                        holidays_list.append(date_range)

                # Add the holidays dictionary to the school_holidays_2015 dictionary
                year_school_holidays[state] = holidays_list


            for state, holidays_list in year_school_holidays.items():
                state_holidays = []
                for date_range in holidays_list:
                    if date_range == '-':       # No holidays in that period
                        continue

                    elif ' + ' in date_range:     # Indicates a discontinued set of dates of ranges 
                        date_parts = date_range.split(' + ')
                        for part in date_parts:
                            if ' - ' in part:   # If part is a range of dates
                                start, end = parse_range(date_range=part, year=year)
                            else:               # Else simply add it
                                start, end = parse_day(day=part, year=year)
                            state_holidays.append((start, end))
                            
                    elif ' - ' in date_range:   # General procedure
                        start, end = parse_range(date_range=date_range, year=year)
                        state_holidays.append((start, end))

                    else:                       # It is a single day (or should be!)
                        try:
                            start, end = parse_day(day=date_range, year=year)
                            state_holidays.append((start, end))
                        except ValueError as e:
                            print(f"Warning! Uncontrolled holiday as {date_range}")
            
                GERMAN_SCHOOL_HOLIDAYS[year][state] = state_holidays


    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2023-12-31")

    date_range = pd.date_range(start_date, end_date, freq='D')
    holiday_indices = []

    for day in date_range:
        holiday_index = 0
        for state, population_percentage in BUNDESLANDER_POP_DICT.items():
            for year, state_holidays in GERMAN_SCHOOL_HOLIDAYS.items():
                for holiday_ranges in state_holidays.get(state, []):
                    if holiday_ranges[0] <= day.floor('D') <= holiday_ranges[1]:
                        holiday_index += population_percentage / 100
        holiday_index = round(holiday_index, 3)
        holiday_indices.append(holiday_index)
        print(f"day {day} has a score of {holiday_index}")

    holiday_series = pd.Series(holiday_indices, index=date_range, name="Holiday_Index")

    holiday_series.to_csv("germany_school_holiday_index.csv", header=True)

    print("Holiday indices saved to germany_school_holiday_index.csv")
    
    for day in date_range:
        holiday_index = 0
        for key, value in BUNDESLANDER_POP_DICT.items():
            holiday_index += value / 100 if day in holidays.country_holidays('DE', subdiv=key) else 0
        holiday_index = round(holiday_index, 3)
        holiday_indices.append(holiday_index)
        print(f"day {day} has a score of {holiday_index}")

    holiday_series = pd.Series(holiday_indices, index=date_range, name="Holiday_Index")

    holiday_series.to_csv("germany_public_holiday_index.csv", header=True)

