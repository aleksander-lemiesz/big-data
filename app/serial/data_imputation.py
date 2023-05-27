import random
from random import uniform

import pandas as pd
from pandas import isnull


def data_imputation(data_frame):
    assert isinstance(data_frame, pd.DataFrame)

    imputation_month_and_hour(data_frame)
    imputation_latitude_longitude(data_frame, 'Latitude')
    imputation_latitude_longitude(data_frame, 'Longitude')
    imputation_race(data_frame, 'VIC_RACE')
    imputation_race(data_frame, 'SUSP_RACE')
    imputation_age(data_frame, 'VIC_AGE_GROUP')
    imputation_age(data_frame, 'SUSP_AGE_GROUP')
    imputation_susp_sex(data_frame)
    imputation_vic_sex(data_frame)
    imputation_boro(data_frame)
    imputation_prem(data_frame)


def imputation_latitude_longitude(data_frame, table_name):
    if table_name == 'Latitude' or table_name == 'Longitude':
        assert isinstance(data_frame, pd.DataFrame)
        len_na = len(data_frame.loc[data_frame[table_name].isna(), table_name])
        data_frame.loc[data_frame[table_name].isna(), table_name] = 0
        x = data_frame[table_name]
        print(sum(x) / (len(x) - len_na), len_na)
        avg = sum(x) / (len(x) - len_na)
        data_frame.loc[data_frame[table_name] == 0, table_name] = avg


def imputation_race(data_frame, race_table_name):
    unknown_record = 'UNKNOWN'

    assert isinstance(data_frame, pd.DataFrame)
    len_na = 0
    if race_table_name == 'VIC_RACE':
        len_na = len(data_frame.loc[data_frame[race_table_name] == unknown_record, race_table_name])
    elif race_table_name == 'SUSP_RACE':
        len_na = len(data_frame.loc[data_frame[race_table_name].isna(), race_table_name])

    known_len = len(data_frame[race_table_name]) - len_na
    white_num = len(data_frame.loc[data_frame[race_table_name] == 'WHITE', race_table_name])
    black_num = len(data_frame.loc[data_frame[race_table_name] == 'BLACK', race_table_name])
    alaska_num = len(data_frame.loc[data_frame[race_table_name] == 'AMERICAN INDIAN/ALASKAN NATIVE', race_table_name])
    asian_num = len(data_frame.loc[data_frame[race_table_name] == 'ASIAN / PACIFIC ISLANDER', race_table_name])

    white_prob = white_num / known_len
    black_prob = black_num / known_len
    alaska_prob = alaska_num / known_len
    asian_prob = asian_num / known_len

    for race in data_frame.index:
        if data_frame[race_table_name][race] == unknown_record or isnull(data_frame[race_table_name][race]):
            prob = uniform(0, 1)
            if prob <= white_prob:
                data_frame[race_table_name][race] = 'WHITE'
            elif white_prob < prob <= (white_prob + black_prob):
                data_frame[race_table_name][race] = 'BLACK'
            elif (white_prob + black_prob) < prob <= (white_prob + black_prob + alaska_prob):
                data_frame[race_table_name][race] = 'AMERICAN INDIAN/ALASKAN NATIVE'
            elif (white_prob + black_prob + alaska_prob) < prob <= 1:
                data_frame[race_table_name][race] = 'ASIAN / PACIFIC ISLANDER'


def imputation_age(data_frame, race_table_name):
    assert isinstance(data_frame, pd.DataFrame)
    len_na = 0
    if race_table_name == 'VIC_AGE_GROUP':
        len_na = len(data_frame.loc[data_frame[race_table_name] == 'UNKNOWN', race_table_name])
    elif race_table_name == 'SUSP_AGE_GROUP':
        len_na = len(data_frame.loc[data_frame[race_table_name].isna(), race_table_name])

    known_len = len(data_frame[race_table_name]) - len_na
    below_18 = len(data_frame.loc[data_frame[race_table_name] == '<18', race_table_name])
    below_24 = len(data_frame.loc[data_frame[race_table_name] == '18-24', race_table_name])
    below_44 = len(data_frame.loc[data_frame[race_table_name] == '25-44', race_table_name])
    below_64 = len(data_frame.loc[data_frame[race_table_name] == '45-65', race_table_name])
    over_64 = len(data_frame.loc[data_frame[race_table_name] == '65+', race_table_name])

    below_18_prob = below_18 / known_len
    below_24_prob = below_24 / known_len
    below_44_prob = below_44 / known_len
    below_64_prob = below_64 / known_len
    over_64_prob = over_64 / known_len

    for age in data_frame.index:
        if data_frame[race_table_name][age] == 'UNKNOWN' or isnull(data_frame[race_table_name][age]):
            prob = uniform(0, 1)
            if prob <= below_18_prob:
                data_frame[race_table_name][age] = '<18'
            elif below_18_prob < prob <= (below_18_prob + below_24_prob):
                data_frame[race_table_name][age] = '18-24'
            elif (below_18_prob + below_24_prob) < prob <= (below_18_prob + below_24_prob + below_44_prob):
                data_frame[race_table_name][age] = '25-44'
            elif (below_18_prob + below_24_prob + below_44_prob) < prob <= (below_18_prob + below_24_prob + below_44_prob + below_64_prob):
                data_frame[race_table_name][age] = '45-65'
            elif (below_18_prob + below_24_prob + below_44_prob + below_64_prob) < prob <= 1:
                data_frame[race_table_name][age] = '65+'


def imputation_susp_sex(data_frame):
    assert isinstance(data_frame, pd.DataFrame)
    sex_table_name = 'SUSP_SEX'
    len_na = len(data_frame.loc[data_frame[sex_table_name].isna(), sex_table_name])

    known_len = len(data_frame[sex_table_name]) - len_na
    m_num = len(data_frame.loc[data_frame[sex_table_name] == 'M', sex_table_name])
    f_num = len(data_frame.loc[data_frame[sex_table_name] == 'F', sex_table_name])
    u_num = len(data_frame.loc[data_frame[sex_table_name] == 'U', sex_table_name])

    m_prob = m_num / known_len
    f_num_prob = f_num / known_len
    u_num_prob = u_num / known_len

    for age in data_frame.index:
        if data_frame[sex_table_name][age] == 'UNKNOWN' or isnull(data_frame[sex_table_name][age]):
            prob = uniform(0, 1)
            if prob <= m_prob:
                data_frame[sex_table_name][age] = 'M'
            elif m_prob < prob <= (m_prob + f_num_prob):
                data_frame[sex_table_name][age] = 'F'
            elif (m_prob + f_num_prob) < prob <= 1:
                data_frame[sex_table_name][age] = 'U'


def imputation_vic_sex(data_frame):
    assert isinstance(data_frame, pd.DataFrame)
    sex_table_name = 'VIC_SEX'
    len_na = len(data_frame.loc[data_frame[sex_table_name].isna(), sex_table_name])

    known_len = len(data_frame[sex_table_name]) - len_na
    m_num = len(data_frame.loc[data_frame[sex_table_name] == 'M', sex_table_name])
    f_num = len(data_frame.loc[data_frame[sex_table_name] == 'F', sex_table_name])
    u_num = len(data_frame.loc[data_frame[sex_table_name] == 'U', sex_table_name])
    e_num = len(data_frame.loc[data_frame[sex_table_name] == 'E', sex_table_name])
    d_num = len(data_frame.loc[data_frame[sex_table_name] == 'D', sex_table_name])

    m_prob = m_num / known_len
    f_num_prob = f_num / known_len
    u_num_prob = u_num / known_len
    e_num_prob = e_num / known_len
    d_num_prob = d_num / known_len

    for age in data_frame.index:
        if data_frame[sex_table_name][age] == 'UNKNOWN' or isnull(data_frame[sex_table_name][age]):
            prob = uniform(0, 1)
            if prob <= m_prob:
                data_frame[sex_table_name][age] = 'M'
            elif m_prob < prob <= (m_prob + f_num_prob):
                data_frame[sex_table_name][age] = 'F'
            elif (m_prob + f_num_prob) < prob <= (m_prob + f_num_prob + u_num_prob):
                data_frame[sex_table_name][age] = 'U'
            elif (m_prob + f_num_prob + u_num_prob) < prob <= (m_prob + f_num_prob + u_num_prob + e_num_prob):
                data_frame[sex_table_name][age] = 'E'
            elif (m_prob + f_num_prob + u_num_prob + e_num_prob) < prob <= 1:
                data_frame[sex_table_name][age] = 'D'


def imputation_boro(data_frame):
    assert isinstance(data_frame, pd.DataFrame)
    brono_table_name = 'BORO_NM'
    len_na = len(data_frame.loc[data_frame[brono_table_name].isna(), brono_table_name])

    known_len = len(data_frame[brono_table_name]) - len_na
    brook_num = len(data_frame.loc[data_frame[brono_table_name] == 'BROOKLYN', brono_table_name])
    man_num = len(data_frame.loc[data_frame[brono_table_name] == 'MANHATTAN', brono_table_name])
    bronx_num = len(data_frame.loc[data_frame[brono_table_name] == 'BRONX', brono_table_name])
    queens_num = len(data_frame.loc[data_frame[brono_table_name] == 'QUEENS', brono_table_name])
    staten_num = len(data_frame.loc[data_frame[brono_table_name] == 'STATEN ISLAND', brono_table_name])

    brook_num_prob = brook_num / known_len
    man_num_prob = man_num / known_len
    bronx_num_prob = bronx_num / known_len
    queens_num_prob = queens_num / known_len
    staten_num_prob = staten_num / known_len

    for age in data_frame.index:
        if data_frame[brono_table_name][age] == 'UNKNOWN' or isnull(data_frame[brono_table_name][age]):
            prob = uniform(0, 1)
            if prob <= brook_num_prob:
                data_frame[brono_table_name][age] = 'BROOKLYN'
            elif brook_num_prob < prob <= (brook_num_prob + man_num_prob):
                data_frame[brono_table_name][age] = 'MANHATTAN'
            elif (brook_num_prob + man_num_prob) < prob <= (brook_num_prob + man_num_prob + bronx_num_prob):
                data_frame[brono_table_name][age] = 'BRONX'
            elif (brook_num_prob + man_num_prob + bronx_num_prob) < prob <= (brook_num_prob + man_num_prob + bronx_num_prob + queens_num_prob):
                data_frame[brono_table_name][age] = 'QUEENS'
            elif (brook_num_prob + man_num_prob + bronx_num_prob + queens_num_prob) < prob <= 1:
                data_frame[brono_table_name][age] = 'STATEN ISLAND'


def imputation_prem(data_frame):
    assert isinstance(data_frame, pd.DataFrame)
    table_name = 'PREM_TYP_DESC'

    len_na = len(data_frame.loc[data_frame[table_name].isna(), table_name])
    known_len = len(data_frame[table_name]) - len_na

    street_num = len(data_frame.loc[data_frame[table_name] == 'STREET', table_name])
    apt_num = len(data_frame.loc[data_frame[table_name] == 'RESIDENCE - APT. HOUSE', table_name])
    house_num = len(data_frame.loc[data_frame[table_name] == 'RESIDENCE-HOUSE', table_name])
    publ_house_num = len(data_frame.loc[data_frame[table_name] == 'RESIDENCE - PUBLIC HOUSING', table_name])
    commercial_num = len(data_frame.loc[data_frame[table_name] == 'COMMERCIAL BUILDING', table_name])
    chain_store_num = len(data_frame.loc[data_frame[table_name] == 'CHAIN STORE', table_name])
    subway_num = len(data_frame.loc[data_frame[table_name] == 'TRANSIT - NYC SUBWAY', table_name])
    department_store_num = len(data_frame.loc[data_frame[table_name] == 'DEPARTMENT STORE', table_name])
    grocery_num = len(data_frame.loc[data_frame[table_name] == 'GROCERY/BODEGA', table_name])
    restaurant_num = len(data_frame.loc[data_frame[table_name] == 'RESTAURANT/DINER', table_name])

    street_num_prob = street_num / known_len
    apt_num_prob = apt_num / known_len
    house_num_prob = house_num / known_len
    publ_house_num_prob = publ_house_num / known_len
    commercial_num_prob = commercial_num / known_len
    chain_store_num_prob = chain_store_num / known_len
    subway_num_prob = subway_num / known_len
    department_store_num_prob = department_store_num / known_len
    grocery_num_prob = grocery_num / known_len
    restaurant_num_prob = restaurant_num / known_len

    for prem in data_frame.index:
        if data_frame[table_name][prem] == 'UNKNOWN' or isnull(data_frame[table_name][prem]):
            prob = uniform(0, 1)
            if prob <= street_num_prob:
                data_frame[table_name][prem] = 'STREET'
            elif street_num_prob < prob <= street_num_prob + apt_num_prob:
                data_frame[table_name][prem] = 'RESIDENCE - APT. HOUSE'
            elif street_num_prob + apt_num_prob < prob <= street_num_prob + apt_num_prob + house_num_prob:
                data_frame[table_name][prem] = 'RESIDENCE-HOUSE'
            elif street_num_prob + apt_num_prob + house_num_prob < prob <= street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob:
                data_frame[table_name][prem] = 'RESIDENCE - PUBLIC HOUSING'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob \
                    < prob <= \
                    street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob:
                data_frame[table_name][prem] = 'COMMERCIAL BUILDING'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob \
                    < prob <= \
                    street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob:
                data_frame[table_name][prem] = 'CHAIN STORE'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob \
                    < prob <= \
                    street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob:
                data_frame[table_name][prem] = 'TRANSIT - NYC SUBWAY'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob \
                    < prob <= \
                    street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob + department_store_num_prob:
                data_frame[table_name][prem] = 'DEPARTMENT STORE'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob + department_store_num_prob \
                    < prob <= \
                    street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob + department_store_num_prob + grocery_num_prob:
                data_frame[table_name][prem] = 'GROCERY/BODEGA'
            elif street_num_prob + apt_num_prob + house_num_prob + publ_house_num_prob + commercial_num_prob + chain_store_num_prob + subway_num_prob + department_store_num_prob \
                    < prob <= 1:
                data_frame[table_name][prem] = 'RESTAURANT/DINER'


def imputation_month_and_hour(df: pd.DataFrame):
    df = df.dropna()
    df['HOUR'] = df['CMPLNT_FR'].dt.hour
    df['MONTH'] = df['CMPLNT_FR'].dt.month
    df = df.drop(columns=['CMPLNT_FR'])
