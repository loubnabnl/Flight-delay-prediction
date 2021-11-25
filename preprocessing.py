import pandas as pd
import numpy as np
from datetime import datetime 
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

""" In this challenge, the goal was to predict flight delay (in 7 bins
where 0 means no delay and 7 is a delay superior to 2 hours) given information
about flights. We had a training dataset and an another dataset por woth some additional
info, the use of external datasets was not allowed.""" 

"""in this file we apply some preprocessing the the data
the new preprocessed datasets are saved in processed_train.csv
and processed_test.csv
- removing and filling missing values in origin and destinaton terminals of airports
-preprocess date information
-create additional features such as:
    -concatenation of departure and arrival countries
    -indicator of domestic flights
    -holiday indicator in the US(US flighst represent most of the traffic)
    -number of flights arriving/departing at the same time in the arrival/departure airport
    -frequency and binary encoding of categorical variables
    -scaling of numerical variables
"""
def drop_na_origin(train_df):
    for o in tqdm(train_df.origin.unique()):
        if train_df[train_df.origin == o].departure_terminal_schd.isna().any():
            #calculate mode
            mode = train_df[train_df.origin == o].departure_terminal_schd.dropna().mode()
            if len(mode) == 0:
                train_df.loc[train_df.origin == o, 'departure_terminal_schd'] = train_df[train_df.origin == o].departure_terminal_schd.fillna('0')
            else:
                train_df.loc[train_df.origin == o, 'departure_terminal_schd'] = train_df[train_df.origin == o].departure_terminal_schd.fillna(mode.values[0])  

def drop_na_destination(train_df):
    for d in tqdm(train_df.destination.unique()):
        if train_df[train_df.destination == d].arrival_terminal_schd.isna().any():
            #calculate mode
            mode = train_df[train_df.destination == d].arrival_terminal_schd.dropna().mode()
            if len(mode) == 0:
                train_df.loc[train_df.destination == d, 'arrival_terminal_schd'] = train_df[train_df.destination == d].arrival_terminal_schd.fillna('0')
            else:
                train_df.loc[train_df.destination == d, 'arrival_terminal_schd'] = train_df[train_df.destination == d].arrival_terminal_schd.fillna(mode.values[0])


def time_to_integer(date_time):
    # convert datetime string to integer
    for i in range(len(date_time)):
        if i> 100 and i//100 == 0:
            print(i)
        dt_time = date_time[i]
        date_time[i] =  int(dt_time[:4] + dt_time[5:7] + dt_time[8:10] + dt_time[11:13] + dt_time[14:16] + dt_time[17:19])
    return date_time

def freq_encode(df, feature):
    # frequency encoding of categorical variables
    fq = df.groupby(feature).size()/len(df)
    df.loc[:, str(feature) + ' encod'] = df[feature].map(fq)
    df.drop(feature, axis = 1, inplace = True)
    
def extract_day(dates):
    days = pd.Series(np.zeros(len(dates)))
    for i in tqdm(range(len(dates))):
        days[i] = str(dates[i])[:10]
    return days

def date_to_str_day(dates):
    # from timestamp to string day
    days = pd.Series(np.zeros(len(dates)))
    for i in tqdm(range(len(dates))):
        date_time_obj = dates[i].to_pydatetime()
        date_time_obj = datetime.strftime(date_time_obj, '%Y-%m-%d %H:%M:%S')
        days[i] = date_time_obj[:10]
    return days

def preprocess_data(train_df, por_df, train = True):
    """for preprocessing the data in train_df
    por_df dataframe with additional information
    set train = False if the dataset doesn't have actual arrival date
    """
    
    # imputing missing values (in departure/ arrival terminals)
    
    print('imputing missing values')
    drop_na_origin(train_df)
    drop_na_destination(train_df)
    #train_df.isna().any()
    #the other features we add don't have nan values
    
    
    # 1: adding country and distance from por_df and drop id
    
    print('adding country and distance variables')
    df2 = por_df[['departure_airport', 'arrival_airport', 'departure_country', 'arrival_country', 'direct_distance']]
    train_df = pd.merge(train_df, df2,  how='left', left_on=['origin','destination'], right_on = ['departure_airport','arrival_airport'])
    train_df.drop(['departure_airport','arrival_airport'], axis = 1, inplace = True)
    train_df.drop(['id'], axis = 1, inplace = True)
    
    # add departure day
    print('adding departure day')
    train_df['departure_day'] = extract_day(train_df['departure_datetime_schd'])
    print('adding number of departing and arriving flights per terminal and day')
    #number of flights departing from terminal 'departure_terminal_schd' in airport 'origin' on day 'departure_day'
    train_df['nb_dep_flights'] = pd.Series(np.zeros(len(train_df)))
    data = train_df.groupby(['departure_day', 'origin', 'departure_terminal_schd'], as_index=False)['nb_dep_flights'].count()
    data.rename(columns={'nb_dep_flights': 'nb_departing_flights', 'origin': 'origin1'}, inplace=True)
    train_df = pd.merge(train_df, data,  how='left', left_on=['departure_day', 'origin', 'departure_terminal_schd'], right_on = ['departure_day', 'origin1', 'departure_terminal_schd'])
    train_df.drop(['origin1', 'nb_dep_flights'], axis=1, inplace=True)

    #number of flights arriving from terminal 'arriving_terminal_schd' in airport 'destination' on day 'departure_day'
    train_df['nb_arr_flights'] = pd.Series(np.zeros(len(train_df)))
    data2 = train_df.groupby(['departure_day', 'destination', 'arrival_terminal_schd'], as_index=False)['nb_arr_flights'].count()
    data2.rename(columns={'nb_arr_flights': 'nb_arriving_flights', 'destination': 'destination1'}, inplace=True)
    train_df = pd.merge(train_df, data2,  how='left', left_on=['departure_day', 'destination', 'arrival_terminal_schd'], right_on = ['departure_day', 'destination1', 'arrival_terminal_schd'])
    train_df.drop(['destination1', 'nb_arr_flights'], axis=1, inplace=True)

    
    # 2: adding interaction features
    
    print('add new variables')
    train_df["origin_destination"] = train_df["origin"] + train_df["destination"]
    train_df["countries_origin_destination"] = train_df["departure_country"] + train_df["arrival_country"]
    train_df['flight_intra'] = 0
    indexes = train_df.loc[train_df['departure_country'] == train_df['arrival_country']].index
    #flights that were delayed
    train_df['flight_intra'].iloc[indexes] = 1
    
    # new feature: is this date a holiday in the US ?
    
    print('adding holiday feature')
    #start and end dates on test set
    train_start = '2019-07-01' 
    train_end = '2019-12-31'
    if train:
        train_start = '2018-01-01'
        train_end = '2019-06-30'
        
    datar = pd.date_range(start=train_start, end=train_end)
    df = pd.DataFrame()
    df['Date'] = datar
    
    cal = calendar()
    holidays = cal.holidays(start=datar.min(), end=datar.max())
    holidays = date_to_str_day(holidays)
    
    train_df['US_holiday_dep']= train_df['departure_day'].isin(holidays)
    
    train_df.loc[train_df['departure_country'] != 'US', 'US_holiday_dep'] = 0
    
    train_df['US_holiday_dep'] = train_df['US_holiday_dep'].astype(int)
    train_df.drop(['departure_day'], axis = 1, inplace = True)
    
    # compute exact delay for training set
    # in this set we had the information about the expected and actual arrival dates

    if train:
        
        # 3: add exact delay column for the training set
        
        print('computing exact delay')
        #delay of flights arriving early is 0
        exact_delay = pd.Series(np.repeat(0, len(train_df['arrival_datetime_act'])))
        for i in tqdm(range(len(train_df['arrival_datetime_act']))):
            if train_df['arrival_datetime_act'][i] < train_df['arrival_datetime_schd'][i]:
                # no delay
                exact_delay[i] = 0
            else:
                exact_delay[i] = pd.Timedelta(pd.to_datetime(train_df['arrival_datetime_act'][i]) 
                                              - pd.to_datetime(train_df['arrival_datetime_schd'][i])).seconds/60
        train_df['exact_delay'] = exact_delay
    
    # 5: convert datetime to numerical values add day, month, dayofweek, hour..
        
    print('converting datetime to numeric values and adding day, month...')
    departure_date = pd.to_datetime(train_df['departure_datetime_schd'], format = '%Y-%m-%d %H:%M:%S')
    arrival_date = pd.to_datetime(train_df['arrival_datetime_schd'], format = '%Y-%m-%d %H:%M:%S')
    train_df['departure_month'] = departure_date.dt.month
    train_df['departure_day']= departure_date.dt.day
    train_df['departure_hour'] = departure_date.dt.hour
    train_df['departure_day_of_week'] = departure_date.dt.dayofweek
    train_df['arrival_hour'] = arrival_date.dt.hour
    # convert to numpy to speed up calculations
    departure_date = train_df['departure_datetime_schd'].to_numpy()
    arrival_date = train_df['arrival_datetime_schd'].to_numpy()
    train_df['departure_datetime_schd'] = time_to_integer(departure_date)
    train_df['arrival_datetime_schd'] = time_to_integer(arrival_date)
    if train:
        actual_arrival_date = train_df['arrival_datetime_act'].to_numpy()
        train_df['arrival_datetime_act'] = time_to_integer(actual_arrival_date)
    
    # 6: scaling numerical variables
        
    print('scaling numerical variables')
    # log normalize the distance
    train_df['distance_log'] = np.log(train_df['direct_distance'])
    scaler = StandardScaler()
    # some features won't be normalized like month dayofweek...
    train_df[['departure_datetime_schd', 'arrival_datetime_schd', 'nb_arriving_flights', 'nb_departing_flights']] = scaler.fit_transform(train_df[['departure_datetime_schd',
       'arrival_datetime_schd', 'nb_arriving_flights', 'nb_departing_flights']])
    if train:
        train_df[['arrival_datetime_act']] = scaler.fit_transform(train_df[['arrival_datetime_act']])

    # 7: Frequency encoding
        
    freq_encode(train_df, 'operating_carrier_encoded')
    freq_encode(train_df, 'operating_flight_no_encoded')
    freq_encode(train_df, 'marketing_carriers_encoded')
    
    # 8: binary encoding of categorical variables
    
    category_cols = ['origin', 'departure_country', 'destination', 'arrival_country', 'origin_destination', 
                     'countries_origin_destination','departure_terminal_schd', 'arrival_terminal_schd']
    binary_transform = BinaryEncoder(cols=category_cols).fit(train_df)
    transformed_train_df = binary_transform.transform(train_df)

    return transformed_train_df


if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    train_ids = train_df['id'].values
    test_df = pd.read_csv('data/test.csv')
    test_ids = test_df['id'].values
    por_df = pd.read_csv('data/por.csv')
    processed_test = preprocess_data(test_df, por_df, False)
    processed_test.index = test_ids
    processed_test.to_csv('data/processed_test.csv', index=True)
    processed_train = preprocess_data(train_df, por_df, True)
    processed_train.index = train_ids
    processed_train.to_csv('data/processed_train.csv')