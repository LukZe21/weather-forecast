import pandas as pd
import pickle

def to_dict_df(data):
    return data.to_dict()

def data_preprocessing(data):
    feats = ['precipprob', 'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility']
    X,y = data[feats], data['icon']
    return X

def forecast_model(data, date):
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    forecast = loaded_model.predict(data)
    forecast_mapped = []
    for val in forecast:
        if val == 0: forecast_mapped.append('rain')
        elif val == 1: forecast_mapped.append('clear-day')
        elif val == 2: forecast_mapped.append('partly-cloudy-day')
        elif val == 3: forecast_mapped.append('wind')
        elif val == 4: forecast_mapped.append('snow')
        else: forecast_mapped.append('cloudy')

    date_dict = date.to_dict()
    dates = []
    for _, val in date_dict.items():
        for key1, val1 in val.items():
            dates.append(val1)
    data_dict = data.to_dict()
    data_dict.update({'forecast': forecast_mapped})
    data_dict.update(date)
    forecast = data_dict.get('forecast')
    date_fc = {}
    for date, fc in zip(dates, forecast):
        date_fc[date] = fc
    return date_fc



print(forecast_model(data_preprocessing(pd.read_csv('data/Tbilisi.csv')),pd.read_csv('data/date.csv')))