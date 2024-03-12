from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from datetime import datetime


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

    date_dict = date['datetime'].to_dict()
    dates = []
    for _, val in date_dict.items():
        dates.append(val)


    temp = [i for k,i in date['temp'].to_dict().items()]
    humidity = [i for k,i in date['humidity'].to_dict().items()]
    windspeed = [i for k,i in date['windspeed'].to_dict().items()]

    data_dict = data.to_dict()
    data_dict.update({'forecast': forecast_mapped})
    data_dict.update(date)

    forecast = data_dict.get('forecast')
    date_fc = {}
    iter = 0
    for date, fc, tp, hmd, wsp in zip(dates, forecast, temp, humidity, windspeed):
        if iter >= 5:
            break
        date_fc[datetime.strptime(date, "%Y-%m-%d").strftime('%A, %d %b')] = {"Forecast":fc, "Temperature": tp, "Humidity": hmd, "Wind Speed": wsp}
        iter += 1
    return date_fc

app = Flask(__name__)


@app.route('/')
def home():
    forecast = forecast_model(data_preprocessing(pd.read_csv('data/Tbilisi.csv')),pd.read_csv('data/Tbilisi.csv'))
    
    df = pd.read_csv('data/Tbilisi.csv')
    x = pd.to_datetime(df['datetime']).dt.strftime('%d%b')
    y = df['temp']
    plt.plot(x, y, marker='.')
    sns.despine(left=False)
    plt.grid(alpha=0.05)
    plt.xticks(rotation=45)
    plt.xlabel('Days')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature in March')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('home.html', predictions=forecast, info=to_dict_df(pd.read_csv('data/Tbilisi.csv')), plot_url=plot_url)

@app.route('/forecast')
def forecast_page():
    pass
    # return render_template('forecast_pg.html', plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)