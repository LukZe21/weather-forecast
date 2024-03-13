from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from datetime import datetime, timedelta
import requests
import sys
import csv
import time



cities = ['Tbilisi', 'Kutaisi', 'Rustavi']

def data_gather():
    date_now = datetime.now().strftime('%Y-%m-%d')
    date_future = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    for city in cities:
        response = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{date_now}/{date_future}?unitGroup=metric&include=days&key=QKR2MZS6XDU84M5T4KH7TFXDL&contentType=csv")
        if response.status_code!=200:
            print('Unexpected Status code: ', response.status_code)
            sys.exit()  
        
        response_hours = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}?unitGroup=metric&include=hours&key=QKR2MZS6XDU84M5T4KH7TFXDL&contentType=csv")

        CSVText = csv.reader(response.text.splitlines(), delimiter=',', quotechar='"')
        CSVText_hours = csv.reader(response_hours.text.splitlines(), delimiter=',', quotechar='"')
        with open(f'data/{city}.csv', mode='w', encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in CSVText:
                writer.writerow(row)
                
        with open(f'data/{city}_hours.csv', mode='w', encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in CSVText_hours:
                writer.writerow(row)


def plot_temp(fut_val=7):
    df = pd.read_csv('data/Tbilisi.csv')
    x_data = [i for i in df['datetime'][:fut_val] if datetime.strptime(i, "%Y-%m-%d").date() >= datetime.now().date()]
    x = pd.to_datetime(x_data).strftime("%d %b")
    y = df[df['datetime'].isin(set(x_data))]['temp']
    plt.figure(facecolor=(244/255, 244/255, 244/255), figsize=(10, 6))
    plt.plot(x, y, marker='.')
    sns.despine(left=False)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=45)
    plt.xlabel('Days')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature Throughout the Days')
    plt.fill_between(x, y, color='lightblue', alpha=0.3)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def to_dict_df(data):
    return data.to_dict()

def data_preprocessing(data):
    feats = ['precipprob', 'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility']
    X,y = data[feats], data['icon']
    X.dropna(inplace=True)
    return X

def forecast_model(data, info, data_hour,count=7):
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    forecast = loaded_model.predict(data)
    forecast_mapped = []
    # Going to give labels to the predicted classes
    for val in forecast:
        if val == 0: forecast_mapped.append('rain')
        elif val == 1: forecast_mapped.append('clear-day')
        elif val == 2: forecast_mapped.append('partly-cloudy-day')
        elif val == 3: forecast_mapped.append('wind')
        elif val == 4: forecast_mapped.append('snow')
        else: forecast_mapped.append('cloudy')

    date_dict = info['datetime'].to_dict()
    dates = []
    # going to append date into dates list
    for _, val in date_dict.items():
        dates.append(val)

    # iterating through data to gather information
    temp = [i for k,i in info['temp'].to_dict().items()]
    humidity = [i for k,i in info['humidity'].to_dict().items()]
    windspeed = [i for k,i in info['windspeed'].to_dict().items()]

    data_dict = data.to_dict()
    data_dict.update({'forecast': forecast_mapped})
    data_dict.update(info)

    forecast = data_dict.get('forecast')
    date_fc = {}
    current_fc = {}
    iter = 0
    for date, fc, tp, hmd, wsp in zip(dates, forecast, temp, humidity, windspeed):
        if iter >= count:
            break
        # if datetime now is less than date loop variable
        if pd.to_datetime(datetime.strptime(date, "%Y-%m-%d")) > datetime.now():
            date_fc[datetime.strptime(date, "%Y-%m-%d").strftime('%A, %d %b')] = {"Forecast":fc, "Temperature": f"{tp}°", "Humidity": f"{hmd}%", "Wind Speed": f"{wsp} kph"}
            iter += 1
        # if datetime now is equal to date loop variable
        elif datetime.strptime(date, "%Y-%m-%d").day == datetime.now().day:
            current_fc[datetime.strptime(date, "%Y-%m-%d").strftime('%A, %d %b')] = {"Forecast":fc, "Temperature": f"{tp}°", "Humidity": f"{hmd}%", "Wind Speed": f"{wsp} kph"}
    

    with open('model_hourly.pkl', 'rb') as f:
        hourly_model = pickle.load(f)

    feats = 'temp, feelslike, dew, humidity, precip, precipprob, snow, snowdepth, windgust, windspeed, winddir, sealevelpressure, cloudcover, visibility, solarradiation, solarenergy'
    feats = feats.split(', ')
    hour_forecast = hourly_model.predict(data_hour[feats])

    hour_forecast_mapped = []

    for fc in hour_forecast:
        if fc == 0: hour_forecast_mapped.append('clear-night')
        elif fc == 1: hour_forecast_mapped.append('clear-day')
        elif fc == 2: hour_forecast_mapped.append('partly-cloudy-night')
        elif fc == 3: hour_forecast_mapped.append('partly-cloudy-day')
        elif fc == 4: hour_forecast_mapped.append('cloudy')
        elif fc == 5: hour_forecast_mapped.append('wind')
        else: hour_forecast_mapped.append('rain')


    temp_hour = [i for k,i in data_hour['temp'].to_dict().items()]
    hours = []
    info_hour = {} 


    for _, val in data_hour['datetime'].to_dict().items():
        hour = datetime.strptime(val, '%Y-%m-%dT%H:%M:%S').strftime('%H')
        hours.append(hour)

    for hr_fc, hr, tph in zip(hour_forecast_mapped, hours, temp_hour):
        info_hour[int(hr)] = {'Forecast': hr_fc, 'Temperature': f"{tph}°"}

    
    return date_fc, current_fc, info_hour

app = Flask(__name__)


time_passed = True
time_past = None

title = 'Tbilisi'
@app.route('/', methods=['GET', 'POST'])
def home():
    global time_passed
    global time_past
    global title
    forecast = forecast_model(data_preprocessing(pd.read_csv(f'data/{title}.csv')),pd.read_csv(f'data/{title}.csv'), data_hour=pd.read_csv(f'data/{title}_hours.csv'))

    # Going to gather the data after every 3 hours from API
    if time_passed:
        time_past = datetime.now().hour
        time_passed = False
        data_gather()
    elif not time_passed and abs(time_past - datetime.now().hour) >= 3:
        time_passed = True

    current_time = time.strftime("%H:%M", time.localtime())
    
    fut_val = 8
    time_frame = '1 Week Forecast'


    if request.method == 'POST':
        selected_item = request.form.get('dropdown')
        if selected_item == '1w' or selected_item == '2w':
            if selected_item == '1w':
                time_frame = '1 Week Forecast'
                forecast = forecast_model(data_preprocessing(pd.read_csv(f'data/{title}.csv')),pd.read_csv(f'data/{title}.csv'), count=7, data_hour=pd.read_csv(f'data/{title}_hours.csv'))
                fut_val = 8
            elif selected_item == '2w':
                time_frame = '2 Weeks Forecast'
                forecast = forecast_model(data_preprocessing(pd.read_csv(f'data/{title}.csv')),pd.read_csv(f'data/{title}.csv'), count=14, data_hour=pd.read_csv(f'data/{title}_hours.csv'))
                fut_val = 15

            plot_url = plot_temp(fut_val)
        
        # Selecting the city data
        elif selected_item == 'tbilisi' or selected_item == 'kutaisi' or selected_item == 'rustavi':
            title = selected_item.capitalize()
            forecast = forecast = forecast_model(data_preprocessing(pd.read_csv(f'data/{title}.csv')),pd.read_csv(f'data/{title}.csv'), data_hour=pd.read_csv(f'data/{title}_hours.csv'))
            plot_url = plot_temp()
        
        return render_template('home.html', predictions=forecast[0], current_fc = forecast[1], hour_fc = forecast[2],
                               info=to_dict_df(pd.read_csv(f'data/{title}.csv')), plot_url=plot_url, time_frame=time_frame,
                               selected_item=selected_item, time=current_time, title=title)

    plot_url = plot_temp(fut_val) 
    
    return render_template('home.html', predictions=forecast[0], current_fc = forecast[1], hour_fc = forecast[2], 
                        info=to_dict_df(pd.read_csv(f'data/{title}.csv')), plot_url=plot_url, time=current_time,
                        title=title)

@app.route('/forecast')
def hourly_forecast():
    hour_now = datetime.now().hour
    forecast = forecast_model(data_preprocessing(pd.read_csv(f'data/{title}.csv')),pd.read_csv(f'data/{title}.csv'), data_hour=pd.read_csv(f'data/{title}_hours.csv'))
    
    df = pd.read_csv(f'data/{title}_hours.csv')
    x_data = [i for i in df['datetime']]
    x = pd.to_datetime(x_data).strftime("%H:%M")
    y = df[df['datetime'].isin(set(x_data))]['temp']
    plt.figure(facecolor=(244/255, 244/255, 244/255), figsize=(10, 6))
    plt.plot(x, y, marker='.', color='black')
    sns.despine(left=False)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=-45)
    plt.xlabel('Hours')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature Throughout the Hours')

    x_past_data = [i for i in df['datetime'] if datetime.strptime(i, "%Y-%m-%dT%H:%M:%S").hour <= hour_now]
    x_past = pd.to_datetime(x_past_data).strftime('%H:%M')
    y_past = df[df['datetime'].isin(set(x_past_data))]['temp']

    x_future_data = [i for i in df['datetime'] if datetime.strptime(i, "%Y-%m-%dT%H:%M:%S").hour >= hour_now]
    x_future = pd.to_datetime(x_future_data).strftime('%H:%M')
    y_future = df[df['datetime'].isin(set(x_future_data))]['temp']

    plt.fill_between(x_past, y_past, color='lightblue', alpha=0.3, label='Passed')
    plt.fill_between(x_future, y_future, color='orange', alpha=0.3, label='Ongoing')

    plt.legend(loc='upper left')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('hourly_fc.html', hour_fc=forecast[2], hour_now=hour_now, plot_url= plot_url, title=title)


if __name__ == '__main__':
    app.run(debug=True)