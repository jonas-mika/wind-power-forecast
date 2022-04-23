import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def dir2radians(data, direction_col='Direction'):
  vals = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
  direction2radians = {vals[i]: 22.5*i for i in range(len(vals))}

  data = data.replace({direction_col: direction2radians})

  return data

def radians2vecs(data, direction_col='Direction'):
  # convert into wind vectors
  wd = data.pop("Direction")
  wv = data["Speed"] 
  wd_rad = wd * np.pi / 180

  data['Wx'] = wv*np.cos(wd_rad)
  data['Wy'] = wv*np.sin(wd_rad)

  return data

def preprocess(data):
  # convert categorical wind direction into radians
  data = dir2radians(data)

  # upsample
  data['Speed'] = data['Speed'].interpolate(method='linear')
  data['Direction'] = data['Direction'].interpolate(method='linear')

  # transform radians to wind vectors
  data = radians2vecs(data)

  # remove all columns that contain nan values in the relevant columns
  data = data[['Speed', 'Wx', 'Wy', 'Total']].dropna()

  # split merged DF into feature matrix and target variable
  X = np.array(data[['Speed', 'Wx', 'Wy']])
  y = np.array(data[['Total']]).reshape(-1)

  return data, X, y

def preprocess_forecasts(forecast_df):
  forecast_df = dir2radians(forecast_df)
  forecast_df = radians2vecs(forecast_df)

  data = forecast_df[['Speed', 'Wx', 'Wy']]
  X = np.array(data)
  
  return data, X
