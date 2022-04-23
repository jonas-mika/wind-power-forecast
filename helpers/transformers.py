# transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
  def __init__(self, columns):
    self.columns = columns

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = X[self.columns]
    # print(f"{self.__class__.__name__}:\n {X}")
    return X

class Direction2Radians(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    vals = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 
        'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    direction2radians = {vals[i]: 22.5*i for i in range(len(vals))}

    X = X.replace({"Direction": direction2radians})
    # print(f"{self.__class__.__name__}:\n {X}")

    return X

class Direction2Vec(BaseEstimator, TransformerMixin):
  def __init__(self):
    return

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    wd = X.pop("Direction")
    wv = X["Speed"] 
    wd_rad = wd * np.pi / 180

    X['Wx'] = wv*np.cos(wd_rad)
    X['Wy'] = wv*np.sin(wd_rad)

    # print(f"{self.__class__.__name__}:\n {X}")
    return X

class InterpolateData(BaseEstimator, TransformerMixin):
  def __init__(self):
    return

  def fit(self, X, y = None):
    return self

  def transform(self, X, y = None):
    X = X.interpolate(strategy='linear')

    # print(f"{self.__class__.__name__}:\n {X}")
    
    return X

class Imputer(BaseEstimator, TransformerMixin):
  def __init__(self):
    return

  def fit(self, X, y = None):
    return self

  def transform(self, X, y=None):
    # get nan rows
    X["Direction"].fillna(X["Direction"].mean(), inplace=True)
    X["Speed"].fillna(X["Speed"].mean(), inplace=True)
    # print(f"{self.__class__.__name__}:\n {X}")

    return X
