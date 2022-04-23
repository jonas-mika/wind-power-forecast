# lsda assignment 3: mlflow ml pipeline
import mlflow
from azureml.core import Workspace

# custom imports 
from helpers import preprocess
from helpers import fetch_logged_data
from helpers import output, working_on, finished

output("Loading Modules.")
# system imports
import os
import sys
import json
import warnings
from timeit import default_timer as timer

# external imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score)

def main():
  total = timer()

  # env setup
  # ws = Workspace.from_config()
  # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
  # mlflow.set_tracking_uri("https://training.itu.dk:5000/")

  # load data
  s = working_on("Loading Data")
  data = pd.read_json("data/dataset.json", orient="split")
  finished("Loading Data", timer() - s)

  # preprocessing
  s = working_on("Preprocessing Data")
  data, X, y = preprocess(data)
  finished("Preprocessing Data", timer() - s)

  # load model configurations for ml experiments
  with open("experiments.json", "r") as f:
    experiments = json.loads(f.read())

  # mlflow experiment settings
  mlflow.set_experiment(f"jsen-wind-power-forecast")
  mlflow.sklearn.autolog()

  s = working_on("Training Models")
  for i, model in enumerate(experiments):
    exp_id = i+1
    name = model["name"]
    params = model["params"]
    metrics = model["metrics"]
  

    reg = globals()[name]()
    pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly_features", PolynomialFeatures()),
            ("reg", reg)])

    grid = GridSearchCV(
        estimator=pipe, 
        param_grid=params,
        scoring=metrics,
        refit="explained_variance",
        cv=5,
        verbose=1,
        n_jobs=-1) 

    grid.__class__.__name__ = name
    grid.fit(X, y)

    #run_id = mlflow.last_active_run().info.run_id
    #print(f"Logged data and model in run: {run_id}")

    #for key, data in fetch_logged_data(run_id).items():
    #    print(f"Logged {key}:\n")
    #    pprint(data)

  finished("Training Models", timer() - s)
  finished("Entire Pipeline", timer() - total)

  print("Get Overview over most recent runs by "\
        "running `mlflow ui`\n"\
        "Watch locally on http://127.0.0.1:5000/#/"\
        "or at https://http://training.itu.dk:5000/")

if __name__ == "__main__":
  main()
