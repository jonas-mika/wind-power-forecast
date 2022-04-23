# lsda assignment 3: mlflow ml pipeline
import mlflow
from azureml.core import Workspace

# custom imports 
from helpers import (
    ColumnSelector,
    Direction2Radians,
    InterpolateData, 
    Imputer,
    Direction2Vec)
from helpers import fetch_logged_data
from helpers import output, working_on, finished

output("Loading Modules.")
# system imports
import os
import sys
import json
import warnings
from datetime import datetime
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
  y = data.pop("Total")
  X = data
  finished("Loading Data", timer() - s)



  # pipe.fit(X, y)
  # test = pd.DataFrame({"Speed": [7, 6], "Direction": ["N", "NW"]})
  # print(pipe.predict(test))
  # return

  # load model configurations for ml experiments
  with open("experiments.json", "r") as f:
    experiments = json.loads(f.read())

  # mlflow experiment settings
  mlflow.set_experiment(f"jsen-wind-power-forecast")
  mlflow.sklearn.autolog(log_input_examples=True, silent=True)

  for i, model in enumerate(experiments):
    exp_id = i+1
    name = model["name"]
    params = model["params"]
    metrics = model["metrics"]

    s = working_on(f"Training {name}")

    reg = globals()[name]()
    pipe = Pipeline([
          ("column_selector", ColumnSelector(["Direction", "Speed"])),
          ("encode_direction", Direction2Radians()),
          ("interpolate", InterpolateData()),
          ("imputer", Imputer()),
          ("transform_direction", Direction2Vec()),
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

    grid.fit(X, y)
    best_estimator = grid.best_estimator_
    best_score = grid.best_score_
    best_params = grid.best_params_

    # update best model if needed
    try: 
      with open("best_model.json", "r") as f:
        best_model = json.loads(f.read())

      if best_score > best_model["score"]:
        print(f"New best model. Saving {name}")
        mlflow.sklearn.save_model(
            best_estimator,
            path="best_model",
            conda_env="conda.yml")

        best_model = {
            "timestamp": datetime.now().timestamp(),
            "date": datetime.today().strftime("%m/%d/%Y"),
            "name": name,
            "score": best_score,
            "params": best_params
            }
        with open("best_model.json", "w") as f:
          json.dump(best_model, f)
      else: 
        print(f"Model {name} not better. Not saving")

    except:
      print(f"No best model saved yet. Saving {name}")
      mlflow.sklearn.save_model(
          best_estimator,
          path="best_model",
          conda_env="conda.yml")

      best_model = {
          "timestamp": datetime.now().timestamp(),
          "date": datetime.today().strftime("%m/%d/%Y"),
          "name": name,
          "score": best_score,
          "params": best_params
          }
      with open("best_model.json", "w") as f:
        json.dump(best_model, f)

    # print logged data of saved model
    # print(f"Logged data and model in run: {run_id}")

    #for key, data in fetch_logged_data(run_id).items():
    #    print(f"Logged {key}:\n")
    #    pprint(data)

    finished(f"Training {name}", timer() - s)


  finished("Entire Pipeline", timer() - total)

  print("Get Overview over most recent runs by "\
        "running `mlflow ui`\n"\
        "Watch locally on http://127.0.0.1:5000/#/"\
        "or at https://http://training.itu.dk:5000/")

if __name__ == "__main__":
  main()
