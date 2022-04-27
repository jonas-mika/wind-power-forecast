# lsda assignment 3: mlflow ml pipeline

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
import shutil
import warnings
from datetime import datetime
from timeit import default_timer as timer

# external imports
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from azureml.core import Workspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
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

  # load model configurations for ml experiments
  with open("experiments.json", "r") as f:
    experiments = json.loads(f.read())

  # mlflow experiment settings
  mlflow.set_experiment(f"jsen-wind-power-forecast")

  for i, model in enumerate(experiments):
    exp_id = i+1
    model_name = model["name"]
    max_degree = model["max_degree"]

    # metrics 
    metrics =  {
        "MAE": (mean_absolute_error, []),
        "MSE": (mean_squared_error, []),
        "Explained Variance": (explained_variance_score, []),
        "R2-Score": (r2_score, [])}

    for degree in range(1, max_degree+1):
      with mlflow.start_run(run_name=f"{model_name} (Degree: {degree})") as run:
        s = working_on(f"Training {model_name} on {degree}-th Polynomial")

        reg = globals()[model_name]()
        pipe = Pipeline([
              ("column_selector", ColumnSelector(["Direction", "Speed"])),
              ("encode_direction", Direction2Radians()),
              ("interpolate", InterpolateData()),
              ("imputer", Imputer()),
              ("transform_direction", Direction2Vec()),
              ("poly_features", PolynomialFeatures(degree=degree)),
              ("reg", reg)])

        # log all parameters
        mlflow.log_params({key: val for key, val
          in pipe.get_params().items() if key!='steps'})
        
        # cross-validate scores
        for train, test in TimeSeriesSplit(3).split(X,y):
          # fit cv split
          pipe.fit(X.iloc[train],y.iloc[train])
          preds = pipe.predict(X.iloc[test])
          truth = y.iloc[test]

          for name, val in metrics.items(): 
            func, scores = val
            score = func(truth, preds)
            scores.append(score)

        # logging model in run
        pipe.fit(X, y)
        mlflow.sklearn.log_model(pipe, 
            artifact_path=f"mlruns/1/{run.info.run_id}")

        for name, val in metrics.items():
          _, scores = val
          mlflow.log_metric(f"Mean {name}", np.mean(scores))

          # final scoring
          if name == 'Explained Variance':
            final_score = np.mean(scores)

        # update best model if needed
        try: 
          with open("best_model.json", "r") as f:
            best_model = json.loads(f.read())

          if final_score > best_model["score"]:
            shutil.rmtree('best_model')
            print(f"New best model. Saving {model_name}")
            mlflow.sklearn.save_model(
                pipe,
                path="best_model",
                conda_env="conda.yml")

            best_model = {
                "timestamp": datetime.now().timestamp(),
                "date": datetime.today().strftime("%m/%d/%Y"),
                "model_name": model_name,
                "score": final_score,
                "degree": degree
                }
            with open("best_model.json", "w") as f:
              json.dump(best_model, f)
          else: 
            print(f"Model {model_name} not better. Not saving")

        except:
          print(f"No best model saved yet. Saving {model_name}")
          mlflow.sklearn.save_model(
              pipe,
              path="best_model",
              conda_env="conda.yml")

          best_model = {
              "timestamp": datetime.now().timestamp(),
              "date": datetime.today().strftime("%m/%d/%Y"),
              "model_name": model_name,
              "score": final_score,
              "degree": degree
              }
          with open("best_model.json", "w") as f:
            json.dump(best_model, f)
        finished(f"Training {model_name}", timer() - s)
  finished("Entire Pipeline", timer() - total)

  print("Get Overview over most recent runs by "\
        "running `mlflow ui`\n"\
        "Watch locally on http://127.0.0.1:5000/#/"\
        "or at https://http://training.itu.dk:5000/")

if __name__ == "__main__":
  main()
