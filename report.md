# Wind Power Forecasting System
## Cloud-Based End-to-End Machine Learning Lifecycle with MLFlow on Azure

*Author: Jonas-Mika Senghaas (jsen@itu.dk), Date: 24.04.2022*

## Project Description

This project revisits Assignment 1 of the course
Large Scale Data Analysis. The goal of the project
was to develop a `sklearn` pipeline that uses
recent [weather data]() and [energy generation
data]() of the region
[Orkney](https://en.wikipedia.org/wiki/Orkney) in
order to build a model predicting the energy
generation through wind energy based on weather
conditions, like the wind speed and direction. The
best-performing model should be served on an
[Azure]() VM as a REST API as a forecasting model
for future weeks. The project is using
  [MLFlow](https://mlflow.org), an open source
  platform to manage the ML lifecycle, to log
  experimentation with different models and
  configuration, allowing for easily reproducible
  results and deployment of the model as a REST
  API in the cloud.

The source code of this project is publicly
available on
[GitHub](https://github.com/jonas-mika/wind-power-forecast).
For details of the implementation and information
about how to reproduce the experiments and results
follow the
[README](https://github.com/jonas-mika/wind-power-forecast/README.md).

## Reproduce Results


MLFlow projects generally offer two entry points
into running projects. The first option is to
simply clone this repository and then resolve the
dependencies as specified in the `conda.yml`
manually. The project can then be run by simply
running the main.py script.

```
git clone https://github.com/jonas-mika/wind-power-prediction-system.git
conda env create -f conda.yml
conda activate wind-power-forecast
python main.py
```

Instead of resolving the dependencies manually, it
is also possible to run the project pipeline as
specified in the `MLProject` environment file
after having cloned the project.

```
mlflow run .
```

MLFlow then first resolves the environment and
then runs the entire project pipeline.

If you don't wish to clone the entire project,
`mlflow run` can be run remotely through SSH
GitHub using the following command:

```
mlflow run git@github.com:jonas-mika/wind-power-prediction-system.git
```

## Data

Unlike Assignment 1, the data is not queried from
a live database, but statically loaded from
`data/dataset.json`, which stores 180 days of
inner-joined weather and energy production data in
JSON-format. The samples range from September 2020
to July 2021. The relevant columns used for
analysis are `Speed`, which stores the wind speed
in m/s, `Direction`, which is a categorical label
denoting the direction of the wind and `Total`,
which measures the total wind energy production.

```
DatetimeIndex: 254933 entries, 2020-10-09 12:31:00 to 2021-04-07 12:30:00
Data columns (total 7 columns):
 #   Column       Non-Null Count   Dtype         
---  ------       --------------   -----         
 0   ANM          254933 non-null  float64       
 1   Non-ANM      254933 non-null  float64       
 2   Total        254933 non-null  float64       
 3   Direction    1318 non-null    object        
 4   Lead_hours   1318 non-null    float64       
 5   Source_time  1318 non-null    datetime64[ns]
 6   Speed        1318 non-null    float64  
```

## Models and Evaluation Metrics

The project is set up in a way, that the `main.py`
script loads a JSON-file, `experiments.json`, that
specifies the `sklearn` models to test for how
suited there are to be used in the forecasting
system. Each model is using the default `sklearn`
set of hyperparameters, but is trained on
different degrees of polynomial features, that are
specified in the `max_degree` field in the JSON
(namely all degrees in the finite interval `[1,
max_degree]`.

```json
[
  {
    "name": "LinearRegression",
    "max_degree": 9
    }
  },
  {
    "name": "KNeighborsRegressor",
    "max_degree": 2
  },
  {
    "name": "GradientBoostingRegressor",
    "max_degree": 2
  }
]
```

*Note, that for the automatic loading of models to
work, the name of the model has to match the
`sklearn` naming conventions and need to be
imported in `main.py`. This has been done for the
models tested out within this project.*

The project tested three ML models in different
hyperparameter configurations, which are specified
in the below table.

| Model | Hyperparameters Grid | #Model Configurations | CV Folds | Total Fits | Scoring |
| :---  | :---: | :---: | :---: | :---: | ---: |
| Linear Regression  | 1, 2, 3, 4, 5, 6, 7, 9 | 9 | 3 | 27  | Explained Variance   |
| KNN Regressor  | 1, 2 | 2 | 3 | 6 | Explained Variance   |
| Gradient Boosting  | 1, 2 | 2 | 3 | 6 | Explained Variance   |

This leads to a total of `13` model configuration
that will be tracked during a single call to
`main`. Each model configuration starts a `mlflow
run` with a name identifying the model name and
the number of polynomial degrees as feature input.
Prior to this, the experiment name was set
globally, so that all runs are stored inside.

```python
# mlflow experiment settings
mlflow.set_experiment(f"jsen-wind-power-forecast")

# iterating over model configurations
[...] 
with mlflow.start_run(run_name=f"{model_name} (Degree: {degree})") as run:
  [...]
```

Each model configuration is then loaded and
inserted as the final estimator (`reg` in the code
snippet) into a preprocessing pipeline of custom
`sklearn` transformers. The preprocessing steps
itself stayed unchanged, involving dropping
irrelevant columns, numerically encoding wind
direction into radians, linearly upsampling
missing weather data, imputing missing data and
finally transforming the data about wind direction
into the more useful feature of wind vectors. The
pipeline furthermore explores the use of
polynomial features.

```python
pipe = Pipeline([
      ('column_selector', ColumnSelector(['Direction', 'Speed'])),
      ('encode_direction', Direction2Radians()),
      ('interpolate', InterpolateData()),
      ('imputer', Imputer()),
      ('transform_direction', Direction2Vec()),
      ('poly_features', PolynomialFeatures()),
      ('reg', reg)])
```

Each run tracks all parameters of all components
in the pipeline and should track metrics. Within
this project four metrics are being tracked, the
`Mean Absolute Error`, `Mean Squared Error`, the
`Explained Variance` and `R2-Score`, which are all
standard metrics to evaluate the performance of
regression problems. 

```python
# metrics 
metrics =  {
    "MAE": (mean_absolute_error, []),
    "MSE": (mean_squared_error, []),
    "Explained Variance": (explained_variance_score, []),
    "R2-Score": (r2_score, [])}
```

To get robust estimates for the out-of-sample
performance, the mean cross-validated scores of
each of the metrics is being tracked. The
cross-validation is a three-fold time series based
cross-validation to ensure to not train on future
data. The `test_size` was left at default.

```python
# use time series split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

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
```

Finally, each model configuration is assigned
a final score, which is used to determine, which
model configuration performed best overall. The
metric used for this was `Explained Variance`.
The project stores metadata about the currently
best saved model in `best_model.json` and the
logged mlflow artifact in `best_model` (both at
the root). After having obtained the
cross-validated metrics for each model
configuration, the mean explained variance is
compared to the currently best saved mean
explained variance as specified in
`best_model.json` and updated, if the new model
outperforms the current best. In that way, the
currently best model can be served easily:

```
mlflow models serve -m best_model [--no-conda] [-h 0.0.0.0] [-p 5000]
```

*Running this command from the VM with an opened
port `5000` serves the model to the public IP
address of the VM. At that stage the model is
publicly served and can be queried from anyone,
from anywhere*

The tracking within this project was tried within
an Azure Machine Learning environment, but found
to be easier to analyse on the locally hosted
tracking server by running:

```
mlflow ui [-h 127.0.0.1] [-p 5000]
```

## Evaluation

The best performing model with a overall best
average cross-validated explained variance score
of `72.9%` is a Gradient Boosting Regressor
trained on just the first degree polynomial features
 It is significantly
outperforming the distance-based K-Nearest
Neighbor Regressor, which only achieves
a maximal performance of  `56%` and is slightly
more performant than the best Linear Regression
model, which achieves a mean explained variance
score of `68%` for a 5-th degree polynomial.

However, the training time of the Linear
Regression is only a fraction of that of the
Gradient Boosting Regressor (15%). If training
time is a limiting factor, the Linear Regression
should therefore be considered as a good
alternative.

## Served Model

I have cloned the project on my Azure VM and
initiated a background process serving the best
model from the experiments explained on port 5000
as a REST API. The model can be queried using the
following syntax:

```
curl http://20.113.156.88:5000/invocations -H 'Content-Type: application/json' -d '{"columns": ["Speed", "Direction"], "data": [[5,"S"]]}'
```

## Discussion of MLFlow

As seen, machine learning experiments typically involve
testing out a lot of different models, in
different flavors of hyperparameters to find the
model that maximises performance. Without the
right tools it is hard to correctly keep track of
all these model configurations, their performance
and to reproduce them. For this reason, using
a platform like MLFlow is useful to manage the
entire ML lifestyle, from experimentation,
reproducability of results to deployment of
models. The main advantages found are:

1. It offers an easy interface to log parameters,
   metrics, artifacts like descriptions,
   visualisations and more. They can be easily
   viewed on a locally or remotely hosted
   tracking server, that offers a UI to view and
   compare the results.

2. MLFlow projects are closely interlinked with
   environments like `conda` or `docker`, which
   specify the packages in the correct versions
   that are used to successfully run the project.
   By allowing clients to run the entire project
   flow by simply running `mlflow run <URL>`,
   which resolves all necessary dependencies and
   runs the entire pipeline automatically, results
   are easily reproducible, even for people with
   little experience in programming.

3. MLFlow offers a unified way to serve models,
   i.e. through a REST API

In conclusion, one can say that MLFlow is not
reinventing the wheel. Everything that it does,
could be done in regular Python. However, it offers
a really complete one-in-all solution for
the entire machine learning pipeline, that greatly
simplifies developing scalable machine learning
projects.
