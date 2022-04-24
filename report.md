# Wind Power Forecasting System
## Cloud-Based End-to-End Machine Learning Lifecycle with MLFlow on Azure

*Author: Jonas-Mika Senghaas (jsen@itu.dk), Date: 24.04.2022*

## Project Description

This project revisits Assignment 1 of the course Large Scale Data Analysis. The goal of the project was to develop a `sklearn` pipeline that uses recent [weather data]() and [energy generation data]() of the region [Orkney](https://en.wikipedia.org/wiki/Orkney) in order to build a model predicting the energy generation through wind energy based on weather conditions, like the wind speed and direction. The best-performing model should be served on an [Azure]() VM as a REST API as a forecasting model for future weeks. The project is using [MLFlow](https://mlflow.org), an open source platform to manage the ML lifecycle, to log experimentation with different models and configuration, allowing for easily reproducible results and deployment of the model as a REST API in the cloud.

## Reproduce Results

The source code of this project is publicly available on [GitHub](https://github.com/jonas-mika/wind-power-forecast). For details of the implementation and information about how to reproduce the experiments and results follow the [README](https://github.com/jonas-mika/wind-power-forecast/README.md).

## Data

Unlike Assignment 1, the data is not queried from a live database, but statically loaded from `data/dataset.json`, which stores 180 days of inner-joined weather and energy production data in JSON-format. The samples range from September 2020 to July 2021. The relevant columns used for analysis are `Speed`, which stores the wind speed in m/s, `Direction`, which is a categorical label denoting the direction of the wind and `Total`, which measures the total wind energy production.

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

The project is set up in a way, that the `main.py` script loads a JSON-file, `experiments.json`, that specifies all the models, hyperparameter grid and evaluation metrics that a single call to main should perform.

```json
  [
    {
      "name": "KNeighborsRegressor",
      "params": {
        "poly_features__degree": [1, 2],
        "reg__n_neighbors": [5, 7, 9, 11, 13, 15],
        "reg__weights": ["uniform", "distance"] 
      },
      "metrics": [
        "explained_variance", 
        "max_error", 
        "neg_mean_absolute_error",
        "neg_mean_squared_error"
      ]   
    },
    ...
  ]
```

*Note, that for the automatic loading of models to work, the name of the model, parameters and metrics need to match the `sklearn` naming conventions and need to be imported in `main.py`. This has been done for the models tested out within this project.*

The project tested three ML models in different hyperparameter configurations, which are specified in the below table.

| Model | Hyperparameters Grid | #Model Configurations | CV Folds | Total Fits | Scoring |
| :---  | :---: | :---: | :---: | :---: | ---: |
| Linear Regression  |  {"poly_features__degree": [1, 2, 3, 4, 5]} | 5 | 5 | 25  | Explained Variance   |
| KNN Regressor  | {"poly_features__degree": [1, 2], "reg__n_neighbors": [5, 7, 9, 11, 13, 15], "reg__weights": ["uniform", "distance"] } | 12 | 5 | 60 | Explained Variance   |
| Gradient Boosting  | {"poly_features__degree": [2], "reg__n_estimators": [10],"reg__max_depth": [2]} | 1 | 5 | 5 | Explained Variance   |

Each specified model is then loaded and inserted as the final estimator (`reg` in the code snippet) into a preprocessing pipeline of custom `sklearn` transformers. The preprocessing steps itself stayed unchanged, involving dropping irrelevant columns, numerically encoding wind direction into radians, linearly upsampling missing weather data, imputing missing data and finally transforming the data about wind direction into the more useful feature of wind vectors. The pipeline furthermore explores the use of polynomial features.

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

Each experiment starts a new MLFlow run, that is automatically tracking and logging parameters, performance metrics and artifacts for all model configurations, both for the cross validation and the final estimator. The logs were connected to a Azure Machine Learning Workspace through a unique tracking URI and could be inspected there. 

After every run, the best estimator is compared to the currently best-performing model (according to the explained variance metric) and saved into `best_model` along with some metadata in `best_model.json`. In that way, the currently best model can be served easily by running 

```
mlflow models serve -m best_model [--no-conda] [-h 0.0.0.0] [-p 5000]
```

## Evaluation

The best performing model with a overall best average cross-validated expalined variance score of `~74%` is a 5-th polynomial regression model. It is significantly outperforming both the distance-based K-Nearest Neighbor regressor, which only achieves a performance of  `~65%` and the gradient boosted ensemble of trees, which reaches similar performance as the KNN Regressor at `~65%`.

## Discussion of MLFlow

Machine Learning experiments typically involve testing out a lot of different models, in different flavors of hyperparameters to find the model that maximises performance. Without the right tools it is hard to correctly keep track of all these model configurations, their performance and to reproduce them. For this reason, using a platform like MLFlow is useful to manage the entire ML lifestyle, from experimentation, reproducability of results to deployment of models.
The main advantages are:

1. It offers an easy interface to log parameters, metrics, artifacts like descriptions, visualisations and more. They can be easily viewed using on a locally or remotely hosted tracking server, that offers a UI to view and compare the results.

2. MLFlow projects are closely interlinked with environments like `conda` or `docker`, which specify the packages in the correct versions that are used to successfully run the project. By allowing clients to run the entire project flow by simply running `mlflow run <URL>`, which resolves all necessary dependencies and runs the entire pipeline automatically, results are easily reproducible, even for people with little experience in programming.

3. MLFlow offers a unified way to serve models, i.e. through a REST API

In conclusion, one can say that MLFlow is not reinventing the wheel. However, it offers a already really complete one-in-all solution for the entire machine learning pipeline, that greatly simpfifies developing scalable machine learning projects - even in teams by offering an easy-to-use, versatile and highly adaptive interface.