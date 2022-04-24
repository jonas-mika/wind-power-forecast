# MLFlow Automatic Wind Power Generation Forecasting System

Repository to store progress on *Assignment 3* of
course *Large Scale Data Analysis*, which explores
using the [MLFlow](https://mlflow.org/) framework for 
organised, reproducible and maintainable machine learning
experiments.

## Running this Project

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
specified in the `MLProject` environment file. Running

```
mlflow run .
```

from the root directory of this project first
resolves the environment and then runs the entire
project pipeline.

If you don't wish to clone the entire project,
`mlflow run` can be run remotely through SSH
GitHub using the following command:

```
mlflow run git@github.com:jonas-mika/wind-power-prediction-system.git
```

## Viewing Results

All experiment results are saved in an Azure
Machine Learning environment. After cloning, the
projects can however easily be run locally and the
results are saved to the filesystem in the
directory `mlruns`. Running `mlflow ui` from the
root of the project opens a MLFlow tracking server
with a user interface to easily view and compare
the results of the different experiments.

```
mlflow ui 
open http://127.0.0.1:5000
```

## Serving Model

To serve the model as a REST API run the following
command after having run the entire project workflow. This will serve the currently best performing model (metadata
stored in `best_model.json` and MLFlow artifact in
directory `best_model`). By default the model is
hosted locally at the address `http://127.0.0.0:5000`.

```
mlflow models serve -m best_model [--no-conda]
```

Get predictions from the model by running

```
curl 127.0.0.1:5000/invocations -H 'Content-Type: application/json'\
  -d '{"columns": ["Speed", "Direction"], "data": [[5,"S"]]}'
```

## Testing the Model

The model is currently hosted on an
[Azure](https://azure.microsoft.com) virtual
machine as a background progress on port `5000`.
The model can be queried live through the
following command:

```
curl http://20.113.156.88:5000/invocations -H 'Content-Type: application/json' -d '{"columns": ["Speed", "Direction"], "data": [[5,"S"]]}'
```
