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

All experiment results are saved in a `PostgreSQL`
database hosted publicly on `https://training.itu.dk:5000/#/`. 
This project has an experiment ID of `jsen-wind-power-forecast`.
This can however be changed as an argument to running the main 
script if desired.

```
open https://training.itu.dk:5000/#/
```

Otherwise running the following command opens a tracking server on localhost:

```
mlflow ui -h 0.0.0.0 -p 8888
open http://0.0.0.0:8888
```

## Serving Model

TBA
