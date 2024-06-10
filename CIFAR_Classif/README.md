# CIFAR_Classif

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Description

This project aims to classify the images from the CIFAR dataset.

## Features

- Generic Class for benchmarking multiple classifiers
    - SVC
    - KNN
    - Logistic Regression
    - Random Forest
    - SVM
    - Decision Tree
    - gradient_boosting

- Feature 2
- Feature 3

## Requirements

- Python 3.6+
- Make
- virtualenv

## Installation

To install the project, you need to run the following commands:
```bash
git clone
cd CIFAR_Classif
mkvirtualenv cifar_classif
pip install -r requirements.txt
make build_lib
```

In order to make the notebook work, you will need to add the CIFAR dataset to the path `data/cifar-10-batches-py`, it can be downloaded here : [cifar-10-batches-py](https://www.cs.toronto.edu/~kriz/cifar.html)  

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make build_lib`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for CIFAR_Classif
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── CIFAR_Classif                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes CIFAR_Classif a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------