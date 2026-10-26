telco_churn
==============================

Singtel AIDA churn case study: rank customers for a 20%-capacity retention list and
test whether contact-centre transcripts improve it.

Deliverables
------------

| File | What it is |
|---|---|
| `notebooks/notebook_v1.2.ipynb` | Technical notebook, brief sections 1-6 plus microsegments |
| `reports/singtel-churn-excom.html` | 8-slide Singtel ExCom deck (self-contained, opens offline) |
| `reports/test_scores_{tabular,combined}.csv` | Test-set readout: churn score vs recorded churn reason |
| `reports/call_list.csv` | Scored base with segment and the top-20% call flag |

RUN Command
------------

Python 3.12 environment: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

    make run        # extract-data -> make-dataset -> build-features -> train-model -> predict
    make notebook   # re-execute the notebook
    make deck       # rebuild the ExCom deck from reports/deck_data.json

| Step | Script | In -> Out |
|---|---|---|
| extract-data | `src/data/extract_dataset.py` | Kaggle + HuggingFace -> `data/raw` |
| make-dataset | `src/data/make_dataset.py` | clean, join calls to customers, T0 filter -> `data/interim` |
| build-features | `src/features/build_features.py` | tabular + call sequence + text score, split -> `data/processed` |
| train-model | `src/models/train_model.py` | logistic models, metrics at 20% -> `models/singtel`, `reports/` |
| predict | `src/models/predict_file.py` | call list with segments -> `reports/call_list.csv` |

The legacy CDR-project modules (`src/pa_lib`, `training_parameter.yaml`) are kept but no longer on the run path.


Model Grid Search Parameters
------------

training_parameter.yaml - This File Contains all the parameters of model

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



