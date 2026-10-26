THIS_FILE := $(lastword $(MAKEFILE_LIST))
PROJECT_ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(PROJECT_ROOT)/.venv/bin/python

raw_data_path ?= data/raw
interim_data_path ?= data/interim
processed_data_path ?= data/processed

export PYTHONPATH := $(PYTHONPATH):$(PROJECT_ROOT)


extract-data:
	$(PYTHON) -m src.data.extract_dataset \
    --raw_data_path $(PROJECT_ROOT)/$(raw_data_path)


make-dataset:
	$(PYTHON) -m src.data.make_dataset \
    --raw_data_path $(PROJECT_ROOT)/$(raw_data_path) \
    --interim_data_path $(PROJECT_ROOT)/$(interim_data_path)


build-features:
	$(PYTHON) -m src.features.build_features \
    --interim_data_path $(PROJECT_ROOT)/$(interim_data_path) \
    --processed_data_path $(PROJECT_ROOT)/$(processed_data_path)


train-model:
	$(PYTHON) -m src.models.train_model \
    --processed_data_path $(PROJECT_ROOT)/$(processed_data_path) \
    --interim_data_path $(PROJECT_ROOT)/$(interim_data_path) \
    --model_logging $(PROJECT_ROOT)/models


predict:
	$(PYTHON) -m src.models.predict_file \
    --processed_data_path $(PROJECT_ROOT)/$(processed_data_path) \
    --interim_data_path $(PROJECT_ROOT)/$(interim_data_path) \
    --model_logging $(PROJECT_ROOT)/models


deck:
	$(PYTHON) -m src.visualization.build_excom_deck


notebook:
	cd $(PROJECT_ROOT)/notebooks && $(PROJECT_ROOT)/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=singtel-churn \
    notebook_v1.2.ipynb


run:
	@$(MAKE) -f $(THIS_FILE) extract-data
	@$(MAKE) -f $(THIS_FILE) make-dataset
	@$(MAKE) -f $(THIS_FILE) build-features
	@$(MAKE) -f $(THIS_FILE) train-model
	@$(MAKE) -f $(THIS_FILE) predict
