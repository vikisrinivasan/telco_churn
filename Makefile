THIS_FILE := $(lastword $(MAKEFILE_LIST))
PROJECT_ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


extract-data:
	export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"; \
    python \
    ${PROJECT_ROOT}/src/data/make_dataset.py \
    --raw_data_path ${PROJECT_ROOT}/${raw_data_path} \
    --interim_data_path ${PROJECT_ROOT}/${interim_data_path}


build-features:
	export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"; \
    python \
    ${PROJECT_ROOT}/src/features/build_features.py \
    --interim_data_path ${PROJECT_ROOT}/${interim_data_path} \
    --processed_data_path ${PROJECT_ROOT}/${processed_data_path}


train-model:
	export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"; \
    python \
    ${PROJECT_ROOT}/src/models/train_model.py \
    --processed_data_path ${PROJECT_ROOT}/${processed_data_path}\
    --model_logging ${PROJECT_ROOT}/models \
    --parameter_path ${PROJECT_ROOT} \
    --teta 4



run:
	@$(MAKE) -f $(THIS_FILE) extract-data
	@$(MAKE) -f $(THIS_FILE) build-features
	@$(MAKE) -f $(THIS_FILE) train-model