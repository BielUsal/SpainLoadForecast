VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: setup lint test spike data eda features train train-test evaluate evaluate-test history joint joint-test e2e e2e-test app-data app

setup:            ## create the virtual environment and install the package
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook,gbm,bayes,rnn,flow,app]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

spike:            ## check the live APIs (needs ESIOS_API_KEY in .env)
	$(PY) scripts/spike_openmeteo.py
	$(PY) scripts/spike_esios.py

data:             ## download what is missing and build the hourly table (needs ESIOS_API_KEY)
	$(PY) -m loadforecast.data.build

eda:              ## re-run the EDA notebook in place
	PATH="$(CURDIR)/$(VENV)/bin:$$PATH" $(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

features:         ## M1/M2
	@echo "not implemented yet (M1/M2)"

train:            ## the NUTS fit of the Bayesian model for the validation year (about an hour; other models train inside evaluate)
	$(PY) -m loadforecast.models.bayes --split validation

train-test:       ## the same fit on data up to the test year
	$(PY) -m loadforecast.models.bayes --split test

evaluate:         ## backtest all models on the validation year and write reports/results
	$(PY) -m loadforecast.evaluation.run --split validation

evaluate-test:    ## the final scores on the test year (done once, on 2026-11-16; needs train-test)
	$(PY) -m loadforecast.evaluation.run --split test --final

history:          ## RNN forecasts over 2022-2024, never scored: training errors for the joint models
	$(PY) -m loadforecast.evaluation.run --history --models rnn

joint:            ## whole-day scenarios (copula, flow) on the validation year (needs history and evaluate)
	$(PY) -m loadforecast.evaluation.run_joint --split validation

joint-test:       ## the same on the test year, once (done 2026-11-17)
	$(PY) -m loadforecast.evaluation.run_joint --split test --final

e2e:              ## end-to-end RNN + flow on the validation year (about 4 h on the GPU), then its scenarios
	$(PY) -m loadforecast.evaluation.run --split validation --models rnn_flow
	$(PY) -m loadforecast.evaluation.run_joint --split validation --e2e

e2e-test:         ## the same on the test year, once
	$(PY) -m loadforecast.evaluation.run --split test --final --models rnn_flow
	$(PY) -m loadforecast.evaluation.run_joint --split test --final --e2e

app-data:         ## copy the predictions the dashboard needs into app/data (run after evaluate-test)
	$(PY) -m loadforecast.viz.app_data --split test

app:              ## run the dashboard locally
	$(PY) -m streamlit run app/streamlit_app.py
