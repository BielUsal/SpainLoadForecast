VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: setup lint test spike data eda features train evaluate app-data app

setup:            ## create the virtual environment and install the package
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,notebook,gbm,bayes,app]"

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

train:            ## the single NUTS fit of the Bayesian model (slow; other models train inside evaluate)
	$(PY) -m loadforecast.models.bayes

evaluate:         ## backtest all models on the validation year and write reports/results
	$(PY) -m loadforecast.evaluation.run --split validation

app-data:         ## copy the predictions the dashboard needs into app/data (run after evaluate)
	$(PY) -m loadforecast.viz.app_data --split validation

app:              ## run the dashboard locally
	$(PY) -m streamlit run app/streamlit_app.py
