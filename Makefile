MAIN_FILE = app.py

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=test $(MAIN_FILE)

format:
	black *.py

lint:
	pylint --disable=R,C $(MAIN_FILE)

run_yaml:
	python -c 'import yaml, sys; print(yaml.safe_load(sys.stdin))' < .github/workflows/main.yml

all: install lint test format