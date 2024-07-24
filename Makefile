MAIN_FILE = app.py

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=test $(MAIN_FILE)

format:
	black lib/*.py *.py

lint:
	pylint --disable=R,C lib/*.py $(MAIN_FILE)

run:
	./app.py

run_yaml:
	python -c 'import yaml, sys; print(yaml.safe_load(sys.stdin))' < .github/workflows/main.yml

all: install lint test format