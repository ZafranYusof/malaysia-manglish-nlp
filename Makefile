.PHONY: test test-all lint build publish clean

test:
	python -m pytest tests/ -q

test-all:
	RUN_HEAVY_TESTS=1 python -m pytest tests/ -v

lint:
	python -m flake8 manglish_nlp/ --max-line-length=120

build:
	python -m build

publish:
	python -m twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info
