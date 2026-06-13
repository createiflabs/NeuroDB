.PHONY: install test lint run docker-build docker-run clean

IMAGE ?= createiflabs/neurodb:dev

install:
	pip install -r requirements-dev.txt -e .

lint:
	ruff check .

test:
	pytest -q

run:
	python -m neurodb

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -v neurodb_data:/data $(IMAGE)

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
