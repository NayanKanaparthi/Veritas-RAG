.PHONY: install test lint format typecheck build benchmark-smoke clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/veritas_rag

build:
	python -m build

benchmark-smoke:
	@mkdir -p /tmp/veritas_smoke_corpus
	@echo "This is a test document." > /tmp/veritas_smoke_corpus/test.txt
	veritas-rag build /tmp/veritas_smoke_corpus --output /tmp/veritas_smoke_artifact
	veritas-rag benchmark /tmp/veritas_smoke_artifact --suite latency
	veritas-rag benchmark /tmp/veritas_smoke_artifact --suite portability

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache dist build *.egg-info
