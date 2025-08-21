.PHONY: tests, lints, docs, format

tests:
	uv run pytest

lints:
	uv run ruff check reconcile examples

format:
	uv run ruff check --select I --fix reconcile examples
	uv run ruff format reconcile examples
