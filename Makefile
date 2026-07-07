.PHONY: help clean check autoformat
.DEFAULT: help

help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run syntax, formatting, and lint checks without changing files"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	python scripts/check_python_syntax.py
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix --show-fixes .
