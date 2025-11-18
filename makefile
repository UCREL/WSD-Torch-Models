.PHONY: lint
lint:
	@echo "ISort:"
	@uv run isort src tests scripts
	@echo "Flake 8:"
	@uv run flake8 --config ./.flake8 src tests
	@echo "MyPy:"
	@uv run mypy
	@echo "Linting finished"

.PHONY: tests
tests:
	@uv run coverage run
	@uv run coverage report