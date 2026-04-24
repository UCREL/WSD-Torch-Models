VERSION_CMD = "uv run scripts/get_version.py ./pyproject.toml"

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

.PHONY: build-python-package
build-python-package:
	@uv lock --check
	@rm -rf ./dist
	@uv build

.PHONY: release-notes
release-notes: build-python-package
	@uv run --no-project --script \
	--with dist/wsd_torch_models-$$("${VERSION_CMD}")-py3-none-any.whl \
	./scripts/release_notes.py

.PHONY: run-cu128
run-cu128:
	@uv run --no-group cpu --group cu128 $(CMD)

.PHONY: add-cu128
add-cu128:
	@uv add --no-sync $(CMD)
	@uv sync --no-group cpu --group cu128 --all-extras

.PHONY: tests-cu128
tests-cu128:
	@uv run --no-group cpu --group cu128 coverage run
	@uv run --no-group cpu --group cu128 coverage report

.PHONY: lint-cu128
lint-cu128:
	@echo "ISort:"
	@uv run --no-group cpu --group cu128 isort src tests scripts
	@echo "Flake 8:"
	@uv run --no-group cpu --group cu128 flake8 --config ./.flake8 src tests
	@echo "MyPy:"
	@uv run --no-group cpu --group cu128 mypy
	@echo "Linting finished"