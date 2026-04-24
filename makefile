VERSION_CMD = "uv run scripts/get_version.py ./pyproject.toml"

.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only src tests scripts
	@uv run ruff check src tests scripts
	@echo "Type checking with Ty"
	@uv run ty check src tests scripts
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
	@echo "Linting with Ruff:"
	@uv run --no-group cpu --group cu128 ruff check --fix-only src tests scripts
	@uv run --no-group cpu --group cu128 ruff check src tests scripts
	@echo "Type checking with Ty"
	@uv run --no-group cpu --group cu128 ty check src tests scripts
	@echo "Linting finished"