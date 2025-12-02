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