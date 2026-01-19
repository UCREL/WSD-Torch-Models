# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Added the arXiv paper to the PyMUSAS BEM model readme, `model_readmes/pymusas_bem.md`, of which this did require the Bib text to be Python string variable in the convert and upload script `scripts/convert_and_upload_bem_model.py`.

## [v0.1.2](https://github.com/UCREL/WSD-Torch-Models/releases/tag/v0.1.2) - 2025-12-10

### Changed

- The version of `numpy` has been relaxed from `numpy>=2.0.0,<3.0` to `numpy>=1.19.0,<3.0` so that we can use the GPU within a spacy pipeline due to spacy's dependency on [cupy](https://github.com/cupy/cupy) version `cupy-cuda12x>=11.5.0,<13.0.0`.

## [v0.1.1](https://github.com/UCREL/WSD-Torch-Models/releases/tag/v0.1.1) - 2025-12-03

### Added

- `tokenizer_kwargs` optional argument to the `wsd_torch_models.bem.BEM.predict` method. This allows users to define key word arguments that can be passed to the sub word tokenizer that is downloaded from HuggingFace through ``transformers.AutoTokenizer.from_pretrained`.
- Added a `ValueError` that is raised within the `wsd_torch_models.bem.BEM.predict` when the number of predicted sense labels does not equal the number of tokens that were given that should have a predicted sense label.
- Added `add_prefix_space=True` argument to the `AutoTokenizer.from_pretrained` method for all examples in the `README.md`, `scripts/convert_and_upload_bem_model.py`, and `model_readmes/pymusas_bem.md`. This is required as this is what the pre-trained `BEM` models expect.
- The devcontainers, found in `.devcontainer`, have been improved so that they use the cached uv packages that have been installed at docker build time.

## [v0.1.0](https://github.com/UCREL/WSD-Torch-Models/releases/tag/v0.1.0) - 2025-12-02

### Added

- First release.
- The Bi-Encoder Model (BEM) from the paper [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://aclanthology.org/2020.acl-main.95.pdf). This model can be found at `wsd_torch_models.bem.BEM`
- The `wsd_torch_models.bem.BEM` class represents a good potential blueprint (abstract class) for other Word Sense Disambiguation methods to inherit from in the future through a parent class.
- Created a script, `scripts/convert_and_upload_bem_model.py`, that converts Pytorch Lightning models that the `wsd_torch_models.bem.BEM` class was created from to be converted into the Pytorch and PyTorchModelHubMixin class that the `wsd_torch_models.bem.BEM` class represents without the need for Pytorch Lightning dependency. This script only requires the checkpoint from the saved Pytorch Lightning model and it will convert the model as well as upload it to the relevant HuggingFace hub repository.
