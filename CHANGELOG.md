# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- First release.
- The Bi-Encoder Model (BEM) from the paper [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://aclanthology.org/2020.acl-main.95.pdf). This model can be found at `wsd_torch_models.bem.BEM`
- The `wsd_torch_models.bem.BEM` class represents a good potential blueprint (abstract class) for other Word Sense Disambiguation methods to inherit from in the future through a parent class.
- Created a script, `scripts/convert_and_upload_bem_model.py`, that converts Pytorch Lightning models that the `wsd_torch_models.bem.BEM` class was created from to be converted into the Pytorch and PyTorchModelHubMixin class that the `wsd_torch_models.bem.BEM` class represents without the need for Pytorch Lightning dependency. This script only requires the checkpoint from the saved Pytorch Lightning model and it will convert the model as well as upload it to the relevant HuggingFace hub repository.
