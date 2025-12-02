# WSD Torch Models

This repository contains the code for the Word Sense Disambiguation (WSD) PyTorch models that have been trained and developed by the [UCREL NLP Group](https://ucrel.lancs.ac.uk/) at Lancaster University, UK.

## Installation

Requires Python `3.10` or greater, it is best that you install the version of PyTorch you would like to use, e.g. CPU/GPU version etc before installing this package else you will get the default version of PyTorch for your operating system/setup, but we do require `torch>=2.2,<3.0`.

``` bash
pip install wsd-torch-models
```

## Models with examples

Here we list the various WSD models we have implemented and how to use them.

### Bi-Encoder Model (BEM)

An inference only implementation of the Bi-Encoder Model (BEM) for Word Sense Disambiguation from the paper [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://aclanthology.org/2020.acl-main.95.pdf). This is a bi-encoder model whereby it encodes the word(s) to disambiguate using the word(s) text context, e.g. whole sentence or document and it will encode every sense definition given and return the most similar sense definition for the given word(s). Unlike the original BEM model we use the same model to encode both the text to disambiguate and the label definitions.

These models were trained using the code from the following GitHub repository [https://github.com/UCREL/experimental-wsd](https://github.com/UCREL/experimental-wsd) and ported over to this library for inference only use with easy saving and loading from the HuggingFace hub.

We currently have 4 pre-trained BEM models that predict sense labels from the [USAS](https://ucrel.lancs.ac.uk/usas/usas_guide.pdf) sense inventory which contains 232 sense categories, which in comparison to WordNet is very coarse (WordNet has approximately 117,000 senses), more details about these models and how they were trained can be found in our forthcoming paper:

* [ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM](https://huggingface.co/ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM) - 17 million parameter English only model.
* [ucrelnlp/PyMUSAS-Neural-Engish-Base-BEM](https://huggingface.co/ucrelnlp/PyMUSAS-Neural-Engish-Base-BEM) - 68 million parameter English only model.
* [ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM](https://huggingface.co/ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM) - 140 million parameter Multilingual model.
* [ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM](https://huggingface.co/ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM) - 307 million parameter Multilingual model.

Of which an example of how to run them can be found below, this particular example uses the Small English model:

``` python
from transformers import AutoTokenizer
import torch

from wsd_torch_models.bem import BEM


if __name__ == "__main__": 
    wsd_model_name = "ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM"
    wsd_model = BEM.from_pretrained(wsd_model_name)
    tokenizer = AutoTokenizer.from_pretrained(wsd_model_name)

    wsd_model.eval()
    # Change this to the device you would like to use, e.g. cpu
    model_device = "cpu"
    wsd_model.to(device=model_device)
    
    sentence = "The river bank was full of fish"
    sentence_tokens = sentence.split()
    
    with torch.inference_mode(mode=True):
        # sub_word_tokenizer can be None when None it will download the appropriate tokenizer
        # but generally it is better to give it the tokenizer as it saves the operation
        # of checking if the tokenizer is already downloaded.
        predictions = wsd_model.predict(sentence_tokens, sub_word_tokenizer=tokenizer, top_n=5)
        
        for sentence_token, semantic_tags in zip(sentence_tokens, predictions):
            print(f"Token: {sentence_token}")
            print("Most likely tags: ")
            for tag in semantic_tags:
                tag_definition = wsd_model.label_to_definition[tag]
                print(f"\t{tag}: {tag_definition}")
            print()
```
<details>
<summary>Output from running the code above:</summary>

``` bash
Token: The
Most likely tags: 
        Z5: title: Grammatical bin description: Prepositions/adverbs/conjunctions, etc
        Z3: title: Other proper names description: Nouns that distinguish/identify a product, company, etc. (note – also includes acronyms)
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        Z2: title: Geographical names description: Nouns that distinguish/identify a specific place (e.g. the name of a road, a city, a country, a continent, etc.)
        A7: title: Definite (+ modals) description: Abstract terms of modality (possibility, necessity, certainty, etc.)

Token: river
Most likely tags: 
        M4: title: Means of transport (Water) description: Terms depicting means of transport/ways of transporting and/or travelling (by water)
        W3: title: Geographical terms description: Geographical terms
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        L2: title: Living creatures generally description: Terms relating to living creatures (e.g. non-human)
        Z2: title: Geographical names description: Nouns that distinguish/identify a specific place (e.g. the name of a road, a city, a country, a continent, etc.)

Token: bank
Most likely tags: 
        M4: title: Means of transport (Water) description: Terms depicting means of transport/ways of transporting and/or travelling (by water)
        I1: title: Money generally description: Terms relating to money generally
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        Z2: title: Geographical names description: Nouns that distinguish/identify a specific place (e.g. the name of a road, a city, a country, a continent, etc.)
        W3: title: Geographical terms description: Geographical terms

Token: was
Most likely tags: 
        M4: title: Means of transport (Water) description: Terms depicting means of transport/ways of transporting and/or travelling (by water)
        W3: title: Geographical terms description: Geographical terms
        M3: title: Means of transport (Land) description: Terms depicting means of transport/ways of transporting and/or travelling (on land)
        K6: title: Children’s games and toys description: Terms relating to children’s games and toys
        H1: title: Architecture & kinds of houses & buildings description: Terms relating to buildings/habitats of various kinds, and their construction

Token: full
Most likely tags: 
        M4: title: Means of transport (Water) description: Terms depicting means of transport/ways of transporting and/or travelling (by water)
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        W3: title: Geographical terms description: Geographical terms
        L3: title: Plants description: Terms relating to plants and plant-life
        Z3: title: Other proper names description: Nouns that distinguish/identify a product, company, etc. (note – also includes acronyms)

Token: of
Most likely tags: 
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        Z3: title: Other proper names description: Nouns that distinguish/identify a product, company, etc. (note – also includes acronyms)
        Z2: title: Geographical names description: Nouns that distinguish/identify a specific place (e.g. the name of a road, a city, a country, a continent, etc.)
        O1.1: title: Substances and materials generally: Solid description: Terms depicting solid substances/materials
        L3: title: Plants description: Terms relating to plants and plant-life

Token: fish
Most likely tags: 
        L2: title: Living creatures generally description: Terms relating to living creatures (e.g. non-human)
        O1.1: title: Substances and materials generally: Solid description: Terms depicting solid substances/materials
        Z1: title: Personal names description: Nouns that distinguish/identify an individual (e.g. a first name and/or surname, a title of address)
        O2: title: Objects generally description: Terms relating to objects generally
        Z3: title: Other proper names description: Nouns that distinguish/identify a product, company, etc. (note – also includes acronyms)
```

</details>

**NOTE**: the pre-trained models we have released come with the sense definitions they have been trained to predict, USAS sense definitions, if you would like to use a different list/set of sense definitions please look at the `wsd_torch_models.bem.BEM.embed_and_set_label_definitions` method which will allow you to change the sense definitions the model will predict. We have not tested how well these models will perform on zero shot sense prediction, e.g. training on one sense inventory and predicting on data using a different sense inventory.

#### Training Data (BEM)

All of these models have been trained on a portion of the [ucrelnlp/English-USAS-Mosaico](https://huggingface.co/datasets/ucrelnlp/English-USAS-Mosaico), specifically [data/wikipedia_shard_0.jsonl.gz](https://huggingface.co/datasets/ucrelnlp/English-USAS-Mosaico/blob/main/data/wikipedia_shard_0.jsonl.gz), which contains 1,083 English Wikipedia articles, with 444,880 sentences, 6.6 million tokens, with 5.3 million silver labelled tokens generated by a English rule based semantic tagger.

#### Model Architecture (BEM)

| Parameter | 17M English | 68M English | 140M Multilingual | 307M Multilingual |
|:----------|:----|:----|:----|:-----|
| Layers | 7 | 19 | 22 | 22 |
| Hidden Size | 256 | 512 | 384 | 768 |
| Intermediate Size | 384 | 768 | 1152 | 1152 |
| Attention Heads | 4 | 8 | 6 | 12 |
| Total Parameters | 17M | 68M | 140M | 307M |
| Non-embedding Parameters | 3.9M | 42.4M | 42M | 110M |
| Max Sequence Length | 8,000 | 8,000 | 8,192 | 8,192 |
| Vocabulary Size | 50,368 | 50,368 | 256,000 | 256,000 |
| Tokenizer | ModernBERT | ModernBERT | Gemma 2 | Gemma 2 |

#### Evaluation (BEM)

We have evaluated the models on 5 datasets from 5 different languages, 4 of these datasets are publicly available whereas one (the Irish data) requires permission from the data owner to access it. The results for these models using top 1 and top 5 accuracy results are shown below, for a more comprehensive comparison please see the technical report.

| Dataset | 17M English | 68M English | 140M Multilingual | 307M Multilingual |
|:----------|:----|:----|:----|:-----|
| **Top 1** |  |  |  |  |
| Chinese | - | - | 42.2 | 47.9 |
| English | 66.4 | 70.1 | 66.0 | 70.2 |
| Finnish | - | - | 15.8 | 25.9 |
| Irish | - | - | 28.5 | 35.6 |
| Welsh | - | - | 21.7 | 42.0 |
| **Top 5** |  |  |  |  |
| Chinese | - | - | 66.3 | 70.4 |
| English | 87.6 | 90.0 | 88.9 | 90.1 |
| Finnish | - | - | 32.8 | 42.4 |
| Irish | - | - | 47.6 | 51.6 |
| Welsh | - | - | 40.8 | 56.4 |

The publicly available datasets can be found on HuggingFace Hub [ucrelnlp/USAS-WSD](https://huggingface.co/datasets/ucrelnlp/USAS-WSD).

**Note** the English models have not been evaluated on the non-English datasets as they are unlikely to be able to represent non-English text well or perform well on non-English data.

## Development

### Setup

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

#### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

#### Local

To run locally first ensure you have the following tools installed locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.9`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, including `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync
```

### Running linters and tests

This code base uses isort, flake8 and mypy to ensure that the format of the code is consistent and contain type hints. ISort and mypy settings can be found within [./pyproject.toml](./pyproject.toml) and the flake8 settings can be found in [./.flake8](./.flake8). To run these linters:

``` bash
make lint
```

To run the tests with code coverage (**NOTE** these are the code coverage tests that the Continuos Integration (CI) reports at the top of this README):

``` bash
make tests
```

### Setting a different default python version

The default or recommended Python version is shown in [.python-version](./.python-version, currently `3.13`, this can be changed using the [uv command](https://docs.astral.sh/uv/reference/cli/#uv-python-pin):

``` bash
uv python pin
# uv python pin 3.13
```

### Converting PyTorch Lightning Model to PyTorch HuggingFace Model

Some of the WSD models were originally trained using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), this section details how we convert these models to PyTorch models with a [HuggingFace Pytorch Model Hub Mixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin), the mixin allows the model to easily be loaded and saved from and to the HuggingFace hub, and then uploads these converted models to HuggingFace Hub.


#### PyMUSAS BEM models

The scripts has various arguments of which these are detailed in the help section of the script:

``` bash
usage: convert_and_upload_bem_model.py [-h] [-r] [-t] [-m] hf_repository_id hf_branch model_checkpoint readme_template_path

Converts a PyTorch Lightning model to a PyTorch HuggingFace model and uploads it to the HuggingFace Hub. The script allows you to just update the model README, model tokenizer, the model itself, or any combinationof these options.

positional arguments:
  hf_repository_id      The repository ID to upload the model too on the HuggingFace Hub, e.g. ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM
  hf_branch             The branch to upload the model too on the HuggingFace Hub, e.g. main, a branch named after the step the model was trained on.
  model_checkpoint      Path to the model checkpoint that you would like to upload
  readme_template_path  File path to the models README template

options:
  -h, --help            show this help message and exit
  -r, --update-readme   update model README
  -t, --update-tokenizer
                        update model tokenizer
  -m, --update-model    update model
```

To upload the model, tokenizer and README for all 4 models to the main branch:

``` bash
uv run scripts/convert_and_upload_bem_model.py ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM main checkpoints/bem_english_small/checkpoints/bem_english_small/model-step=532637-validation_accuracy=0.99394.ckpt model_readmes/pymusas_bem.md -rmt

uv run scripts/convert_and_upload_bem_model.py ucrelnlp/PyMUSAS-Neural-Engish-Base-BEM main checkpoints/bem_english_base/model-step=532637-validation_accuracy=0.99669.ckpt model_readmes/pymusas_bem.md -rmt

uv run scripts/convert_and_upload_bem_model.py ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM main checkpoints/bem_multilingual_small/model-step=392261-validation_accuracy=0.99615.ckpt model_readmes/pymusas_bem.md -rmt

uv run scripts/convert_and_upload_bem_model.py ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM main checkpoints/bem_multilingual_base/model-step=240947-validation_accuracy=0.99625.ckpt model_readmes/pymusas_bem.md -rmt
```

To upload only an updated/new README:

``` bash
uv run scripts/convert_and_upload_bem_model.py ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM main checkpoints/bem_english_small/model-step=532637-validation_accuracy=0.99394.ckpt model_readmes/pymusas_bem.md -r
```

### Python packages that can be removed and replaced

As of Python version `3.11`:
* `from typing_extensions import Self` - the `typing_extensions` package can be removed and this can be replaced with `from typing import Self`

## Citation

Technical report is forthcoming.

## Contact Information

* Paul Rayson (p.rayson@lancaster.ac.uk)
* Andrew Moore (a.p.moore@lancaster.ac.uk / andrew.p.moore94@gmail.com)
* UCREL Research Centre (ucrel@lancaster.ac.uk) at Lancaster University.