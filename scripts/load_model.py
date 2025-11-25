from pathlib import Path

from safetensors.torch import save_file
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import ModelCard

from wsd_torch_models.bem import BEM


if __name__ == "__main__": 
    model_checkpoint = Path("./checkpoints/model-step=532637-validation_accuracy=0.99394.ckpt")
    torch_lightning_checkpoint = torch.load(model_checkpoint)

    hyper_parameters_keys = [
        "base_model_name",
        "freeze_base_model",
        "number_transformer_encoder_layers",
        "add_scalar_mixer",
        "scalar_mix_layer_norm",
        "transformer_encoder_hidden_dim",
        "transformer_encoder_num_heads",
        "batch_first"
    ]
    hyper_parameters_dict = {
        hyper_parameter_key: torch_lightning_checkpoint["hyper_parameters"][hyper_parameter_key]
        for hyper_parameter_key in hyper_parameters_keys
    }
    wsd_model = BEM(**hyper_parameters_dict)
    wsd_model.load_state_dict(torch_lightning_checkpoint["state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(hyper_parameters_dict["base_model_name"])
    
    hf_repoistory_id = "ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM"
    hf_repoistory_branch = "main"
    wsd_model.push_to_hub(hf_repoistory_id, branch=hf_repoistory_branch)
    tokenizer.push_to_hub(hf_repoistory_id, branch=hf_repoistory_branch)
    card = ModelCard(content)



      
  


"""
---
license: cc-by-nc-sa-4.0
base_model: jhu-clsp/ettin-encoder-17m
base_model_relation: finetune
datasets:
- ucrelnlp/English-USAS-Mosaico
language:
- en
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- pytorch
- word-sense-disambiguation
- lexical-semantics
---

# Model Card for PyMUSAS Neural English Small BEM

A fine tuned 17 million parameter English ModernBERT architecture semantic tagger. The tagger outputs semantic tags at the token level from the [USAS tagset](https://ucrel.lancs.ac.uk/usas/usas_guide.pdf).

The semantic tagger is a variation of the [Bi-Encoder Model (BEM) from Blevins and Zettlemoyer 2020](https://aclanthology.org/2020.acl-main.95.pdf) a Word Sense Disambiguation (WSD) model.

## Table of contents

## Quick start

### Installation

### Usage

## Model Description

### Model Sources

- Training Repository:
- Inference/Usage Respoistory:

### Model Architecture

| Parameter | 17M English | 68M English | 140M Multilingual | 307M Multilingual |
|:----------|:----|:----|:----|:-----|
| Layers | 7 | 19 | 22 | 22 |
| Hidden Size | 256 | 512 | 384 | 768 |
| Intermediate Size | 384 | 768 | 1152 | 1152 |
| Attention Heads | 4 | 8 | 6 | 12 |
| Total Parameters | 17M | 68M | 140M | 307M |
| Non-embedding Parameters | 42M | 110M |
| Max Sequence Length | 8,000 | 8,000 | 8,192 | 8,192 |
| Vocabulary Size | 50,368 | 50,368 | 256,000 | 256,000 |
| Tokenizer | ModernBERT | ModernBERT | Gemma 2 | Gemma 2 |

## Training Data

## Contact Information

The scripts described in this README allow you to train a variation of the [Bi-Encoder Model (BEM) from Blevins and Zettlemoyer 2020](https://aclanthology.org/2020.acl-main.95.pdf) a Word Sense Disambiguation (WSD) model. The only difference between the original and this version is that this version ties the weights of the context and gloss encoder. The model is trained to find the most relevant gloss/description for a given contextualised token that is to be disambiguated. The description comes from the semantic tagset, which in this case is USAS, whereby the description describes a semantic tag.

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: [More Information Needed]
- Paper: [More Information Needed]
- Docs: [More Information Needed]
"""