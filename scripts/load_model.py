import argparse
from pathlib import Path

from huggingface_hub import ModelCard
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from wsd_torch_models.bem import BEM
from wsd_torch_models.data_utils import load_usas_mapper


def generate_bem_readme(readme_template_path: Path) -> str:
    with readme_template_path.open("r", encoding="utf-8") as fp:
        content = fp.read()
        return content



if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("readme_template_path", type=Path, help="File path to the models README template")
    args = parser.parse_args()
    content = generate_bem_readme(args.readme_template_path)

    usas_tags_to_filter_out = set([
        "Z99"
    ])
    hf_repoistory_id = "ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM"
    hf_repoistory_branch = "main"
    repoistory_type = "model"
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
    import pdb
    pdb.set_trace()
    raise ValueError()
    wsd_model.load_state_dict(torch_lightning_checkpoint["state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(hyper_parameters_dict["base_model_name"])
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    label_definitions =load_usas_mapper(tags_to_filter_out=usas_tags_to_filter_out)
    wsd_model.embed_and_set_label_definitions(label_definitions, tokenizer)
    
    
    wsd_model.push_to_hub(hf_repoistory_id, branch=hf_repoistory_branch)
    tokenizer.push_to_hub(hf_repoistory_id, revision=hf_repoistory_branch)
    card = ModelCard(content)
    card.push_to_hub(hf_repoistory_id, repo_type=repoistory_type, revision=hf_repoistory_branch)
