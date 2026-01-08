import argparse
import logging
from pathlib import Path

from huggingface_hub import ModelCard, model_info, create_branch, list_repo_refs
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from wsd_torch_models.bem import BEM
from wsd_torch_models.data_utils import load_usas_mapper


logger = logging.getLogger(__name__)


def generate_bem_readme(readme_template_path: Path,
                        base_model_id: str,
                        languages: str,
                        model_title: str,
                        model_size: str,
                        base_model_language: str,
                        model_id: str) -> str:
    """
    Gets the readme template and formats it with the model information and
    returns the formatted readme.

    Args:
        readme_template_path (Path): The path to the readme template
        base_model_id (str): The base model id, e.g.
            jhu-clsp/ettin-encoder-17m
        languages (str): The languages supported by the model, e.g.
            `\n- en\n- es`
        model_title (str): The title of the model, e.g.
            PyMUSAS Neural English Small BEM
        model_size (str): The size of the model, e.g.
            17 Million (17M)
        base_model_language (str): The language of the base model, e.g.
            English or Multilingual
        model_id (str): The model id, e.g.
            ucrelnlp/PyMUSAS-Neural-English-Small-BEM

    Returns:
        str: The formatted readme
    """
    with readme_template_path.open("r", encoding="utf-8") as fp:
        content = fp.read().format(
            base_model_id=base_model_id,
            languages=languages,
            model_title=model_title,
            model_size=model_size,
            base_model_language=base_model_language,
            model_id=model_id)
        return content



if __name__ == "__main__":

    description = (
        "Converts a PyTorch Lightning model to a PyTorch HuggingFace model and "
        "uploads it to the HuggingFace Hub. The script allows you to just "
        "update the model README, model tokenizer, the model itself, or any combination"
        "of these options."
    )

    hf_repository_id_help = (
        "The repository ID to upload the model too on the HuggingFace Hub, e.g. "
        "ucrelnlp/PyMUSAS-Neural-English-Small-BEM"
    )
    hf_branch_help = (
        "The branch to upload the model too on the HuggingFace Hub, e.g. main, "
        "a branch named after the step the model was trained on. If the branch "
        "does not exist in the model repository, the branch is created before "
        "uploading the model to it."
    )
    model_checkpoint_help = (
        "Path to the model checkpoint that you would like to upload"
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("hf_repository_id", type=str, help=hf_repository_id_help)
    parser.add_argument("hf_branch", type=str, help=hf_branch_help)
    parser.add_argument("model_checkpoint", type=Path, help=model_checkpoint_help)
    parser.add_argument("readme_template_path", type=Path,
                        help="File path to the models README template")
    parser.add_argument("-r", "--update-readme", action="store_true", help="update model README")
    parser.add_argument("-t", "--update-tokenizer", action="store_true", help="update model tokenizer")
    parser.add_argument("-m", "--update-model", action="store_true", help="update model")
    args = parser.parse_args()

    hf_repository_id = args.hf_repository_id
    assert isinstance(hf_repository_id, str)

    hf_branch = args.hf_branch
    assert isinstance(hf_branch, str)

    model_checkpoint = args.model_checkpoint
    assert isinstance(model_checkpoint, Path)

    readme_template_path = args.readme_template_path
    assert isinstance(readme_template_path, Path)

    update_readme = args.update_readme
    assert isinstance(update_readme, bool)

    update_tokenizer = args.update_tokenizer
    assert isinstance(update_tokenizer, bool)

    update_model = args.update_model
    assert isinstance(update_model, bool)

    usas_tags_to_filter_out = set([
        "Z99"
    ])

    logging.basicConfig(level=logging.INFO)

    supported_model_repositories = set([
        "ucrelnlp/PyMUSAS-Neural-English-Small-BEM",
        "ucrelnlp/PyMUSAS-Neural-English-Base-BEM",
        "ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM",
        "ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM",
    ])

    if hf_repository_id not in supported_model_repositories:
        raise ValueError(f"The repository ID {hf_repository_id} is not supported"
                         "by this script. Supported repository IDs are: "
                         f"{supported_model_repositories}")

    logger.info(f"HF repository ID: {hf_repository_id}")
    logger.info(f"HF branch: {hf_branch}")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Readme template path: {readme_template_path}")
    logger.info(f"Update readme: {update_readme}")
    logger.info(f"Update tokenizer: {update_tokenizer}")
    logger.info(f"Update model: {update_model}")
    logger.info(f"Filtering out tags: {usas_tags_to_filter_out}")
        

    
    
    repository_type = "model"
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

    tokenizer = AutoTokenizer.from_pretrained(hyper_parameters_dict["base_model_name"],
                                              add_prefix_space=True)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    if hf_branch != "main":
        for branch in list_repo_refs(hf_repository_id).branches:
            if branch.name == hf_branch:
                logger.info(f"Branch {hf_branch} already exists")
                break
        else:
            logger.info(f"Creating branch {hf_branch}")
            create_branch(hf_repository_id, branch=hf_branch, exist_ok=False)

    if update_model:
        logger.info("Updating model and uploading to the HuggingFace Hub")
        wsd_model = BEM(**hyper_parameters_dict)
        wsd_model.load_state_dict(torch_lightning_checkpoint["state_dict"])
        
        label_definitions = load_usas_mapper(tags_to_filter_out=usas_tags_to_filter_out)
        wsd_model.embed_and_set_label_definitions(label_definitions, tokenizer)
        
        
        wsd_model.push_to_hub(hf_repository_id, branch=hf_branch)
    
    if update_tokenizer:
        logger.info("Updating tokenizer and uploading to the HuggingFace Hub")
        tokenizer.push_to_hub(hf_repository_id, revision=hf_branch)

    if update_readme:
        logger.info("Updating README and uploading to the HuggingFace Hub")
        wsd_base_model_id = hyper_parameters_dict["base_model_name"]
        base_model_data = model_info(wsd_base_model_id)
        wsd_model_languages = base_model_data.card_data.language
        wsd_model_languages_yaml_format = "\n- " + "\n- ".join(wsd_model_languages)

        model_title: str | None = None
        model_size: str | None = None
        base_model_language: str | None = None

        if hf_repository_id == "ucrelnlp/PyMUSAS-Neural-English-Small-BEM":
            model_title = "PyMUSAS Neural English Small BEM"
            model_size = "17 Million (17M)"
            base_model_language = "English"
        elif hf_repository_id == "ucrelnlp/PyMUSAS-Neural-English-Base-BEM":
            model_title = "PyMUSAS Neural English Base BEM"
            model_size = "68 Million (68M)"
            base_model_language = "English"
        elif hf_repository_id == "ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM":
            model_title = "PyMUSAS Neural Multilingual Small BEM"
            model_size = "140 Million (140M)"
            base_model_language = "Multilingual"
        elif hf_repository_id == "ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM":
            model_title = "PyMUSAS Neural Multilingual Base BEM"
            model_size = "307 Million (307M)"
            base_model_language = "Multilingual"

        if model_title is None or model_size is None or base_model_language is None:
            raise ValueError(f"The repository ID {hf_repository_id} is not supported"
                             "by this script for generating a README")

        content = generate_bem_readme(readme_template_path, 
                                      base_model_id=wsd_base_model_id,
                                      languages=wsd_model_languages_yaml_format,
                                      model_title=model_title,
                                      model_size=model_size,
                                      base_model_language=base_model_language,
                                      model_id=hf_repository_id
                                      )
        card = ModelCard(content)
        card.push_to_hub(hf_repository_id, repo_type=repository_type, revision=hf_branch)

    logger.info("Done")
