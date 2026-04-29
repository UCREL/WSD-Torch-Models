import logging
from typing import Annotated

import typer
from transformers import AutoTokenizer

from wsd_torch_models.bem import BEM

logger = logging.getLogger(__name__)


def main(huggingface_model_id: Annotated[str, typer.Argument(help="The HuggingFace ID of the model and tokenizer to download")]) -> None:
    """
    Downloads the pre-trained model and tokenizer for the given HuggingFace Model ID
    """
    logger.info(f"Downloading model; {huggingface_model_id} and tokenizer.")
    BEM.from_pretrained(huggingface_model_id)
    AutoTokenizer.from_pretrained(huggingface_model_id)
    logger.info(f"Download; {huggingface_model_id}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
    
