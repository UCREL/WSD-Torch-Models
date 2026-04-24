import argparse
import logging
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from wsd_torch_models.bem import BEM

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    description = (
        "Downloads the pre-trained models and tokenizers that is used in testing"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("cache_directory", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cache_directory = args.cache_directory
    assert isinstance(cache_directory, Path)

    models = ["jhu-clsp/ettin-encoder-17m", "ucrelnlp/PyMUSAS-Neural-English-Small-BEM"]
    for model in models:
        logger.info(f"Caching model; {model} and tokenizer too: {cache_directory}")
        if model == "jhu-clsp/ettin-encoder-17m":
            AutoModel.from_pretrained(model, cache_dir=cache_directory)
        else:
            BEM.from_pretrained(model, cache_dir=cache_directory)
        AutoTokenizer.from_pretrained(model, cache_dir=cache_directory)
        logger.info(f"Download; {model}")
