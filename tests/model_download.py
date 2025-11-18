import argparse
import logging
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    description = (
        "Downloads the pre-trained model and tokenizer that is used in testing"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("cache_directory", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cache_directory = args.cache_directory
    assert isinstance(cache_directory, Path)

    model = "jhu-clsp/ettin-encoder-17m"
    logger.info(f"Caching model; {model} and tokenizer too: {cache_directory}")
    AutoModel.from_pretrained(model, cache_dir=cache_directory)
    AutoTokenizer.from_pretrained(model, cache_dir=cache_directory)  # type: ignore
    logger.info(f"Download; {model}")
