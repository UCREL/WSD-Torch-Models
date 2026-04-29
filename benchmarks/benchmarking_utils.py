import json
import os
import warnings
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

import psutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from wsd_torch_models.bem import BEM


class Languages(str, Enum):
    english = "English"
    multilingual = "Multilingual"


class TaggerSizes(str, Enum):
    small = "Small"
    base = "Base"

@contextmanager
def track_memory_usage(memory_statistic_name: str,
                       gpu_memory_statistic_name: str,
                       memory_statistics: dict[str, float | int | str],
                       device: str) -> Iterator[None]:
    """
    Context manager to track memory usage of the enclosed code.

    The RAM is tracked using the psutil library, specifically `psutil.virtual_memory`.

    The GPU memory is tracked using the torch library, specifically `torch.cuda.memory.max_memory_allocated`
    of which the max memory peak is reset before the enclosed code is executed.

    Args:
        memory_statistic_name (str): name of the key to store the memory statistic
            too in the memory_statistics dictionary.
        gpu_memory_statistic_name (str): name of the key to store the GPU memory statistic
            too in the memory_statistics dictionary.
        memory_statistics (dict[str, float | int | str]]): dictionary to store the memory statistics
        device (str): If `cuda` then GPU memory will also be tracked else the
            reported value will be 0.0 for `gpu_memory_statistic_name`.

    Yields:
        None
    """
    if device == "cuda":
        torch.cuda.memory.reset_peak_memory_stats()

    memory_checkpoint = psutil.virtual_memory().used
    yield None
    memory_used = psutil.virtual_memory().used - memory_checkpoint
    memory_used_mb = memory_used / (1024 ** 2)
    memory_statistics[memory_statistic_name] = round(memory_used_mb, 2)
    
    if device == "cuda":
        gpu_memory_used = torch.cuda.memory.max_memory_allocated() / (1024 ** 2)
        memory_statistics[gpu_memory_statistic_name] = round(gpu_memory_used, 2)

def load_tagger(model_id: str, device: str) -> tuple[BEM, PreTrainedTokenizerBase]:
    """
    Loads the tagger and tokenizer for the given `model_id`, whereby the model
    will be loaded to the given device.

    Args:
        model_id: The HuggingFace model ID for the model to load and return.
        device (str): The device to load the model too. This should be a
            [torch device string like cpu or cuda](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device).

    Returns:
        The model and the tokenizer for the given model id.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wsd_model = BEM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        wsd_model.to(device)
        wsd_model.eval()
        return (wsd_model, tokenizer)


def wikipedia_dataset_to_directory(huggingface_dataset_id: str,
                                   directory: str,
                                   file_prefix: str,
                                   number_tokens: int) -> int:
    """
    Saves a subset of English Wikipedia articles to the specified directory,
    whereby the number of articles saved is based on the number of tokens.

    Tokens come from a whitespace tokenizer and is only used for token counting.

    Args:
        huggingface_dataset_id (str): The Hugging Face dataset ID of the Wikipedia dataset, e.g. HuggingFaceFW/finewiki
        directory (str): The directory to which the files should be saved.
        file_prefix (str): The prefix of the file names. Each prefix is appended with a unique article number.
        number_tokens (int): The minimum number of tokens to be saved. Once the
            number of tokens is reached no more articles are saved.

    Returns:
        int: The number of tokens saved.
    """
    language_code = "en"
    split = "train"
    wikipedia_language_dataset = load_dataset(huggingface_dataset_id,
                                              language_code,
                                              split=split,
                                              streaming=True,
                                              columns=["text"])
    article_count = 0
    token_count = 0
    for object in wikipedia_language_dataset:
        text = object["text"]
        # We skip any article that contains a table
        if "| -" in text:
            continue
        # Removes markdown headers
        text = text.replace("#", "")
        # Tried to remove markdown lists, but I think this creates a worse format
        # text = re.sub(r"\s*-\s+", "", text)
        token_count += len(text.split())
        article_count += 1

        temp_file = Path(directory, f"{file_prefix}{article_count}")
        with temp_file.open("w", encoding="utf-8") as f:
            f.write(text)
        if token_count > number_tokens:
            break
    return token_count


def text_from_files(file_directory: Path,
                    file_prefix: str) -> Iterable[str]:
    """
    Yields lines of non empty text from files in a directory whereby the file
    names start with the given file prefix.

    All lines of text are stripped of leading and trailing whitespace.

    Args:
        file_directory (Path): The directory to read files from.
        file_prefix (str): The prefix of the file names to read.

    Yields:
        An iterable of strings, where each string is a non-empty line from
        one of the files with leading and trailing whitespace stripped.
    """
    for file in file_directory.iterdir():
        if file.name.startswith(file_prefix):
            with file.open("r", encoding="utf-8") as file_fp:
                for line in file_fp:
                    line = line.strip()
                    if line:
                        yield line


def tagger_speed_test(neural_model: BEM,
                      sub_word_tokenizer: PreTrainedTokenizerBase,
                      wikipedia_data_directory: Path,
                      file_prefix: str,
                      max_texts: int = -1) -> None:
    """
    Tests the speed of a neural model.
    The speed test is performed on the given Wikipedia dataset text files, whereby the
    tagger has to tag all tokens in each file.

    Args:
        neural_model: The neural model that will tag the text.
        sub_word_tokenizer: The tokenizer for the neural model.
        wikipedia_data_directory (Path): The directory containing the Wikipedia
            dataset text files.
        file_prefix (str): The prefix of the file names to read from the directory.
        max_texts (int): The maximum number of Wikipedia articles to process.
            If -1 then all Wikipedia articles are processed. Defaults to -1.

    Returns:
        None
    """
    with torch.inference_mode(mode=True):
        if max_texts == -1:
            for text in text_from_files(wikipedia_data_directory, file_prefix):
                tokens = text.split()
                neural_model.predict(tokens, sub_word_tokenizer=sub_word_tokenizer, top_n=5)
        else:
            for text_index, text in enumerate(text_from_files(wikipedia_data_directory, file_prefix)):
                tokens = text.split()
                neural_model.predict(tokens, sub_word_tokenizer=sub_word_tokenizer, top_n=5)
                if text_index == max_texts:
                    break


def to_json_file(path: Path,
                 data: dict[str, int | float | str]
                 ) -> None:
    """
    Writes data to a JSON file, whereby the values of the data will be saved in
    a list, thus allowing future data to be appended to it, if required.

    If the file already exists, it will append the given data to the existing data.
    The keys of the existing data and the new data must match, otherwise a KeyError is raised.

    Args:
        path (Path): The path to the JSON file.
        data (dict[str, int | float | str]): The data to write to the JSON file.

    Returns:
        None

    Raises:
        KeyError: If the keys of the existing data and the new data do not match.
    """
    data_with_list_values = {
        key: [value] for key, value in data.items()
    }
    if path.exists():
        # First check that the file is not empty
        if os.stat(str(path.resolve())).st_size != 0:
            with path.open("r", encoding="utf-8") as json_fp:
                additional_data = json.load(json_fp)
                if additional_data.keys() != data.keys():
                    raise KeyError(f"The keys of the existing JSON data in {path} "
                                   "does not match the keys of the new data.")
                for key, value in additional_data.items():
                    data_with_list_values[key] = value + [data[key]]
    with path.open("w", encoding="utf-8") as json_fp:
        json.dump(data_with_list_values, json_fp)