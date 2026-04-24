import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from transformers import AutoTokenizer

from wsd_torch_models.bem import BEM


@pytest.fixture
def test_data_dir() -> Path:
    return (Path(__file__).parent / "data").resolve()


@pytest.fixture
def large_corpus(test_data_dir: Path) -> Path:
    return test_data_dir / "large_corpus.txt"


@pytest.fixture
def large_corpus_tagged(test_data_dir: Path) -> Path:
    return test_data_dir / "large_corpus_tags.txt"


def cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    # Returns

    `bool`
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bem_tagger(device: Literal["cpu", "cuda"]) -> None:
    """
    This is only to test the `ucrelnlp/PyMUSAS-Neural-English-Small-BEM`. It is
    mainly to check that changes from Python package updates do not change the
    output of the model or degrade the model in anyway. Packages that could do this
    are likely to be `torch` or `transformers`. In addition it will allow us to
    check that using a different implementation of attention does not affect results.
    """

    if not cuda_available() and device == "cuda":
        pytest.skip("CUDA not available")

    test_text = "The river bank was full of fish"
    test_tokens = test_text.split()
    
    wsd_model_name = "ucrelnlp/PyMUSAS-Neural-English-Small-BEM"
    wsd_model = BEM.from_pretrained(wsd_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(wsd_model_name, add_prefix_space=True)
    output_tokens = tokenizer(test_tokens, padding=False, truncation=False, is_split_into_words=True)
    expected_token_ids: list[int] = [50281, 380, 8281, 4310, 369, 2120, 273, 6773, 50282]
    assert output_tokens.input_ids == expected_token_ids
    assert output_tokens.attention_mask == [1] * len(expected_token_ids)

    wsd_model.eval()
    wsd_model.to(device=device)
    
    expected_semantic_tags = [
        ["Z5", "N5", "Z3", "Z2", "A11.1"],
        ["M4", "W3", "N5", "Z2", "M3"],
        ["M4", "W3", "I2.1", "A9", "I1"],
        ["A3", "Z5", "A5.1", "A6.2", "O4.2"],
        ["N5.1", "I3.2", "B1", "K5.1", "I3.1"],
        ["Z5", "N5", "A9", "I1.1", "I2.1"],
        ["L2", "F1", "S2", "O2", "F2"]
    ]
    with torch.inference_mode(mode=True):
        predictions = wsd_model.predict(test_tokens, sub_word_tokenizer=tokenizer, top_n=5)
        assert len(predictions) == len(expected_semantic_tags)
        
        for token_index, (token, semantic_tags) in enumerate(zip(test_tokens, predictions)):
            assert token == test_tokens[token_index]
            assert semantic_tags == expected_semantic_tags[token_index]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_on_large_corpus(device: Literal["cpu", "cuda"],
                         large_corpus: Path,
                         large_corpus_tagged: Path) -> None:
    """
    This is very similar to the `test_bem_tagger` test but it uses a corpus that
    contains 3,269 tokens.

    The tagged corpus was created using version v0.1.2 of this codebase with
    `ucrelnlp/PyMUSAS-Neural-English-Small-BEM` on the 24th of April 2026, it
    was created using a CPU but tested to ensure it was the same as the output
    from CUDA.
    """

    if not cuda_available() and device == "cuda":
        pytest.skip("CUDA not available")

    wsd_model_name = "ucrelnlp/PyMUSAS-Neural-English-Small-BEM"
    wsd_model = BEM.from_pretrained(wsd_model_name)
    tokenizer = AutoTokenizer.from_pretrained(wsd_model_name, add_prefix_space=True)
    output: list[list[list[str]]] = []
    with torch.inference_mode(mode=True):
        with large_corpus.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                predictions = wsd_model.predict(tokens, sub_word_tokenizer=tokenizer, top_n=1)
                output.append(predictions)
    
    expected_output_line_count = 0
    with large_corpus_tagged.open("r", encoding="utf-8") as fp:
        for line_index, line in enumerate(fp):
            assert json.loads(line) == output[line_index]
            expected_output_line_count += 1
    assert len(output) == expected_output_line_count
