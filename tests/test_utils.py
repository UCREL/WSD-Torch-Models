import typing

import pytest
import torch

from wsd_torch_models import utils


def test_tiny_value_of_dtype() -> None:
    valid_dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    for valid_dtype in valid_dtypes:
        assert isinstance(utils.tiny_value_of_dtype(valid_dtype), float)

    # Non float dtype
    with pytest.raises(TypeError):
        utils.tiny_value_of_dtype(torch.int32)

    # Not supported float dtype
    with pytest.raises(TypeError):
        utils.tiny_value_of_dtype(torch.float8_e4m3fn)


def test_get_linear_schedule_with_warmup() -> None:
    """
    Test taken from HuggingFace Transformers.

    Reference:
    https://github.com/huggingface/transformers/blob/c40b370bd01539ba9a05a35995a2fb5dc467f373/tests/optimization/test_optimization.py#L123
    """
    def unwrap_schedule(scheduler: torch.optim.lr_scheduler.LambdaLR,
                        num_steps: int) -> list[float]:
        lrs = []
        for _ in range(num_steps):
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        return typing.cast(list[float], lrs)
    test_model = torch.nn.Linear(50, 50)
    test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=10.0)
    test_scheduler = utils.get_linear_schedule_with_warmup(test_optimizer, 2, 10)
    assert [0.0] == test_scheduler.get_lr()
    
    expected_lrs = [0.0, 5.0, 10.0, 8.75, 7.5, 6.25, 5.0, 3.75, 2.5, 1.25]

    assert expected_lrs == unwrap_schedule(test_scheduler, len(expected_lrs))
