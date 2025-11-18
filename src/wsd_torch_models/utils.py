import functools

import torch


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given floating point PyTorch
    data type that is used to avoid numerical issues such as division by zero.


    This is different from `torch.finfo(dtype).tiny` because it causes some NaN bugs.

    Reference:
    https://github.com/MaksymDel/allennlp-light/blob/main/allennlp_light/nn/util.py#L1999

    Args:
        dtype (torch.dtype): A torch float dtype.
    Returns:
        A small float value that is not too small to causes any NaN bugs.
    Raises:
        TypeError: If the dtype is not one of the following: torch.float64,
            torch.float32, torch.float16, or torch.bfloat16
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float32 or dtype == torch.float64 or dtype == torch.bfloat16:
        return 1e-13
    elif dtype == torch.float16:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def get_linear_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    last_epoch: int = -1) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Reference:
    https://github.com/huggingface/transformers/blob/a5923d4de7df2fbd1f373dfcfe983216b79b6937/src/transformers/optimization.py#L105

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    lr_lambda = functools.partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
