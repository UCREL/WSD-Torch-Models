# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
# Reference from AllenNLP Light: https://github.com/MaksymDel/allennlp-light/blob/main/allennlp_light/modules/scalar_mix.py
from typing import List, Optional

import torch

from wsd_torch_models.utils import tiny_value_of_dtype


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of `N` tensors,
    `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.

    In addition, if `do_layer_norm=True` then apply layer normalization
    to each tensor before weighting.

    This ScalarMix came out of the
    [Deep contextualized word representations](https://aclanthology.org/N18-1202.pdf)
    (ELMo) see equation 1 on page 3.
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: Optional[List[float]] = None,
        trainable: bool = True,
    ) -> None:
        """
        Args:
            mixture_size (int): The number of tensors that will be mixed (`N`).
            do_layer_norm (bool): Whether to apply layer normalisation to each
                tensor before weighting.
            initial_scalar_parameters (Optional[List[float]]): Initial scalar
                parameters for the mixing on the `N` tensors.
            trainable (bool): Whether the scalar parameters should be trained.
                Default True.

        Returns:
            None

        Raises:
            RuntimeError: If the number of `initial_scalar_parameters` in the
                list is different to the `mixture_size`.
        """
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise RuntimeError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters, mixture_size)
            )

        self.scalar_parameters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]),
                    requires_grad=trainable,
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(
        self, tensors: List[torch.Tensor], mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`. The input tensors can be any shape
        with at least two dimensions, but must all be the same shape.

        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.

        When `do_layer_norm=False` the `mask` is ignored.

        Args:
            tensors (List[torch.Tensor]): Tensors to compute average weightings
                from. Should be a list of Tensors with the same share whereby
                each tensor should be at least two dimensions. The list must
                be of the same length as `self.mixture_size`
            mask (Optional[torch.BoolTensor]): A mask to determine which
                values to be ignored when calculating layer normalisation.
                Default None.

        Returns:
            torch.Tensor: A tensor of the same shape as one of the tensors in
                the `tensors` input.

        Raises:
            RuntimeError: If the length of `tensors` is not the same as
                `self.mixture_size`.
            AssertionError: If `mask` is `None` and `self.do_layer_norm` is
                True.
            
        """
        if len(tensors) != self.mixture_size:
            raise RuntimeError(
                "{} tensors were passed, but the module was initialized to "
                "mix {} tensors.".format(len(tensors), self.mixture_size)
            )

        def _do_layer_norm(tensor: torch.Tensor,
                           broadcast_mask: torch.Tensor,
                           num_elements_not_masked: torch.Tensor
                           ) -> torch.Tensor:
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(
                variance + tiny_value_of_dtype(variance.dtype)
            )

        scalar_normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        split_scalar_normed_weights = torch.split(scalar_normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(split_scalar_normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            assert mask is not None
            broadcast_mask = mask.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask) * input_dim

            pieces = []
            for weight, tensor in zip(split_scalar_normed_weights, tensors):
                pieces.append(
                    weight
                    * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return self.gamma * sum(pieces)
