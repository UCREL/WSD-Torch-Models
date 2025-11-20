import math

import pytest
import torch
from transformers import PreTrainedModel

from wsd_torch_models.bem import BEM


class TestBEM:

    BASE_MODEL_NAME = "jhu-clsp/ettin-encoder-17m"

    @pytest.fixture
    def bem_model(self) -> BEM:
        bem_kwargs = {
            "base_model_name": self.BASE_MODEL_NAME,
            "freeze_base_model": True,
            "number_transformer_encoder_layers": 0,
            "add_scalar_mixer": False,
            "batch_first": True,
            "base_model": None
        }
        model = BEM(**bem_kwargs)  # type: ignore
        model.eval()
        return model

    def test__get_base_model(self) -> None:
        base_model = BEM._get_base_model(self.BASE_MODEL_NAME)
        assert isinstance(base_model, PreTrainedModel)
        total_number_parameters = 0
        for parameter in base_model.parameters():
            total_number_parameters += torch.numel(parameter)
        assert total_number_parameters > 15000000
        assert total_number_parameters < 17000000

    @pytest.mark.parametrize("batch_dimension", [1, 2])
    def test__average_token_embedding_pooling(self, batch_dimension: int) -> None:
        """
        Args:
            batch_dimension (int): Denotes the number of dimensions the batch shape
                should be.

        Written mostly by Mistral Codestral, verified by apmoore1.
        """

        # Single sequence example
        token_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        token_attention_mask = torch.tensor([1, 1, 0])
        expected_output = torch.tensor([2.0, 3.0])
        for _ in range(batch_dimension):
            token_embeddings = token_embeddings.unsqueeze(0)
            token_attention_mask = token_attention_mask.unsqueeze(0)
            expected_output = expected_output.unsqueeze(0)

        output = BEM._average_token_embedding_pooling(
            token_embeddings, token_attention_mask
        )
        torch.testing.assert_close(expected_output, output)

        # multiple sequences
        token_embeddings = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
        )

        token_attention_mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
        expected_output = torch.tensor([[2.0, 3.0], [9.0, 10.0]])
        for _ in range(1, batch_dimension):
            token_embeddings = token_embeddings.unsqueeze(0)
            token_attention_mask = token_attention_mask.unsqueeze(0)
            expected_output = expected_output.unsqueeze(0)

        output = BEM._average_token_embedding_pooling(
            token_embeddings, token_attention_mask
        )
        torch.testing.assert_close(expected_output, output)

        # All padding (handle the case of potentially dividing by 0.0)
        # Single sequence example
        token_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        token_attention_mask = torch.tensor([0, 0, 0])
        expected_output = torch.tensor([0.0, 0.0])
        for _ in range(batch_dimension):
            token_embeddings = token_embeddings.unsqueeze(0)
            token_attention_mask = token_attention_mask.unsqueeze(0)
            expected_output = expected_output.unsqueeze(0)

        output = BEM._average_token_embedding_pooling(
            token_embeddings, token_attention_mask
        )
        torch.testing.assert_close(expected_output, output)

        # Handle empty sequences
        token_embeddings = torch.tensor([[]])

        token_attention_mask = torch.tensor([])
        expected_output = torch.tensor([])
        expected_output_shape = [0]
        for _ in range(batch_dimension):
            token_embeddings = token_embeddings.unsqueeze(0)
            token_attention_mask = token_attention_mask.unsqueeze(0)
            expected_output = expected_output.unsqueeze(0)
            expected_output_shape.insert(0, 1)

        output = BEM._average_token_embedding_pooling(
            token_embeddings, token_attention_mask
        )
        torch.testing.assert_close(expected_output, output)
        assert tuple(expected_output_shape) == output.shape

        # Test that a value error is raised if the incorrect embedding and attention
        # mask dimensions are given

        with pytest.raises(ValueError):
            token_embeddings = token_embeddings.squeeze(0)
            BEM._average_token_embedding_pooling(
                token_embeddings, token_attention_mask
            )
        token_embeddings = token_embeddings.unsqueeze(0)
        output = BEM._average_token_embedding_pooling(
            token_embeddings, token_attention_mask
        )
        torch.testing.assert_close(expected_output, output)

        with pytest.raises(ValueError):
            token_attention_mask = token_attention_mask.unsqueeze(0)
            BEM._average_token_embedding_pooling(
                token_embeddings, token_attention_mask
            )

    @torch.inference_mode(mode=True)
    @pytest.mark.parametrize("text_token_masking", [True, False])
    @pytest.mark.parametrize("text_word_ids_masking", [True, False])
    @pytest.mark.parametrize("label_definitions_attention_masking", [True, False])
    def test_forward_with_no_masking(self, bem_model: BEM,
                                     text_token_masking: bool,
                                     text_word_ids_masking: bool,
                                     label_definitions_attention_masking: bool) -> None:
        # B = 3 T = 4 S = 2 ST = 5
        # B x T
        text_input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]])
        
        text_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        if text_token_masking:
            text_attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        
        text_word_ids_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        if text_word_ids_masking:
            text_word_ids_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]])
        
        # B x S x ST
        label_definitions_input_ids = torch.tensor([[[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]],
                                                    [[7, 8, 9, 10, 11], [10, 11, 12, 13, 14]],
                                                    [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]])
        
        label_definitions_attention_mask = torch.tensor([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                         [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                         [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]])
        if label_definitions_attention_masking:
            label_definitions_attention_mask = torch.tensor([[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
                                                             [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                             [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]])

        # B x S
        expected_output_shape = (3, 2)
        expected_output_dtype = torch.float32
        result = bem_model.forward(text_input_ids,
                                   text_attention_mask,
                                   text_word_ids_mask,
                                   label_definitions_input_ids,
                                   label_definitions_attention_mask)
        assert result.shape == expected_output_shape
        assert result.dtype == expected_output_dtype
        assert math.fabs(result.sum().item()) > 0.0

        if label_definitions_attention_masking:
            assert result[0][1].item() == 0.0
        else:
            assert result[0][1].item() > 0.0

        if text_token_masking or text_word_ids_masking or label_definitions_attention_masking:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(result[0], result[2])
        else:
            torch.testing.assert_close(result[0], result[2])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(result[0], result[1])
