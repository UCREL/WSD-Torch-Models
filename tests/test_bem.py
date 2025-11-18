import pytest
import torch

from wsd_torch_models.bem import BEM


class TestBEM:

    # def test__get_base_model() -> None:
    # BEM._get_base_model("jhu-clsp/ettin-encoder-17m")

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
