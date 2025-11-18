from collections import OrderedDict
import inspect
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Union

from huggingface_hub import PyTorchModelHubMixin, constants
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.file_download import hf_hub_download
from safetensors.torch import save_model as save_model_as_safetensor
import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from wsd_torch_models.scalar_mix import ScalarMix
from wsd_torch_models.utils import tiny_value_of_dtype


logger = logging.getLogger(__name__)


class BEM(torch.nn.Module, PyTorchModelHubMixin):
    """
    An inference only implementation of the Bi-Encoder Model (BEM)
    for Word Sense Disambiguation from the paper
    [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders]
    (https://aclanthology.org/2020.acl-main.95.pdf).

    This is a bi-encoder model whereby it encodes the word(s) to disambiguate
    using the word(s) text context, e.g. whole sentence or document and it will
    encode every sense definition given and return the most similar sense definition
    for the given word(s).
    """

    @staticmethod
    def _get_base_model(base_model_name: str) -> PreTrainedModel:
        """
        Downloads the `base_model_name` from HuggingFace Hub.

        Checks if a pooling layer would be added when loaded and if so the
        pooling layer is removed as it is not required in the BEM setup.

        Args:
            base_model_name (str): HuggingFace model id, e.g.
                `jhu-clsp/ettin-encoder-17m` from the HuggingFace Hub.
        
        Returns:
            AutoModel: The loaded `base_model_name` as a HuggingFace
                AutoModel.
        """
        base_model = AutoModel.from_pretrained(base_model_name)
        base_model_type = type(base_model)
        if "add_pooling_layer" in inspect.getfullargspec(base_model_type.__init__).args:
            base_model = AutoModel.from_pretrained(base_model_name,
                                                   add_pooling_layer=False)
        if not isinstance(base_model, PreTrainedModel):
            raise TypeError(f"The model loaded; {base_model} should have "
                            "inherited from transformers.PreTrainedModel")
        return base_model

    def __init__(self,
                 base_model_name: str,
                 freeze_base_model: bool,
                 number_transformer_encoder_layers: int,
                 add_scalar_mixer: bool = True,
                 scalar_mix_layer_norm: bool = True,
                 transformer_encoder_hidden_dim: int = 512,
                 transformer_encoder_num_heads: int = 8,
                 batch_first: bool = True,
                 base_model: PreTrainedModel | None = None,
                 **kwargs: Any
                 ) -> None:
        """
        Args:
            base_model_name (str): The name of the HuggingFace base model
                to use, e.g. FacebookAI/roberta-base.
            freeze_base_model (bool): (Ignore as it is only relevant for training
                but is useful as metadata on how the model was trained)
                Whether the base model should not be trained
                (the model weights are frozen).
            number_transformer_encoder_layers (int): The number of transformer
                encoder layers to add to the base model. Can be 0.
            add_scalar_mixer (bool): Whether to use a ScalarMix to generate a
                base model hidden state rather than using the base model's final
                layer as the hidden state. Default True.
            scalar_mix_layer_norm (bool): Whether the scalar mixer should normalise
                each transformer hidden layer before weighting or not. Default
                True. Not used if `add_scalar_mixer` is False.
            transformer_encoder_hidden_dim (int): The hidden dimension size of the
                `number_transformer_encoder_layers` layers. Default is 512.
            transformer_encoder_num_heads (int): The number of heads of the
                `number_transformer_encoder_layers` layers. Default is 8.
            batch_first (bool): If the batch should be first dimension,
                (batch, seq, feature) else False will be (seq, batch, feature).
                This is only used for the `number_transformer_encoder_layers`
                layers.
            base_model (PreTrainedModel | None): If this is not None then instead of
                downloading the base model using the `base_model_name` this
                is used instead as the base_model, this is required for loading
                the model using `from_pretrained`. Default is None.
            **kwargs (Any): This is required for the `from_pretrained` method,
                all of these `kwargs` are ignored.

        Returns:
            None
        """

        super().__init__()
        self.base_model_name = base_model_name
        if base_model is None:
            logger.info(f"Downloading the base model from the HuggingFace hub: {base_model_name}")
            self.base_model = self._get_base_model(self.base_model_name)
        else:
            self.base_model = base_model
        self.base_model_hidden_size = self.base_model.config.hidden_size
        self.freeze_base_model = freeze_base_model
        logger.info(f"Base model: {self.base_model_name} loaded")
        # Add 1 for the embedding layer
        self.base_model_number_hidden_layers = (
            self.base_model.config.num_hidden_layers + 1
        )
        logger.info(
            "Number of hidden layers in base model: "
            f"{self.base_model_number_hidden_layers}"
        )

        self.scalar_mix: Optional[ScalarMix] = None
        self.scalar_mix_layer_norm = scalar_mix_layer_norm
        if add_scalar_mixer:
            self.scalar_mix = ScalarMix(
                self.base_model_number_hidden_layers,
                do_layer_norm=self.scalar_mix_layer_norm,
            )

        # Optional list of layers to further encode the tokens after embedding
        # from the base transformer model.
        token_model_layers_list: list[tuple[str, torch.nn.Module]] = []
        self.token_model_layers: torch.nn.Sequential | None = None

        self.batch_first = batch_first
        self.number_transformer_encoder_layers = number_transformer_encoder_layers
        self.transformer_encoder_hidden_dim = transformer_encoder_hidden_dim
        self.transformer_encoder_num_heads = transformer_encoder_num_heads
        self.linear_bridge: torch.nn.Linear | None = None
        self.transformer: torch.nn.TransformerEncoder | None = None

        if self.number_transformer_encoder_layers:
            logger.info(
                f"Adding {self.number_transformer_encoder_layers} "
                "transformer encoder layers to the base model."
            )
            if self.transformer_encoder_hidden_dim != self.base_model_hidden_size:
                logger.info(
                    "Base model hidden dimension "
                    f"{self.base_model_hidden_size} does not match "
                    "Transformer encoder model hidden dimension "
                    f"{self.transformer_encoder_hidden_dim}"
                    "creating a linear layer bridge between the two."
                )
                self.linear_bridge = torch.nn.Linear(
                    self.base_model_hidden_size, self.transformer_encoder_hidden_dim
                )
                token_model_layers_list.append(("Linear Bridge", self.linear_bridge))

            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
                self.transformer_encoder_hidden_dim,
                self.transformer_encoder_num_heads,
                batch_first=self.batch_first,
            )
            self.token_transformer = torch.nn.TransformerEncoder(
                transformer_encoder_layer, num_layers=number_transformer_encoder_layers
            )
            token_model_layers_list.append(
                ("Token Transformer", self.token_transformer)
            )
            self.token_model_layers = torch.nn.Sequential(
                OrderedDict(token_model_layers_list)
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Save weights from a Pytorch model to a local directory. In addition it
        saves the configuration of the `base_model`.

        Required to save a model using `save_pretrained` a method that is overridden
        from `PyTorchModelHubMixin`.

        Reference:
        https://github.com/huggingface/huggingface_hub/blob/c8992647f02e254281f45afd01a0d28aaeee08ab/src/huggingface_hub/hub_mixin.py#L753

        Args:
            save_directory (Path): The directory to save the model too.
        
        Returns:
            None
        """
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / constants.SAFETENSORS_SINGLE_FILE))  # type: ignore [arg-type]
        auto_model_config = self.base_model.config
        auto_model_config.save_pretrained(str(save_directory / "base_model_config"))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs: Any,
    ) -> "BEM":
        """
        Loads the PyTorch model weights from a local directory or from the
        HuggingFace hub.

        Required to load a model using `from_pretrained` a method that is overridden
        from `PyTorchModelHubMixin`.

        Reference:
        https://github.com/huggingface/huggingface_hub/blob/c8992647f02e254281f45afd01a0d28aaeee08ab/src/huggingface_hub/hub_mixin.py#L759

        Args:
            model_id (str): Either a local directory or model id from the HuggingFace
                hub.
            revision (Optional[str]): The specific model version to use.
                It can be a branch name, a tag name, or a commit id,
                since we use a git-based system for storing models
                and other artifacts on huggingface.co, so revision
                can be any identifier allowed by git.
                Defaults to `main`.
            cache_dir (Optional[Union[str, os.PathLike]]): Path to a directory
                in which a downloaded pretrained model configuration
                should be cached if the standard cache should not be used.
            force_download (bool): Whether or not to force the (re-)download of the
                model weights and configuration files,
                overriding the cached versions if they exist.
            local_files_only (bool): Whether or not to only look at local files
                (i.e., do not try to download the model).
            token (str | bool | None): The token to use as HTTP bearer authorization
                for remote files. If True, or not specified,
                will use the token generated when running
                hf auth login (stored in ~/.huggingface).
            map_location (str): Where the tensors should be loaded too. Defaults `cpu`.
            strict (bool): Defaults to False.
            **model_kwargs (Any): The arguments to pass to the model to initialise
                the model. Please note that the `base_model` argument is passed
                by this method internally as we load the `base_model` within this
                method.
            
        Returns:
            None
        """
        config_directory_name = "base_model_config"
        if os.path.isdir(model_id):
            logger.info("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            auto_model_config_directory = os.path.join(model_id, config_directory_name)
            auto_model_config = AutoConfig.from_pretrained(auto_model_config_directory)
            auto_model = AutoModel.from_config(auto_model_config)  # type: ignore
            model_kwargs["base_model"] = auto_model
            model = cls(**model_kwargs)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_config_directory = Path(temp_dir, config_directory_name)
                auto_model_config = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                    subfolder=config_directory_name,
                    local_dir=temp_model_config_directory
                )
                temp_model_config_directory_str = str(temp_model_config_directory.resolve())
                auto_model_config = AutoConfig.from_pretrained(temp_model_config_directory_str)
                auto_model = AutoModel.from_config(auto_model_config)  # type: ignore
                model_kwargs["base_model"] = auto_model

            model = cls(**model_kwargs)
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)

    def _token_encoding(
        self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the token IDs and the attention masks for those token ID's it
        returns a contextualised embedding for each token using the following
        applicable layers:
        1. self.base_model - Typically a pre-train language Model.
        2. self.scalar_mix - The average embedding for that token based on a learnt
            weighting of all of the base model layers.
        3. self.token_model_layers - to be learnt transformer layers.

        Args:
            token_input_ids (torch.Tensor): The token IDs to embed. torch.Long.
                Shape (Batch, Sequence Length).
            token_attention_mask (torch.Tensor): The attention mask associated
                with the token IDs. torch.Long. Shape (Batch, Sequence Length).
        Returns:
            torch.Tensor: A contextualised embedding for each token. torch.Float.
                Shape (Batch, Sequence Length, Embedding Dimension)
        """

        base_model_output: BaseModelOutput = (
            self.base_model(
                token_input_ids, token_attention_mask, output_hidden_states=True
            )
        )

        # self.base_model_number_hidden_layers of hidden layers of
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        token_model_embedding = base_model_output.last_hidden_state
        if token_model_embedding is None:
            raise TypeError("The last hidden state from the base model of "
                            "the BEM model is None when it should be a tensor. "
                            f"Token input ids: {token_input_ids} and token "
                            f"attention mask: {token_attention_mask}")
        
        if self.scalar_mix is not None:
            base_model_hidden_layers = base_model_output.hidden_states
            token_model_embedding = self.scalar_mix(
                base_model_hidden_layers, token_attention_mask
            )

        # Further token encoding through the token model layers
        if self.token_model_layers:
            token_model_embedding = self.token_model_layers(token_model_embedding)

        return token_model_embedding

    @staticmethod
    def _average_token_embedding_pooling(
        token_embeddings: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        The token embeddings average pooled over the sequence length taking
        into account the attention mask.

        Written with Mistral Codestral.

        Args:
            token_embeddings (torch.Tensor): The embeddings/encodings for a
                batch of tokens whereby the batch can be of various shapes.
                torch.Float. Shape (..., Sequence Length, Embedding Dimension).
            token_attention_mask (torch.Tensor): The attention mask for the
                token embeddings. The mask is expected to be a Binary mask (1 or 0).
                torch.Long. Shape (..., Sequence Length).
        Returns:
            torch.Tensor: The average token embeddings pooled over each sequence
                in the batch taking into account the attention mask. torch.Float.
                Shape (..., Embedding Dimension).
        Raises:
            ValueError: If token_embeddings doesn't have at least 3 dimensions,
                or if token_attention_mask's dimension
                doesn't match the expected shape (one less than token_embeddings).
        """

        # Validate input shapes
        if token_embeddings.dim() < 3:
            raise ValueError(
                "token_embeddings must have at least 3 dimensions "
                "(... , Sequence Length, Embedding Dimension), "
                f"got {token_embeddings.shape}"
            )
        if (
            token_attention_mask.dim() < 2
            or token_attention_mask.dim() != token_embeddings.dim() - 1
        ):
            raise ValueError(
                "token_attention_mask must have one less dimension "
                "than token_embeddings (... , Sequence Length), "
                f"got {token_attention_mask.shape} for the attention mask and "
                f"{token_embeddings.shape} for the embeddings."
            )

        # Float Tensor, shape (..., Sequence Length, 1)
        broadcast_token_attention_mask = token_attention_mask.unsqueeze(-1).to(
            token_embeddings
        )
        # Float tensor, shape (..., Embedding Dimension)
        masked_embedding_sum = torch.mul(
            token_embeddings, broadcast_token_attention_mask
        ).sum(dim=-2)
        # Float tensor of (..., 1) represents the number of tokens that make up the given word.
        number_token_vectors = broadcast_token_attention_mask.sum(-2)
        # Stops dividing by zero which causes nan values
        tiny_value_to_stop_nan = tiny_value_of_dtype(number_token_vectors.dtype)
        number_token_vectors = torch.clamp(
            number_token_vectors, min=tiny_value_to_stop_nan
        )

        # Float tensor, shape (..., Embedding Dimension)
        average_token_embeddings = masked_embedding_sum / number_token_vectors
        return average_token_embeddings
    
    def label_definition_encoding(self,
                                  label_definitions_input_ids: torch.Tensor,
                                  label_definitions_attention_mask: torch.Tensor,
                                  ) -> torch.Tensor:
        
        BATCH_SIZE, S, ST = label_definitions_input_ids.shape

        # Encoding the label definition sequences, these need to be reshaped
        # so that they can be processed by the token encoding model/layers
        definition_input_ids_encoding = label_definitions_input_ids.view(-1, ST)
        definition_attention_mask_encoding = label_definitions_attention_mask.view(-1, ST)
        definition_token_embedding = self._token_encoding(
            definition_input_ids_encoding, definition_attention_mask_encoding
        )
        average_definition_token_embeddings = self._average_token_embedding_pooling(
            definition_token_embedding, definition_attention_mask_encoding
        )
        # View the embeddings back to shape:
        # (Batch, S, Embedding Dimension)
        average_definition_token_embeddings = average_definition_token_embeddings.view(
            BATCH_SIZE, S, -1
        )
        return average_definition_token_embeddings
    
    def text_encoding(self,
                      text_input_ids: torch.Tensor,
                      text_attention_mask: torch.Tensor,
                      ) -> torch.Tensor:
        # Encoding the text sequence
        # Shape (B, D)
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        return text_encoding
    
    def token_encoding_using_text_encoding(self,
                                           text_encoding: torch.Tensor,
                                           text_word_ids_mask: torch.Tensor
                                           ) -> torch.Tensor:
        # Expanded so that we have a text embedding per positive sample
        # Current Shape (Batch, Sequence Length, Dimension)
        # New Shape (B, M, T, D)
        # expanded_text_encoding = text_encoding.unsqueeze(1).expand(-1, S, -1, -1)
        # Shape (B, D)
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding
    
    def token_text_encoding(self,
                            text_input_ids: torch.Tensor,
                            text_attention_mask: torch.Tensor,
                            text_word_ids_mask: torch.Tensor
                            ) -> torch.Tensor:
        # Encoding the text sequence
        # Shape (B, D)
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        # Expanded so that we have a text embedding per positive sample
        # Current Shape (Batch, Sequence Length, Dimension)
        # New Shape (B, M, T, D)
        # expanded_text_encoding = text_encoding.unsqueeze(1).expand(-1, S, -1, -1)
        # Shape (B, D)
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding
    
    def token_label_similarity(self,
                               label_definition_embedding: torch.Tensor,
                               token_text_embedding: torch.Tensor) -> torch.Tensor:
        # Expand the text encoding so that we can get token similarity for each
        # label definition
        expanded_average_text_encoding = token_text_embedding.unsqueeze(-1)
        expanded_similarity_score = torch.matmul(label_definition_embedding, expanded_average_text_encoding)
        similarity_score = expanded_similarity_score.squeeze(-1)
        return similarity_score

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_word_ids_mask: torch.Tensor,
        label_definitions_input_ids: torch.Tensor,
        label_definitions_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Finds the most similar label definition for the given text word ids
        which are contextualised by itself and it's surrounding text (`text_input_ids`).

        B represents the batch size. This is the number of text sequences.
        S represents the largest number of label definitions within one text
            sequence within the batch.
        T represents the largest token length for the text sample.
        ST represents the largest token length for the label definitions sentences.
        
        Args:
            text_input_ids (torch.Tensor): Tokenized text sample
                which contains all of the tokens a single set of tokens will be
                encoded and matched against the label definitions to determine
                which definition is the most similar.
                torch.Long Shape (B, T).
            text_attention_mask (torch.Tensor): 1 or 0 attention mask for the
                text samples. torch.Long tensor. 1 represents a token to
                attend to, 0 a token to ignore. Shape (B, T).
            text_word_ids_mask (torch.Tensor): A token mask for the
                single set of tokens used to average the encoded text inputs.
                These are the tokens you want to find the most similar definitions
                for. torch.Long. Shape (B, T).
            label_definitions_input_ids (torch.Tensor): The input ids for the
                sentences to match the token encoding against to determine which
                is the most similar. torch.Long. Shape (B, S, ST).
            label_definitions_attention_mask (torch.Tensor): The attention
                mask (1 or 0) for the `label_definitions_input_ids`. torch.Long.
                Shape (B, S, ST)
            
        Returns:
            torch.Tensor: A floating point tensor of shape (B, S). Higher the more
                similar.
        """
        average_definition_token_embeddings = self.label_definition_encoding(label_definitions_input_ids, label_definitions_attention_mask)

        average_text_encoding = self.token_text_encoding(text_input_ids,
                                                         text_attention_mask,
                                                         text_word_ids_mask)

        similarity_score = self.token_label_similarity(average_definition_token_embeddings, average_text_encoding)
        
        return similarity_score
