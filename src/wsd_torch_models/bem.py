from collections import OrderedDict
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union, overload
import warnings

from huggingface_hub import PyTorchModelHubMixin, constants
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.file_download import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import save_model as save_model_as_safetensor
import torch
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput
from typing_extensions import Self

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

    Unlike the original BEM model we use the same model to encode both the
    text to disambiguate and the label definitions.

    The layers of the model that encode all tokens (text to disambiguate and the
    label definitions) is first the `base_model` and then optionally followed
    by the `linear_bridge` and `token_model_layers`. The `base_model` may encode
    the text using either the final layer of the base model or a ScalarMix.

    Attributes:
        base_model (transformers.PreTrainedModel): The base model to encode the word(s) to
            disambiguate using the word(s) text context. This is used for both
            the text that is to be disambiguated and the label definitions.
        base_model_hidden_size (int): The hidden dimension size of the
            `base_model`.
        base_model_number_hidden_layers (int): The number of hidden layers in
            the `base_model`.
        base_model_name (str): The name of the HuggingFace base model
            to use, e.g. FacebookAI/roberta-base.
        freeze_base_model (bool): (Ignore as it is only relevant for training
            the base model).
            Whether the base model should not be trained
        add_scalar_mixer (bool): Whether to use a ScalarMix to generate a
            base model hidden state rather than using the base model's final
            layer as the hidden state.
        scalar_mix_layer_norm (bool): Whether the scalar mixer should normalise
            its output.
        number_transformer_encoder_layers (int): The number of transformer encoder
            layers to use. Can be 0, when 0 then the following two attributes
            are ignore, `transformer_encoder_hidden_dim` and `transformer_encoder_num_heads`
            even if they contain a value.
        transformer_encoder_hidden_dim (int): The hidden dimension size of the
            transformer encoder.
        transformer_encoder_num_heads (int): The number of heads in the
            transformer encoder.
        linear_bridge (torch.nn.Linear | None): A linear layer to bridge the
            output of the base model and the input of the first transformer encoder
            within the `token_model_layers`.
            This is None when `number_transformer_encoder_layers` is 0 or when
            the output dimension of the base model is the same as the input
            dimension of the transformer encoder.
        token_model_layers (torch.nn.Sequential | None): A list of transformer encoder
            layers to further encode tokens after the base model layers.
            This is None when `number_transformer_encoder_layers` is 0.
        inference_ready (bool): Whether the model is ready for inference.
            This is based on whether the `label_definition_embeddings`,
            `label_to_definition` and `embedding_index_to_label` attributes
            have been set. These can be set by calling the method
            `embed_and_set_label_definitions` or at initialisation if the
            `label_definitions_directory_path` is not None.
        label_definition_embeddings (torch.Tensor | None): The embeddings
            of the label/senses definitions.
        label_to_definition (dict[str, str] | None): A dictionary mapping
            labels/senses to their definitions.
        embedding_index_to_label (dict[int, str] | None): A dictionary mapping
            embedding indices to labels/senses.
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
                 label_definitions_directory_path: str | os.PathLike | None = None,
                 # model_type: str = "modernbert",
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
            label_definitions_directory_path (str | os.PathLike | None): The path
                to the directory containing the label definitions data. This data
                should include the sense labels to their definitions, the
                embedded definitions, and the embedding index to the sense
                labels, this data should be saved to the following file names
                within this directory: `label_to_definition.json`,
                `label_definitions_embeddings.safetensors` and
                `embedding_index_to_label.json`. Default is None, if this is
                not `None` then the attribute `inference_ready` is set to True.
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
        
        self.inference_ready = False
        self.label_definition_embeddings: torch.Tensor | None = None
        self.label_to_definition: dict[str, str] | None = None
        self.embedding_index_to_label: dict[int, str] | None = None
        if label_definitions_directory_path:
            label_definitions_embeddings_path = Path(label_definitions_directory_path,
                                                     "label_definitions_embeddings.safetensors")
            if not label_definitions_embeddings_path.exists():
                raise FileNotFoundError(
                    f"Could not find the label definitions embeddings SAFETENSORS file: "
                    f"in the directory: {label_definitions_directory_path} "
                    f"files in the directory: {list(Path(label_definitions_directory_path).iterdir())}"
                )
            label_definitions_tensor_dict = load_safetensors(label_definitions_embeddings_path,
                                                             device=str(self.base_model.device))
            if "label_definitions_embeddings" in label_definitions_tensor_dict:
                self.label_definition_embeddings = label_definitions_tensor_dict[
                    "label_definitions_embeddings"
                ]
            else:
                raise ValueError(
                    "Could not find label_definitions_embeddings in the "
                    f"SAFETENSORS file: {label_definitions_embeddings_path}"
                )
            label_to_definition_path = Path(label_definitions_directory_path,
                                            "label_to_definition.json")
            with label_to_definition_path.open("r", encoding="utf-8") as read_fp:
                self.label_to_definition = json.load(read_fp)
            embedding_index_to_label_path = Path(label_definitions_directory_path,
                                                 "embedding_index_to_label.json")
            with embedding_index_to_label_path.open("r", encoding="utf-8") as read_fp:
                self.embedding_index_to_label = {int(embedding_index): label
                                                 for embedding_index, label in json.load(read_fp).items()}
            self.inference_ready = True

    def embed_and_set_label_definitions(self,
                                        label_definitions: dict[str, str],
                                        tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Tokenizes the label definitions using the `tokenizer` and
        embeds the label definitions using the `label_definition_encoding` method.
        The embedded label definitions are then saved self.label_definition_embeddings.

        This will set the following attributes:
        inference_ready (bool): True
        label_definition_embeddings (torch.Tensor): The embeddings of the label/senses definitions.
        label_to_definition (dict[str, str]): A dictionary mapping
            labels/senses to their definitions.
        embedding_index_to_label (dict[int, str]): A dictionary mapping
            embedding indices to labels/senses.

        NOTE: we recommend set the model to evaluation mode and running this
        method within a `torch.inference_mode` context. e.g.
        
        ``` python
        model.eval()
        with torch.inference_mode():
            model.embed_and_set_label_definitions(label_definitions, tokenizer)
        ```

        Args:
            label_definitions (dict[str, str]): The label definitions to embed.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use to tokenize
                the label definitions.

        Returns:
            None
        """
        label_definition_embeddings_list: list[torch.Tensor] = []
        embedding_index_to_label: dict[int, str] = {}
        
        for index, label_definition in enumerate(label_definitions.items()):
            label, definition = label_definition
            tokenized_definition = tokenizer(definition, return_tensors="pt", padding=False, truncation=False)
            tokenized_definition_input_ids: torch.Tensor = tokenized_definition.input_ids.to(self.base_model.device).unsqueeze(0)
            tokenized_definition_attention_mask: torch.Tensor = tokenized_definition.attention_mask.to(self.base_model.device).unsqueeze(0)

            definition_embedding = self.label_definition_encoding(tokenized_definition_input_ids,
                                                                  tokenized_definition_attention_mask)
            label_definition_embeddings_list.append(definition_embedding)
            embedding_index_to_label[index] = label
        
        label_definition_embeddings = torch.vstack(label_definition_embeddings_list)
        NUM_DESC, DESC_BATCH, EMBEDDING_DIM = label_definition_embeddings.shape
        label_definition_embeddings = label_definition_embeddings.view(DESC_BATCH, NUM_DESC, EMBEDDING_DIM)
        
        self.inference_ready = True
        self.label_definition_embeddings = label_definition_embeddings
        self.label_to_definition = label_definitions
        self.embedding_index_to_label = embedding_index_to_label

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
        if self.inference_ready:
            logger.info("Saving label definitions embeddings, label definitions and embeddings index.")
            label_definitions_directory_path = save_directory / "label_definitions"
            label_definitions_directory_path.mkdir(parents=True, exist_ok=True)
            
            assert self.label_definition_embeddings is not None
            label_definition_embedding_dict = {"label_definitions_embeddings": self.label_definition_embeddings}
            label_definition_embedding_path = label_definitions_directory_path / "label_definitions_embeddings.safetensors"
            save_safetensors(label_definition_embedding_dict, label_definition_embedding_path)
            
            label_to_definition_path = label_definitions_directory_path / "label_to_definition.json"
            with label_to_definition_path.open("w", encoding="utf-8") as write_fp:
                json.dump(self.label_to_definition, write_fp)
            
            embedding_index_to_label_path = label_definitions_directory_path / "embedding_index_to_label.json"
            with embedding_index_to_label_path.open("w", encoding="utf-8") as write_fp:
                json.dump(self.embedding_index_to_label, write_fp)

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
                method. Also note that the `label_definitions_directory_path` argument
                is also passed by this method internally if the Path exists, the
                Path should be at `model_id/label_definitions`.
            
        Returns:
            None
        """
        config_directory_name = "base_model_config"
        label_definitions_directory_name = "label_definitions"
        
        if os.path.isdir(model_id):
            logger.info("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            auto_model_config_directory = os.path.join(model_id, config_directory_name)
            auto_model_config = AutoConfig.from_pretrained(auto_model_config_directory)
            auto_model = AutoModel.from_config(auto_model_config)  # type: ignore
            model_kwargs["base_model"] = auto_model

            label_definitions_directory_path = Path(os.path.join(model_id, label_definitions_directory_name))
            if label_definitions_directory_path.is_dir():
                model_kwargs["label_definitions_directory_path"] = label_definitions_directory_path
            
            model = cls(**model_kwargs)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            auto_model_config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
                subfolder=config_directory_name,
                repo_type="model"
            )
            auto_model_config = AutoConfig.from_pretrained(auto_model_config_file)
            auto_model = AutoModel.from_config(auto_model_config)  # type: ignore
            model_kwargs["base_model"] = auto_model
  
            try:
                files_to_download = [
                    "label_definitions_embeddings.safetensors",
                    "label_to_definition.json",
                    "embedding_index_to_label.json"
                ]
                file_downloaded_folder: Path | None = None
                for file_name in files_to_download:
                    file_name_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        token=token,
                        local_files_only=local_files_only,
                        subfolder=label_definitions_directory_name,
                        repo_type="model"
                    )
                    if file_downloaded_folder is None:
                        file_downloaded_folder = Path(file_name_path).parent
                assert file_downloaded_folder is not None
                downloaded_label_definitions_directory_path = str(file_downloaded_folder.resolve())
                model_kwargs["label_definitions_directory_path"] = downloaded_label_definitions_directory_path
                
            except EntryNotFoundError:
                logging.info(f"Could not find {label_definitions_directory_name} directory "
                             f"for model {model_id}, thus `inference_ready` "
                             "will be set to `False`")

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
                    repo_type="model"
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
                    repo_type="model"
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
        """
        Encodes the label definitions given their token ids and attention mask.
        This returns an encoding per label definition and not per token.

        Args:
            label_definitions_input_ids (torch.Tensor): The label definition token IDs to
                embed. torch.Long.
                Shape(Batch, Largest number of label definitions (S),
                Sequence length (ST)).
            label_definitions_attention_mask (torch.Tensor): The attention
                mask associated with the token IDs. torch.Long.
                Shape(Batch, Largest number of label definitions (S),
                Sequence length (ST)).
        Returns:
            torch.Tensor: A contextualised embedding for each label definition.
                Label definitions that are not relevant (padded) are ignored
                as they will have an embedding of zero (zero values).
                torch.Float. Shape (Batch, Largest number of label definitions (S),
                Embedding Dimension)
        """
        
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
        """
        Encodes the text sequence using the token encoding model/layers.
        
        Args:
            text_input_ids (torch.Tensor): The token IDs to embed. torch.Long.
                Shape (Batch, Sequence Length).
            text_attention_mask (torch.Tensor): The attention mask associated with the token IDs.
                torch.Long. Shape (Batch, Sequence Length).
        
        Returns:
            torch.Tensor: A contextualised embedding for each token in the text sequence.
                torch.Float. Shape (Batch, Sequence Length, Embedding Dimension)
        """
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        return text_encoding
    
    def token_encoding_using_text_encoding(self,
                                           text_encoding: torch.Tensor,
                                           text_word_ids_mask: torch.Tensor
                                           ) -> torch.Tensor:
        
        """
        Given the text encodings it returns the average embedding for each token
        within the text word ids mask for each given text encoding in the batch.
        
        Args:
            text_encoding (torch.Tensor): The contextualised embedding for each text sequence.
                torch.Float. Shape (Batch, Sequence Length, Embedding Dimension).
            text_word_ids_mask (torch.Tensor): The attention mask associated with the token IDs
                that make up the average embedding for the given sequence.
                torch.Long. Shape (Batch, Sequence Length).
        
        Returns:
            torch.Tensor: An average embedding for each token in the IDS for
            each given text encoding. torch.Float. Shape (Batch, Embedding Dimension)
        """
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding
    
    def token_text_encoding(self,
                            text_input_ids: torch.Tensor,
                            text_attention_mask: torch.Tensor,
                            text_word_ids_mask: torch.Tensor
                            ) -> torch.Tensor:
        """
        Encoding the text sequence with the token encoding and then
        pooling the token embeddings with the attention mask to get the
        average embedding for each token which is used to calculate the
        similarity with the label definitions.

        Args:
            text_input_ids (torch.Tensor): The token IDs to embed. torch.Long.
                Shape (BATCH, SEQUENCE).
            text_attention_mask (torch.Tensor): The attention mask associated
                with the token IDs. torch.Long. Shape (BATCH, SEQUENCE).
            text_word_ids_mask (torch.Tensor): The attention mask associated
                with the token IDs. torch.Long. Shape (BATCH, SEQUENCE).

        Returns:
            torch.Tensor: The average token embeddings pooled over each sequence
                in the batch taking into account the attention mask. torch.Float.
                Shape (BATCH, D).
        """
        # Shape (B, S, D)
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        # Shape (B, D)
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding
    
    def token_label_similarity(self,
                               label_definition_embedding: torch.Tensor,
                               token_text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculates the similarity score between each label definition and a given
        token encoding. This is used to determine which label definition is the
        most similar to a given token.

        Label definitions that are not relevant (padded) are ignored as they
        will have a similarity score of zero.

        Args:
            label_definition_embedding (torch.Tensor): The contextualised embedding of
                the label definitions. Shape (Batch, Largest number of label definitions (S),
                Embedding Dimension).
            token_text_embedding (torch.Tensor): The contextualised embedding of the
                token. Shape (Batch, Embedding Dimension).

        Returns:
            torch.Tensor: The similarity score between each label definition and the token
                encoding. Shape (Batch, Largest number of label definitions (S))
        """
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

        Tensor dimension definitions:

        B: represents the batch size. This is the number of text sequences.
        S: represents the largest number of label definitions within one text
            sequence within the batch.
        T: represents the largest token length for the text sample.
        ST: represents the largest token length for the label definitions sentences.
        
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
    
    def predict(self,
                tokens: list[str],
                sub_word_tokenizer: PreTrainedTokenizerBase | None = None,
                top_n: int = -1
                ) -> list[list[str]]:
        """
        Predicts the `top_n` sense labels per token whereby those sense labels
        can be mapped to their corresponding sense definitions.

        NOTE: we recommend that the number of tokens in the list should represent
        a sentence, in addition the more tokens in the list the more
        memory the model requires and on CPU at least the more time it will
        take to predict the sense labels.

        NOTE: we recommend to put the model in evaluation model and to run
        it within a `torch.inference_mode(mode=True)` context, e.g.

        ``` python
        model.eval()
        with torch.inference_mode(mode=True):
            predictions = model.predict(tokens)
        ```

        Args:
            tokens (list[str]): A list of tokens to predict the sense labels for.
            sub_word_tokenizer (PreTrainedTokenizerBase | None): The sub-word tokenizer
                used to split the tokens into sub-word tokens that can be used
                by the model. Default None. If None the tokenizer used
                is `transformers.AutoTokenizer.from_pretrained(self.base_model_name)`.
            top_n (int): The number of sense labels to predict. Default -1 which
                predicts all sense labels. If 0 will raise a ValueError.

        Returns:
            list[list[str]]: A list of sense labels per token, where the
                number of sense labels predicted per token is `top_n`.

        Raises:
            ValueError: If `top_n` is 0 or less than -1.
            ValueError: If the model does not have the `label_definition_embeddings`
                attribute set using either the method `embed_and_set_label_definitions`,
                at initialisation by setting the `label_definitions_directory_path`
                argument, or by downloading a pre-trained model from the HuggingFace
                hub.
        """
        if not self.inference_ready:
            raise ValueError(
                "The model requires the `label_definition_embeddings` attribute "
                "to be set using either the method `embed_and_set_label_definitions`, "
                "at initialisation by setting the `label_definitions_directory_path` "
                "argument, or by downloading a pre-trained model from the HuggingFace "
                "hub."
            )
        if top_n == 0 or top_n < -1:
            raise ValueError(f"The top_n argument cannot be {top_n}, has to be either "
                             "-1 or a positive integer > 0.")
        if sub_word_tokenizer is None:
            sub_word_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)  # type: ignore
            assert isinstance(sub_word_tokenizer, PreTrainedTokenizerBase)
        model_device = self.base_model.device
        number_tokens = len(tokens)  # This can be seen as the batch size.

        sub_word_tokens = sub_word_tokenizer(tokens, return_tensors="pt",
                                             padding=False, truncation=False,
                                             is_split_into_words=True)
        sub_word_token_ids: torch.Tensor = sub_word_tokens.input_ids.to(device=model_device)
        sub_word_attention_mask: torch.Tensor = sub_word_tokens.attention_mask.to(device=model_device)
        sub_word_ids_to_token_ids: list[int | None] = sub_word_tokens.word_ids()
        number_subwords = len(sub_word_ids_to_token_ids)
        
        output = self.text_encoding(sub_word_token_ids, sub_word_attention_mask)

        word_id_mask: torch.Tensor = torch.zeros((number_tokens, number_subwords),
                                                 dtype=torch.long,
                                                 device=model_device)
        for sub_word_index, token_index in enumerate(sub_word_ids_to_token_ids):
            if token_index is None:
                continue
            word_id_mask[token_index, sub_word_index] = 1

        token_embeddings = self.token_encoding_using_text_encoding(output, word_id_mask)
        label_similarity_scores = self.token_label_similarity(self.label_definition_embeddings,  # type: ignore
                                                              token_embeddings)
        # torch does not support negative indexing like numpy or python does.
        if top_n == -1:
            top_n = label_similarity_scores.shape[-1]

        sorted_label_similarity_scores = torch.argsort(label_similarity_scores, dim=-1, descending=True)
        top_n_label_similarity_scores = sorted_label_similarity_scores[:, :top_n].cpu().tolist()

        predicted_labels: list[list[str]] = []
        for top_n_label_similarity_score in top_n_label_similarity_scores:
            predicted_labels.append(
                [self.embedding_index_to_label[top_n_index]  # type: ignore
                 for top_n_index in top_n_label_similarity_score]
            )

        return predicted_labels

    @overload
    def to(  # noqa: E704
        self,
        device: Optional[DeviceLikeType] = ...,
        dtype: Optional[torch.dtype] = ...,
        non_blocking: bool = ...,
    ) -> Self: ...

    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = ...) -> Self: ...  # noqa: E704

    @overload
    def to(self, tensor: torch.Tensor, non_blocking: bool = ...) -> Self: ...  # noqa: E704

    def to(self, *args, **kwargs):  # type: ignore
        r"""Move and/or cast the parameters and buffers.

        This has been taken from torch.nn.Module:
        https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/module.py#L1244

        This has been adapted so that the `label_definition_embeddings`
        are also moved to the correct device if `inference_ready` is `True`.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError(
                    "nn.Module.to only accepts floating point or complex "
                    f"dtypes, but got desired dtype={dtype}"
                )
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected."
                )

        def convert(t: torch.Tensor) -> torch.Tensor:
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise

        if self.inference_ready:
            self.label_definition_embeddings = convert(self.label_definition_embeddings)
        return self._apply(convert)
