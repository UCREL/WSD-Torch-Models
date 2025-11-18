import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

class NeuralInferenceModel():
    def __init__(self,
                 model_path: Path,
                 usas_mapper_path: Path,
                 usas_tags_to_filter_out: set[str] | None
                 ) -> None:
        logger.debug(f"Neural model path: {model_path}")
        logger.debug(f"USAS mapper path: {usas_mapper_path}")

        usas_tag_to_description_mapper: dict[str, str] | None = None
        self.usas_tags_to_filter_out = usas_tags_to_filter_out

        if usas_mapper_path.suffix == ".yaml":
            usas_tag_to_description_mapper = load_usas_mapper(usas_mapper_path,
                                                            self.usas_tags_to_filter_out)
        elif usas_mapper_path.suffix == ".json":
            import json
            with usas_mapper_path.open("r", encoding="utf-8") as mapper_fp:
                usas_tag_to_description_mapper = json.load(mapper_fp)
                if self.usas_tags_to_filter_out is not None:
                    for filter_tag in self.usas_tags_to_filter_out:
                        del usas_tag_to_description_mapper[filter_tag]
        self.usas_tag_to_description_mapper = usas_tag_to_description_mapper
        self.model = TokenSimilarityVariableNegatives.load_from_checkpoint(str(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.base_model_name, add_prefix_space=True)

        usas_embedded_descriptions_tensor: torch.Tensor | None = None
        usas_index_to_tag: dict[int, str] = {}
        self.model.eval()
        with torch.inference_mode(mode=True):
            usas_embedded_descriptions = []
            
            for index, usas_tag_description in enumerate(usas_tag_to_description_mapper.items()):
                usas_tag, usas_description = usas_tag_description
                tokenized_usas_description = self.tokenizer(usas_description, truncation=False, padding=False, return_tensors="pt")
                description_input_ids = tokenized_usas_description.input_ids.to(self.model.device).unsqueeze(0)
                description_attention_mask = tokenized_usas_description.attention_mask.to(self.model.device).unsqueeze(0)
                definition_embedding = self.model.label_definition_encoding(description_input_ids, description_attention_mask)
                usas_embedded_descriptions.append(definition_embedding)
                usas_index_to_tag[index] = usas_tag
            usas_embedded_descriptions_tensor = torch.vstack(usas_embedded_descriptions)
            NUM_DESC, DESC_BATCH, EMBEDDING_DIM = usas_embedded_descriptions_tensor.shape
            usas_embedded_descriptions_tensor = usas_embedded_descriptions_tensor.view(DESC_BATCH, NUM_DESC, EMBEDDING_DIM)
        assert isinstance(usas_embedded_descriptions_tensor, torch.Tensor)
        self.usas_embedded_descriptions_tensor = usas_embedded_descriptions_tensor
        self.usas_index_to_tag = usas_index_to_tag

    def inference(self,
                  tokens: list[str],
                  token_indexes: list[tuple[int, int]],
                  top_n: int) -> list[list[USASTagGroup]]:
        self.model.eval()
        prediction_labels: list[list[USASTagGroup]] = []
        with torch.inference_mode(mode=True):
            tokenized_text = self.tokenizer(tokens, truncation=False, padding=False, return_tensors="pt", is_split_into_words=True)
            print(tokenized_text)
            if tokenized_text.input_ids.shape[1] > self.tokenizer.model_max_length:
                raise ValueError(f"Text token length too large for model.")
            text_input_ids = tokenized_text.input_ids.to(self.model.device)
            text_attention_mask = tokenized_text.attention_mask.to(self.model.device)
            text_embedding = self.model.text_encoding(text_input_ids, text_attention_mask)
            
            for start_end_indexes in token_indexes:

                token_offset_indexes = set(range(*start_end_indexes))
                text_word_ids_mask = []
                for word_id in tokenized_text.word_ids():
                    if word_id is None:
                        text_word_ids_mask.append(0)
                    elif word_id in token_offset_indexes:
                        text_word_ids_mask.append(1)
                    else:
                        text_word_ids_mask.append(0)
                text_word_ids_mask = torch.tensor(text_word_ids_mask, dtype=torch.long)
                text_word_ids_mask = text_word_ids_mask.unsqueeze(0).to(device=self.model.device)
                if text_word_ids_mask.sum() == 0:
                    raise ValueError("Cannot find the token offsets in the given sample.")
                token_embedding = self.model.token_encoding_using_text_encoding(text_embedding, text_word_ids_mask)
                label_similarity_score = self.model.token_label_similarity(self.usas_embedded_descriptions_tensor, token_embedding)[0]
                top_n_sorted_label_similarity_score = torch.argsort(label_similarity_score, descending=True)[:top_n].cpu().tolist()
                predicted_usas_tags = [self.usas_index_to_tag[top_n_index] for top_n_index in top_n_sorted_label_similarity_score]
                prediction_usas_tag_groups = [USASTagGroup(tags=[USASTag(tag=predicted_usas_tag)]) for predicted_usas_tag in predicted_usas_tags]
                prediction_labels.append(prediction_usas_tag_groups)
        return prediction_labels
    
    def get_post_tagger_inference(self,
                                  top_n: int
                                  ) -> Callable[[list[str], tuple[int, int], list[str]], list[str]]:
        
        def post_tagger_inference(text_tokens: list[str],
                                  token_offsets: tuple[int, int],
                                  tagger_label_predictions: list[USASTagGroup]
                                  ) -> list[USASTagGroup]:
            inference_output: list[USASTagGroup] | None = None
            default_return = tagger_label_predictions
            if not tagger_label_predictions:
                inference_output = self.inference(text_tokens, [token_offsets], top_n)[0]
            elif len(tagger_label_predictions) == 0:
                inference_output = self.inference(text_tokens, [token_offsets], top_n)[0]
            elif len(tagger_label_predictions) == 1:
                tagger_label_prediction_tags = [tag.tag for tag in tagger_label_predictions[0].tags]
                if tagger_label_prediction_tags == ["Z99"]:
                    inference_output = self.inference(text_tokens, [token_offsets], top_n)[0]
                else:
                    return default_return
            else:
                return default_return

            assert isinstance(inference_output, list)
            return inference_output
            #return [usas_tag.tag for usas_tag_group in inference_output for usas_tag in usas_tag_group.tags]
        return post_tagger_inference