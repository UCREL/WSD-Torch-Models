import torch
from transformers import AutoTokenizer

from wsd_torch_models.bem import BEM


if __name__ == "__main__": 
    wsd_model_name = "ucrelnlp/PyMUSAS-Neural-Engish-Small-BEM"
    wsd_model = BEM.from_pretrained(wsd_model_name)
    tokenizer = AutoTokenizer.from_pretrained(wsd_model_name)

    wsd_model.eval()
    # Change this to the device you would like to use, e.g. cpu
    model_device = "cpu"
    wsd_model.to(device=model_device)
    
    sentence = "The river bank was full of fish"
    sentence_tokens = sentence.split()
    
    with torch.inference_mode(mode=True):
        # sub_word_tokenizer can be None when None it will download the appropriate tokenizer
        # but generally it is better to give it the tokenizer as it saves the operation
        # of checking if the tokenizer is already downloaded.
        predictions = wsd_model.predict(sentence_tokens, sub_word_tokenizer=tokenizer, top_n=5)
        
        for sentence_token, semantic_tags in zip(sentence_tokens, predictions):
            print(f"Token: {sentence_token}")
            print("Most likely tags: ")
            for tag in semantic_tags:
                tag_definition = wsd_model.label_to_definition[tag]
                print(f"\t{tag}: {tag_definition}")
            print()