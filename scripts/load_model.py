from pathlib import Path

from safetensors.torch import save_file
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from wsd_torch_models.bem import BEM


if __name__ == "__main__": 
    model_checkpoint = Path("./checkpoints/model-step=532637-validation_accuracy=0.99394.ckpt")
    torch_lightning_checkpoint = torch.load(model_checkpoint)

    hyper_parameters_keys = [
        "base_model_name",
        "freeze_base_model",
        "number_transformer_encoder_layers",
        "add_scalar_mixer",
        "scalar_mix_layer_norm",
        "transformer_encoder_hidden_dim",
        "transformer_encoder_num_heads",
        "batch_first"
    ]
    hyper_parameters_dict = {
        hyper_parameter_key: torch_lightning_checkpoint["hyper_parameters"][hyper_parameter_key]
        for hyper_parameter_key in hyper_parameters_keys
    }

    model_weights = torch_lightning_checkpoint["state_dict"]
    # update keys by dropping `auto_encoder.`
    #for key in list(model_weights):
    #    model_weights[key.replace("base_model.", "")] = model_weights.pop(key)

    #save_file(model_weights, "model.safetensors")

    #config = AutoConfig.from_pretrained("./bem_config")
    #a_model = AutoModel.from_config(config)
    
    #a_model = AutoModel.from_pretrained("model.safetensors", config=config)

    #hyper_parameters_dict["base_model"] = a_model
    
    wsd_model = BEM(**hyper_parameters_dict)
    wsd_model.save_pretrained("./test_bem")

    #model_save_path = "bem_model"
    #wsd_model.save_pretrained(model_save_path, config=hyper_parameters_dict, push_to_hub=False)


    #tokenizer = AutoTokenizer.from_pretrained(hyper_parameters_dict["base_model_name"])