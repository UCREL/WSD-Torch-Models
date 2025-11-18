from wsd_torch_models.bem import BEM


if __name__ == "__main__": 
    wsd_model = BEM.from_pretrained("./test_bem")