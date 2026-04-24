neural_model_huggingface_ids=(
    "ucrelnlp/PyMUSAS-Neural-English-Small-BEM"
    "ucrelnlp/PyMUSAS-Neural-English-Base-BEM"
    "ucrelnlp/PyMUSAS-Neural-Multilingual-Small-BEM"
    "ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM"
)
echo "Installing the neural PyMUSAS models"
for neural_model_huggingface_id in "${neural_model_huggingface_ids[@]}"; do
    uv run --no-group cpu --group cu128 download_model.py ${neural_model_huggingface_id}
done
echo "Finished installing the neural PyMUSAS models for all languages"