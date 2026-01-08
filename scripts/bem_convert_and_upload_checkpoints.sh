#!/bin/bash

usage() {
    echo "Usage: $0 <CHECKPOINT_DIR> <HUB_REPO>"
    echo "This script uploads the model checkpoints with step numbers in their checkpoint file names "
    echo "to the given HuggingFace Hub repository. The model checkpoints that are uploaded are "
    echo "all checkpoint file names in the given checkpoint directory with \"step=\" in their name."
    echo ""
    echo "  checkpoint_dir: Path to the directory that contains the model checkpoints to upload"
    echo "  hub_repo: The ID of the HuggingFace Hub repository to upload the model checkpoints too"
    exit 1
}

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

CHECKPOINT_DIR="$1"
HUB_REPO_ID="$2"

echo "checkpoint dir $CHECKPOINT_DIR"
echo "huggingface repo id $HUB_REPO_ID"

for file in "$CHECKPOINT_DIR"/*; do
    # Skip if not a file (e.g., directories)
    if [ ! -f "$file" ]; then
        continue
    fi

    # Extract just the filename
    filename=$(basename "$file")
    echo "Filename: \"$filename\""
    step_number=$(echo "$filename" | sed 's/.*step=\([0-9]*\)-.*/\1/')
    echo "Step number: \"$step_number\""
    
    uv run python scripts/convert_and_upload_bem_model.py "$HUB_REPO_ID" "$step_number" "$file" model_readmes/pymusas_bem.md -rmt
done