export HF_HOME=/mnt/l/Datasets/HuggingFace
export HF_DATASETS_CACHE=/mnt/l/Datasets/HuggingFace/cache
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"
#mkdir -p /mnt/l/Datasets/HuggingFace/cache
#echo "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" > /mnt/l/Datasets/HuggingFace/cache/stored_tokens
export HF_TOKEN="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"

huggingface-cli login --token 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
python train_data_prep.py
python train_from_scratch.py