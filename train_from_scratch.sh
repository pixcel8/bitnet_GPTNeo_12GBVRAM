export HF_HOME=/mnt/l/Datasets/HuggingFace
export HF_DATASETS_CACHE=/mnt/l/Datasets/HuggingFace/cache
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"
export HF_TOKEN=[HF TOKEN NEEDED]

huggingface-cli login --token [HF TOKEN NEEDED]
python train_data_prep.py
python train_from_scratch.py
