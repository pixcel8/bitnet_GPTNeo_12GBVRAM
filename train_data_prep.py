#!/usr/bin/env python3
# Data prep → tokenizer
# Target VRAM: ~12 GB on 16 GB RTX 5060

# ─── 0. Imports & Environment ───────────────────────────────────────────────────
import os
import json
import logging
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, Features, Value, DatasetDict
from tokenizers import ByteLevelBPETokenizer
import random
import glob
from huggingface_hub import list_repo_files, hf_hub_download
import re
import pandas as pd

# Environment variables
os.environ["HF_DATASETS_CACHE"] = "/mnt/l/Datasets/HuggingFace/cache"
os.environ["HF_HOME"] = "/mnt/l/Datasets/HuggingFace"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = "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"
hf_auth_token=os.getenv("HF_TOKEN")


# ─── 1. Config ──────────────────────────────────────────────────────────────────
stream_dataset = False
sample_dataset_size = 1000000
datasets_dir = os.path.join('/mnt/l/Datasets/', "working_sample_datasets")
os.makedirs(datasets_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

# ─── 2. Dataset Preparation ─────────────────────────────────────────────────────
def load_any_dataset(
    source,
    local_dir=None,
    match_subfolder=None,
    preferred_ext="*",
    hf_token=None,
    langs=None,
    sample_dataset_size=0,
    split="train",
    text_columns=None,
    map_fn=None,
    verbose=False
):
    """
    Universal loader for file-based, script-based, or local datasets.
    Returns a Dataset or DatasetDict with 'text' and 'meta_source' columns.
    """
    def replace_slashes(s): return re.sub(r"[\\/]", "___", s)

    # Local file
    if os.path.exists(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".csv":
            ds = Dataset.from_csv(source)
        elif ext == ".json":
            ds = Dataset.from_json(source)
        elif ext == ".parquet":
            ds = Dataset.from_parquet(source)
        else:
            raise ValueError(f"Unsupported local file type: {ext}")
        ds = ds.add_column("meta_source", [os.path.basename(source)] * len(ds))
    else:
        # Try HF static first
        files = list_repo_files(source, repo_type="dataset")
        supported_exts = ["parquet", "json", "csv"]
        available_exts = {ext: [] for ext in supported_exts}
        for f in files:
            for ext in supported_exts:
                if f.endswith(f".{ext}"):
                    available_exts[ext].append(f)

        if not any(available_exts.values()):
            if verbose: print("ℹ Script-based dataset detected — using load_dataset()")
            ds = load_dataset(
                source,
                split=split,
                cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                use_auth_token=hf_token
            )
            ds = ds.add_column("meta_source", [source] * len(ds))
        else:
            if preferred_ext == "*":
                for ext in supported_exts:
                    if available_exts[ext]:
                        preferred_ext = ext
                        if verbose: print(f"🔍 Inferred extension: {preferred_ext}")
                        break
            lang_dirs = set()
            for f in files:
                parts = f.split("/")
                if len(parts) >= 3 and parts[0] == match_subfolder:
                    lang_dirs.add(parts[1])
            if not lang_dirs:
                langs = [""]
            elif langs == ["default"] or not langs:
                langs = list(lang_dirs)
            dataset_path = os.path.join(local_dir or ".", replace_slashes(source))
            sel_files = [
                f for lang in langs
                for f in available_exts[preferred_ext]
                if f.startswith(f"{match_subfolder}/{lang}/")
                or (lang == "" and f.startswith(f"{match_subfolder}/"))
            ]
            for fpath in sel_files:
                hf_hub_download(
                    repo_id=source,
                    filename=fpath,
                    repo_type="dataset",
                    token=hf_token,
                    local_dir=dataset_path,
                    local_dir_use_symlinks=False
                )
            datasets_list = []
            for lang in langs:
                lang_path = os.path.join(dataset_path, match_subfolder, lang) if lang else os.path.join(dataset_path, match_subfolder)
                matched_files = glob.glob(os.path.join(lang_path, f"*.{preferred_ext}"))
                if preferred_ext == "parquet":
                    lang_datasets = [Dataset.from_parquet(fp) for fp in matched_files]
                elif preferred_ext == "json":
                    lang_datasets = [Dataset.from_json(fp) for fp in matched_files]
                elif preferred_ext == "csv":
                    lang_datasets = [Dataset.from_csv(fp) for fp in matched_files]
                datasets_list.append(concatenate_datasets(lang_datasets))
            ds = concatenate_datasets(datasets_list)
            ds = ds.add_column("meta_source", [source] * len(ds))

    # Apply mapping if given
    if map_fn:
        ds = ds.map(map_fn, load_from_cache_file=False)

    # Standardise text column
    if text_columns and "text" not in ds.column_names:
        ds = ds.map(lambda ex: {"text": " ".join(str(ex[col]) for col in text_columns if col in ex)})

    # Sample
    if sample_dataset_size > 0:
        ds = ds.shuffle(seed=42).select(range(min(sample_dataset_size, len(ds))))

    # Split
    if isinstance(ds, Dataset):
        split_ds = ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
        return DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})
    return ds


def huggingface_select_downloader(repo, local_dir, match_subfolder, preferred_ext, hf_token, langs, sample_dataset_size=0, verbose=False):
    def replace_slashes(source: str) -> str:
        return re.sub(r"[\\/]", "___", source)

    dataset_path = os.path.join(local_dir, replace_slashes(repo))
    print(f"⏳ Downloading selected subfolders from {repo}")

    files = list_repo_files(repo, repo_type="dataset")
    supported_exts = ["parquet", "json", "csv"]
    available_exts = {ext: [] for ext in supported_exts}

    # Index available files by extension
    for f in files:
        for ext in supported_exts:
            if f.endswith(f".{ext}"):
                available_exts[ext].append(f)

    # Handle wildcard extension
    if preferred_ext == "*":
        for ext in supported_exts:
            if available_exts[ext]:
                preferred_ext = ext
                print(f"🔍 Inferred file extension: .{preferred_ext}")
                break
        else:
            print("🚫 No supported file types found in dataset.")
            print("📦 Dataset structure overview:")
            for f in files:
                print(f" - {f}")
            raise ValueError("❌ No compatible files found. Check dataset format or structure.")

    # Auto-detect language subfolders
    lang_dirs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == match_subfolder:
            lang_dirs.add(parts[1])

    if not lang_dirs:
        print("⚠️ No language subfolders detected. Assuming flat structure.")
        langs = [""]
    elif langs == ["default"] or not langs:
        langs = list(lang_dirs)
        print(f"🌐 Auto-detected languages: {langs}")

    # Select files matching subfolder, language, and extension
    sel_files = [
        f for lang in langs
        for f in available_exts[preferred_ext]
        if f.startswith(f"{match_subfolder}/{lang}/") or (lang == "" and f.startswith(f"{match_subfolder}/"))
    ]

    if not sel_files:
        print("🚫 No matching files found for selected languages and extension.")
        print("📦 Dataset structure overview:")
        for f in files:
            print(f" - {f}")
        raise ValueError("❌ No compatible files found. Check dataset format or language filters.")

    # Download selected files
    for fpath in sel_files:
        local_path = hf_hub_download(
            repo_id=repo,
            filename=fpath,
            repo_type="dataset",
            token=hf_token,
            local_dir=dataset_path,
            local_dir_use_symlinks=False
        )
        if verbose:
            print(f"📥 Downloaded: {local_path}")

    print(f"📂 Loading and concatenating from {dataset_path} using .{preferred_ext}")

    datasets_list = []
    for lang in langs:
        lang_path = os.path.join(dataset_path, match_subfolder, lang) if lang else os.path.join(dataset_path, match_subfolder)
        file_pattern = os.path.join(lang_path, f"*.{preferred_ext}")
        matched_files = glob.glob(file_pattern)

        if not matched_files:
            print(f"⚠️ No .{preferred_ext} files found for language: {lang or '[flat]'}")
            continue

        if preferred_ext == "parquet":
            lang_datasets = [Dataset.from_parquet(fp) for fp in matched_files]
        elif preferred_ext == "json":
            lang_datasets = [Dataset.from_json(fp) for fp in matched_files]
        elif preferred_ext == "csv":
            lang_datasets = [Dataset.from_csv(fp) for fp in matched_files]

        datasets_list.append(concatenate_datasets(lang_datasets))

    if not datasets_list:
        print("⚠️ No datasets loaded. Returning empty dataset.")
        return Dataset.from_dict({})

    datasets_merge = concatenate_datasets(datasets_list)

    if sample_dataset_size > 0:
        sample_size = min(sample_dataset_size, len(datasets_merge))
        datasets_merge = datasets_merge.shuffle(seed=42).select(range(sample_size))
        print(f"🎯 Sampled {sample_size} examples from full dataset")

    return datasets_merge




def sample_streaming_dataset(streaming_ds, sample_size=1000000, chunk_size=10000, seed=42, name="dataset"):
    np.random.seed(seed)
    buffer = []
    retries = 3
    iterator = iter(streaming_ds)
    pbar = tqdm(total=sample_size, desc=f"Sampling {name}")

    while len(buffer) < sample_size:
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(iterator))
            except StopIteration:
                break
            except Exception as e:
                logging.warning(f"Error while streaming {name}: {e}")
                time.sleep(2)
                continue

        if not chunk:
            break

        buffer.extend(chunk)
        pbar.update(len(chunk))

        if len(buffer) >= sample_size:
            break

    np.random.shuffle(buffer)
    return Dataset.from_list(buffer[:sample_size])

def load_random_shards(dataset_name, config=None, split="train", num_shards=10, seed=42):
    builder = load_dataset(dataset_name, config, split=split, streaming=True)
    shard_paths = builder._data_files  # internal, but works
    random.seed(seed)
    selected = random.sample(shard_paths, num_shards)
    return load_dataset(dataset_name, config, split=split, data_files=selected)

def load_random_cached_shards(dataset_name, config, cache_root, num_shards):
    # Define the path to the cached dataset shards
    train_dir = os.path.join(cache_root, config, "train")
    val_dir = os.path.join(cache_root, config, "validation")

    # List all available shard files
    train_files = sorted([
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.endswith(".json.gz")
    ])
    val_files = sorted([
        os.path.join(val_dir, f) for f in os.listdir(val_dir)
        if f.endswith(".json.gz")
    ])

    # Randomly select shards
    selected_train = random.sample(train_files, min(num_shards, len(train_files)))
    selected_val = random.sample(val_files, min(num_shards, len(val_files)))

    selected_files = {
        "train": selected_train,
        "validation": selected_val
    }

    # Load the dataset using local files only
    dataset = load_dataset(
        path=dataset_name,
        name=config,
        data_files=selected_files,
        cache_dir=cache_root,
        local_files_only=True
    )

    return dataset

def shuffle_dataset(ds, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(ds))
    return ds.select(indices)

def nq_to_text(example):
    question = example["question"]["text"]
    doc = example["document"]["text"]
    return {"text": f"Q: {question}\n\n{doc}"}

def ensure_text_column(ds, fallback_keys=["content", "prompt", "code", "text"]):
    def to_text(example):
        for key in fallback_keys:
            if key in example and isinstance(example[key], str):
                return {"text": example[key]}
        return {"text": ""}
    
    if isinstance(ds, DatasetDict):
        for split in ds:
            if "text" not in ds[split].column_names:
                ds[split] = ds[split].map(to_text, load_from_cache_file=False)
    else:
        if "text" not in ds.column_names:
            ds = ds.map(to_text, load_from_cache_file=False)
    return ds

def length_filter(ex):
    n = len(ex["text"].split())
    return 128 < n < 512


if not os.path.exists(os.path.join(datasets_dir, "combined")):
            
    # If the full datasets have been downloaded, you can add keep_in_memory=True with local_files_only=True to load_dataset
    #raw_en_ds = load_dataset("allenai/c4", "en", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"], streaming=True)
    #raw_en_ds = load_random_shards("allenai/c4", config="en", num_shards=10)
    #en_ds = sample_streaming_dataset(raw_en_ds, sample_size=sample_dataset_size, name="C4_English")
    #raw_en_ds = load_random_cached_shards("allenai/c4", config="en", cache_root=en_ds_path, num_shards=10)

    en_ds_path = os.path.join(os.environ["HF_HOME"], "datasets--allenai--c4/en")
    if not os.path.exists(os.path.join(datasets_dir, "en_ds")): #If dataset path does not exist regenerate dataset
        print("⏳ Loading English (C4)")
        features = Features({
            "text": Value("string"),
            "metadata": Value("string")  # adjust as needed
        })
        # Grab all matching shard files
        train_files = glob.glob(f"{en_ds_path}/train/c4-train.*.json.gz")
        val_files = glob.glob(f"{en_ds_path}/validation/c4-validation.*.json.gz")

        # Randomly select shards
        selected_train = random.sample(train_files, min(10, len(train_files)))
        selected_val = random.sample(val_files, min(10, len(val_files)))
        print("Train shards:", selected_train)
        print("Validation shards:", selected_val)

        selected_files = {
            "train": selected_train,
            "validation": selected_val
        }

        # Load using the generic JSON loader
        raw_en_ds = load_dataset(
            path="json",
            data_files=selected_files,
            #features=features,
            cache_dir=os.environ["HF_DATASETS_CACHE"],
        )
        #en_ds = shuffle_dataset(raw_en_ds)
        en_ds = DatasetDict({
            split: ds.shuffle(seed=42)
            for split, ds in raw_en_ds.items()
        })
        en_ds.save_to_disk(os.path.join(datasets_dir, "en_ds"))
    else:
        print("⏳ Loading English (C4) - Sampled")
        en_ds = load_from_disk(os.path.join(datasets_dir, "en_ds"))
        

    if not os.path.exists(os.path.join(datasets_dir, "python_ds")): #If dataset path does not exist regenerate dataset
        print("⏳ Loading bigcode/the-stack Python")
        langs = ["css", "batchfile", "python", "assembly"]
        #python_ds = huggingface_select_downloader("bigcode/the-stack",os.environ["HF_HOME"],"data", "parquet", os.environ["HF_TOKEN"], langs, sample_dataset_size )
        # bigcode/the-stack from HF static
        python_ds = load_any_dataset(
            source="bigcode/the-stack",
            local_dir=os.environ["HF_HOME"],
            match_subfolder="data",
            preferred_ext="parquet",
            hf_token=os.environ["HF_TOKEN"],
            langs=["css", "batchfile", "python", "assembly"],
            sample_dataset_size=sample_dataset_size
        )
        #nq_ds = sample_streaming_dataset(raw_nq_ds, sample_size=sample_dataset_size, name="NaturalQuestions")
        #python_ds = load_dataset("bigcode/the-stack", split="train.shuffle(seed=42)[:sample_dataset_size]", cache_dir=os.environ["HF_DATASETS_CACHE"], streaming=stream_dataset, use_auth_token=hf_auth_token)
        #sciphi_ds = load_dataset("SciPhi/textbooks-are-all-you-need-lite", split="train.shuffle(seed=42)[:sample_dataset_size]", cache_dir=os.environ["HF_DATASETS_CACHE"], streaming=stream_dataset, use_auth_token=hf_auth_token)
        #nq_ds = load_dataset("natural_questions", split="train.shuffle(seed=42)[:sample_dataset_size]", cache_dir=os.environ["HF_DATASETS_CACHE"], streaming=stream_dataset, use_auth_token=hf_auth_token).map(nq_to_text)
        #raw_python_ds = concatenate_datasets([load_dataset("bigcode/the-stack", lang, split="train") for lang in langs])
        #raw_python_ds = load_dataset("bigcode/the-stack", "python", cache_dir=os.environ["HF_DATASETS_CACHE"], streaming=stream_dataset)
        #raw_python_ds = load_random_shards("bigcode/the-stack", num_shards=10)
        #raw_python_ds = raw_python_ds.filter(lambda x: x.get("lang") == "python")
        #python_ds = sample_streaming_dataset(raw_python_ds, sample_size=sample_dataset_size, name="Python")
        #raw_python_ds = load_random_cached_shards("bigcode/the-stack", config=none, cache_root=os.environ["HF_DATASETS_CACHE"], num_shards=10)
        #python_ds = shuffle_dataset(raw_python_ds)
        
        # Shuffle and split
        #split_python_ds = python_ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)

        # Wrap into DatasetDict for consistency with en_ds
        #python_ds = DatasetDict({
        #    "train": split_python_ds["train"],
        #    "validation": split_python_ds["test"]
        #})

        # Save to disk
        python_ds.save_to_disk(os.path.join(datasets_dir, "python_ds"))
    else:
        print("⏳ Loading bigcode/the-stack Python - Sampled")
        python_ds = load_from_disk(os.path.join(datasets_dir, "python_ds"))


    if not os.path.exists(os.path.join(datasets_dir, "sciphi_ds")): #If dataset path does not exist regenerate dataset
        print("⏳ Loading SciPhi/textbooks-are-all-you-need-lite Python")
        langs = ["default"]
        #sciphi_ds = huggingface_select_downloader("SciPhi/textbooks-are-all-you-need-lite",os.environ["HF_HOME"],"data", "*", os.environ["HF_TOKEN"], langs, sample_dataset_size )
        # SciPhi from HF static
        sciphi_ds = load_any_dataset(
            source="SciPhi/textbooks-are-all-you-need-lite",
            local_dir=os.environ["HF_HOME"],
            match_subfolder="data",
            preferred_ext="*",
            hf_token=os.environ["HF_TOKEN"],
            langs=["default"],
            sample_dataset_size=sample_dataset_size
        )
        
        # Shuffle and split
        #split_sciphi_ds = sciphi_ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
        # Wrap into DatasetDict for consistency with en_ds
        #sciphi_ds = DatasetDict({
        #    "train": split_sciphi_ds["train"],
        #    "validation": split_sciphi_ds["test"]
        #})

        # Save to disk
        sciphi_ds.save_to_disk(os.path.join(datasets_dir, "sciphi_ds"))
    else:
        print("⏳ Loading SciPhi/textbooks-are-all-you-need-lite - Sampled")
        sciphi_ds = load_from_disk(os.path.join(datasets_dir, "sciphi_ds"))  


    if not os.path.exists(os.path.join(datasets_dir, "nq_ds")):
        print("⏳ Loading Natural Questions from CSV")
        
        csv_path = "/mnt/l/Datasets/NaturalQuestions/Natural-Questions-Base.csv"
        # Load with pandas
        df_nq = pd.read_csv(csv_path)

        # Map to single `text` column
        df_nq["text"] = df_nq.apply(
            lambda row: f"Q: {row['question']}\n\nLong Answer: {row['long_answers']}\n\nShort Answer: {row['short_answers']}",
            axis=1
        )

        # Convert to Hugging Face Dataset
        nq_ds = Dataset.from_pandas(df_nq[["text"]])

        # Shuffle + split
        split_nq = nq_ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
        nq_ds = DatasetDict({
            "train": split_nq["train"],
            "validation": split_nq["test"]
        })

        nq_ds.save_to_disk(os.path.join(datasets_dir, "nq_ds"))

    else:
        print("⏳ Loading Natural Questions - Sampled")
        nq_ds = load_from_disk(os.path.join(datasets_dir, "nq_ds"))
    
    print("⏳ Generate Combined dataset")
        
    print("🔍 Ensuring 'text' column exists in all datasets…")
    python_ds = ensure_text_column(python_ds)
    sciphi_ds = ensure_text_column(sciphi_ds)
    nq_ds     = ensure_text_column(nq_ds)
    en_ds     = ensure_text_column(en_ds)    


    print("🔍 Filtering datasets by length (128–512 tokens)…")
    en_ds   = en_ds.filter(length_filter)
    python_ds = python_ds.filter(length_filter)
    sciphi_ds = sciphi_ds.filter(length_filter)
    nq_ds = nq_ds.filter(length_filter)

    print("Prepare for Weighted Sampling")
    en_ds     = en_ds.map(lambda x: {**x, "source": "c4"})
    python_ds = python_ds.map(lambda x: {**x, "source": "python"})
    sciphi_ds = sciphi_ds.map(lambda x: {**x, "source": "sciphi"})
    nq_ds     = nq_ds.map(lambda x: {**x, "source": "nq"})

    print("🔗 Concatenating datasets…")
    #combined = concatenate_datasets([ en_ds, python_ds, sciphi_ds, nq_ds])
    #Tag with source datasets
    def tag_source(ds, name):
        return ds.map(lambda ex: {"source": name}, load_from_cache_file=False)
    combined_train = concatenate_datasets([
        tag_source(en_ds["train"], "en"),
        tag_source(python_ds["train"], "python"),
        tag_source(sciphi_ds["train"], "sciphi"),
        tag_source(nq_ds["train"], "nq")
    ])
    combined_val = concatenate_datasets([
        tag_source(en_ds["validation"], "en"),
        tag_source(python_ds["validation"], "python"),
        tag_source(sciphi_ds["validation"], "sciphi"),
        tag_source(nq_ds["validation"], "nq")
    ])

    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_val
    })

    print(f"   → {len(combined['train']) + len(combined['validation'])} examples total")

    print("Saving Working Datasets")
    combined.save_to_disk(os.path.join(datasets_dir, "combined"))
else: 
    print("⏳ Combined Dataset found")

# ─── 3. Tokenizer Training ──────────────────────────────────────────────────────

tokeniser_path = os.path.join(datasets_dir, "combined", "tokenizer_bpe")
os.makedirs(tokeniser_path, exist_ok=True)
if not os.path.exists(tokeniser_path):
    print("⏳ Training tokeniser from scratch")

    raw_txt = os.path.join(tokeniser_path, "all_texts.txt")
    min_tok_freq = 2
    vocab_size = 150_000

    print("✍️  Writing raw texts for tokenizer…")

    with open(raw_txt, "w", encoding="utf-8") as f:
        for split in combined:
            for ex in combined[split]:
                if "text" in ex and isinstance(ex["text"], str):
                    f.write(ex["text"].replace("\n", " ") + "\n")

    print("🚀 Training Byte-Level BPE tokenizer…")
    tokenizer_bpe = ByteLevelBPETokenizer()
    tokenizer_bpe.train(
        files=[raw_txt],
        vocab_size=vocab_size,
        min_frequency=min_tok_freq,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer_bpe.save_model(tokeniser_path)

    #Save configuration of tokeniser parameters
    with open(os.path.join(tokeniser_path,"config.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "min_frequency": min_tok_freq}, f)
else: 
    print("⏳ Tokeniser found")
print("\U0001F4BE Combined Dataset and Tokeniser ready - proceed to training")  # Unicode for 💾