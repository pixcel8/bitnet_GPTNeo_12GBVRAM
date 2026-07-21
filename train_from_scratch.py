#!/usr/bin/env python3
# Train 4-bit “BitNet” model → eval → best-model saving
# Target VRAM: ~12 GB on 16 GB RTX 5060

# ─── 0. Imports & Environment ───────────────────────────────────────────────────
import os
import json
import logging
import math
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from tokenizers import ByteLevelBPETokenizer
from packaging import version
import transformers
from transformers import (
    RobertaTokenizerFast,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from bitsandbytes.optim import Adam8bit
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import EarlyStoppingCallback
from transformers import AutoModelForCausalLM


# ─── 1. Config ──────────────────────────────────────────────────────────────────
#datasets_dir = os.path.join(os.getcwd(), "working_datasets")
datasets_dir = os.path.join('/mnt/l/Datasets/', "working_sample_datasets")
use_modern_training = False
is_modern = version.parse(transformers.__version__) >= version.parse("4.4.0") and use_modern_training
print(f"🤖 Transformers {transformers.__version__} | modern branch: {is_modern}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

def measure_gpu_throughput(test_hidden=1024, test_layers=4, test_seq_len=256, batch_size=1, steps=5):
    """Return seconds per micro-batch for a small GPT-Neo-like block."""
    config = GPTNeoConfig(
        vocab_size=50257,
        max_position_embeddings=test_seq_len,
        hidden_size=test_hidden,
        num_hidden_layers=test_layers,
        num_attention_heads=test_hidden // 64,
        intermediate_size=test_hidden * 4
    )
    model = GPTNeoForCausalLM(config).to("cuda")
    model.train()

    dummy_input = torch.randint(0, config.vocab_size, (batch_size, test_seq_len), device="cuda")
    dummy_labels = dummy_input.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warm-up
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(dummy_input, labels=dummy_labels).loss
        loss.backward()
        optimizer.step()

    # Timed run
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(dummy_input, labels=dummy_labels).loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / steps  # seconds per micro-batch
    
combined = load_from_disk(os.path.join(datasets_dir, "combined"))
tokenizer_path = os.path.join(datasets_dir, "combined", "tokenizer_bpe")

print("🔄 Wrapping in RobertaTokenizerFast…")
tokenizer = RobertaTokenizerFast.from_pretrained(
    tokenizer_path,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
)

# ✅ Set max length explicitly to match training
tokenizer.model_max_length = 512
def preconfigure_training(train_dataset, tokenizer, target_hours=None, target_params_millions=None, eval_fraction=0.05, save_fraction=0.05, vram_safety_margin_gb=2):
    """
    Preconfigure model + training args based on desired training time or model size.
    Reads dataset size from train_dataset to avoid hardcoding.
    
    Args:
        train_dataset: HF Dataset object for training split
        target_hours: float, total wall-clock hours you want to spend training
        target_params_millions: float, desired model size in millions of parameters
    """
    gpu_training_perf_eval(num_examples, total_vram_gb):
        # Your measured throughput baseline
        measured_s_per_microbatch = 0.043
        test_params_m = (1024**2 * 12 * 4) / 1e6  # ~50M params for test config
        test_seq_len = 256
        per_device_batch_size = 1
        grad_accum = 8
        effective_batch_size = per_device_batch_size * grad_accum
        steps_per_epoch = math.ceil(num_examples/ effective_batch_size)

        # VRAM model (FP16 weights + grads + ~2GB overhead)
        def estimate_vram_usage_gb(params_b, seq_len):
            weights_gb = params_b * 2 * 2 / 1024  # params × bytes × (weights+grads)
            activations_gb = params_b * (seq_len / 512) * 0.5  # rough scaling
            return weights_gb + activations_gb + 2  # +2GB overhead

        # Time the model
        def estimate_train_time_hours(params_b, seq_len):
            s_per_microbatch = measured_s_per_microbatch * (params_b*1000 / test_params_m) * (seq_len / test_seq_len)
            s_per_step = s_per_microbatch * grad_accum
            return (s_per_step * steps_per_epoch) / 3600

        # Grid
        params_range = np.linspace(0.7, 7.0, 50)  # B params
        seq_range = np.linspace(128, 768, 50)
        P, S = np.meshgrid(params_range, seq_range)

        GPUVRAM = estimate_vram_usage_gb(P, S)
        GPUTIME = estimate_train_time_hours(P, S)

        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        c = ax.contourf(P, S, TIME, levels=20, cmap='viridis')
        cb = fig.colorbar(c, ax=ax)
        cb.set_label("Training Time [hours] (1 epoch)")

        # VRAM limit contour
        vram_limit = total_vram_gb
        ax.contour(P, S, VRAM, levels=[vram_limit], colors='red', linewidths=2)
        ax.clabel(ax.contour(P, S, VRAM, levels=[vram_limit], colors='red'), fmt=f"{vram_limit}GB VRAM limit", colors='red')

        ax.set_xlabel("Params [B]")
        ax.set_ylabel("Sequence Length")
        ax.set_title("Training Time & VRAM Constraints (1 epoch)")

        plt.show()
        
        
    # --- 1. Dataset size ---
    num_examples = len(train_dataset)
    print(f"📊 Training examples: {num_examples:,}")
    
    # --- 2. Hardware throughput baseline (adjust for your GPU) ---
    # GPU VRAM
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_vram_gb = torch.cuda.memory_allocated(0) / 1024**3
    usable_vram_gb = total_vram_gb - vram_safety_margin_gb
    print(f"💾 GPU VRAM: {total_vram_gb:.1f} GB (usable ~{usable_vram_gb:.1f} GB)")

    # Measure throughput
    print("⚡ Measuring GPU throughput...")
    measured_s_per_microbatch = measure_gpu_throughput()
    print(f"   → {measured_s_per_microbatch:.3f} s/micro-batch (test config)")

    # Baseline params for scaling
    test_params_m = (1024**2 * 12 * 4) / 1e6  # rough param count for test config
    test_seq_len = 256

    # --- 3. Decide model size ---
    if target_params_millions:
        params_m = target_params_millions
    elif target_hours:
        steps_per_epoch = math.ceil(num_examples / 8)
        total_steps = steps_per_epoch * 3
        total_microbatches = total_steps * 8
        total_seconds_budget = target_hours * 3600
        params_m = test_params_m * (total_seconds_budget / (total_microbatches * measured_s_per_microbatch))
        params_m = max(125, min(params_m, 6000))
    else:
        params_m = 1300

    # --- 4. Derive model dimensions from params ---
    if params_m <= 1500:
        num_layers = 24
    elif params_m <= 3000:
        num_layers = 28
    else:
        num_layers = 32

    hidden_size = int(((params_m * 1e6) / (12 * num_layers)) ** 0.5)
    hidden_size = max(64, (hidden_size // 64) * 64)
    # Now set heads so it divides evenly
    num_heads = max(1, hidden_size // 64)

    # Sequence length tuning
    vram_per_token_gb = (hidden_size * num_layers * 2) / 1e9
    max_seq_len = min(512, int(usable_vram_gb / (vram_per_token_gb * 1.5)))
    max_seq_len = max(128, (max_seq_len // 8) * 8)
    tokenizer.model_max_length = max_seq_len

    # Gradient accumulation tuning
    grad_accum = 8
    if usable_vram_gb > 14:
        grad_accum = 4
    elif usable_vram_gb < 10:
        grad_accum = 16

    # --- 5. Estimate training speed and adjust epochs ---
    steps_per_epoch = math.ceil(num_examples / (1 * grad_accum))
    s_per_microbatch = measured_s_per_microbatch * (params_m / test_params_m) * (max_seq_len / test_seq_len)
    s_per_step = s_per_microbatch * grad_accum
    if target_hours:
        total_steps_budget = int((target_hours * 3600) / s_per_step)
        num_epochs = max(1, total_steps_budget // steps_per_epoch)
    else:
        num_epochs = 3

    # Dynamic eval/save steps
    eval_steps = max(1, int(steps_per_epoch * eval_fraction))
    save_steps = max(1, int(steps_per_epoch * save_fraction))

    # --- 6. Build config and args ---
    config = GPTNeoConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        model_type="gpt_neo",
        gradient_checkpointing=True,
        attention_layers=["global" if i % 2 == 0 else "local" for i in range(num_layers)]
    )
    #config = GPTNeoConfig(
    #vocab_size=tokenizer.vocab_size,
    #max_position_embeddings=1024,
    #hidden_size=4096,
    #num_hidden_layers=32,
    #num_attention_heads=32,
    #intermediate_size=4096 * 4,
    #model_type="gpt_neo" # Added for switch to AutoModelForCausalLM
    #)
    
    # Base Training args

    # Build TrainingArguments
    common_args = dict(
        output_dir="models",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        fp16=False,
        bf16=True,
        num_train_epochs=num_epochs,
        logging_steps=100,
        save_total_limit=2,
        logging_dir="logs",
        learning_rate=5e-5,
    )

    # TrainingArguments
    if is_modern:
        training_args = TrainingArguments(
            **common_args,
            evaluation_strategy="steps",
            eval_steps = eval_steps,
            save_strategy="steps",
            save_steps = save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="perplexity",
            greater_is_better=False,
        )
    else:
        training_args = TrainingArguments(
            **common_args,
            do_eval=True,                 # legacy eval switch
            # Do NOT set evaluation_strategy here (not supported on legacy)
            # Avoid load_best_model_at_end to prevent strategy mismatch assertions
            load_best_model_at_end=False,
            # metric_for_best_model not required in legacy since we use our own callback
        )

    # Summary
    print(f"📐 Model: ~{params_m/1000:.2f}B params | hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    print(f"📏 Seq length: {max_seq_len} | grad_accum={grad_accum}")
    print(f"🗂 Steps/epoch: {steps_per_epoch} | epochs={num_epochs}")
    print(f"🔍 Eval every {eval_steps} steps | 💾 Save every {save_steps} steps")
    print(f"⏱ Est. {s_per_step:.1f}s/step → ~{(steps_per_epoch*num_epochs*s_per_step)/3600:.1f}h total")
    return config, training_args

# ─── 4. Tokenization & Splitting ────────────────────────────────────────────────
def tokenize_and_label(examples):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    out["labels"] = out["input_ids"].copy()
    return out

print("🔄 Tokenizing dataset…")
tokenized = combined.map(tokenize_and_label, batched=True, remove_columns=[])
print(tokenized["train"].column_names)

print("✂️  Splitting train/validation")
train_ds = tokenized["train"]
eval_ds  = tokenized["validation"]
assert "source" in train_ds.column_names, "'source' column missing from training set"

source_weights = {
    "en": 0.3,
    "python": 0.35,
    "sciphi": 0.25,
    "nq": 0.1
}
weights = [source_weights.get(ex["source"], 0.1) for ex in train_ds]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# Reassign after formatting
train_ds = tokenized["train"]
eval_ds  = tokenized["validation"]

#splits = tokenized.train_test_split(test_size=0.2, seed=42)
#train_ds = splits["train"]
#eval_ds  = splits["test"]
print(f"   → Train: {len(train_ds)} examples")
print(f"   → Eval:  {len(eval_ds)} examples")


# ─── 5. Model Setup ─────────────────────────────────────────────────────────────

config, training_args = preconfigure_training(
    train_dataset=train_ds,
    tokenizer=tokenizer,
    target_hours=24 # or target_params_millions=1300
)

# ✅ Enable gradient checkpointing to reduce VRAM usage
config.gradient_checkpointing = True
config.attention_layers = ["global" if i % 2 == 0 else "local" for i in range(config.num_hidden_layers)]


print("🔄 Initializing quantized model…")
model = GPTNeoForCausalLM(config)

model = model.half().to(DEVICE) #Half precision
torch.cuda.empty_cache()

vram_used = torch.cuda.memory_allocated() / 1024**3
print(f"   → VRAM used by weights: {vram_used:.2f} GB (≈12 GB expected)")

# ─── 6. Trainer Setup ───────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

def restore_best_model_if_needed(trainer, model):
    args = trainer.args
    if getattr(args, "load_best_model_at_end", False):
        print("✅ Best model already restored by Trainer.")
        return
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, LegacyBestModelCallback) and cb.best_model_path:
            print(f"🔁 Restoring best model from {cb.best_model_path}")
            model.load_state_dict(torch.load(os.path.join(cb.best_model_path, "pytorch_model.bin")))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    mask = shift_labels != tokenizer.pad_token_id
    preds = np.argmax(shift_logits[mask], axis=-1)
    acc = (preds == shift_labels[mask]).mean()
    #loss = torch.nn.functional.cross_entropy(
    #    torch.tensor(shift_logits[mask]), torch.tensor(shift_labels[mask]), reduction="mean"
    #).item()
    loss = torch.nn.functional.cross_entropy(
    torch.from_numpy(shift_logits[mask]),
    torch.from_numpy(shift_labels[mask]),
    reduction="mean").item()
    perplexity = math.exp(loss)
    return {"accuracy": acc, "perplexity": perplexity}

class CustomTrainer(Trainer):
    def __init__(self, *args, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
        )

class LegacyBestModelCallback(TrainerCallback):
    def __init__(self, metric_name="eval_loss", greater_is_better=False, patience=2):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.patience = patience
        self.best_metric = None
        self.best_model_path = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return control  # nothing to do

        current_metric = metrics[self.metric_name]
        is_better = (
            self.best_metric is None or
            (self.greater_is_better and current_metric > self.best_metric) or
            (not self.greater_is_better and current_metric < self.best_metric)
        )

        if is_better:
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            # Save best model
            self.best_model_path = os.path.join(args.output_dir, f"best_checkpoint_epoch_{int(state.epoch)}")
            os.makedirs(self.best_model_path, exist_ok=True)
            model.save_pretrained(self.best_model_path)
            torch.save(model.state_dict(), os.path.join(self.best_model_path, "pytorch_model.bin"))
            print(f"✅ Saved new best model to {self.best_model_path} (metric: {self.best_metric:.4f})")
        else:
            self.num_bad_epochs += 1
            print(f"⚠️ No improvement in {self.metric_name} for {self.num_bad_epochs} eval(s)")

        # Early stopping check
        if self.num_bad_epochs >= self.patience:
            print(f"🛑 Early stopping triggered after {self.num_bad_epochs} bad eval(s)")
            control.should_training_stop = True

        return control


print("🛠️  Preparing Trainer…")

optimizer = Adam8bit(model.parameters(), lr=training_args.learning_rate)

# Callbacks (branch-aware)
trainer_callbacks = []
if is_modern:
    # Modern: built-in EarlyStopping + built-in best-model loading
    trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
else:
    # Legacy: our own best-model saver + early stopping (no reliance on evaluation_strategy)
    trainer_callbacks.append(
        LegacyBestModelCallback(metric_name="perplexity", greater_is_better=False, patience=2)
    )

if not training_args.load_best_model_at_end:
    trainer_callbacks.append(
        LegacyBestModelCallback(metric_name="perplexity", greater_is_better=False, patience=2)
    )

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    sampler=sampler,
    callbacks=trainer_callbacks,
)
# ─── 7. Train & Save Best Model ────────────────────────────────────────────────
print("🏁 Starting training…")
trainer.train()
restore_best_model_if_needed(trainer, model) # This function handles legacy trainer with callback, or leaves the loaded best model intact

print("💾 Saving best model & tokenizer to models/best_model…")
best_model_dir = "models/best_model"
os.makedirs(best_model_dir, exist_ok=True)
shutil.copytree(self.best_model_path, best_model_dir, dirs_exist_ok=True)
#trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

# ─── 8. Save Summary ────────────────────────────────────────────────────────────
summary_path = os.path.join(best_model_dir, "model_summary.txt")

#source_counts = Counter(ex["source"] for ex in train_ds if "source" in ex)
source_counts = Counter(ex.get("source", "unknown") for ex in train_ds)
source_stats = dict(source_counts)

print("📊 Evaluating final model on validation set…")
final_metrics = trainer.evaluate(eval_dataset=eval_ds)

model_config = config.to_dict()
tokenizer_stats = {
    "vocab_size": tokenizer.vocab_size,
    "max_length": tokenizer.model_max_length,
    "special_tokens": tokenizer.all_special_tokens,
}
pad_id = tokenizer.pad_token_id
total_tokens = sum((ex["input_ids"] != pad_id).sum().item() for ex in train_ds)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

with open(summary_path, "w") as f:
    f.write("📊 Training Set Composition:\n")
    for source, count in source_stats.items():
        f.write(f"  {source}: {count}\n")

    f.write("\n📈 Final Evaluation Metrics:\n")
    for metric, value in final_metrics.items():
        f.write(f"  {metric}: {value:.4f}\n")

    f.write("\n🧠 Model Configuration:\n")
    for k, v in model_config.items():
        f.write(f"  {k}: {v}\n")

    f.write("\n🔤 Tokenizer Stats:\n")
    for k, v in tokenizer_stats.items():
        f.write(f"  {k}: {v}\n")

    f.write(f"\n📚 Total Tokens Trained On: {total_tokens}\n")
    f.write(f"🔢 Total Parameters: {total_params:,}\n")
    f.write(f"🛠️  Trainable Parameters: {trainable_params:,}\n")

print(f"📝 Model summary saved to: {summary_path}")
print("✅ Training complete. Best model is in:", best_model_dir)
