#!/usr/bin/env python3
# Train BitNet-style ternary-weight Transformer → eval → best-model saving
# Target VRAM: ~12 GB on 16 GB RTX-class GPU
# GPU throughput benchmarking, dynamic model scaling, visualization, weighted sampling, early stopping, best-model restore, summary logging.

# ─── 0. Imports & Environment ───────────────────────────────────────────────────
import os
import math
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import Counter
from datasets import load_from_disk
from packaging import version

import transformers
from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)

from torch.utils.data import DataLoader, WeightedRandomSampler

# ─── 1. Paths & Global setup ────────────────────────────────────────────────────
datasets_dir = os.path.join('/mnt/l/Datasets/', "working_sample_datasets")
combined = load_from_disk(os.path.join(datasets_dir, "combined"))
tokenizer_path = os.path.join(datasets_dir, "combined", "tokenizer_bpe")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

use_modern_training = False
is_modern = version.parse(transformers.__version__) >= version.parse("4.4.0") and use_modern_training
print(f"🤖 Transformers {transformers.__version__} | modern branch: {is_modern}")

# ─── 2. Tokenizer ───────────────────────────────────────────────────────────────
print("🔄 Wrapping in RobertaTokenizerFast…")
tokenizer = RobertaTokenizerFast.from_pretrained(
    tokenizer_path,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
)
tokenizer.model_max_length = 512  # default, may be lowered by auto-sizer later

# ─── 3. Dataset & Weighted sampling ─────────────────────────────────────────────
train_ds = combined["train"]
eval_ds  = combined["validation"]

assert "source" in train_ds.column_names, "'source' column missing from training set"
print(f"   → Train: {len(train_ds)} examples")
print(f"   → Eval:  {len(eval_ds)} examples")

source_weights = {
    "en": 0.3,
    "python": 0.35,
    "sciphi": 0.25,
    "nq": 0.1
}
weights = [source_weights.get(ex["source"], 0.1) for ex in train_ds]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ─── 4. Tokenization & formatting ───────────────────────────────────────────────
def tokenize_and_label(examples):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
    )
    out["labels"] = out["input_ids"].copy()
    return out

print("🔄 Tokenizing dataset…")
tokenized = combined.map(tokenize_and_label, batched=True, remove_columns=[])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_ds = tokenized["train"]
eval_ds  = tokenized["validation"]

print(f"   → Train: {len(train_ds)} examples")
print(f"   → Eval:  {len(eval_ds)} examples")

# ─── 5. GPU throughput benchmark ────────────────────────────────────────────────
def measure_gpu_throughput(test_hidden=1024, test_layers=4, test_seq_len=256, batch_size=1, steps=6):
    """Return seconds per micro-batch for a small transformer-like block on your GPU."""
    class TinyBlock(nn.Module):
        def __init__(self, h, heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(h, h*4), nn.GELU(), nn.Linear(h*4, h))
            self.ln1 = nn.LayerNorm(h)
            self.ln2 = nn.LayerNorm(h)
        def forward(self, x, attn_mask=None):
            a, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = self.ln1(x + a)
            f = self.ff(x)
            return self.ln2(x + f)

    torch.cuda.empty_cache()
    h = test_hidden
    heads = max(1, h // 64)
    blocks = nn.ModuleList([TinyBlock(h, heads) for _ in range(test_layers)]).to(DEVICE)
    lm_head = nn.Linear(h, 50257).to(DEVICE)
    emb = nn.Embedding(50257, h).to(DEVICE)
    pos = nn.Parameter(torch.zeros(1, test_seq_len, h)).to(DEVICE)
    params = list(blocks.parameters()) + list(lm_head.parameters()) + list(emb.parameters()) + [pos]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    dummy_input = torch.randint(0, 50257, (batch_size, test_seq_len), device=DEVICE)
    dummy_labels = dummy_input.clone()

    def fwd(x_ids):
        x = emb(x_ids) + pos[:, :x_ids.size(1)]
        for b in blocks:
            x = b(x)
        logits = lm_head(x)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = dummy_labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )
        return loss

    # Warm-up
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = fwd(dummy_input)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = fwd(dummy_input)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / steps  # seconds per micro-batch

# ─── 6. BitNet architecture ─────────────────────────────────────────────────────
class BitLinear(nn.Module):
    """
    Ternary-weight linear layer: weights constrained to {-1, 0, +1}.
    Uses sign as a projection with a straight-through estimator for gradients.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
        self.threshold = 0.0  # Set >0.0 if you want sparsity (0s)

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.5, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def ternary(self, w):
        if self.threshold > 0.0:
            w_abs = torch.abs(w)
            mask = (w_abs < self.threshold)
            s = torch.sign(w)
            s[mask] = 0.0
            return s
        return torch.sign(w)

    def forward(self, x):
        w_tern = self.ternary(self.weight)
        # STE: gradient flows through original weights
        return F.linear(x, w_tern, self.bias)

class BitNetBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            BitLinear(hidden_size, hidden_size * 4),
            nn.GELU(),
            BitLinear(hidden_size * 4, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + a)
        f = self.ff(x)
        return self.ln2(x + f)

class BitNetForCausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, gradient_checkpointing=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.blocks = nn.ModuleList([BitNetBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = BitLinear(hidden_size, vocab_size)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        x = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1)]
        if self.gradient_checkpointing:
            # checkpoint each block to reduce activation memory
            for b in self.blocks:
                x = torch.utils.checkpoint.checkpoint(b, x, attention_mask)
        else:
            for b in self.blocks:
                x = b(x, attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
        # Return in Trainer-friendly format
        return {"loss": loss, "logits": logits}

# ─── 7. Dynamic model scaling & visualization ───────────────────────────────────
def preconfigure_training(train_dataset, tokenizer, target_hours=None, target_params_millions=None,
                          eval_fraction=0.05, save_fraction=0.05, vram_safety_margin_gb=2):
    """
    Decide model dims and training args based on dataset size, measured GPU throughput, and VRAM.
    Produces a time/VRAM contour visualization.
    """
    num_examples = len(train_dataset)
    print(f"📊 Training examples: {num_examples:,}")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    usable_vram_gb = max(1.0, total_vram_gb - vram_safety_margin_gb)
    print(f"💾 GPU VRAM usable: ~{usable_vram_gb:.1f} GB")

    print("⚡ Measuring GPU throughput…")
    measured_s_per_microbatch = measure_gpu_throughput()
    print(f"   → {measured_s_per_microbatch:.3f} s/micro-batch")

    test_params_m = 50.0
    test_seq_len = 256
    grad_accum = 8
    steps_per_epoch = max(1, math.ceil(num_examples / (1 * grad_accum)))

    def estimate_vram_usage_gb(params_m, seq_len, batch_size):
        weights_gb = (params_m * 1e6 * 2) / (1024**3)
        activations_gb = seq_len * batch_size * 0.00002
        return weights_gb + activations_gb + 2.0

    def estimate_train_time_hours(params_m, seq_len, batch_size):
        s_per_microbatch = measured_s_per_microbatch * (params_m / test_params_m) * (seq_len / test_seq_len)
        s_per_step = s_per_microbatch * grad_accum
        steps_per_epoch = math.ceil(num_examples / (batch_size * grad_accum))
        return (s_per_step * steps_per_epoch) / 3600

    # --- Plot 1: model size vs seq length ---
    params_range = np.linspace(100, 1500, 40)
    seq_range = np.linspace(128, 768, 40)
    P, S = np.meshgrid(params_range, seq_range)
    VRAM = estimate_vram_usage_gb(P, S, 1)
    TIME = estimate_train_time_hours(P, S, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(P, S, TIME, levels=20, cmap='viridis')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Training Time [hours] (1 epoch)")
    cs = ax.contour(P, S, VRAM, levels=[usable_vram_gb], colors='red', linewidths=2)
    ax.clabel(cs, fmt=f"{usable_vram_gb:.1f}GB VRAM limit", colors='red')
    ax.set_xlabel("Params [millions]")
    ax.set_ylabel("Sequence length")
    ax.set_title("BitNet: Time & VRAM vs Seq Length")
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "bitnet_time_vram_contours.png"))
    plt.close(fig)

    # --- Plot 2: model size vs dataset size ---
    dataset_range = np.linspace(num_examples * 0.25, num_examples * 2, 40)
    P2, D = np.meshgrid(params_range, dataset_range)
    VRAM2 = estimate_vram_usage_gb(P2, 512, 1)  # fixed seq_len for comparison
    TIME2 = (D / num_examples) * estimate_train_time_hours(P2, 512, 1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    c2 = ax2.contourf(P2, D, TIME2, levels=20, cmap='plasma')
    cb2 = fig2.colorbar(c2, ax=ax2)
    cb2.set_label("Training Time [hours] (1 epoch)")
    cs2 = ax2.contour(P2, D, VRAM2, levels=[usable_vram_gb], colors='red', linewidths=2)
    ax2.clabel(cs2, fmt=f"{usable_vram_gb:.1f}GB VRAM limit", colors='red')
    ax2.set_xlabel("Params [millions]")
    ax2.set_ylabel("Dataset size (examples)")
    ax2.set_title("BitNet: Time & VRAM vs Dataset Size")
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "bitnet_dataset_capacity.png"))
    plt.close(fig2)
    
    # --- Printout table: max feasible dataset size for chosen model size ---
    print("\n📋 Dataset Capacity Table (BitNet)")
    print(f"{'Params (M)':>12} | {'Max Examples':>15} | {'Est. Hours/Epoch':>18}")
    print("-" * 50)

    for pm in [params_m * 0.5, params_m, params_m * 1.5]:
        # Sweep dataset sizes until VRAM limit is hit
        max_examples = None
        est_hours = None
        for ds in np.linspace(num_examples * 0.25, num_examples * 2, 50):
            vram_use = estimate_vram_usage_gb(pm, 512, 1)
            if vram_use <= usable_vram_gb:
                max_examples = int(ds)
                est_hours = (ds / num_examples) * estimate_train_time_hours(pm, 512, 1)
        if max_examples:
            print(f"{pm:12.0f} | {max_examples:15,d} | {est_hours:18.2f}")
        else:
            print(f"{pm:12.0f} | {'OOM':>15} | {'-':>18}")

    # Choose params_m
    if target_params_millions:
        params_m = target_params_millions
    elif target_hours:
        total_seconds_budget = target_hours * 3600
        s_per_step = measured_s_per_microbatch * grad_accum
        total_steps_budget = int(total_seconds_budget / s_per_step)
        params_m = test_params_m * (total_steps_budget / (steps_per_epoch * 3))
        params_m = float(np.clip(params_m, 125, 1500))
    else:
        params_m = 1300.0

    # Derive dims
    if params_m <= 600:
        num_layers = 24
    elif params_m <= 1200:
        num_layers = 28
    else:
        num_layers = 32

    hidden_size = int(((params_m * 1e6) / (12 * num_layers)) ** 0.5)
    hidden_size = max(64, (hidden_size // 64) * 64)
    num_heads = max(1, hidden_size // 64)
    max_seq_len = tokenizer.model_max_length
    grad_accum = 8
    num_epochs = 3
    eval_steps = max(1, int(steps_per_epoch * eval_fraction))
    save_steps = max(1, int(steps_per_epoch * save_fraction))

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_seq_len": max_seq_len,
        "grad_accum": grad_accum,
        "num_epochs": num_epochs,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
    }, TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        num_train_epochs=num_epochs,
        logging_steps=100,
        save_total_limit=2,
        logging_dir="logs",
        learning_rate=5e-5,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
    )


# ─── 8. Build config and args ───────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
config_dict, training_args = preconfigure_training(
    train_dataset=train_ds,
    tokenizer=tokenizer,
    target_hours=24,  # or target_params_millions=1300
)

hidden_size   = config_dict["hidden_size"]
num_layers    = config_dict["num_layers"]
num_heads     = config_dict["num_heads"]
max_seq_len   = config_dict["max_seq_len"]
grad_accum    = config_dict["grad_accum"]
num_epochs    = config_dict["num_epochs"]
eval_steps    = config_dict["eval_steps"]
save_steps    = config_dict["save_steps"]
s_per_step    = config_dict["s_per_step"]

# ─── 9. Model setup ─────────────────────────────────────────────────────────────
print("🔄 Initializing BitNet model…")
model = BitNetForCausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    max_seq_len=max_seq_len,
    gradient_checkpointing=True,
).to(DEVICE)

# Mixed precision for compute; weights remain float for training (STE applies ternary at forward)
model = model.bfloat16()  # compute in bf16 when possible

torch.cuda.empty_cache()
vram_used = torch.cuda.memory_allocated() / 1024**3
print(f"   → VRAM currently allocated: {vram_used:.2f} GB")

# ─── 10. Trainer setup ──────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = (shift_labels != tokenizer.pad_token_id)

    # argmax predictions
    preds = np.argmax(shift_logits[mask], axis=-1)
    acc = (preds == shift_labels[mask]).mean()

    # cross-entropy loss & perplexity
    sl = torch.from_numpy(shift_logits[mask]).float()
    sy = torch.from_numpy(shift_labels[mask]).long()
    loss = F.cross_entropy(sl, sy, reduction="mean").item()
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
            return control
        current_metric = metrics[self.metric_name]
        is_better = (
            self.best_metric is None or
            (self.greater_is_better and current_metric > self.best_metric) or
            (not self.greater_is_better and current_metric < self.best_metric)
        )
        if is_better:
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.best_model_path = os.path.join(args.output_dir, f"best_checkpoint_epoch_{int(state.epoch or 0)}")
            os.makedirs(self.best_model_path, exist_ok=True)
            # Save best weights
            torch.save(model.state_dict(), os.path.join(self.best_model_path, "pytorch_model.bin"))
            print(f"✅ Saved new best model to {self.best_model_path} (metric: {self.best_metric:.4f})")
        else:
            self.num_bad_epochs += 1
            print(f"⚠️ No improvement in {self.metric_name} for {self.num_bad_epochs} eval(s)")
        if self.num_bad_epochs >= self.patience:
            print(f"🛑 Early stopping triggered after {self.num_bad_epochs} bad eval(s)")
            control.should_training_stop = True
        return control

def restore_best_model_if_needed(trainer, model):
    args = trainer.args
    if getattr(args, "load_best_model_at_end", False):
        print("✅ Best model already restored by Trainer.")
        return
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, LegacyBestModelCallback) and cb.best_model_path:
            path = os.path.join(cb.best_model_path, "pytorch_model.bin")
            if os.path.exists(path):
                print(f"🔁 Restoring best model from {cb.best_model_path}")
                sd = torch.load(path, map_location=DEVICE)
                model.load_state_dict(sd)

# Callbacks (branch-aware)
trainer_callbacks = []
if is_modern:
    trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
else:
    trainer_callbacks.append(LegacyBestModelCallback(metric_name="perplexity", greater_is_better=False, patience=2))

if not getattr(training_args, "load_best_model_at_end", False):
    trainer_callbacks.append(LegacyBestModelCallback(metric_name="perplexity", greater_is_better=False, patience=2))

optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

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

# ─── 11. Train & save best model ────────────────────────────────────────────────
print("🏁 Starting BitNet training…")
trainer.train()
restore_best_model_if_needed(trainer, model)

print("💾 Saving best model & tokenizer to models/best_model…")
best_model_dir = "models/best_model"
os.makedirs(best_model_dir, exist_ok=True)
# Save weights
torch.save(model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
# Save a minimal config for reproducibility
bitnet_config = {
    "vocab_size": tokenizer.vocab_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "max_seq_len": max_seq_len,
    "gradient_checkpointing": True,
}
with open(os.path.join(best_model_dir, "config.bitnet.json"), "w") as f:
    json.dump(bitnet_config, f, indent=2)
tokenizer.save_pretrained(best_model_dir)

# ─── 12. Summary ────────────────────────────────────────────────────────────────
summary_path = os.path.join(best_model_dir, "finetune_summary.txt")
final_metrics = trainer.evaluate(eval_dataset=eval_ds)
pad_id = tokenizer.pad_token_id
# Token count (approx; dataset already tensorized with padding)
total_tokens = 0
for ex in train_ds:
    ids = ex["input_ids"]
    total_tokens += int((ids != pad_id).sum().item())

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Source composition
source_counts = Counter([ex["source"] for ex in combined["train"]])

with open(summary_path, "w") as f:
    f.write("📊 Training Set Composition:\n")
    for source, count in source_counts.items():
        f.write(f"  {source}: {count}\n")

    f.write("\n📈 Final Evaluation Metrics:\n")
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            f.write(f"  {metric}: {value:.4f}\n")
        else:
            f.write(f"  {metric}: {value}\n")

    f.write("\n🔢 Model Parameters:\n")
    f.write(f"  Total: {total_params:,}\n")
    f.write(f"  Trainable: {trainable_params:,}\n")
    f.write(f"\n📚 Total Tokens Trained On: {total_tokens}\n")

print(f"📝 Summary saved to: {summary_path}")
print("✅ Training complete. Best model saved to:", best_model_dir)
print("🖼️ Contour plot saved to logs/bitnet_time_vram_contours.png")
