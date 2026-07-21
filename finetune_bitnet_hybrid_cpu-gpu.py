#!/usr/bin/env python3
# Fine-tune BitNet-pretrained model with hybrid CPU–GPU offloading, weighted sampling, and introspective logging

# ─── 0. Imports ────────────────────────────────────────────────────────────────
import os, json, math, torch, logging
import torch.nn.functional as F
from collections import Counter
from datasets import load_from_disk
from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn

# ─── 1. BitNet architecture (import or inline) ─────────────────────────────────
# If you have bitnet_model.py, replace this section with:
# from bitnet_model import BitNetForCausalLM, BitNetBlock, BitLinear

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, threshold=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.threshold = threshold
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.uniform_((self.weight), -0.5, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def ternary(self, w):
        if self.threshold > 0.0:
            w_abs = torch.abs(w)
            s = torch.sign(w)
            s[w_abs < self.threshold] = 0.0
            return s
        return torch.sign(w)
    def forward(self, x):
        w_tern = self.ternary(self.weight)
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
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_seq_len, gradient_checkpointing=True):
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

        # Hybrid offload map (index -> device str), constructed via set_device_map()
        self._device_map = None

    def set_device_map(self, device_map):
        """
        device_map: dict with keys among:
          - "embed", "pos_embed", "ln_f", "head"
          - "block.0", "block.1", ..., "block.N-1"
          Values are "cuda" or "cpu".
        Moves modules accordingly and stores map to route activations at runtime.
        """
        self._device_map = device_map.copy()

        def move(m, dev_str):
            if isinstance(m, nn.Parameter):
                # pos_embed is a Parameter
                with torch.no_grad():
                    setattr(self, 'pos_embed', nn.Parameter(self.pos_embed.detach().to(dev_str)))
            else:
                m.to(dev_str)

        # Move non-blocks
        if "embed" in device_map: self.embed.to(device_map["embed"])
        if "ln_f" in device_map: self.ln_f.to(device_map["ln_f"])
        if "head" in device_map: self.head.to(device_map["head"])
        if "pos_embed" in device_map:
            with torch.no_grad():
                self.pos_embed = nn.Parameter(self.pos_embed.detach().to(device_map["pos_embed"]))

        # Move blocks
        for i, b in enumerate(self.blocks):
            key = f"block.{i}"
            if key in device_map:
                b.to(device_map[key])

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Determine device for embeddings
        d_embed = self._device_map.get("embed", input_ids.device if self._device_map else input_ids.device)
        x = self.embed(input_ids.to(d_embed)) + self.pos_embed[:, :input_ids.size(1)].to(d_embed)

        # Iterate blocks, moving activations as needed
        for i, b in enumerate(self.blocks):
            d_block = self._device_map.get(f"block.{i}", d_embed if self._device_map else d_embed)
            if x.device.type != d_block:
                x = x.to(d_block, non_blocking=True)
            # Note: attention_mask should follow x device
            am = attention_mask.to(d_block) if attention_mask is not None else None

            if self.gradient_checkpointing and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(b, x, am)
            else:
                x = b(x, am)

        d_ln = self._device_map.get("ln_f", x.device if self._device_map else x.device)
        if x.device.type != d_ln:
            x = x.to(d_ln, non_blocking=True)
        x = self.ln_f(x)

        d_head = self._device_map.get("head", d_ln)
        if x.device.type != d_head:
            x = x.to(d_head, non_blocking=True)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # Compute on logits device to avoid transfers
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # adjust if pad_token_id != 0
            )
        return {"loss": loss, "logits": logits}

# ─── 2. Paths & Environment ─────────────────────────────────────────────────────
datasets_dir = "/mnt/l/Datasets/working_sample_datasets"
model_dir = os.path.join(datasets_dir, "models", "best_model")   # pretrained BitNet output
output_dir = os.path.join(datasets_dir, "models", "finetuned_bitnet_hybrid")
os.makedirs(output_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

# ─── 3. Load Tokenizer & Dataset ────────────────────────────────────────────────
print("🔄 Loading tokenizer and dataset…")
tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
tokenizer.model_max_length = 512

combined = load_from_disk(os.path.join(datasets_dir, "combined"))
train_ds = combined["train"]
eval_ds  = combined["validation"]

assert "source" in train_ds.column_names, "'source' column missing from training set"
print(f"   → Train: {len(train_ds)} examples")
print(f"   → Eval:  {len(eval_ds)} examples")

# ─── 4. Weighted Sampling ───────────────────────────────────────────────────────
source_counts = Counter([ex["source"] for ex in train_ds])
total = sum(source_counts.values())
source_weights = {source: round(1.0 - (count / total), 4) for source, count in source_counts.items()}
weights = [source_weights[ex["source"]] for ex in train_ds]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ─── 5. Load BitNet model & weights ─────────────────────────────────────────────
print("⚙️  Loading BitNet model from pretrained weights…")
with open(os.path.join(model_dir, "config.bitnet.json")) as f:
    cfg = json.load(f)

model = BitNetForCausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=cfg["hidden_size"],
    num_layers=cfg["num_layers"],
    num_heads=cfg["num_heads"],
    max_seq_len=cfg["max_seq_len"],
    gradient_checkpointing=cfg.get("gradient_checkpointing", True)
)
state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state_dict)

# ─── 6. Hybrid CPU–GPU device map ───────────────────────────────────────────────
def build_blockwise_device_map(model, gpu_vram_gb_budget=12.0):
    """
    Simple heuristic: place as many blocks on GPU as the VRAM budget allows, rest on CPU.
    Embeddings, ln_f, head go to GPU by default. You can tune this split.
    """
    # Rough size estimate per block (in GB). Adjust based on hidden/layers.
    h = model.hidden_size
    per_block_gb = max(0.15, (h * 4 * h + h * h) / 1e9)  # heuristic
    max_gpu_blocks = int(gpu_vram_gb_budget // per_block_gb)
    max_gpu_blocks = max(1, min(max_gpu_blocks, model.num_layers))

    device_map = {"embed": "cuda", "pos_embed": "cuda", "ln_f": "cuda", "head": "cuda"}
    for i in range(model.num_layers):
        device_map[f"block.{i}"] = "cuda" if i < max_gpu_blocks else "cpu"
    return device_map, max_gpu_blocks

# Detect available VRAM and build map
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
usable_vram_gb = max(4.0, total_vram_gb - 4.0)  # leave margin
device_map, gpu_blocks = build_blockwise_device_map(model, gpu_vram_gb_budget=usable_vram_gb)
print(f"🧭 Hybrid device map: {gpu_blocks}/{model.num_layers} blocks on GPU, rest on CPU")

# Apply map and send model parts to devices
model.set_device_map(device_map)

# ─── 7. Trainer setup ───────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    mask = shift_labels != tokenizer.pad_token_id
    preds = torch.argmax(torch.tensor(shift_logits[mask]), axis=-1)
    acc = (preds == torch.tensor(shift_labels[mask])).float().mean().item()
    loss = F.cross_entropy(
        torch.tensor(shift_logits[mask]), torch.tensor(shift_labels[mask]), reduction="mean"
    ).item()
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

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
    num_train_epochs=5,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",
    greater_is_better=False,
    save_total_limit=2,
    logging_dir=os.path.join(output_dir, "logs"),
    learning_rate=3e-5,
)

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ─── 8. Train & Save ────────────────────────────────────────────────────────────
print("🏁 Starting BitNet fine-tuning (hybrid offload)…")
trainer.train()

print("💾 Saving fine-tuned model…")
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
tokenizer.save_pretrained(output_dir)
with open(os.path.join(output_dir, "config.bitnet.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# ─── 9. Summary ─────────────────────────────────────────────────────────────────
summary_path = os.path.join(output_dir, "finetune_summary.txt")
final_metrics = trainer.evaluate(eval_dataset=eval_ds)
pad_id = tokenizer.pad_token_id
total_tokens = sum((ex["input_ids"] != pad_id).sum().item() for ex in train_ds)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

with open(summary_path, "w") as f:
    f.write("📊 Training Set Composition:\n")
    for source, count in source_counts.items():
        f.write(f"  {source}: {count}\n")
    f.write("\n📈 Final Evaluation Metrics:\n")
    for metric, value in final_metrics.items():
        f.write(f"  {metric}: {value:.4f}\n")
    f.write("\n🔢 Model Parameters:\n")
    f.write(f"  Total: {total_params:,}\n")
    f.write(f"  Trainable: {trainable_params:,}\n")
    f.write(f"\n📚 Total Tokens Trained On: {total_tokens}\n")

print(f"📝 Summary saved to: {summary_path}")
print("✅ BitNet fine-tuning complete (hybrid). Model saved to:", output_dir)
