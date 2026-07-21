#!/usr/bin/env python3
# Fine-tune BitNet-pretrained model with introspective logging and weighted sampling

# ─── 0. Imports ────────────────────────────────────────────────────────────────
import os, json, math, torch, logging
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

# Import your BitNet architecture from pretraining
from bitnet_model import BitNetForCausalLM  # adjust import path to where you saved it

# ─── 1. Paths & Environment ─────────────────────────────────────────────────────
datasets_dir = "/mnt/l/Datasets/working_sample_datasets"
model_dir = os.path.join(datasets_dir, "models", "best_model")  # pretrained BitNet
output_dir = os.path.join(datasets_dir, "models", "finetuned_bitnet")
os.makedirs(output_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

# ─── 2. Load Tokenizer & Dataset ────────────────────────────────────────────────
print("🔄 Loading tokenizer and dataset…")
tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
tokenizer.model_max_length = 512

combined = load_from_disk(os.path.join(datasets_dir, "combined"))
train_ds = combined["train"]
eval_ds  = combined["validation"]

assert "source" in train_ds.column_names, "'source' column missing from training set"
print(f"   → Train: {len(train_ds)} examples")
print(f"   → Eval:  {len(eval_ds)} examples")

# ─── 3. Weighted Sampling ───────────────────────────────────────────────────────
source_counts = Counter([ex["source"] for ex in train_ds])
total = sum(source_counts.values())
source_weights = {
    source: round(1.0 - (count / total), 4)
    for source, count in source_counts.items()
}
weights = [source_weights[ex["source"]] for ex in train_ds]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ─── 4. Load BitNet Model ───────────────────────────────────────────────────────
print("⚙️  Loading BitNet model from pretrained weights…")
# Load config if you saved one during pretraining
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
model.to(DEVICE)

model.config = cfg  # optional: attach config for Trainer compatibility
model.config.use_cache = False

# ─── 5. Trainer Setup ───────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    mask = shift_labels != tokenizer.pad_token_id
    preds = torch.argmax(torch.tensor(shift_logits[mask]), axis=-1)
    acc = (preds == torch.tensor(shift_labels[mask])).float().mean().item()
    loss = torch.nn.functional.cross_entropy(
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
    bf16=True,  # BitNet can run well in bf16
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

# ─── 6. Train & Save ────────────────────────────────────────────────────────────
print("🏁 Starting BitNet fine-tuning…")
trainer.train()

print("💾 Saving fine-tuned model…")
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
tokenizer.save_pretrained(output_dir)
with open(os.path.join(output_dir, "config.bitnet.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# ─── 7. Summary ─────────────────────────────────────────────────────────────────
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
print("✅ BitNet fine-tuning complete. Model saved to:", output_dir)
