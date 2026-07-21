#!/usr/bin/env python3
# Fine-tune GPT-Neo model with 4-bit quantization, offloading, and introspective logging

# ─── 0. Imports ────────────────────────────────────────────────────────────────
import os, json, math, torch, logging
from collections import Counter
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from bitsandbytes.optim import Adam8bit
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import init_empty_weights, infer_auto_device_map

# ─── 1. Paths & Environment ─────────────────────────────────────────────────────
datasets_dir = "/mnt/l/Datasets/working_sample_datasets"
model_dir = os.path.join(datasets_dir, "models", "best_model")
offload_dir = os.path.join(datasets_dir, "offload")
output_dir = os.path.join(datasets_dir, "models", "finetuned")
os.makedirs(offload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This script expects a GPU!"
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)

# ─── 2. Load Tokenizer & Dataset ────────────────────────────────────────────────
print("🔄 Loading tokenizer and dataset…")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
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

# ─── 4. Quantized Model Loading with Accelerate ─────────────────────────────────
print("⚙️  Loading model with 4-bit quantization and offloading…")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

with init_empty_weights():
    model_skeleton = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )

max_memory = {
    0: "16GB",
    "cpu": "64GB"
}
device_map = infer_auto_device_map(
    model_skeleton,
    max_memory=max_memory,
    no_split_module_classes=["GPTNeoBlock"],
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=device_map,
    offload_folder=offload_dir,
    torch_dtype=torch.float16,
    quantization_config=bnb_cfg,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

try:
    model.enable_xformers_memory_efficient_attention()
    print("✅ xFormers attention enabled.")
except Exception:
    print("⚠️ xFormers not available. Continuing without it.")

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
    fp16=True,
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

optimizer = Adam8bit(model.parameters(), lr=training_args.learning_rate)

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
print("🏁 Starting fine-tuning…")
trainer.train()

print("💾 Saving fine-tuned model…")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

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
print("✅ Fine-tuning complete. Model saved to:", output_dir)
