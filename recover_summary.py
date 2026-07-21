import os
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# ─── 1. Reload Model & Tokenizer ───────────────────────────────────────────────
best_model_dir = "models/best_model"
model = AutoModelForCausalLM.from_pretrained(best_model_dir)
tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
config = model.config

# ─── 2. Rebuild Tokenizer Stats ────────────────────────────────────────────────
tokenizer_stats = {
    "vocab_size": tokenizer.vocab_size,
    "max_length": tokenizer.model_max_length,
    "special_tokens": tokenizer.all_special_tokens,
}
pad_id = tokenizer.pad_token_id

# ─── 3. Reload Datasets ────────────────────────────────────────────────────────
# Replace with your actual dataset loading logic
from train_data_prep import load_train_dataset, load_eval_dataset
train_ds = load_train_dataset()
eval_ds = load_eval_dataset()

# ─── 4. Recompute Token & Parameter Stats ──────────────────────────────────────
total_tokens = sum((ex["input_ids"] != pad_id).sum().item() for ex in train_ds)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─── 5. Re-evaluate Model ──────────────────────────────────────────────────────
trainer = Trainer(model=model, tokenizer=tokenizer, eval_dataset=eval_ds)
final_metrics = trainer.evaluate()

# ─── 6. Rebuild Source Composition ─────────────────────────────────────────────
source_counts = Counter(ex.get("source", "unknown") for ex in train_ds)

# ─── 7. Write Summary ──────────────────────────────────────────────────────────
summary_path = os.path.join(best_model_dir, "model_summary.txt")
with open(summary_path, "w") as f:
    f.write("📊 Training Set Composition:\n")
    for source, count in source_counts.items():
        f.write(f"  {source}: {count}\n")

    f.write("\n📈 Final Evaluation Metrics:\n")
    for metric, value in final_metrics.items():
        f.write(f"  {metric}: {value:.4f}\n")

    f.write("\n🧠 Model Configuration:\n")
    for k, v in config.to_dict().items():
        f.write(f"  {k}: {v}\n")

    f.write("\n🔤 Tokenizer Stats:\n")
    for k, v in tokenizer_stats.items():
        f.write(f"  {k}: {v}\n")

    f.write(f"\n📚 Total Tokens Trained On: {total_tokens}\n")
    f.write(f"🔢 Total Parameters: {total_params:,}\n")
    f.write(f"🛠️  Trainable Parameters: {trainable_params:,}\n")

print(f"✅ Recovered summary saved to: {summary_path}")
