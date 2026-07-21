import torch, json
from transformers import RobertaTokenizerFast
from bitnet_model import BitNetForCausalLM  # your BitNet class

# Paths
model_dir = "/mnt/l/Datasets/working_sample_datasets/models/finetuned_bitnet_hybrid"

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)

# Load config & model
with open(f"{model_dir}/config.bitnet.json") as f:
    cfg = json.load(f)

model = BitNetForCausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=cfg["hidden_size"],
    num_layers=cfg["num_layers"],
    num_heads=cfg["num_heads"],
    max_seq_len=cfg["max_seq_len"],
    gradient_checkpointing=cfg.get("gradient_checkpointing", False)
)
state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict)

# Build device map (same logic as training)
def build_blockwise_device_map(model, gpu_vram_gb_budget=12.0):
    per_block_gb = max(0.15, (model.hidden_size * 4 * model.hidden_size) / 1e9)
    max_gpu_blocks = int(gpu_vram_gb_budget // per_block_gb)
    max_gpu_blocks = max(1, min(max_gpu_blocks, model.num_layers))
    device_map = {"embed": "cuda", "pos_embed": "cuda", "ln_f": "cuda", "head": "cuda"}
    for i in range(model.num_layers):
        device_map[f"block.{i}"] = "cuda" if i < max_gpu_blocks else "cpu"
    return device_map

total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
usable_vram_gb = max(4.0, total_vram_gb - 4.0)
device_map = build_blockwise_device_map(model, gpu_vram_gb_budget=usable_vram_gb)
model.set_device_map(device_map)

model.eval()

# Inference
prompt = "Once upon a time in Belfast,"
inputs = tokenizer(prompt, return_tensors="pt").to(device_map["embed"])
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
