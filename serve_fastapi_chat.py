#!/usr/bin/env python3
import os, json, torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizerFast, TextStreamer
import uvicorn

# Try to import BitNet
try:
    from bitnet_model import BitNetForCausalLM
    BITNET_AVAILABLE = True
except ImportError:
    BITNET_AVAILABLE = False

MODEL_DIR = "/path/to/your/saved/model"
SYSTEM_PROMPT = "You are a helpful assistant with deep technical knowledge of AI training pipelines."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SESSION_FILE = "sessions.json"

# ─── Detect model type ──────────────────────────────────────────────────────────
def detect_model_type(model_dir):
    if os.path.exists(os.path.join(model_dir, "config.bitnet.json")):
        return "bitnet"
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "gpt_neo":
            return "gptneo"
    return "unknown"

# ─── Build hybrid device map ────────────────────────────────────────────────────
def build_blockwise_device_map(num_layers, hidden_size, gpu_vram_gb_budget=12.0, block_prefix="block"):
    per_block_gb = max(0.15, (hidden_size * 4 * hidden_size) / 1e9)
    max_gpu_blocks = int(gpu_vram_gb_budget // per_block_gb)
    max_gpu_blocks = max(1, min(max_gpu_blocks, num_layers))
    device_map = {}
    for i in range(num_layers):
        device_map[f"{block_prefix}.{i}"] = "cuda" if i < max_gpu_blocks else "cpu"
    return device_map

# ─── Load model & tokenizer ─────────────────────────────────────────────────────
def load_model_and_tokenizer(model_dir):
    model_type = detect_model_type(model_dir)
    print(f"📦 Detected model type: {model_type}")

    if model_type == "bitnet" and BITNET_AVAILABLE:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "config.bitnet.json")) as f:
            cfg = json.load(f)
        model = BitNetForCausalLM(
            vocab_size=tokenizer.vocab_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            max_seq_len=cfg["max_seq_len"],
            gradient_checkpointing=cfg.get("gradient_checkpointing", False)
        )
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usable_vram_gb = max(4.0, total_vram_gb - 4.0)
        if usable_vram_gb < (cfg["num_layers"] * 0.15):
            device_map = build_blockwise_device_map(cfg["num_layers"], cfg["hidden_size"], usable_vram_gb)
            model.set_device_map(device_map)
            print(f"🧭 Running in hybrid mode: {sum(1 for d in device_map.values() if d=='cuda')} blocks on GPU")
        else:
            model.to(DEVICE)
            print("🚀 Running fully on GPU")

    elif model_type == "gptneo":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb < 16:
            from accelerate import infer_auto_device_map, init_empty_weights
            with init_empty_weights():
                skeleton = AutoModelForCausalLM.from_pretrained(model_dir)
            device_map = infer_auto_device_map(skeleton, max_memory={0: f"{int(total_vram_gb)}GB", "cpu": "64GB"})
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.float16)
            print("🧭 GPT‑Neo hybrid CPU–GPU mode enabled")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).to(DEVICE)
            print("🚀 GPT‑Neo fully on GPU")

    else:
        raise RuntimeError("Unknown or unsupported model type")

    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

# ─── Persistent sessions ───────────────────────────────────────────────────────
if os.path.exists(SESSION_FILE):
    with open(SESSION_FILE, "r") as f:
        chat_histories = json.load(f)
else:
    chat_histories = {}

def save_sessions():
    with open(SESSION_FILE, "w") as f:
        json.dump(chat_histories, f, indent=2)

# ─── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    history = chat_histories.get(req.session_id, SYSTEM_PROMPT + "\n")
    history += f"User: {req.message}\nAssistant:"

    inputs = tokenizer(history, return_tensors="pt")
    if hasattr(model, "_device_map") and model._device_map:
        first_dev = list(model._device_map.values())[0]
        inputs = {k: v.to(first_dev) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    def token_generator():
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer
            )
        yield "data: [DONE]\n\n"

    chat_histories[req.session_id] = history
    save_sessions()

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/sessions")
def list_sessions():
    """Return all active session IDs and last message snippet."""
    sessions_info = []
    for sid, history in chat_histories.items():
        lines = [line for line in history.strip().split("\n") if line.strip()]
        last_line = lines[-1] if lines else ""
        sessions_info.append({"session_id": sid, "last_line": last_line})
    return {"sessions": sessions_info}

# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("chat_server:app", host="0.0.0.0", port=8000, reload=True)
