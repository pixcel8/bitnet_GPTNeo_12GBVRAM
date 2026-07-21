#!/usr/bin/env python
"""
app.py

Flask server that:
  • Serves the above index.html + style.css
  • Streams token-by-token responses from your hybrid CPU/GPU 8-bit BitNet
  • Emits SSE events: `token`, `done`, `end`, and `error`
  • Computes per-response metrics (tokens, time, throughput)
"""

import os, sys, argparse, logging, time
import torch, torch.nn.functional as F
from flask import Flask, request, Response, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Arguments & environment
parser = argparse.ArgumentParser()
parser.add_argument("--model-path",   default="./my_bitnet_model")
parser.add_argument("--offload-dir",  default="./offload")
parser.add_argument("--host",         default="0.0.0.0")
parser.add_argument("--port",         default=5000, type=int)
parser.add_argument("--max-length",   default=512, type=int)
parser.add_argument("--top-p",        default=0.9, type=float)
parser.add_argument("--temperature",  default=0.8, type=float)
args = parser.parse_args()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HOME"]             = os.path.expanduser("~/.cache/huggingface")
os.makedirs(args.offload_dir, exist_ok=True)

# 2. Logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s | %(levelname)s | %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bitnet_chat")

# 3. Flask init
app = Flask(__name__, template_folder="templates", static_folder="static")

# 4. Load model & tokenizer
logger.info("Loading tokenizer from %s", args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

logger.info("Loading model with CPU/GPU offload + 8-bit quant…")
model = AutoModelForCausalLM.from_pretrained(
  args.model_path,
  device_map="auto",
  load_in_8bit=True,
  offload_folder=args.offload_dir,
  offload_state_dict=True,
  torch_dtype=torch.float16
)
model.config.use_cache = True

try:
  model.enable_xformers_memory_efficient_attention()
  logger.info("xFormers attention enabled")
except:
  logger.warning("xFormers unavailable, continuing")

# 5. Serve UI
@app.route("/")
def home():
  return render_template("index.html")

# 6. SSE generator
def generate_stream(prompt):
  start = time.time()
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
  past = None
  count = 0

  for _ in range(args.max_length):
    out = model(input_ids=input_ids, past_key_values=past)
    logits, past = out.logits, out.past_key_values
    next_logits = logits[:, -1, :] / args.temperature
    probs = F.softmax(next_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)

    token = tokenizer.decode(next_id[0], skip_special_tokens=False)
    count += 1

    yield f"event: token\ndata: {token.__repr__()}\n\n"
    input_ids = next_id
    if next_id[0].item() == tokenizer.eos_token_id:
      break

  duration = time.time() - start
  metrics = {
    "tokens":      count,
    "durationSec": round(duration, 3),
    "toksPerSec":  round(count / duration, 1)
  }
  yield f"event: done\ndata: {metrics}\n\n"
  yield "event: end\ndata: [STREAM_END]\n\n"

# 7. Stream endpoint
@app.route("/stream", methods=["POST"])
def stream():
  data = request.get_json(force=True, silent=True) or {}
  prompt = data.get("prompt", "").strip()
  if not prompt:
    return Response(
      "event: error\ndata: Missing prompt\n\n",
      mimetype="text/event-stream"
    )

  logger.info("New request (%d chars)", len(prompt))
  return Response(
    generate_stream(prompt),
    mimetype="text/event-stream",
    headers={"Cache-Control": "no-cache"}
  )

# 8. Run server
if __name__ == "__main__":
  logger.info("Starting server on %s:%d", args.host, args.port)
  app.run(host=args.host, port=args.port)
