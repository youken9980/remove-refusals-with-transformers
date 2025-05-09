import torch
import gc
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from tqdm import tqdm

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

MODEL_ID = "Qwen/Qwen3-32B"
# More samples can help find the direction better.
NUM_PROMPT_SAMPLES = 32
# Used to skip the first and last layers for the modifications.
SKIP_BEGIN_LAYERS = 1  # Don't mess with the first layer.
SKIP_END_LAYERS = 0
# The layer we will use for the refusal_dir calculation will be floor(LAYER_FRACTION_TO_USE * model.layers).
LAYER_FRACTION_TO_USE = 0.6

TORCH_DTYPE = torch.float16
DEVICE_MAP = "cpu"

torch.inference_mode()
torch.set_default_device(DEVICE_MAP)
torch.set_grad_enabled(False)

# Load the model on the GPU in quantized type if we can.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    # quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=TORCH_DTYPE),
    low_cpu_mem_usage=True,
    device_map=DEVICE_MAP
)
model.requires_grad_(False)
if torch.backends.mps.is_available():
    TORCH_DTYPE = torch.float32
    DEVICE_MAP = "mps"
    model = model.to(TORCH_DTYPE).to(DEVICE_MAP).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

layer_idx = int(len(model.model.layers) * LAYER_FRACTION_TO_USE)
print("Layer index for refusal direction: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, min(NUM_PROMPT_SAMPLES, len(harmful)))
harmless_instructions = random.sample(harmless, min(NUM_PROMPT_SAMPLES, len(harmless)))

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

bar_generate = tqdm(total = len(harmful_instructions) + len(harmless_instructions), desc = "Generating samples")

# Only return the final hidden state of the layer we care about, and use 'cpu' to save VRAM.
def generate(toks):
    output = model.generate(
        toks.to(model.device),
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id
    )
    bar_generate.update(n=1)
    return output.hidden_states[0][layer_idx][:, -1, :].to(DEVICE_MAP) # Final hidden state = -1.

harmful_hidden = [generate(toks) for toks in harmful_toks]
harmless_hidden = [generate(toks) for toks in harmless_toks]

bar_generate.close()

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir.squeeze() / refusal_dir.norm()

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
