import jaxtyping
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import einops
from tqdm import tqdm
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

MODEL_ID = "Qwen2.5-0.5B-Instruct"
TORCH_DTYPE = torch.bfloat16
DEVICE_MAP = "cpu"

torch.inference_mode()
torch.set_default_device(DEVICE_MAP)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, device_map=DEVICE_MAP)
if torch.backends.mps.is_available():
    TORCH_DTYPE = torch.float32
    DEVICE_MAP = "mps"
    model = model.to(TORCH_DTYPE).to(DEVICE_MAP).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# settings:
instructions = 32
layer_idx = int(len(model.model.layers) * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)

def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)

harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

print(harmful_hidden)

harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
