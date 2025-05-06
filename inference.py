from typing import Optional, Tuple
import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
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

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj


class AblationDecoderLayer(nn.Module):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        # noinspection PyTypeChecker
        return outputs


for idx in reversed(range(len(model.model.layers))):  # for qwen 1 this needs to be changed to model.transformer.h
    model.model.layers.insert(idx, AblationDecoderLayer())

conversation=[]

streamer = TextStreamer(tokenizer)

print(f"Chat with {MODEL_ID}:")
while True:
    prompt = input()
    conversation.append({"role": "user", "content": prompt})
    toks = tokenizer.apply_chat_template(conversation=conversation,
        add_generation_prompt=True, return_tensors="pt")

    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1337)

    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": "".join(decoded)})

