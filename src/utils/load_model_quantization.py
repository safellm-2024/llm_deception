# Import stuff
import torch
import tqdm.auto as tqdm
import plotly.express as px

from tqdm import tqdm
from jaxtyping import Float

# import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.set_grad_enabled(False)

LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-70b-chat-hf"
inference_dtype = torch.float32
# inference_dtype = torch.float32
# inference_dtype = torch.float16

print("loading from hf")
hf_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_2_7B_CHAT_PATH,
    torch_dtype=inference_dtype,
    device_map="cuda",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)

print(
    "free(Gb):",
    torch.cuda.mem_get_info()[0] / 1000000000,
    "total(Gb):",
    torch.cuda.mem_get_info()[1] / 1000000000,
)
print("%%%%%%%%%%%%%%%%%%", end="/n/n")


print("loading from transformer lens")
model = HookedTransformer.from_pretrained(
    LLAMA_2_7B_CHAT_PATH,
    hf_model=hf_model,
    dtype=inference_dtype,
    fold_ln=False,
    fold_value_biases=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=tokenizer,
)
print("before delete")
print(
    "free(Gb):",
    torch.cuda.mem_get_info()[0] / 1000000000,
    "total(Gb):",
    torch.cuda.mem_get_info()[1] / 1000000000,
)
print("%%%%%%%%%%%%%%%%%%", end="/n/n")

del hf_model
del tokenizer
gc.collect()
torch.cuda.empty_cache()


print("after delete")
print(
    "free(Gb):",
    torch.cuda.mem_get_info()[0] / 1000000000,
    "total(Gb):",
    torch.cuda.mem_get_info()[1] / 1000000000,
)
print("%%%%%%%%%%%%%%%%%%", end="/n/n")

print(model.generate("The capital of Germany is", max_new_tokens=2, temperature=0))

print("Done")
