# Import stuff
import torch
import tqdm.auto as tqdm
import plotly.express as px
import os
import json
from peft import LoraConfig
from tqdm import tqdm
from jaxtyping import Float
import pickle

# import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import json

torch.set_grad_enabled(False)
load_path = os.path.join(
    "D:\Data\jailbreak", "runs", "activation_pca\gemma-2-9b-it\completions"
)
load_name_j = "gemma-2-9b-it_completions_evil_confidant"
load_name_h = "gemma-2-9b-it_completions_HHH"


def get_hhh_data(completions_hhh, completions_jailbreak):
    completion_choice = []

    for completion_h, completion_j in zip(completions_hhh, completions_jailbreak):
        category = completion_h["category"]
        response_h = completion_h["response"]
        response_j = completion_j["response"]
        completion_c = {}
        if category == "HHH":
            completion_c["request"] = completion_h["request"]
            completion_c["response_HHH"] = response_h[:200]
            completion_c["response_jailbreak"] = response_j[:200]
            completion_c["label"] = completion_h["label"]
            completion_c["ID"] = completion_h["ID"]
            completion_c["category"] = completion_h["category"]

            completion_choice.append(completion_c)
    return completion_choice


def get_jailbreak_data(completions_hhh, completions_jailbreak):
    # 1. categories: HHH vs other
    # Criteria (only for non HHH): 1. refusal score = 1 for HHH, refusal score = 0 for jailbreak

    completion_choice = []
    for completion_h, completion_j in zip(completions_hhh, completions_jailbreak):
        category = completion_h["category"]
        response_h = completion_h["response"]
        response_j = completion_j["response"]

        if completion_h["refusal_score"] == 1 and completion_j["refusal_score"] == 0:
            completion_c = {}
            completion_c["request"] = completion_h["request"]
            completion_c["response_HHH"] = response_h[:200]
            completion_c["response_jailbreak"] = response_j[:200]
            completion_c["label"] = completion_h["label"]
            completion_c["ID"] = completion_h["ID"]
            completion_c["category"] = completion_h["category"]
            completion_choice.append(completion_c)
    return completion_choice


# 1. Load from completion

# Load HHH completion
with open(os.path.join(load_path, load_name_h + ".json"), "rb") as f:
    data_HHH = json.load(f)
# Load jailbreak completion
with open(os.path.join(load_path, load_name_j + ".json"), "rb") as f:
    data_jailbreak = json.load(f)


completions_hhh = data_HHH["completions"]
completions_jailbreak = data_jailbreak["completions"]


# HHH response

completion_choice_hhh = get_hhh_data(completions_hhh, completions_jailbreak)

save_name = "harmless_choice"
# save_path = os.path.join("D:", "Code", "deception_and_jailbreak", "dataset", "choice")
with open(save_name + ".json", "w") as f:
    json.dump(completion_choice_hhh, f, indent=4)


### Jailbreak response
completion_choice_jailbreak = get_jailbreak_data(completions_hhh, completions_jailbreak)

save_name = "harmful_choice"
with open(save_name + ".json", "w") as f:
    json.dump(completion_choice_jailbreak, f, indent=4)
