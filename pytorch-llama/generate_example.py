# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb

import os
import time
import urllib.request

import torch

from model import Llama3Model, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text

#######################################
# Model settings

MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"

MODEL_CONTEXT_LENGTH = 131_072  # Supports up to 131_072

# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "How many helicopters can a human eat in one sitting?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.
TOP_K = 1
#######################################


###################################
# Initialize model
##################################

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE}...")
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Downloaded to {MODEL_FILE}")


if "1B" in MODEL_FILE:
    from model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from model import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location=torch.device('cpu')))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    # torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

model.to(device)

###################################
# Initialize tokenizer
##################################
TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)

###################################
# Generate text
##################################

torch.manual_seed(42)

print("Here")

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

elapsed = time.time() - start

print(f"Time: {elapsed:.2f} sec")

print(f"TPS: {token_ids.shape[1] / elapsed:.2f}")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
