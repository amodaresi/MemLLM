import os
import re
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, default=1)
argparser.add_argument('--checkpoint_dir', type=str, default='', help='Directory to save the output files')
argparser.add_argument('--dataset_path', type=str)
argparser.add_argument('--shard_size', type=int, default=10, help='Number of articles per shard')
argparser.add_argument('--start_idx', type=int, default=0, help='Start doc_idx:')
argparser.add_argument('--stop_idx', type=int, default=-1, help='Stop doc_idx:')
argparser.add_argument('--force_generate', action='store_true')
argparser.add_argument('--max_prompt_length', type=int, default=512, help='Maximum token length for prompts')
argparser.add_argument('--max_generation_length', type=int, default=512, help='Maximum token length for generated outputs')
argparser.add_argument('--store_inputs', type=str, default="index")
argparser.add_argument('--cache_dir', type=str, default='', help='Cache directory for the model')
argparser.add_argument('--output_dir', type=str, default="")
args = argparser.parse_args()
print(args)

from peft import PeftModel
from datasets import load_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast
import transformers
import torch
import json
import nltk
import pickle
import numpy as np
from copy import copy
from tqdm import tqdm
import argparse

CACHE_DIR = args.cache_dir
MODEL_DIR = args.checkpoint_dir
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
print(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model_config = transformers.AutoConfig.from_pretrained(BASE_MODEL_NAME)

device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    config=model_config,
    torch_dtype=torch.bfloat16,
    # trust_remote_code=True,
    cache_dir=CACHE_DIR,
    # device="auto"
).to(device)
model = PeftModel.from_pretrained(
    model,
    MODEL_DIR,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
).to(device)

model.config.pad_token_id = tokenizer.pad_token_id  # unk
model.config.eos_token_id = tokenizer.eos_token_id

tokenizer.add_special_tokens({"additional_special_tokens": [
    "({", "})", "-->"
]})

CONTINUE_TOKEN_ID = tokenizer.convert_tokens_to_ids(["-->"])[0]

import tempfile

temp_dir = tempfile.TemporaryDirectory()
print(temp_dir.name)

merged_model_path = os.path.join(temp_dir.name, "merged_model")
print(merged_model_path)

merged = model.merge_and_unload()
merged.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Model temporarily merged")

del model, merged
torch.cuda.empty_cache()

dataset_path = args.dataset_path
if dataset_path.endswith(".json"):
    ds = load_dataset("json", data_files=dataset_path, keep_in_memory=True)["train"]
elif dataset_path.endswith(".pkl"):
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
else:
    raise Exception("Dataset either filtered json or augmented pkl")

START_IDX = args.start_idx
STOP_IDX = len(ds) if args.stop_idx < 0 else args.stop_idx
print(START_IDX, STOP_IDX)

MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_OUTPUT_LENGTH = args.max_generation_length

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.0, 
    max_tokens=MAX_OUTPUT_LENGTH, 
    skip_special_tokens=False, 
    logprobs=5, 
    spaces_between_special_tokens=True, 
    special_token_ignore=1542, 
    special_token_stop_after=4
)

llm = LLM(model=merged_model_path, dtype=torch.bfloat16, tensor_parallel_size=args.num_gpus)
temp_dir.cleanup()

tokenizer = llm.get_tokenizer()
tokenizer.truncation_side = "left"

# bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir=CACHE_DIR)


def add_whitespace_around_dashes(input_string):
    # Pattern to find hyphens, en dashes, and em dashes between numbers
    # and ensure spaces around the dash. This matches a digit followed by any dash and another digit
    pattern = r'(?<=\d)\s*[-–—]\s*(?=\d)'
    # Replace pattern with en dash surrounded by a single space on each side
    # Using an en dash (–) for replacement, but you can adjust as needed
    output_string = re.sub(pattern, ' – ', input_string)
    return output_string

def converter(text):
    output = " ".join(nltk.word_tokenize(text))
    output = add_whitespace_around_dashes(output)
    return output

def preprocess_function(example):
    full_text = example["pretext"] + "({USER_ST})" + example["index"] + "({USER_END})"
    return tokenizer.encode(full_text, max_length=MAX_PROMPT_LENGTH)

def batched_preprocess_function(examples):
    full_texts = [example["pretext"] + "({USER_ST})" + example["index"] + "({USER_END})" for example in examples]
    return tokenizer.batch_encode_plus(full_texts, max_length=MAX_PROMPT_LENGTH)["input_ids"]

# Build shard
no_of_shards = int(np.ceil((STOP_IDX - START_IDX) / args.shard_size))

for shard_idx in tqdm(range(no_of_shards)):
    shard_start = START_IDX + shard_idx * args.shard_size
    shard_end = min(START_IDX + (shard_idx + 1) * args.shard_size, STOP_IDX)
    shard_data = []
    for data_idx in tqdm(range(shard_start, shard_end)):
        data_sample = copy(ds[data_idx])

        text = data_sample["original_prompt"] + " It is or they are " + '"'+data_sample["edit"].strip()+'"'

        # sent_text = [converter(t) for t in nltk.sent_tokenize(text)]

        # for sent_id in range(len(sent_text)):
            # example = copy(data_sample)
            # example["pretext"] = " ".join(sent_text[max(0, sent_id-24):sent_id])
            # example["index"] = sent_text[sent_id]
            # example["sent_id"] = sent_id
        #     shard_data.append(example)
        example = copy(data_sample)
        example["pretext"] = ""
        example["index"] = text
        example["sent_id"] = 0
        shard_data.append(example)
        
        # prompt_token_ids.append(preprocess_function(example))
    prompt_token_ids = batched_preprocess_function(shard_data)
    print(len(prompt_token_ids))
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    final_outputs = []
    for idx, el in enumerate(shard_data):
        try:
            cuttingpoints = list(outputs[idx].outputs[0].cutting_points.keys())
            cuttingpoint_vals = list(outputs[idx].outputs[0].cutting_points.values())
            if len(cuttingpoint_vals) != 0:
                _argmax = cuttingpoints[int(np.argmax(cuttingpoint_vals))]
                if args.force_generate and outputs[idx].outputs[0].token_ids[_argmax-1] == CONTINUE_TOKEN_ID:
                    cuttingpoints = cuttingpoints[1:]
                    cuttingpoint_vals = cuttingpoint_vals[1:]
                    _argmax = cuttingpoints[int(np.argmax(cuttingpoint_vals))]
                first_text = tokenizer.decode(outputs[idx].outputs[0].token_ids[:_argmax]) + "})</s>"
            else:
                first_text = tokenizer.decode(outputs[idx].outputs[0].token_ids)
            

            el["raw_generation"] = first_text
            if args.store_inputs == "index":
                del el["pretext"]
            elif args.store_inputs == "none":
                del el["pretext"]
                del el["index"]
        except:
            el["raw_generation"] = ""

        final_outputs.append(el)
    
    with open(os.path.join(args.output_dir, f"shard_output_{shard_start}_{shard_end}.json"), "w") as f:
        json.dump(final_outputs, f, indent=2)
    