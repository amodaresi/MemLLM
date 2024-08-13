import os, re
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.environ["CUDA_VISIBLE_DEVICES"])

import torch
from tqdm.auto import tqdm
import transformers
from datasets import load_dataset
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
# from sklearn.metrics.pairwise import cosine_similarity
# from utils.redisDB import RedisDBForRelations

# from utils.memory_controller import MemoryControllerForRelations
from utils.memory.memory_controller_v3 import MemoryControllerForRelations
from inference.memllm import MEMLLM

import nltk, re, json
nltk.download("punkt")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

CACHE_DIR = "HF_CACHE"

MODEL_DIR = "MR_MODEL_PATH"
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LOC = "MEMORY_WRITE_OUTPUTS/outputs.json"

from utils.contr.src.contriever import Contriever
from transformers import AutoTokenizer

contriever = Contriever.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR).to(device)
contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR) 

def get_representation(phrases, batch_size=128, pbar=False):
    steps = int(np.ceil(len(phrases) * 1.0 / batch_size))
    all_embs = []
    for i in (tqdm(range(steps)) if pbar else range(steps)):
        inputs = contriever_tokenizer(phrases[i*batch_size:min((i+1)*batch_size, len(phrases))], padding=True, truncation=True, return_tensors="pt").to(device)
        all_embs.append(contriever(**inputs).cpu().detach().numpy())
    return np.concatenate(all_embs, axis=0) if len(all_embs) > 0 else []

uri = "PYRO:MEMORY_SERVER_URI"

def get_new_memory():
    RelMem = MemoryControllerForRelations(uri,
                                        embedding_function=get_representation,
                                        embedding_function_rel=get_representation,
                                        caching_strategy="func",
                                        dist_thr=0.15,
                                        dist_rel_thr=0.8)

    RelMem.index_entities(embedding_hnsw_ef_runtime=300, max_elements=30000)
    RelMem.index_relation_types(embedding_hnsw_ef_runtime=300, max_elements=256)
    return RelMem

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    config=model_config,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    device_map="auto"
)

mr_model = PeftModel.from_pretrained(
    model,
    MODEL_DIR,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
)

if tokenizer.pad_token_id:
    mr_model.config.pad_token_id = tokenizer.pad_token_id  # unk
else:
    mr_model.config.pad_token_id = 0
mr_model.config.eos_token_id = tokenizer.eos_token_id

tokenizer.add_special_tokens({"additional_special_tokens": [
    "({", "})", "-->"
]})
tokenizer.model_max_length = 1024

memllm = MEMLLM(
    mr_model,
    None,
    tokenizer,
    memory_controller=get_new_memory(),
    memory_write_late_stopping=5,
    memory_write_generation_config=GenerationConfig(max_length=512, pad_token_id=mr_model.config.pad_token_id),
    memory_read_text_generation_config=GenerationConfig(max_new_tokens=64, pad_token_id=mr_model.config.pad_token_id),
    memory_read_api_generation_config=GenerationConfig(max_new_tokens=128, pad_token_id=mr_model.config.pad_token_id),
)

REL_SIM = 0.5
T = 1000
N = 1000
SEED = 123
MEM_DISABLED = False

# PROMPT = """Who was the founder of Apple Inc.?({MEM_READ(>>founder>>Apple Inc.)-->Steve Jobs;Steve Wozniak;Ronald Wayne})Apple Inc. was founded by Steve Jobs, Steve Wozniak and Ronald Wayne.</s>"""
# PROMPT = """Who was the founder of Saffron Inc.?({MEM_READ(>>founder>>Saffron Inc.)-->Josh Weasley})Saffron Inc. was founded by Josh Weasley.</s>"""
# PROMPT = """on what year did the cyber wars happen? The cyber wars happened in 1992.</s>Where did Franco Staccioni born? Franco Staccioni was born in Hawaii, United States.</s>Who was the founder of Saffron Inc.?({MEM_READ(>>founder>>Saffron Inc.)-->Josh Weasley})Saffron Inc. was founded by Josh Weasley.</s>"""
# PROMPT = """in which county is the city los santos located at? The city los santos is located at Jacksonville county.</s>Which company developed compound VI? Compound VI was developed by Hexacorp Inc.</s>on what year did the cyber wars happen? The cyber wars happened in 1992.</s>Where did Franco Staccioni born? Franco Staccioni was born in Hawaii, United States.</s>Who was the founder of Saffron Inc.?({MEM_READ(>>founder>>Saffron Inc.)-->Josh Weasley})Saffron Inc. was founded by Josh Weasley.</s>"""
# PROMPT = """in which county is the city los santos located at? Jacksonville county</s>Which company developed compound VI? Hexacorp Inc.</s>on what year did the cyber wars happen? 1992</s>Where did Franco Staccioni born? Hawaii, United States</s>Who was the founder of Saffron Inc.?({MEM_READ(>>founder>>Saffron Inc.)-->Josh Weasley})Josh Weasley</s>"""
PROMPT = """in which county is the city los santos located at? Jacksonville county</s>Which company developed compound VI? Hexacorp Inc.</s>on what year did the cyber wars happen? 1992</s>Where did Franco Staccioni born? Hawaii, United States</s>What was Josh Weasly's occupation?({MEM_READ(Josh Weasly>>position held>>)-->member of the workers union})member of the workers union</s>"""


random.seed(SEED)
np.random.seed(SEED)


with open(LOC, "r") as f:
    model_edit_data = json.load(f)

with_edits_idxes = [i for i, d in  enumerate(model_edit_data) if d["generated_relations"] != []]

print(len(with_edits_idxes))

# selected_items = np.random.choice(with_edits_idxes, N, replace=False).tolist()
selected_items = np.arange(1000).tolist()

def remove_between_brackets(s):
    pattern = r'\(\{.*?\}\)'
    return re.sub(pattern, '', s)

SPACES = 9 # The tokenizer will add spaces after and before special tokens so we add this

def get_output(input_text, filtered_relationships=[], get_raw=False):
    full_prompt = PROMPT + input_text + "({"
    full_path = []
    for x, status in memllm.lazy_memory_read(full_prompt, REL_SIM=REL_SIM, stop_after_first_success=True, filtered_relationships=filtered_relationships):
        full_path.append(memllm.tokenizer.decode(x))
        # pass
    final_output = memllm.tokenizer.decode(x)
    removed_memapi_output = remove_between_brackets(final_output)
    full_prompt_rem_memapi = remove_between_brackets(full_prompt)
    final_text_output = removed_memapi_output[len(full_prompt_rem_memapi)+SPACES:]
    if get_raw:
        return final_text_output, final_output, full_path
    return final_text_output, full_path

def get_output_for_debug(input_text):
    full_prompt = PROMPT + input_text + "({"
    for x, status in memllm.lazy_memory_read(full_prompt, REL_SIM=REL_SIM, stop_after_first_success=True):
        # pdb.set_trace()
        pass
    final_output = memllm.tokenizer.decode(x)
    removed_memapi_output = remove_between_brackets(final_output)
    full_prompt_rem_memapi = remove_between_brackets(full_prompt)
    final_text_output = removed_memapi_output[len(full_prompt_rem_memapi)+SPACES:]
    return final_text_output

new_results = []
acc = []
if MEM_DISABLED:
    NAME = "MEM_DISABLE_1000_outputs_shortPrompt4shot-001-WISE.json"
    print(LOC.replace(".json",NAME))
    print(PROMPT)
    # We should store one-by-one and evaluate the individual edit
    for idx in tqdm(selected_items):
        sample = model_edit_data[idx]
        sample["original_prompt_result"], sample["generation_path"] = get_output(sample["original_prompt"])
        acc.append(sample["edit"].lower() in sample["original_prompt_result"].lower())
        print(np.mean(acc))
        sample["rephrase_prompt_result"], _ = get_output(sample["rephrase_prompt"])
        sample["loc_prompt_result"], _ = get_output(sample["loc_prompt"])
        new_results.append(sample)
elif T == 1:
    NAME = "rand_T1_1000_outputs_shortPrompt4shot-001-WISE.json"
    print(LOC.replace(".json",NAME))
    print(PROMPT)
    # We should store one-by-one and evaluate the individual edit
    for idx in tqdm(selected_items):
        sample = model_edit_data[idx]
        memllm.memory_controller = get_new_memory()
        memllm.memory_controller.store_relationship_batched_v2(sample["generated_relations"], verbose=False)
        
        sample["original_prompt_result"], sample["generation_path"] = get_output(sample["original_prompt"])
        acc.append(sample["edit"].lower() in sample["original_prompt_result"].lower())
        print(np.mean(acc))
        sample["rephrase_prompt_result"], _ = get_output(sample["rephrase_prompt"])
        sample["loc_prompt_result"], _ = get_output(sample["loc_prompt"])
        new_results.append(sample)

else:
    # We should apply all edits and evaluate them all at once
    NAME = f"rand_T1000_1000_outputs_shortPrompt4shot-001-WISE-{0.15}-{REL_SIM}-LR5-wRAW-wSteps.json"
    print(LOC.replace(".json",NAME))
    memllm.memory_controller = get_new_memory()
    for idx in tqdm(selected_items):
        sample = model_edit_data[idx]
        memllm.memory_controller.store_relationship_batched_v2(sample["generated_relations"], verbose=False)
    
    mem_dump = {}
    idx = 1
    while True:
        try:
            rel_triple = memllm.memory_controller.RelMem.get_relationships_by_id([idx], replace_id_w_txt=True)
            if len(rel_triple) > 0:
                mem_dump[idx] = rel_triple[0][1:]
            else:
                break
        except KeyError:
            break
        idx += 1

    print(mem_dump[10])
    print(mem_dump[100])
    print(mem_dump[1000])
    
    filtered_relationships = {}
    for idx in tqdm(selected_items):
        sample = model_edit_data[idx]
        filtered_relationships[idx] = []
        for rel in sample["generated_relations"]:
            if rel[0] == sample["edit"]:
                for q_rel_id, q_rel in mem_dump.items():
                    if q_rel[1] == rel[1] and q_rel[2] == rel[2] and q_rel[0] != sample["edit"]:
                        filtered_relationships[idx].append(q_rel_id)
            elif rel[2] == sample["edit"]:
                for q_rel_id, q_rel in mem_dump.items():
                    if q_rel[1] == rel[1] and q_rel[0] == rel[0] and q_rel[2] != sample["edit"]:
                        filtered_relationships[idx].append(q_rel_id)
    
    memllm.memory_controller = get_new_memory()
    for idx in tqdm(selected_items):
        sample = model_edit_data[idx]
        memllm.memory_controller.store_relationship_batched_v2(sample["generated_relations"], verbose=False)

        # sample["original_prompt_result"] = get_output(sample["original_prompt"])
        # sample["rephrase_prompt_result"] = get_output(sample["rephrase_prompt"])
        # sample["loc_prompt_result"] = get_output(sample["loc_prompt"])
        sample["original_prompt_result"], sample["original_prompt_result_RAW"], sample["original_prompt_result_all_steps"] = get_output(sample["original_prompt"], filtered_relationships=filtered_relationships[idx], get_raw=True)
        acc.append(sample["edit"].lower() in sample["original_prompt_result"].lower())
        print(np.mean(acc))
        sample["rephrase_prompt_result"], sample["rephrase_prompt_result_RAW"], sample["rephrase_prompt_result_all_steps"] = get_output(sample["rephrase_prompt"], filtered_relationships=filtered_relationships[idx], get_raw=True)
        sample["loc_prompt_result"], sample["loc_prompt_result_RAW"], sample["loc_prompt_result_all_steps"] = get_output(sample["loc_prompt"], get_raw=True)
        new_results.append(sample)

with open(LOC.replace(".json",NAME), "w") as f:
    json.dump(new_results, f, indent=2)

    