import os
import numpy as np
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

CACHE_DIR = "HF_CACHE_DIR"
ReDOCRED_EVAL_LOC = "test_eval.withOther.json"

MODEL_DIR = "PATH_TO_MR_MODEL"
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

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
# print("##### EVAL DOCRED (MW Generated) #####")

RelMem = MemoryControllerForRelations(uri,
                                      embedding_function=get_representation,
                                      embedding_function_rel=get_representation,
                                      caching_strategy="func",
                                      dist_thr=0.15,
                                      dist_rel_thr=0.15)
# RelMem.index_entities(embedding_hnsw_ef_runtime=300, max_elements=30000)
# RelMem.index_relation_types(embedding_hnsw_ef_runtime=300, max_elements=256)


ds = load_dataset("json", data_files=ReDOCRED_EVAL_LOC, cache_dir=CACHE_DIR, keep_in_memory=True)["train"]
ds_per_doc = []
prev_doc_idx = -1
for example in ds:
    if example["data_idx"] != prev_doc_idx:
        ds_per_doc.append([])
        prev_doc_idx = example["data_idx"]
    ds_per_doc[-1].append(example)

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
    memory_controller=RelMem,
    memory_write_late_stopping=5,
    memory_write_generation_config=GenerationConfig(max_length=512, pad_token_id=mr_model.config.pad_token_id),
    memory_read_text_generation_config=GenerationConfig(max_length=128, pad_token_id=mr_model.config.pad_token_id),
    memory_read_api_generation_config=GenerationConfig(max_new_tokens=128, pad_token_id=mr_model.config.pad_token_id),
)

USE_LABEL_FOR_QUERY = "N"
USE_LABEL_ONLY_FOR_ANSWER = False
USE_LABEL_FOR_ANSWER = False
USE_LABEL_FOR_ANSWER_IF_NOT_EMPTY = False
USE_LABEL_FOR_MR_POS = False
MEM_DISABLED = False

REL_SIM = 0.1

P_BRANCHING = True

print("USE_LABEL_FOR_QUERY", USE_LABEL_FOR_QUERY)
print("USE_LABEL_FOR_ANSWER", USE_LABEL_FOR_ANSWER)
print("USE_LABEL_ONLY_FOR_ANSWER", USE_LABEL_ONLY_FOR_ANSWER)
print("USE_LABEL_FOR_ANSWER_IF_NOT_EMPTY", USE_LABEL_FOR_ANSWER_IF_NOT_EMPTY)
print("USE_LABEL_FOR_MR_POS", USE_LABEL_FOR_MR_POS)
print("MEM_DISABLED", MEM_DISABLED)
print("P_BRANCHING", P_BRANCHING)

print("MEM_URI", uri)

print("MODEL_NAME:", MODEL_DIR)
print("Dataset:", ReDOCRED_EVAL_LOC)
print("REL_SIM:", REL_SIM)

PPLs_per_doc = []
total_loss = 0
total_token_length = 0

target_loss_stream = []
entity_loss_stream = []

def get_token_span(pos, word_to_end_token, bos=True):
    start = None
    for p in range(pos[0], pos[1]):
        if p >= len(word_to_end_token):
            token_span = (word_to_end_token[p-1], 1000000000000000)
        elif p == 0:
            token_span = (1 if bos else 0, word_to_end_token[p])
        else:
            token_span = (word_to_end_token[p-1], word_to_end_token[p])
        if start is None:
            start = token_span[0]
        end = token_span[1]
    return start, end

def get_entity_perplexities(text_loss_stream, doc):
    entity_losses = []
    for ds in doc:
        if ds["eot"]:
            whole_text = ds["full_text_tokenized"]
            entities_loc = ds["entities_loc"]

    all_tokenized = tokenizer([" ".join(whole_text[:i]) for i in range(1, len(whole_text))])
    word_to_end_token = [len(all_tokenized["input_ids"][i]) for i in range(len(all_tokenized["input_ids"]))]

    for ent_loc in entities_loc:
        tok_loc = get_token_span(ent_loc, word_to_end_token)
        entity_losses.append(text_loss_stream[tok_loc[0]-1:tok_loc[1]-1])
    return entity_losses

i = 0
for doc in tqdm(ds_per_doc):
    loss_stream, text_only_loss_stream, token_length, target_losses = memllm.perplexity_eval(doc, GOLD_POS=USE_LABEL_FOR_MR_POS, GOLD_Q=USE_LABEL_FOR_QUERY, GOLD_A=USE_LABEL_FOR_ANSWER, GOLD_A_ONLY=USE_LABEL_ONLY_FOR_ANSWER, MEM_DISABLED=MEM_DISABLED, REL_SIM=REL_SIM, p_branching=P_BRANCHING, GOLD_A_IF_NOT_EMPTY=USE_LABEL_FOR_ANSWER_IF_NOT_EMPTY)
    # l, _, tl = memllm.perplexity_eval_wo_mem(doc)
    total_loss += np.sum(loss_stream)
    total_token_length += token_length
    PPLs_per_doc.append(np.sum(loss_stream) / token_length)

    for t in target_losses:
        target_loss_stream.append(np.mean(t))

    entity_losses = get_entity_perplexities(text_only_loss_stream, doc)
    for t in entity_losses:
        entity_loss_stream.append(np.mean(t))

    i += 1
    if i % 10 == 0:
        print("PPL (Total):", total_loss/total_token_length, " -- PPL (Target): ", np.mean(target_loss_stream), " -- PPL (Entities): ", np.mean(entity_loss_stream))
print("DONE")
print("PPL (Total):", total_loss/total_token_length, " -- PPL (Target): ", np.mean(target_loss_stream), " -- PPL (Entities): ", np.mean(entity_loss_stream))