import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
import json
import numpy as np
from tqdm.auto import tqdm

PATH = "PATH_TO_GATHERED_RESULTS/all_relations_0.json"

with open(PATH, "r") as f:
    triples = json.load(f)

print(len(triples))

device = "cuda:0"
EMB_BATCH_SIZE = 24
INDEX_BATCH = 400000
MAX_LENGTH_EMB = 128

CACHE_DIR = "HF_CACHE_DIR"

from utils.contr.src.contriever import Contriever
from transformers import AutoTokenizer
import torch

contriever = Contriever.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR).to(device)
contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR) 

def get_representation(phrases, batch_size=EMB_BATCH_SIZE, pbar=False):
    steps = int(np.ceil(len(phrases) * 1.0 / batch_size))
    all_embs = []
    with torch.no_grad():
        for i in (tqdm(range(steps)) if pbar else range(steps)):
            inputs = contriever_tokenizer(phrases[i*batch_size:min((i+1)*batch_size, len(phrases))], padding=True, truncation=True, return_tensors="pt", max_length=MAX_LENGTH_EMB).to(device)
            all_embs.append(contriever(**inputs).cpu().detach().numpy())
    return np.concatenate(all_embs, axis=0) if len(all_embs) > 0 else []


from utils.memory.memory_controller_v3 import MemoryControllerForRelations

uri = "PYRO:MEMORY_SERVER_URI"
RelMem = MemoryControllerForRelations(uri,
                                      embedding_function=get_representation,
                                      embedding_function_rel=get_representation,
                                      caching_strategy="func",
                                      dist_thr=0.35,
                                      dist_rel_thr=0.3)

print("Controller loaded")
RelMem.index_entities(embedding_hnsw_ef_runtime=300, max_elements=25000000)
RelMem.index_relation_types(embedding_hnsw_ef_runtime=200, max_elements=5000)
# RelMem.RelMem.load_snapshot()


steps = int(np.ceil(len(triples) * 1.0 / INDEX_BATCH))
for i in range(steps):
    print(f"Batch #{i+1} / {steps}")
    relations = triples[i*INDEX_BATCH:min((i+1)*INDEX_BATCH, len(triples))]
    RelMem.store_relationship_batched_v2(relations)
print("Done")