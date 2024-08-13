import os, json
from tqdm.auto import tqdm
from datasets import load_dataset

DATASET_PATH = "PATH_TO_DATASET/test_filtered.json"
OUTPUT_NAME = "PATH_TO_GATHERED_DATA/test_relations_to_store.json"

ds = load_dataset("json", data_files=DATASET_PATH, keep_in_memory=True)["train"]

import json
with open("../../utils/wikidata_props_en.json", "r") as f:
    wikiprops = json.load(f)

def convert_relation_to_readable(rel, example):
    h = example["vertexSet"][rel['h']]
    t = example["vertexSet"][rel['t']]

    relations = []
    r_name = wikiprops[rel["r"]]
    for h_alias in h:
        for t_alias in t:
            relations.append([h_alias["name"], r_name, t_alias["name"]])
 
    return relations

triples_set = set()
triples = []

for d in ds:
    for r in d["labels"]:
        rel_readable = convert_relation_to_readable(r, d)
        for r_hr in rel_readable:
            r_hr_str = ">>".join(r_hr)
            if r_hr_str not in triples_set:
                triples_set.add(r_hr_str)
                triples.append(r_hr)

print(len(triples))
with open(OUTPUT_NAME, "w") as f:
    json.dump(triples, f, indent=2)
