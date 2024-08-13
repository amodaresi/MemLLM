from datasets import load_dataset
import numpy as np
import random, os
from itertools import product
import json
from tqdm import tqdm

FILTERED_FILEPATH = "FILTERED_DATA_DIR/test_filtered.json"
OUTPUT_FILEPATH = "FILTERED_DATA_DIR_PROCESSED"

is_all = False
subset = "test"

with open("utils/wikidata_props_en.json", "r") as f:
    wikiprops = json.load(f)

# is_all = True returns all memwrite queries with different formats of entities as well: X Inc, X, etc. False picks a random one.
def get_memwrites(element, doc_idx, is_all=False, is_random=True, return_entities=False, drop_annotated=False, return_relidx=False):
    memwrite_queries = []
    prev_len = 0
    if is_random:
        index_len = random.randint(1,5)
    else:
        index_len = 1
    while prev_len==0 or prev_len<len(element["sents"]):
        prefix = " ".join([" ".join(x) for x in element["sents"][:prev_len]])
        document = " ".join([" ".join(x) for x in element["sents"][prev_len:prev_len+index_len]])
        selected_relations = []
        for rel_idx, relation in enumerate(element["labels"]):
            evidence = relation["evidence"]
            if len(evidence) == 0:
                selected_relations.append(rel_idx)
            elif not drop_annotated:
                for ev_sent in evidence:
                    if ev_sent>=prev_len and ev_sent<(prev_len+index_len):
                        selected_relations.append(rel_idx)
                        break

        entities = []
        if return_entities:
            for vs in element["vertexSet"]:
                for vertex in vs:
                    if vertex["sent_id"]<(prev_len)+index_len and vertex["name"] not in entities:
                        entities.append(vertex["name"])

        memwrite = []
        location = []
        memwrite_idxes = []
        for sel_rel in selected_relations:
            head = element["labels"][sel_rel]["h"]
            tail = element["labels"][sel_rel]["t"]
            unique_combinations = [x for x in product(element["vertexSet"][head], element["vertexSet"][tail])]
            random.shuffle(unique_combinations)
            for el in unique_combinations:
                head_in_span = el[0]["sent_id"]>=prev_len and el[0]["sent_id"]<(prev_len)+index_len
                tail_in_span = el[1]["sent_id"]>=prev_len and el[1]["sent_id"]<(prev_len)+index_len
                head_and_tail_in_range = el[1]["sent_id"]<(prev_len)+index_len and el[0]["sent_id"]<(prev_len)+index_len

                # if el[0]["name"].lower() not in (prefix.lower()+" "+document.lower()):
                #     continue
                # if el[1]["name"].lower() not in (prefix.lower()+" "+document.lower()):
                #     continue

                if (head_in_span or tail_in_span) and head_and_tail_in_range:
                    if [el[0]["name"], wikiprops[element["labels"][sel_rel]["r"]], el[1]["name"]] in memwrite:
                        continue
                    memwrite.append([el[0]["name"], wikiprops[element["labels"][sel_rel]["r"]], el[1]["name"]])
                    memwrite_idxes.append(sel_rel)
                    head_loc = (el[0]['sent_id'], el[0]['pos'][0])
                    tail_loc = (el[1]['sent_id'], el[1]['pos'][0])
                    if not (head_in_span and tail_in_span):
                        if head_in_span:
                            location.append(head_loc)
                        elif tail_in_span:
                            location.append(tail_loc)
                    else:
                        if head_loc>tail_loc:
                            location.append(head_loc)
                        else:
                            location.append(tail_loc)
                    if not is_all:
                        break
                    
        # Sorting based on the location
        memwrite = [x for _, x in sorted(zip(location, memwrite))]
        rel_idxes = [x for _, x in sorted(zip(location, memwrite_idxes))]
        memwrite_queries.append({"pretext": prefix, "index": document, "relations":memwrite, "doc_idx":doc_idx, "entities": entities, "sent_id": prev_len})
        if return_relidx:
            memwrite_queries[-1]["rel_idxes"] = rel_idxes
        prev_len = (prev_len+index_len)
        index_len = random.randint(1,5) if is_random else 1
    return memwrite_queries


ds = load_dataset("json", data_files=FILTERED_FILEPATH, keep_in_memory=True)["train"]

memwrite_queries = []
random.seed(42)
for doc_idx, element in tqdm(enumerate(ds)):
    # if doc_idx == 2:
    #     print("a")
    x = get_memwrites(element, doc_idx=doc_idx, is_all=is_all, is_random=False, return_entities=False)
    memwrite_queries.extend(x)

fn = f"{OUTPUT_FILEPATH}/{subset}" + ".json"
print(fn)
with open(fn, "w") as f:
    json.dump(memwrite_queries, f, indent=2)