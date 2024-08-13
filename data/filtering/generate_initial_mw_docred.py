from datasets import load_dataset
import numpy as np
import random
from itertools import product
import json
from tqdm import tqdm

dataset = load_dataset("json", data_files="PATH_TO_SEEDDATASET", keep_in_memory=True)["train"]

FILEPATH = "PATH_TO_OUTPUT"

is_all = False
subset = "validation"
distant = True if "distant" in subset else False
print("Distant", distant)

with open("utils/wikidata_props_en.json", "r") as f:
    wikiprops = json.load(f)

# is_all = True returns all memwrite queries with different formats of entities as well: X Inc, X, etc. False picks a random one.
def get_memwrites(element, doc_idx, distant=False, is_all=False, is_random=True, return_entities=False, drop_annotated=False):
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
        memwrites_sorted = sorted(zip(location, memwrite, memwrite_idxes))
        memwrite = [x for _, x, _ in memwrites_sorted]
        rel_idxes = [x for _, _, x in memwrites_sorted]
        memwrite_queries.append({"pretext": prefix, "index": document, "relations":memwrite, "doc_idx":doc_idx, "entities": entities, "sent_id": prev_len, "rel_idxes": rel_idxes})
        prev_len = (prev_len+index_len)
        index_len = random.randint(1,5) if is_random else 1
    return memwrite_queries


ds = dataset


memwrite_queries = []
random.seed(42)
for doc_idx, element in tqdm(enumerate(ds)):
    # if doc_idx == 2:
    #     print("a")
    x = get_memwrites(element, doc_idx=doc_idx, distant=distant, is_all=is_all, is_random=False, return_entities=False, drop_annotated=False)
    memwrite_queries.extend(x)

fn = f"{FILEPATH}/{subset}" + ".json"
print(fn)
with open(fn, "w") as f:
    json.dump(memwrite_queries, f, indent=2)