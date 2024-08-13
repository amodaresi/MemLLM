import json, os
from copy import copy

LOC = "MEMORY_WRITE_RAWOUTPUTS.json"

with open(LOC, "r") as f:
    ds = json.load(f)

gathered_data = []
prev_doc_idx = -1

def filter_(d):
    if "sex" in d["original_prompt"] or "gender" in d["original_prompt"]:
        return False
    return True

for _d in ds:
    if _d["data_idx"] != prev_doc_idx:
        if prev_doc_idx != -1:
            del example["sent_id"]
            del example["index"]
            triples_set = set()
            triples = []

            raw_generation = example["raw_generation"]
            if len(raw_generation) > 0:
                triples_raw = raw_generation.split("--> ")[-1].split("})")[0]
            else:
                triples_raw = ""

            for x in triples_raw.split(";"):
                if len(x.split(">>")) == 3:
                    relation = x.split(">>")
                    if relation[0] == "" or relation[1] == "" or relation[2] == "":
                        continue
                    if x not in triples_set:
                        triples.append(relation)
                        triples_set.add(x)
                else:
                    pass

            example["generated_relations"] = triples
            if filter_(example):
                gathered_data.append(example)
        prev_doc_idx = _d["data_idx"]
        
    example = copy(_d)

with open(os.path.join(os.path.dirname(LOC), "outputs.json"), "w") as f:
    json.dump(gathered_data, f, indent=2)