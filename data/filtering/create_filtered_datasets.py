import os, re, json
from tqdm.auto import tqdm
from datasets import load_dataset
from copy import deepcopy

redocred = load_dataset("tonytan48/Re-DocRED", keep_in_memory=True)

split = "train"
filtering_results_dir = f"[PATH_DIR]"

with open(os.path.join(filtering_results_dir, f"{split}.json"), "r") as f:
    filtering_input = json.load(f)

# Gather filtering results
all_filtered_results = []
json_files = os.listdir(os.path.join(filtering_results_dir, f"filtered_{split}"))
json_files = sorted(json_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))
for file in tqdm(json_files):
    with open(os.path.join(filtering_results_dir, f"filtered_{split}", file), "r") as f:
        data = json.load(f)
    for d in data:
        all_filtered_results.append({
            "result": "The answer is Yes" in d["text"],
            "relation": d["relation"],
            "doc_idx": d["doc_idx"],
            "sentence_idx": d["sentence_idx"]
        })

filtering_results_structured = {}
filtering_results_structured_possible = {}
i = 0
for _input in filtering_input:
    if _input["doc_idx"] not in filtering_results_structured:
        filtering_results_structured[_input["doc_idx"]] = {}
        filtering_results_structured_possible[_input["doc_idx"]] = {}
    for j, r in enumerate(_input["relations"]):
        rel_idx = _input["rel_idxes"][j]
        if rel_idx not in filtering_results_structured[_input["doc_idx"]]:
            filtering_results_structured[_input["doc_idx"]][rel_idx] = []
            filtering_results_structured_possible[_input["doc_idx"]][rel_idx] = []
        assert _input["doc_idx"] == all_filtered_results[i]["doc_idx"]
        assert r == all_filtered_results[i]["relation"]
        if all_filtered_results[i]["result"]:
            filtering_results_structured[_input["doc_idx"]][rel_idx].append(_input["sent_id"])
        filtering_results_structured_possible[_input["doc_idx"]][rel_idx].append(_input["sent_id"])
        i += 1
    
filtering_results_structured_selected = {}
for _input in filtering_input:
    if _input["doc_idx"] not in filtering_results_structured_selected:
        filtering_results_structured_selected[_input["doc_idx"]] = {}
    for j, r in enumerate(_input["relations"]):
        rel_idx = _input["rel_idxes"][j]
        filtering_results_structured_selected[_input["doc_idx"]][rel_idx] = [max(filtering_results_structured_possible[_input["doc_idx"]][rel_idx])] if len(filtering_results_structured[_input["doc_idx"]][rel_idx]) == 0 else filtering_results_structured[_input["doc_idx"]][rel_idx]
        assert type(filtering_results_structured_selected[_input["doc_idx"]][rel_idx]) is list

new_dataset = []
for i in range(len(redocred[split])):
    example = redocred[split][i]
    new_example = deepcopy(example)
    for rel_idx, relation in enumerate(new_example["labels"]):
        evidence = relation["evidence"]
        if len(evidence) > 0:
            continue
        else:
            if rel_idx in filtering_results_structured_selected[i]:
                new_example["labels"][rel_idx]["evidence"] = filtering_results_structured_selected[i][rel_idx]
    new_dataset.append(new_example)

with open(os.path.join(filtering_results_dir, f"{split}_filtered.json"), "w") as f:
    json.dump(new_dataset, f, indent=2)


