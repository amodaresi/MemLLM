import numpy as np
import json
from tqdm.auto import tqdm
from datasets import load_dataset
from mr_data_config import *

split = "test"

FILTERED_PATH = f"FILTERED_DATA_DIR/{split}_filtered_entFix.json"
SAVE_PATH = f"FILTERED_DATA_DIR_PROCESSED/{split}_eval.withOther.json"

ds = load_dataset("json", data_files=FILTERED_PATH, keep_in_memory=True)["train"]

def convert_pos(v_dict, sent_lengths, idx):
    shift = sent_lengths[:v_dict["sent_id"]].sum()
    text_pos = [int(v_dict["pos"][0] + shift), int(v_dict["pos"][1] + shift)]
    v_dict["text_pos"] = text_pos
    v_dict["idx"] = idx
    return v_dict

with open("utils/wikidata_props_en.json", "r") as f:
    rel_type_map = json.load(f)

all_queries = []
entities_per_doc = {}
for data_idx in tqdm(range(len(ds))):
    relation_labels = ds[data_idx]['labels']
    entities_per_doc[data_idx] = []
    
    sents = ds[data_idx]["sents"]
    full_text = [s for sent in sents for s in sent]
    sent_lengths = np.array([len(s) for s in sents])
    vertexSet = [convert_pos(v, sent_lengths, idx) for idx, vers in enumerate(ds[data_idx]['vertexSet']) for v in vers]
    vertexSet = sorted(vertexSet, key=lambda x: (x['text_pos'][0], x["idx"]))
    
    relation_queries = []
    covered_relations = np.zeros(len(relation_labels))
    seen_entities = {}

    for ent_id, entity in enumerate(vertexSet):
        if entity["idx"] not in seen_entities:
            seen_entities[entity["idx"]] = []
        seen_entities[entity["idx"]].append(ent_id)
        entities_per_doc[data_idx].append(entity["text_pos"])
        for rel_idx in range(len(covered_relations)):
            if covered_relations[rel_idx] == 0:
                if relation_labels[rel_idx]["h"] in seen_entities and relation_labels[rel_idx]["t"] in seen_entities:
                    if relation_labels[rel_idx]["h"] == entity["idx"] or relation_labels[rel_idx]["t"] == entity["idx"]:
                        covered_relations[rel_idx] = 1
                        head_idx = relation_labels[rel_idx]["h"]
                        tail_idx = relation_labels[rel_idx]["t"]
                        if head_idx == tail_idx:
                            continue
                        relation_text = rel_type_map[relation_labels[rel_idx]["r"]]
                        target_type = "HEAD" if entity["idx"] == head_idx else "TAIL"
                        target_vertex = entity
                        to_add_relation_queries = []
                        filtered = False
                        for vertex_id in seen_entities[head_idx if target_type == "TAIL" else tail_idx]:
                            vertex = vertexSet[vertex_id]
                            to_add_relation_queries.append({
                                "data_idx": data_idx,
                                "head": vertex["name"] if target_type == "TAIL" else target_vertex["name"],
                                "head_text_pos": vertex["text_pos"] if target_type == "TAIL" else target_vertex["text_pos"],
                                "relation_type": relation_text,
                                "tail": vertex["name"] if target_type == "HEAD" else target_vertex["name"],
                                "tail_text_pos": vertex["text_pos"] if target_type == "HEAD" else target_vertex["text_pos"],
                                "target_pos": entity["pos"],
                                "target_text_pos": [int(entity["text_pos"][0]), int(entity["text_pos"][1])],
                                "target_sent": entity["sent_id"],
                                "target_type": target_type
                            })
                            head_text = " ".join(full_text[to_add_relation_queries[-1]["head_text_pos"][0]:to_add_relation_queries[-1]["head_text_pos"][1]])
                            tail_text = " ".join(full_text[to_add_relation_queries[-1]["tail_text_pos"][0]:to_add_relation_queries[-1]["tail_text_pos"][1]])
                            to_add_relation_queries[-1]["head"] = head_text
                            to_add_relation_queries[-1]["tail"] = tail_text

                        if not filtered:
                            relation_queries.extend(to_add_relation_queries)

    relation_queries_strs = set()
    for q in relation_queries:
        q_str = json.dumps(q)
        if q_str not in relation_queries_strs:
            relation_queries_strs.add(q_str)
            all_queries.append(q)
            
print(len(all_queries))

all_queries = sorted(all_queries, key=lambda x: (x['data_idx'], x['target_pos'][0], x['head'] if x['target_type'] != "HEAD" else x['tail']))

# merge queries
all_queries_merged = []
offset = 0
for q in all_queries:
    new_q = {
        "data_idx": q["data_idx"],
        'target_pos': q['target_pos'],
        'target_text_pos': q['target_text_pos'],
        'target_sent': q['target_sent'],
        'target_text': q['head'] if q['target_type'] == "HEAD" else q['tail'],
        'target_types': [q['target_type']],
        'relation_type': [q['relation_type']],
        'source_entities': [q['head'] if q['target_type'] == "TAIL" else q['tail']],
        'source_entities_text_pos': [q['head_text_pos'] if q['target_type'] == "TAIL" else q['tail_text_pos']],
    }
    if len(all_queries_merged) == 0:
        all_queries_merged.append(new_q)
    else:
        offset = len(all_queries_merged) - 1
        merged = False
        while offset >= 0 and all_queries_merged[offset]["data_idx"] == q["data_idx"]:
            if all_queries_merged[offset]["target_pos"] == q["target_pos"] and all_queries_merged[offset]["target_sent"] == q["target_sent"]:
                merged = True
                if all_queries_merged[offset]['source_entities'][-1] == new_q["source_entities"][0] and new_q["relation_type"][0] == all_queries_merged[offset]['relation_type'][-1]:
                    if all_queries_merged[offset]['source_entities_text_pos'][-1][0] < new_q["source_entities_text_pos"][-1][0]:
                        all_queries_merged[offset]['source_entities_text_pos'][-1] = new_q["source_entities_text_pos"][-1]
                else:       
                    all_queries_merged[offset]['source_entities'].extend(new_q["source_entities"])
                    all_queries_merged[offset]['source_entities_text_pos'].extend(new_q["source_entities_text_pos"])
                    all_queries_merged[offset]['relation_type'].extend(new_q["relation_type"])
                    all_queries_merged[offset]['target_types'].extend(new_q["target_types"])
            offset -= 1
        if not merged:
            all_queries_merged.append(new_q)

all_queries_merged_sorted = sorted(all_queries_merged, key=lambda x: (x['data_idx'], x['target_sent'], x['target_pos'][0]))


query_results = []
full_raw_queries = []
for q_idx in tqdm(range(len(all_queries_merged_sorted))):
    q = all_queries_merged_sorted[q_idx]
    data_idx = q["data_idx"]
    query_results.append([])
    full_raw_queries.append([])

    for h_id, q_h in enumerate(q["source_entities"]):
        raw_query = (q_h, q["relation_type"][h_id], None) if q["target_types"][h_id] == "TAIL" else (None, q["relation_type"][h_id], q_h)
        query_res = []
        query_results[-1].append(query_res)
        full_raw_queries[-1].append(raw_query)

data_idxes = np.array([q["data_idx"] for q in all_queries_merged_sorted])
query_results_maxes = np.array([max([len(q_r) for q_r in q_res]) for q_res in query_results])

included_queries = []
included_queries_full = []
priorities = []
# included_targets = []
has_query = []
for q_idx in tqdm(range(len(all_queries_merged_sorted))):
    q = all_queries_merged_sorted[q_idx]
    data_idx = q["data_idx"]
    # target_type = q["target_type"].lower()
    # query_entity_type = "head" if target_type == "tail" else "tail"
    sentences = ds[data_idx]['sents']
    # full_text = " ".join([" ".join(sent) for sent in sentences])
    full_text_tokenized = [w for sent in sentences for w in sent]

    included_queries.append([])
    included_queries_full.append([])
    priorities.append([])
    # included_targets.append(None)
    has_query.append(0)

    raw_query_translator = {json.dumps(full_raw_queries[q_idx][h_id]): query_results[q_idx][h_id] for h_id in range(len(full_raw_queries[q_idx]))}

    pretext = " ".join(full_text_tokenized[:q["target_text_pos"][0]])

    if q["target_text"] not in pretext:
        for h_id, q_h in enumerate(q["source_entities"]):
            # query_res = query_results[q_idx][h_id]
            raw_query = (q_h, q["relation_type"][h_id], None) if q["target_types"][h_id] == "TAIL" else (None, q["relation_type"][h_id], q_h)
            query_res = raw_query_translator[json.dumps(raw_query)]
            raw_query_masked = ('X', q["relation_type"][h_id], None) if q["target_types"][h_id] == "TAIL" else (None, q["relation_type"][h_id], 'X')
            # if len(query_res) > 0 and q[target_type][0] in query_res:
            if raw_query_masked not in SKIP_QUERIES:
                if len(query_res) <= MAX_QUERY_RES_LEN:
                    included_queries[-1].append(h_id)
                    included_queries_full[-1].append(json.dumps(raw_query))
                    priorities[-1].append(len(query_res) if len(query_res) > 0 else 5)
                    has_query[-1] = 1

print(np.sum(has_query))

included_queries_sorted = []
for l_idx, l in enumerate(included_queries):
    included_queries_sorted.append([x for _,x in sorted(zip(priorities[l_idx],l))])

included_queries_full_sorted = []
for l_idx, l in enumerate(included_queries_full):
    included_queries_full_sorted.append([x for _,x in sorted(zip(priorities[l_idx],l))])

has_query_per_idx = {}
has_query_per_idx_arg = {}
for q_idx in tqdm(range(len(all_queries_merged_sorted))):
    q = all_queries_merged_sorted[q_idx]
    data_idx = q["data_idx"]
    if data_idx not in has_query_per_idx:
        has_query_per_idx[data_idx] = []
        has_query_per_idx_arg[data_idx] = []

    if has_query[q_idx] == 1:
        if len(has_query_per_idx_arg[data_idx]) > 0:
            if included_queries_full_sorted[has_query_per_idx_arg[data_idx][-1]][0] != included_queries_full_sorted[q_idx][0]:
                has_query_per_idx_arg[data_idx].append(q_idx)
        else:
            has_query_per_idx_arg[data_idx].append(q_idx)

    has_query_per_idx[data_idx].append(has_query[q_idx])

for k in has_query_per_idx.keys():
    has_query_per_idx[k] = np.array(has_query_per_idx[k])
    has_query_per_idx_arg[k] = np.array(has_query_per_idx_arg[k])

np.random.seed(42)
data_mem = []
prev_written_data_idx = -1
for q_idx in tqdm(range(len(all_queries_merged_sorted))):
    q = all_queries_merged_sorted[q_idx]
    if has_query[q_idx] == 0:
        continue
    
    if q_idx + 1 < len(all_queries_merged_sorted):
        next_q = all_queries_merged_sorted[q_idx + 1]
    else:
        next_q = None
    data_idx = q["data_idx"]

    sentences = ds[data_idx]["sents"]
    # full_text = " ".join([" ".join(sent) for sent in sentences])
    full_text_tokenized = [w for sent in sentences for w in sent]

    read_point = q["target_text_pos"][0]

    pre_text = " ".join(full_text_tokenized[:read_point])
    if prev_written_data_idx != data_idx:
        preapi_text = pre_text
        pre_text = ""
    else:
        preapi_text = ""

    if np.sum(has_query_per_idx_arg[data_idx] > q_idx):
        next_qs = has_query_per_idx_arg[data_idx][has_query_per_idx_arg[data_idx] > q_idx]
        next_q = all_queries_merged_sorted[int(next_qs[0])]
        next_point = next_q["target_text_pos"][0]
    else:
        next_point = 10000000000000

    continuation = " ".join(full_text_tokenized[read_point:next_point])
    eot = next_point == 10000000000000
    
    all_query_res = []
    all_queries = []

    api_count = 0
    for h_id, q_h in enumerate(q["source_entities"]):
        if h_id in included_queries_sorted[q_idx]:
            query_res = query_results[q_idx][h_id]
            if len(query_res) < MAX_QUERY_RES_LEN and api_count < MAX_QUERIES:
                all_query_res.append(query_res)
                
                api_count += 1
                if q["target_types"][h_id] == "TAIL":
                    all_queries.append((q_h, q["relation_type"][h_id], None))
                else:
                    all_queries.append((None, q["relation_type"][h_id], q_h))

    if api_count > 0:
        if len([x for res in all_query_res for x in res]) == 0:
            all_query_res = [[q["target_text"]] for _ in all_query_res]
        data_mem.append({
                    "data_idx": data_idx,
                    "pre_text": pre_text,
                    "preapi_text": preapi_text,
                    "queries": all_queries,
                    "target_result": q["target_text"],
                    "continuation": continuation,
                    "full_text_tokenized": full_text_tokenized if eot else [],
                    "entities_loc": entities_per_doc[data_idx] if eot else [],
                    "eot": eot
                })
        prev_written_data_idx = data_idx

i = 1
data_mem[0]["target_result"] = [data_mem[0]["target_result"]]
while i < len(data_mem):
    data_mem[i]["target_result"] = [data_mem[i]["target_result"]]
    if data_mem[i]["continuation"] in data_mem[i-1]["continuation"] and data_mem[i-1]["data_idx"] == data_mem[i]["data_idx"] and len(data_mem[i]["continuation"]) > 0:
        data_mem[i-1]["target_result"].extend(data_mem[i]["target_result"])
        del data_mem[i]
    else:
        i += 1

with open(SAVE_PATH, "w") as f:
    json.dump(data_mem, f, indent=2)