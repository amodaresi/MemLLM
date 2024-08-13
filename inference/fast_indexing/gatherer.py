import os, json
from tqdm.auto import tqdm

PATH = "MEMORY_WRITE_OUTPUT_DIR/redocred/test/"

triples_set = set()
triples = []
agg_shard_count = 0
agg_idx = 0
SHARD_COUNT = 10000000
MIN_INDEX = 0
MAX_INDEX = 7000000
for filepath in tqdm(sorted([x for x in os.listdir(PATH) if x.startswith("shard")], key=lambda x: int(x.split("_")[-2]))):
    starting_point = int(filepath.split("_")[-2])
    if starting_point >= MAX_INDEX or starting_point < MIN_INDEX:
        continue
    with open(PATH + filepath, "r") as f:
        examples = json.load(f)
        for example in examples:
            raw_generation = example["raw_generation"]
            if len(raw_generation) > 0:
                triples_raw = raw_generation.split("--> ")[-1].split("})")[0]
            else:
                triples_raw = ""

            for x in triples_raw.split(";"):
                if len(x.split(">>")) == 3:
                    relation = x.split(">>")
                    if x not in triples_set:
                        triples.append(relation)
                        triples_set.add(x)
                else:
                    pass
    
    agg_shard_count += 1
    if agg_shard_count == SHARD_COUNT:
        agg_shard_count = 0
        print(len(triples))
        with open(os.path.join(PATH, f"all_relations_{agg_idx}.json"), "w") as f:
            json.dump(triples, f, indent=2)
        triples_set = set()
        triples = []
        agg_idx += 1

print(len(triples))
with open(os.path.join(PATH, f"all_relations_{agg_idx}.json"), "w") as f:
    json.dump(triples, f, indent=2)
