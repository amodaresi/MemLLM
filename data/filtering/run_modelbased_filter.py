import os
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import argparse
import random
import numpy as np

PROMPT = """To determine whether the main sentence contains information about the given relation, both the main sentence and the context will be provided. The goal is to identify whether there is evidence of the relation in the main sentence, supported by the context. If there is no relation or the evidence exists solely in the context without requiring the main sentence, respond with No. Otherwise, respond with Yes. Provide reasoning to support your response.

Context: 
Main Sentence: James Michael Osting ( born April 7 , 1977 ) is a former Major League Baseball pitcher .
Relation: ("Osting", "date of birth", "April 7 , 1977")
Evidence: The relation indicates that Osting was born on April 7, 1977. The main sentence explicitly mentions that Osting was born on April 7, 1977. The answer is Yes.


Context: Splashdown is a Hot Tuna album released in 1984 containing the tracks from a previously unreleased live acoustic performance that had been played on the short - lived radio station WQIV in the mid-1970s . During the recording , news of the Apollo - Soyuz mission returning to Earth after the first USA - USSR rendezvous in space reached the station , and the astronauts ' radio transmissions were played at the same time as Jorma and Jack continued with " Police Dog Blues . " The transmissions mixed with the song were preserved for this release as the last track of side 1 .
Main Sentence: The album was Hot Tuna 's first release on Relix Records , and one of the first Relix releases .
Relation: ("Hot Tuna", "country of origin", "USA")
Evidence: The relation indicates that the origin of Hot Tuna is the country of the United States. There is no evidence in the main sentence regarding the country of origin of Hot Tuna. The answer is No.


Context: 
Main sentence: The Chemung Canal Bank Building is located at 415 East Water Street in Elmira , Chemung County , New York , United States .
Relation: ("Chemung County", "capital", "Elmira")
Evidence: The relation indicates that Elmira is the capital of Chemung County. The main sentence only specifies the location of Elmira within Chemung County but does not mention Elmira as the capital of Chemung County. The answer is No.


Context: Carrie Lam Cheng Yuet - ngor , GBM , GBS (; born 13 May 1957 ) is the 4th and current Chief Executive of Hong Kong . Before that she was the Chief Secretary for Administration , the most senior rank of principal officials of Hong Kong , from 2012 to 2017 .
Main sentence: After graduating from the University of Hong Kong , Lam joined the civil service in 1980 and served in various bureaux and departments .
Relation: ("Lam", "educated at", "University of Hong Kong")
Evidence: The relation indicates that Lam received education at the University of Hong Kong. The main sentence mentions that Carrie Lam Cheng Yuet-ngor graduated from the University of Hong Kong. The answer is Yes.


Context: Pacific Fair is a major shopping centre in Broadbeach Waters on the Gold Coast , Queensland , Australia . It was Queensland 's largest regional shopping centre until 2006 . Pacific Fair was developed by Hooker Retail Developments and opened in 1977 on what was swampland with 96 specialty stores and two anchor tenants . Since then , Pacific Fair has undergone numerous expansions and has grown to have more than 300 specialty stores and four anchor tenants . In January 2014 , work began on a major redevelopment project to meet the predicted regional growth on the Gold Coast . Prior to the redevelopment , the shopping centre had four main major stores including a four - level Myer , Kmart , Target , Coles and Toys ' R ' Us . Daimaru operated in the centre before its Australian withdrawal , albeit briefly .
Main Sentence: It also had a 12-screen Birch Carroll and Coyle Cinema ( re - opened as Event Cinemas in late 2015 ) .
Relation: ("Event Cinemas", "country", "Australia")
Evidence: The relation indicates that Event Cinemas is located in the country of Australia. The main sentence mentions that Event Cinemas is part of Pacific Fair which is located in Australia. The answer is Yes.


Context: Benjamin Winslow Harris ( November 10 , 1823 - February 7 , 1907 ) was a nineteenth - century politician , lawyer and judge from Massachusetts . He was the father of Robert Orr Harris . Born in East Bridgewater , Massachusetts , Harris pursued an academic course at Phillips Academy , Andover , graduating in 1847 . He graduated from Dane Law School of Harvard University in 1849 . He was admitted to the bar in Boston , Massachusetts in 1850 , commencing practice in East Bridgewater . He served in the Massachusetts Senate in 1857 , was a member of the Massachusetts House of Representatives in 1858 , was district attorney for the southeastern district of Massachusetts from 1858 to 1866 and was collector of internal revenue for the second district of Massachusetts from 1866 to 1873 . Harris was elected a Republican to the United States House of Representatives in 1872 , serving from 1873 to 1883 , not being a candidate for renomination in 1882 . There , he served as chairman of the Committee on Naval Affairs from 1881 to 1883 . Afterwards , he resumed practicing law in East Bridgewater , Massachusetts and was judge of probate for Plymouth County , Massachusetts from 1887 to 1906 .
Main Sentence: Harris died in East Bridgewater on February 7 , 1907 and was interred in Central Cemetery in East Bridgewater .
Relation: ("Benjamin Winslow Harris", "place of birth", "East Bridgewater")
Evidence: The relation indicates that Benjamin Winslow Harris was born in East Bridgewater. The main sentence lacks information about Benjamin Winslow Harris's place of birth. The evidence for East Bridgewater as his birthplace is exclusively found in the context, not in the main sentence. The answer is No.


Context: Greatest Hats is the first compilation album by the Canadian new wave / synthpop group Men Without Hats , released in 1996 .
Main Sentence: A slightly modified version of the album was released in the US in 1996 , entitled Collection .
Relation: ("Collection", "performer", "Men Without Hats")
Evidence: The relation indicates that Men Without Hats is the performer of the Collection album. The main sentence says that Men Without Hats released slightly modified version of the Greatest Hats album which is the album Collection. The answer is Yes.


Context: Aaron Hobart ( June 26 , 1787 - September 19 , 1858 ) was a U.S. Representative from Massachusetts . Born in Abington , Massachusetts , Hobart pursued classical studies and graduated from Brown University in 1805 . He studied law , was admitted to the bar and commenced practice in Abington . He served as member of the Massachusetts House of Representatives and served in the Massachusetts State Senate . Hobart was elected as a Democratic - Republican to the Sixteenth Congress to fill the vacancy caused by the resignation of Zabdiel Sampson . He was reelected as a Democratic - Republican to the Seventeenth Congress , elected as an Adams - Clay Republican to the Eighteenth Congress , and reelected as an Adams candidate to the Nineteenth Congress , and served from November 24 , 1820 , to March 3 , 1827 . He declined to be a candidate for renomination in 1826 .
Main Sentence: He then served as an Executive councilor 1827 - 1831 and served as probate judge 1843 - 1858 .
Relation: ("Aaron Hobart", "date of death", "1858")
Evidence: The relation indicates that Aaron Hobart passed away in the year 1858. The main sentence does not contain information about the given relation. The evidence of Aaron Hobart's date of death in 1858 is solely present in the context and is not mentioned in the provided main sentence. The answer is No.


Context: {context}
Main Sentence: {index}
Relation: ("{entity1}", "{relation}", "{entity2}")
Evidence:"""

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, default=4)
argparser.add_argument('--shard_size', type=int, default=10, help='Number of documents per shard')
argparser.add_argument('--start_idx', type=int, default=0, help='Start doc_idx:')
argparser.add_argument('--stop_idx', type=int, default=-1, help='Stop doc_idx:')
argparser.add_argument('--max_length', type=int, default=384, help='Maximum number of tokens to generate')
argparser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset')
argparser.add_argument('--output_dir', type=str, default='', help='Directory to save the output files')
argparser.add_argument('--cache_dir', type=str, default='', help='Cache directory for the model')
argparser.add_argument('--overwrite', action='store_true', help='Overwrite the json file if it exists. If you have changed the shard size, you should change this True.', default=False)
args = argparser.parse_args()

print("Overwrite:", args.overwrite)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(args.dataset_path, "r") as f:
    training_examples = json.load(f)

print("Reformatting training examples...")
target_examples = []
prev_doc_id = -1
sentence_idx = 0
for example in tqdm(training_examples):
    if prev_doc_id != example["doc_idx"]:
        sentence_idx = 0
    else:
        sentence_idx += 1
    prev_doc_id = example["doc_idx"]
    for rel_idx, relation in enumerate(example["relations"]):
        target_examples.append({"pretext": example["pretext"], "index": example["index"], "relation": relation, "doc_idx": example["doc_idx"], "sentence_idx": sentence_idx, "rel_idx": example["rel_idxes"][rel_idx]})

print("Number of target examples:", len(target_examples))

print("Sharding with size", args.shard_size, "...")
shards = []
doc_counter = 0
prev_doc = -1
cur_shard = []
doc_range = np.arange(target_examples[-1]["doc_idx"] + 1).tolist()
doc_range = doc_range[args.start_idx:(args.stop_idx if args.stop_idx > 0 else len(doc_range))]
print("Running from doc id:", min(doc_range), max(doc_range))

for el in tqdm(target_examples):
    if el["doc_idx"] >= doc_range[0] and el["doc_idx"] <= doc_range[-1]:
        if el["doc_idx"] != prev_doc:
            if doc_counter == args.shard_size:
                shards.append(cur_shard)
                cur_shard = []
                doc_counter = 1
            else:
                doc_counter += 1
        cur_shard.append(el)
        prev_doc = el["doc_idx"]
if len(cur_shard) > 0:
    shards.append(cur_shard)


shard_prompts = []
for shard in shards:
    cur_shard_prompts = []
    for example in shard:
        cur_shard_prompts.append(PROMPT.format(context=example["pretext"], index=example["index"], entity1=example["relation"][0], relation=example["relation"][1], entity2=example["relation"][2]).strip())
    shard_prompts.append(cur_shard_prompts)

print("Number of shards:", len(shards))
print("Average number of requests per shard:", sum([len(shard) for shard in shard_prompts])/len(shard_prompts))


sampling_params = SamplingParams(temperature=0, max_tokens=args.max_length, stop=["\n\n\n"], include_stop_str_in_output=True)

llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", download_dir=args.cache_dir, tensor_parallel_size=args.num_gpus, gpu_memory_utilization=0.90)


for shard_idx, shard_prompt in tqdm(enumerate(shard_prompts), total=len(shard_prompts)):
    if os.path.exists(os.path.join(args.output_dir, f"output_doc_{min(doc_range)}_{max(doc_range)}_"+str(shard_idx)+".json")) and not args.overwrite:
        print("Skipping shard", shard_idx)
        continue
    outputs = llm.generate(shard_prompt, sampling_params)
    results = []
    for o, example in zip(outputs, shards[shard_idx]):
        temp = vars(o.outputs[0])
        del temp["token_ids"]
        temp = {**temp, **example}
        temp["prompt"] = o.prompt
        results.append(temp)
    fn = os.path.join(args.output_dir, f"output_doc_{min(doc_range)}_{max(doc_range)}_"+str(shard_idx)+".json")
    with open(fn, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)