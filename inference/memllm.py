import os, re
import nltk
import torch
import tempfile
import numpy as np
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel, LogitsProcessorList, LogitsProcessor, StoppingCriteria, StoppingCriteriaList

class LateStoppingProcessor(LogitsProcessor):
    def __init__(self, stopping_delay: int, stopping_token: int):
        self.stopping_delay = stopping_delay
        self.stopping_token = stopping_token

        self.cutting_points = {}
        self.cumlogprob = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_tokens = torch.argsort(scores[0], dim=-1, descending=True)[:2]
        logsoftscores = torch.nn.functional.log_softmax(scores, dim=-1)[0]
        if top_tokens[0] == self.stopping_token:
            cutting_points_cumlogprobs = list(self.cutting_points.values())

            best_cumlogprob_arg = 0
            if len(cutting_points_cumlogprobs) > 0:
                best_cumlogprob_arg = int(np.argmax(cutting_points_cumlogprobs))

            if len(self.cutting_points) < self.stopping_delay or best_cumlogprob_arg > len(cutting_points_cumlogprobs) - self.stopping_delay:
                self.cutting_points[len(input_ids[0])] = ((self.cumlogprob + logsoftscores[self.stopping_token]) / len(input_ids[0])).cpu().numpy()
                self.cumlogprob += logsoftscores[top_tokens[1]].cpu().numpy()
                scores[0, self.stopping_token] = -float("inf")
        else:
            self.cumlogprob += logsoftscores[top_tokens[0]].cpu().numpy()
        
        return scores
    

class ReadStopper(StoppingCriteria):
    def __init__(self, stop_token: int) -> None:
        self.stop_token = stop_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return input_ids[:, -1] == self.stop_token


class MEMLLM:
    def __init__(
        self,
        mr_model: PreTrainedModel,
        mw_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        memory_controller,
        memory_write_generation_config: GenerationConfig = None,
        memory_read_api_generation_config: GenerationConfig = None,
        memory_read_text_generation_config: GenerationConfig = None,
        memory_write_late_stopping: int = 5,
        special_tokens_map: dict = {"START": "({", "WAIT_FOR_RES": "-->", "END": "})"}
    ):
        self.mr_model = mr_model
        self.mw_model = mw_model
        self.tokenizer = tokenizer
        self.memory_controller = memory_controller
        self.memory_write_generation_config = memory_write_generation_config
        self.memory_write_late_stopping = memory_write_late_stopping

        self.memory_read_api_generation_config = memory_read_api_generation_config
        self.memory_read_text_generation_config = memory_read_text_generation_config
        self.special_tokens_map = special_tokens_map
        self.special_tokens_map_enc = {k: self.tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens_map.items()}

        self.max_length = tokenizer.model_max_length

        if memory_read_api_generation_config:
            self.max_length = min(self.max_length, memory_read_api_generation_config.max_length)

        if memory_read_text_generation_config:
            self.max_length = min(self.max_length, memory_read_text_generation_config.max_length)

        if memory_write_generation_config:
            self.max_length = min(self.max_length, memory_write_generation_config.max_length)

    def memwr_text_preprocessor(self, text: str):
        output = " ".join(nltk.word_tokenize(text))
        pattern = r'(?<=\d)\s*[-–—]\s*(?=\d)'
        output = re.sub(pattern, ' – ', output)
        return output
    
    def batched_memwr_preprocess_function(self, examples):
        max_prompt_length = int(self.tokenizer.model_max_length / 2)
        full_texts = [example["pretext"] + "({USER_ST})" + example["index"] + "({USER_END})" for example in examples]
        return self.tokenizer.batch_encode_plus(full_texts, max_length=max_prompt_length)["input_ids"]
    
    
    _late_stopgenerate_cache = {}
    def hf_late_stopgenerate(self, prompt_token_ids, cache=False):
        final_outputs = []
        for prompt_token_id in prompt_token_ids:
            prompt_token_id_strd = ','.join(map(str, prompt_token_id))
            if cache and prompt_token_id_strd in self._late_stopgenerate_cache:
                final_outputs.append(self._late_stopgenerate_cache[prompt_token_id_strd])
            else:
                lateStopper = LateStoppingProcessor(stopping_delay=self.memory_write_late_stopping, stopping_token=self.tokenizer.convert_tokens_to_ids(self.special_tokens_map["END"]))
                outputs = self.mw_model.generate(
                    input_ids=torch.tensor([prompt_token_id], device=self.mw_model.device), 
                    generation_config=self.memory_write_generation_config,
                    logits_processor=LogitsProcessorList([
                        lateStopper
                    ]))
                try:
                    cuttingpoints = list(lateStopper.cutting_points.keys())
                    cuttingpoint_vals = list(lateStopper.cutting_points.values())
                    if len(cuttingpoint_vals) != 0:
                        _argmax = cuttingpoints[int(np.argmax(cuttingpoint_vals))]
                        first_text = self.tokenizer.decode(outputs[0].cpu().numpy()[1:_argmax]) + "})"
                    else:
                        first_text = self.tokenizer.decode(outputs[0].cpu().numpy()[1:-1])

                    for token in self.tokenizer.all_special_tokens:
                        first_text = first_text.replace(f"{token} ", token)
                    final_outputs.append(first_text)
                except:
                    final_outputs.append("")
                if cache:
                    self._late_stopgenerate_cache[prompt_token_id_strd] = final_outputs[-1]
        return final_outputs


    def batch_memory_write(self, texts, output_raw=False):
        max_sents = int(self.tokenizer.model_max_length / 20 / 2)
        shard_data = []
        inverse_idx = []
        for text_id, text in enumerate(texts):
            sent_text = [self.memwr_text_preprocessor(t) for t in nltk.sent_tokenize(text)]
            for sent_id in range(len(sent_text)):
                example = {}
                example["pretext"] = " ".join(sent_text[max(0, sent_id-max_sents):sent_id])
                example["index"] = sent_text[sent_id]
                example["sent_id"] = sent_id
                shard_data.append(example)
                inverse_idx.append(text_id)

        prompt_token_ids = self.batched_memwr_preprocess_function(shard_data)
        final_outputs = self.hf_late_stopgenerate(prompt_token_ids)

        doc_relations = [set() for _ in texts]
        if output_raw:
            doc_generated_outputs = [[] for _ in texts]
        for idx, raw_generation in enumerate(final_outputs):
            if len(raw_generation) > 0:
                # triplets_raw = raw_generation[1:]
                triplets_raw = raw_generation.split("-->")[-1].split("})")[0]
            else:
                triplets_raw = ""

            triplets = []
            for x in triplets_raw.split(";"):
                if len(x.split(">>")) == 3:
                    triplets.append(tuple(x.split(">>")))
                else:
                    pass
            doc_relations[inverse_idx[idx]].update(triplets)
            if output_raw:
                doc_generated_outputs[inverse_idx[idx]].append(raw_generation)

            if len(doc_relations[inverse_idx[idx]]) > 0:
                self.memory_controller.store_relationship_batched_v2(list(doc_relations[inverse_idx[idx]]), verbose=False)
        if output_raw:
            return doc_relations, doc_generated_outputs
        return doc_relations


    def lazy_memory_write(self, text):
        max_sents = int(self.tokenizer.model_max_length / 20 / 2)
        shard_data = []
        sent_text = [self.memwr_text_preprocessor(t) for t in nltk.sent_tokenize(text)]
        for sent_id in range(len(sent_text)):
            example = {}
            example["pretext"] = " ".join(sent_text[max(0, sent_id-max_sents):sent_id])
            example["index"] = sent_text[sent_id]
            example["sent_id"] = sent_id
            shard_data.append(example)

        prompt_token_ids = self.batched_memwr_preprocess_function(shard_data)
        for i, prompt_token_id in enumerate(prompt_token_ids):
            raw_generation = self.hf_late_stopgenerate([prompt_token_id])[0]
            if len(raw_generation) > 0:
                # triplets_raw = raw_generation[1:]
                triplets_raw = raw_generation.split("-->")[-1].split("})")[0]
            else:
                triplets_raw = ""

            triplets = []
            for x in triplets_raw.split(";"):
                if len(x.split(">>")) == 3:
                    triplets.append(tuple(x.split(">>")))
                else:
                    pass
            if len(triplets) > 0:
                self.memory_controller.store_relationship_batched_v2(triplets, verbose=False)
            yield triplets, raw_generation, i == (len(prompt_token_ids) - 1)


    _mem_read_cache = {}
    _mem_read_cache_w_skip = {}
    _mem_read_api_cache = {}
    def lazy_memory_read(self, text, REL_SIM=0.1, cache=False, stop_after_first_success=False, filtered_relationships=[]):
        prev_query_raw = None
        prev_queries_results = None

        START_TOKEN = self.tokenizer.convert_tokens_to_ids(self.special_tokens_map["START"])
        WAIT_FOR_RES_TOKEN = self.tokenizer.convert_tokens_to_ids(self.special_tokens_map["WAIT_FOR_RES"])
        END_TOKEN = self.tokenizer.convert_tokens_to_ids(self.special_tokens_map["END"])

        prefix = self.tokenizer.encode_plus(text)["input_ids"]
        _input = prefix
        skip = False
        mr_start_pos = None
        mr_end_pos = None
        got_first_success = False
        initial_length = len(_input)
        while len(_input) - initial_length < self.memory_read_text_generation_config.max_new_tokens:
            _input_strd = ','.join(map(str, _input))
            if cache and skip and _input_strd in self._mem_read_cache_w_skip:
                output_tokens_pre_api = self._mem_read_cache_w_skip[_input_strd]
            elif cache and not skip and _input_strd in self._mem_read_cache:
                output_tokens_pre_api = self._mem_read_cache[_input_strd]
            else:
                if _input[-1] != START_TOKEN:
                    output_tokens_pre_api = self.mr_model.generate(
                                input_ids=torch.tensor([_input], device=self.mr_model.device), 
                                generation_config=self.memory_read_text_generation_config,
                                stopping_criteria=StoppingCriteriaList([
                                    ReadStopper(stop_token=START_TOKEN)
                                ]),
                                begin_suppress_tokens=[START_TOKEN] if skip else [])[0].cpu().numpy().tolist()
                else:
                    output_tokens_pre_api = _input
                if cache:
                    if skip:
                        self._mem_read_cache_w_skip[_input_strd] = output_tokens_pre_api
                    else:
                        self._mem_read_cache[_input_strd] = output_tokens_pre_api
            # if mr_start_pos:
            #     output_tokens_pre_api = output_tokens_pre_api[:mr_start_pos] + output_tokens_pre_api[mr_end_pos:]
            skip = False
            if got_first_success and stop_after_first_success:
                yield output_tokens_pre_api, True
                break
            yield output_tokens_pre_api, False
            if "MEM_READ" in text:
                mr_start_pos = int(np.argwhere(np.array(output_tokens_pre_api) == START_TOKEN).flatten()[0])
                mr_end_pos = int(np.argwhere(np.array(output_tokens_pre_api) == END_TOKEN).flatten()[0]) + 1

            if START_TOKEN == output_tokens_pre_api[-1]:
                _input_for_api_generation = output_tokens_pre_api[:mr_start_pos] + output_tokens_pre_api[mr_end_pos:] if mr_start_pos else output_tokens_pre_api
                _input_for_api_generation_strd = ','.join(map(str, _input_for_api_generation))
                if cache and _input_for_api_generation_strd in self._mem_read_api_cache:
                    output_tokens = self._mem_read_api_cache[_input_for_api_generation_strd]
                else:
                    output_tokens = self.mr_model.generate(
                            input_ids=torch.tensor([_input_for_api_generation], device=self.mr_model.device), 
                            generation_config=self.memory_read_api_generation_config,
                            stopping_criteria=StoppingCriteriaList([
                                ReadStopper(stop_token=WAIT_FOR_RES_TOKEN)
                            ]))[0].cpu().numpy().tolist()
                    if cache:
                        self._mem_read_api_cache[_input_for_api_generation_strd] = output_tokens
                yield output_tokens, False
                start_of_api = int(np.argwhere(np.array(output_tokens) == START_TOKEN).flatten()[0])
                api_gen_output = self.tokenizer.decode(output_tokens[start_of_api:])
                queries_raw = api_gen_output[12:-4]
                try:
                    queries_raw = queries_raw.split(";")
                    queries = []
                    for query_raw in queries_raw:
                        query = query_raw.split(">>")
                        if query_raw[:2] == ">>":
                            query[0] = None
                        else:
                            query[2] = None
                        if len(query) != 3 or (query[0] is not None and query[2] is not None):
                            pass
                        else:
                            queries.append(query)
                except:
                    _input = output_tokens_pre_api[:-1]
                    # mr_start_pos = None
                    # mr_end_pos = None
                    skip = True
                    continue
                target_results = []
                if queries_raw != prev_query_raw:
                    for query in queries:
                        query_result, too_broad = self.memory_controller.query_relationship(
                            query,
                            q_thr=REL_SIM,
                            filtered_relationships=filtered_relationships
                        )
                        target_results.extend([q["relationship"][0] if query[0] is None else q["relationship"][2] for q in query_result])
                target_results = list(set(target_results))[:30]
                
                if len(target_results) == 0 or queries_raw == prev_query_raw:
                    # yield output_tokens + [END_TOKEN], False
                    _input = output_tokens_pre_api[:-1]
                    yield _input, False
                    # mr_start_pos = None
                    # mr_end_pos = None
                    skip = True
                    continue
                else:
                    query_result_encoded = self.tokenizer(";".join(target_results))["input_ids"][1:]
                    if mr_start_pos:
                        mr_start_pos = len(output_tokens_pre_api[:mr_start_pos] + output_tokens_pre_api[mr_end_pos:]) - 1
                    else:
                        mr_start_pos = len(output_tokens_pre_api) - 1
                    _input = output_tokens + query_result_encoded + [END_TOKEN]
                    mr_end_pos = len(_input)
                    prev_query_raw = queries_raw
                    yield _input, False
                    got_first_success = True
            else:
                break
        yield output_tokens_pre_api, True
    

    def compute_loss_segment(self, inputs, continued_segment, disable_api_start=False, disable_api_start_first=False):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        full_inputs = {"input_ids": inputs + continued_segment}
        full_inputs["labels"] = [-100] * len(inputs) + continued_segment
        full_inputs = {k: torch.tensor([v], dtype=torch.long, device=self.mr_model.device) for k,v in full_inputs.items()}
        with torch.no_grad():
            outputs = self.mr_model(input_ids=full_inputs["input_ids"])
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if disable_api_start:
                logits[0, :, self.special_tokens_map_enc["START"]] = -10000
            if disable_api_start_first:
                logits[0, len(inputs) - 1, self.special_tokens_map_enc["START"]] = -10000

            labels = full_inputs["labels"].to(logits.device)
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            return loss_fn(logits.view((-1, logits.size(-1))), labels.view(-1)).view(-1, labels.size(-1))[0].cpu().float().numpy().tolist()[max(len(inputs)-1, 0):-1]
    
    def get_target_length(self, continuation_encoded, target_encoded):
        tar_len = len(target_encoded)
        if continuation_encoded[:tar_len] == target_encoded:
            return tar_len
        else:
            raise Exception("Tokenization mismatch")
        
    def validate_query(self, q):
        if len(q) > 3:
            return False
        if q[0] is None and q[2] is None:
            return False
        if q[0] is not None and q[2] is not None:
            return False
        if q[1] is None or q[1] in ["", " "]:
            return False
        if q[0] is not None and q[0] in ["", " "]:
            return False
        if q[2] is not None and q[2] in ["", " "]:
            return False
        return True

    def perplexity_eval(self, example, GOLD_POS=False, GOLD_Q="N", GOLD_A=False, GOLD_A_IF_NOT_EMPTY=False, GOLD_A_ONLY=False, p_branching=False, REL_SIM=0.15, MEM_DISABLED=False):
        # Gold Q -- N: Generate query, P: assign gold_query but also consider probs, Y: assign gold_query without probs consideration
        loss_stream = []
        text_only_loss_stream = []
        target_losses = []
        text_length = 0

        for segment_data in example:
            if segment_data["preapi_text"] != "":
                pretext_encoded = self.tokenizer(segment_data["preapi_text"] + self.special_tokens_map["START"])["input_ids"]
                text_length += len(pretext_encoded[1:-1])
                losses = self.compute_loss_segment([], pretext_encoded, disable_api_start=MEM_DISABLED or GOLD_POS)
                loss_stream.extend(losses)
                text_only_loss_stream.extend(losses[:-1])
                if GOLD_POS:
                    loss_stream = loss_stream[:-1]
            else:
                pretext_encoded = self.tokenizer(segment_data["preapi_text"] + segment_data["pre_text"] + self.special_tokens_map["START"])["input_ids"]

            queries = []
            if GOLD_Q == "N" and not MEM_DISABLED:
                output_tokens = self.mr_model.generate(
                    input_ids=torch.tensor([pretext_encoded], device=self.mr_model.device), 
                    generation_config=self.memory_read_api_generation_config,
                    stopping_criteria=StoppingCriteriaList([
                        ReadStopper(stop_token=self.special_tokens_map_enc["WAIT_FOR_RES"])
                    ]))[0].cpu().numpy().tolist()
                start_of_api = int(np.argwhere(np.array(output_tokens) == self.special_tokens_map_enc["START"]).flatten()[0])
                api_gen_output = self.tokenizer.decode(output_tokens[start_of_api:])
                queries_raw = api_gen_output[12:-4]
                try:
                    queries_raw = queries_raw.split(";")
                    for query_raw in queries_raw:
                        query = query_raw.split(">>")
                        if query_raw[:2] == ">>":
                            query[0] = None
                        else:
                            query[2] = None
                        if not self.validate_query(query):
                            pass
                        else:
                            queries.append(query)
                except:
                    # Skip -- compute continuation loss with pretext
                    pass
            elif not MEM_DISABLED:
                queries = segment_data["queries"]

            api_text = "MEM_READ("
            for q in queries:
                if q[-1] is None:
                    api_text += q[0] + ">>" + q[1] + ">>" + ";"
                else:
                    api_text += ">>" + q[1] + ">>" + q[2] + ";"
            api_text = api_text[:-1] + ")" + self.special_tokens_map["WAIT_FOR_RES"]
            api_text_encoded = self.tokenizer(api_text)["input_ids"][1:]

            if GOLD_Q == "P":
                query_loss = self.compute_loss_segment(pretext_encoded, api_text_encoded, disable_api_start=MEM_DISABLED or GOLD_POS)
                loss_stream.extend(query_loss)
            
            target_results = []
            if GOLD_A_ONLY:
                target_results.extend(segment_data["target_result"])
            else:
                for query in queries:
                    query_results, _ = self.memory_controller.query_relationship(
                        query,
                        q_thr=REL_SIM
                    )
                    target_results.extend([q["relationship"][0] if query[0] is None else q["relationship"][2] for q in query_results])
                target_results = list(set(target_results))[:30]
                if (len(target_results) > 0 and GOLD_A_IF_NOT_EMPTY) or GOLD_A:
                    for target_res in segment_data["target_result"]:
                        if target_res not in target_results:
                            target_results.append(target_res)
            
            continuation = segment_data["continuation"] + ("" if segment_data["eot"] else self.special_tokens_map["START"])
            continuation_encoding = self.tokenizer(continuation)
            continuation_encoded = continuation_encoding["input_ids"][1:]

            target_char_indices = [(segment_data["continuation"].find(word), segment_data["continuation"].find(word) + len(word) - 1) for word in segment_data["target_result"]]
            target_tok_indices = []

            for start_char, end_char in target_char_indices:
                start_token = continuation_encoding.char_to_token(start_char)-1
                end_token = continuation_encoding.char_to_token(end_char)-1
                assert start_token is not None
                assert end_token is not None
                target_tok_indices.append((start_token, end_token))

            text_length += len(continuation_encoded)
            if not segment_data["eot"]:
                text_length -= 1
                
            if len(queries) == 0 or len(target_results) == 0:
                if GOLD_Q == "P":
                    # Remove query loss
                    loss_stream = loss_stream[:-len(query_loss)]
                if not GOLD_POS:
                    # Remove last starting loss
                    loss_stream = loss_stream[:-1]

                losses = self.compute_loss_segment(pretext_encoded[:-1], continuation_encoded, disable_api_start=MEM_DISABLED or GOLD_POS, disable_api_start_first=True)
                loss_stream.extend(losses)
                text_only_loss_stream.extend(losses)

                for start_tok, end_tok in target_tok_indices:
                    target_losses.append(losses[start_tok:end_tok+1])
                
                if not segment_data["eot"]:
                    text_only_loss_stream = text_only_loss_stream[:-1]
                if GOLD_POS and not segment_data["eot"]:
                    loss_stream = loss_stream[:-1]
            else:
                if p_branching and not GOLD_POS:
                    losses_wo_mem = self.compute_loss_segment(pretext_encoded[:-1], continuation_encoded, disable_api_start=MEM_DISABLED or GOLD_POS, disable_api_start_first=True)
                query_result_encoded = self.tokenizer(";".join(target_results))["input_ids"][1:] + [self.special_tokens_map_enc["END"]]
                losses = self.compute_loss_segment(pretext_encoded + api_text_encoded + query_result_encoded, continuation_encoded, disable_api_start=MEM_DISABLED or GOLD_POS)
                if p_branching:
                    p_mem = np.exp(-loss_stream[-1])
                    p_x_if_mem = np.exp(-np.array(losses))
                    p_x_if_NotMem = np.exp(-np.array(losses_wo_mem))
                    merged_losses = p_mem * p_x_if_mem + (1-p_mem) * p_x_if_NotMem
                    merged_losses = -np.log(merged_losses)
                    losses = merged_losses.tolist()
                    loss_stream = loss_stream[:-1]
                loss_stream.extend(losses)
                text_only_loss_stream.extend(losses)
                for start_tok, end_tok in target_tok_indices:
                    target_losses.append(losses[start_tok:end_tok+1])
                if not segment_data["eot"]:
                    text_only_loss_stream = text_only_loss_stream[:-1]
                if GOLD_POS and not segment_data["eot"]:
                    loss_stream = loss_stream[:-1]

        return loss_stream, text_only_loss_stream, text_length, target_losses
    
    def perplexity_eval_wo_mem(self, example):
        for segment_data in example:
            if segment_data["eot"]:
                full_text = segment_data["pre_text"] + " " + segment_data["continuation"]
                encoded = self.tokenizer(full_text)["input_ids"]
                losses = self.compute_loss_segment([], encoded, disable_api_start=True)
                return losses, losses, len(encoded) - 1