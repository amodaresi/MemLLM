import os
import sys
from typing import List

import fire
import torch
import transformers
import datasets
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets

import numpy as np

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
# from transformers import LlamaForCausalLM, LlamaTokenizer

def main(
    # model/data params
    base_model: str = "tiiuae/falcon-7b-instruct",  # the only required argument
    train_data_path: str = "",
    train_aug_path: str = "",
    eval_data_path: str = "",
    output_dir: str = "",
    cache_dir: str = "",
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 2,
    augmented_buckets: int = 10,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 300,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    # lora_target_modules: List[str] = [
    #     "q_proj",
    #     "v_proj",
    # ],
    lora_target_modules: List[str] = ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training MemLLM-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"train_aug_path: {train_aug_path}\n"
            f"augmented_buckets: {augmented_buckets}\n"
            f"eval_data_path: {eval_data_path}"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    print(gradient_accumulation_steps, device_map, world_size)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    train_raw_data = load_dataset("json", data_files=train_data_path)["train"]
    train_aug_data = []
    for i in range(augmented_buckets):
        train_aug_data.append(load_dataset("json", data_files=train_aug_path + f"train_aug_sample_bucket_{i+1}.json")["train"])
    eval_raw_data = load_dataset("json", data_files=eval_data_path)["train"]
    print(train_raw_data)

    model_config = transformers.AutoConfig.from_pretrained(base_model)
    # model_config.attn_config['attn_impl'] = 'triton'
    # model_config.init_device = 'cuda:0'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        device_map=device_map
    )
    # model.to("cuda:0")

    # tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    special_tokens = ["({", "})", "-->"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    START_API, END_API, API_CONTINUE = tokenizer.convert_tokens_to_ids(special_tokens)
    print(START_API, END_API, API_CONTINUE)

    tokenizer.truncation_side='left'

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    # tokenizer.padding_side = "left"  # Allow batched inference

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="lora_only",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    np.random.seed(seed=42)

    # raw_datasets = datasets.load_from_disk(train_data_path)
    # print(len(raw_datasets))
    # tokenized_datasets = raw_datasets.train_test_split(test_size=0.001, seed=42)

    # print(tokenized_datasets["train"][0])

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )
        return result

    def preprocess_function(example):
        full_text = example["pretext"] + "({USER_ST})" + example["index"] + "({USER_END})" + "({MEM_WRITE-->"
        for relation in example["relations"]:
            full_text += ">>".join(relation) + ";"
        if len(example["relations"]) > 0:
            full_text = full_text[:-1]
        full_text += "})" + tokenizer.eos_token

        tokenized = tokenize(full_text)
        input_ids = np.array(tokenized["input_ids"])
        labels = input_ids.copy()
        try:
            start_api_loc = np.argwhere(input_ids == START_API).flatten()
            start_api_loc = start_api_loc[2]
            labels[:int(start_api_loc)] = -100
        except:
            print(example)
            print(input_ids)
            print(len(input_ids))
            print(len(example["relations"]))
            print(tokenizer.decode(input_ids))
            print(start_api_loc)
            raise Exception()

        processed = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        processed["labels"] = labels.tolist()
        return processed
    
    
    tokenized_train_dataset = train_raw_data.filter(lambda example: len(example["relations"]) < 32)
    tokenized_train_dataset = tokenized_train_dataset.map(preprocess_function)

    tokenized_train_aug_datasets = []
    for i in range(augmented_buckets):
        tokenized_train_aug_datasets.append(train_aug_data[i].filter(lambda example: len(example["relations"]) < 32))
        tokenized_train_aug_datasets[-1] = tokenized_train_aug_datasets[-1].map(preprocess_function).remove_columns(["original_doc_idx"])

    tokenized_eval_dataset = eval_raw_data.filter(lambda example: len(example["relations"]) < 32).map(preprocess_function)

    tokenized_train_dataset = concatenate_datasets([tokenized_train_dataset] + tokenized_train_aug_datasets)
    tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=43)

    print(tokenized_train_dataset[0])
    
    lengths = [len(tokenized_train_dataset[i]["input_ids"]) for i in range(len(tokenized_train_dataset))]
    max_length = max(lengths)
    print(len(lengths))
    print("Max length:", max_length)
    print("Max length:", np.median(lengths), np.quantile(lengths, q=0.75))
    print(tokenized_train_dataset[0])

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")


    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt, cache_dir=cache_dir)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt, cache_dir=cache_dir)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, cache_dir=cache_dir)
    #     val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        print("ASDASDASD")
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.01,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            save_safetensors=False,
            bf16=True,
            logging_steps=25,
            optim="adamw_torch",
            weight_decay=0.001,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=0.05,
            save_steps=0.05,
            output_dir=output_dir,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(
        #     tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt"
        # )
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    # model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(main)
