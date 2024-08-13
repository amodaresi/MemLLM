python main-model-editing.py \
    --num_gpus 1 \
    --shard_size 999999 \
    --start_idx 0 \
    --stop_idx -1 \
    --max_prompt_length 192 \
    --max_generation_length 256 \
    --force_generate \
    --dataset_path /PATH_TO_EDIT_DATASET/zsre_mend_edit_preproc_reformatted.json \
    --checkpoint_dir PATH_TO_MW_MODEL \
    --output_dir /PATH_FOR_MW_OUTPUTS/forced-wise-fullSent-wQuotes-contModel/ \
    --cache_dir HF_CACHE