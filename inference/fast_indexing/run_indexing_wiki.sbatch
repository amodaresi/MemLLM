python main-wiki.py \
    --num_gpus 1 \
    --shard_size 1000 \
    --start_idx 6000000 \
    --stop_idx -1 \
    --abstracts \
    --max_prompt_length 192 \
    --max_generation_length 256 \
    --checkpoint_dir PATH_TO_MW_MODEL \
    --output_dir OUTPUT_DIR \
    --cache_dir HF_CACHE