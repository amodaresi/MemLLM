python -u run_modelbased_filter.py \
    --num_gpus 4 \
    --shard_size 100 \
    --start_idx 0 \
    --stop_idx -1 \
    --max_length 256 \
    --dataset_path DATASET_PATH \
    --output_dir OUTPUT_DIR \
    --cache_dir CACHE_DIR