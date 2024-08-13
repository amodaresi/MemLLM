torchrun --nproc_per_node=4 --master_port 21321 main_write_mistral_w_annotated_w_augmented.py \
    --base_model 'mistralai/Mistral-7B-v0.1' \
    --micro_batch_size 3 \
    --batch_size 96 \
    --learning_rate 2e-5 \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --cutoff_len 1024 \
    --num_epochs 2 \
    --augmented_buckets 0 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]' \
    --train_data_path PATH_TO_PROCESSED_MW_DATA/data/train.json \
    --train_aug_path PATH_TO_PROCESSED_MW_DATA/data/augmented/ \
    --eval_data_path PATH_TO_PROCESSED_MW_DATA/data/validation.json \
    --cache_dir HF_CACHE_DIR \
    --resume_from_checkpoint 'OUTPUT_DIR/model/CHECKPOINT_TO_CONTINUE_FROM' \
    --output_dir OUTPUT_DIR