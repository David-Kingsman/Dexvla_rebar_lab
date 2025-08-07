#!/bin/bash
LLM=qwen2_vl   #qwen2_vl  paligemma
LLM_MODEL_SIZE=2B #3B
ACTION_HEAD=scale_dp_policy  #unet_diffusion_policy or scale_dp_policy

# path to the pre-trained ScaleDP weights

DIT_PRETRAIN=/home/zekaijin/DexVLA/models/scale_dp_h/open_scale_dp_h_backbone.ckpt
# change to the official qwen2_vl weights
MNOP=/home/zekaijin/DexVLA/models/Qwen2-VL-2B-Instruct

# Task name
TASKNAME=rebar_insertion

# replace with your task name, e.g., rebar_insertion
OUTPUT=/home/zekaijin/DexVLA/output/qwen2_rebar_insertion_stage2

# Create output directory if it does not exist
mkdir -p $OUTPUT

echo "=== start 4-GPU training  ==="
echo "Configuration:"
echo "- Number of GPUs: 4"
echo "- LoRA: Disabled"
echo "- Flash Attention: Disabled"
echo "- Batch Size per GPU: 2"
echo "- Total Batch Size: 64 (4 GPUs × 2 batch × 8 accumulation)"
echo "- pretrained weights: ScaleDP_H"
echo "- Output Directory: $OUTPUT"

# training parameters 
deepspeed --master_port 29604 --num_gpus=4 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning False \
  --lora_enable False \
  --action_dim 7 \
  --state_dim  7 \
  --flash_attn False \
  --chunk_size 50 \
  --load_pretrain_dit True \
  --pretrain_dit_path $DIT_PRETRAIN \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_H" \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --task_name ${TASKNAME} \
  --model_name_or_path $MNOP \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 2000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant" \
  --logging_steps 25 \
  --tf32 True \
  --model_max_length 512 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --dataloader_pin_memory True \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log 2>&1 | tee $OUTPUT/log.log


# Copy the preprocessor and chat template configuration files to each checkpoint directory
for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo "=== Training Complete ==="
echo "Output Directory: $OUTPUT"
