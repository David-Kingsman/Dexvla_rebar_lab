#!/bin/bash
LLM=qwen2_vl   #qwen2_vl  paligemma
LLM_MODEL_SIZE=2B #3B

ACTION_HEAD=scale_dp_policy  #unet_diffusion_policy or scale_dp_policy

# path to the pre-trained ScaleDP weights
# DIT_PRETRAIN=/path/to/pretrained/ScaleDP  # path to the pre-trained ScaleDP weights
# MNOP=/path/to/pretrained/qwen2_vl # change to the official qwen2_vl weights

# use a large model (more parameters, more powerful)
DIT_PRETRAIN=/home/zekaijin/DexVLA/models/scale_dp_l/open_scale_dp_l_backbone.ckpt
# # use a small model (fewer parameters, less powerful)
# DIT_PRETRAIN=/home/zekaijin/DexVLA/models/scale_dp_l/open_scale_dp_l_backbone.ckpt
MNOP=/home/zekaijin/DexVLA/models/Qwen2-VL-2B-Instruct

# Task name
TASKNAME=rebar_insertion

# Output directory ("lora" must be included when training LoRA)
OUTPUT=/home/zekaijin/DexVLA/output/train_dexvla_stage2


# Create output directory if it does not exist
mkdir -p $OUTPUT

echo "=== start single GPU training (without Flash Attention + ScaleDP_L) ==="
echo "Configuration:"
echo "- Number of GPUs: 1"
echo "- Flash Attention: Disabled"
echo "- Batch Size: 2"
echo "- pretrained weights: ScaleDP_L"
echo "- Output Directory: $OUTPUT"


# training parameters 
deepspeed --master_port 29604 --num_gpus=1 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning False \
  --lora_enable False \
  --action_dim 14 \
  --state_dim 14 \
  --flash_attn False \
  --chunk_size 50 \
  --load_pretrain_dit True \
  --pretrain_dit_path $DIT_PRETRAIN \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_L" \
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
  --max_steps 50000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "constant" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
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
