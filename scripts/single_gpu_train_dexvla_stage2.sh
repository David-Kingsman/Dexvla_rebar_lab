# #!/bin/bash

# # 🔥 必需环境变量
# export TASKNAME="rebar_insertion"
# export MNOP="/home/zekaijin/DexVLA/models/Qwen2-VL-2B-Instruct"  # Qwen2-VL权重路径
# export DIT_PRETRAIN="/home/zekaijin/DexVLA/models/scale_dp_h/pytorch_model.bin"  # ScaleDP权重路径
# export ACTION_HEAD="diffusion_policy"

# Output directory (LoRA训练必须包含"lora")
OUTPUT=/home/zekaijin/DexVLA/output/qwen2_lora_rebar_insertion

# 🔥 Stage2训练 - 单GPU LoRA配置
python ./train_vla.py \
  --use_reasoning False \
  --lora_enable True \              # LoRA训练
  --load_pretrain_dit True \        # 🔥 必需：加载预训练ScaleDP
  --pretrain_dit_path $DIT_PRETRAIN \  # 🔥 必需：ScaleDP权重路径
  --action_dim 7 \
  --state_dim 7 \
  --flash_attn True \
  --chunk_size 10 \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_L" \  # 使用小模型节省显存
  --image_size_stable "(240,320)" \
  --image_size_wrist "(240,320)" \
  --task_name ${TASKNAME} \
  --model_name_or_path $MNOP \      # 🔥 必需：Qwen2-VL权重路径
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 30000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --lazy_preprocess True \
  --report_to tensorboard