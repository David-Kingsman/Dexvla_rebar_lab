#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import gc
import torch

# ====== 稳定性：禁用不稳定注意力内核，提升 matmul 精度 ======
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision("high")

# ====== 第三方/本项目模块 ======
from transformers import AutoConfig
from qwen2_vla.model_load_utils import load_model_for_eval
from qwen2_vla.utils.image_processing_qwen2_vla import *
from policy_heads import *
from arm_control_base.cam_base.camera_redis_interface import CameraRedisSubInterface
from arm_control_base.utils.util_transform import convert_pos_axis_angle_singularity, convert_pose_rep, action_interpolation
from xarm.wrapper import XArmAPI


# ==============================
# 工具函数
# ==============================
def cast_norms_fp32(m):
    """将所有 LayerNorm/RMSNorm 类层转为 FP32，提高数值稳定性"""
    for _, mod in m.named_modules():
        if type(mod).__name__ in ["LayerNorm", "RMSNorm", "LlamaRMSNorm", "T5LayerNorm"]:
            mod.to(dtype=torch.float32)

from contextlib import nullcontext

def _cast_norms_fp32(m):
    for _, mod in m.named_modules():
        if type(mod).__name__ in ["LayerNorm","RMSNorm","LlamaRMSNorm","T5LayerNorm"]:
            mod.to(dtype=torch.float32)

def _make_fp32_island(block, out_back_to):
    block.to(dtype=torch.float32)

    def pre_hook(_, args, kwargs):
        # 兼容 positional / keyword 传参
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            kwargs["hidden_states"] = kwargs["hidden_states"].to(torch.float32)
            return args, kwargs
        if args and torch.is_tensor(args[0]):
            hs = args[0].to(torch.float32)
            return (hs,) + args[1:], kwargs
        return args, kwargs

    def post_hook(_, args, kwargs, output):
        def _cast(o):
            if torch.is_tensor(o): return o.to(out_back_to)
            if isinstance(o, (list, tuple)): return type(o)(_cast(x) for x in o)
            if isinstance(o, dict): return {k: _cast(v) for k, v in o.items()}
            return o
        return _cast(output)

    # 关键：with_kwargs=True（老版 PyTorch 不支持就 try/except 降级）
    try:
        block.register_forward_pre_hook(pre_hook, with_kwargs=True)
        block.register_forward_hook(post_hook, with_kwargs=True)
    except TypeError:
        block.register_forward_pre_hook(lambda m, a: pre_hook(m, a, {})[0])
        block.register_forward_hook(lambda m, a, out: post_hook(m, a, {}, out))


def set_precision(policy, vit_dtype=torch.bfloat16, head_fp32=False, last2_fp32=False, fp32_visual_proj=False):
    """
    - vit_dtype：视觉主干基础精度（bf16 推荐）
    - last2_fp32：将视觉塔最后两层做成 FP32 小岛（自动处理 dtype 边界）
    - fp32_visual_proj：把 visual_proj 也做成小岛（视觉→LLM 投影稳定）
    - head_fp32：动作头/扩散头走 FP32
    """
    model = policy.policy
    vt = model.visual

    # 基础：大部分模块保持 vit_dtype（省显存/快）
    vt.to(dtype=vit_dtype)
    if hasattr(model, "visual_merger"):
        model.visual_merger.to(dtype=vit_dtype)
    if hasattr(model, "visual_proj"):
        model.visual_proj.to(dtype=vit_dtype)

    # Norm 固定 FP32（提升稳定性）
    _cast_norms_fp32(vt)
    if hasattr(model, "visual_merger"):
        _cast_norms_fp32(model.visual_merger)

    # 最后两层 ViT 做 FP32 小岛（避免上一层 bf16 → 本层 fp32 的 matmul 冲突）
    if last2_fp32:
        num_blocks = len(vt.blocks)
        for i in range(max(0, num_blocks-2), num_blocks):
            _make_fp32_island(vt.blocks[i], out_back_to=vit_dtype)

    # 视觉→LLM 投影也做小岛（可选，很多模型里这一步也容易放大）
    if fp32_visual_proj and hasattr(model, "visual_proj"):
        _make_fp32_island(model.visual_proj, out_back_to=vit_dtype)

    # 动作/扩散头按需 FP32（通常建议 True）
    for name, mod in model.named_modules():
        if any(k in name.lower() for k in ["policy_head","action_head","diffusion","unet","value_head"]):
            try:
                mod.to(dtype=torch.float32 if head_fp32 else vit_dtype)
            except Exception:
                pass

    # 同步 config（防止 forward 里读 config.torch_dtype 又 cast 回去）
    if hasattr(model, "config"):
        try:
            model.config.torch_dtype = vit_dtype
        except Exception:
            pass
    if vit_dtype == torch.float32:
        for blk in vt.blocks:
            _make_fp32_island(blk, out_back_to=torch.float32)
        # （可选）视觉投影也做成小岛
        if hasattr(model, "visual_proj"):
            _make_fp32_island(model.visual_proj, out_back_to=torch.float32)

def _force_linear_input_dtype(linear):
    def _pre(mod, args):
        x = args[0]
        if torch.is_tensor(x) and x.dtype != mod.weight.dtype:
            return (x.to(mod.weight.dtype),)
        return args
    linear.register_forward_pre_hook(_pre)

def fix_qkv_dtype(policy):
    for blk in policy.policy.visual.blocks:
        if hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
            _force_linear_input_dtype(blk.attn.qkv)

def fix_policy_head_inputs(policy):
    if not hasattr(policy.policy, "policy_head"):
        return
    head = policy.policy.policy_head
    try:
        head_dtype = next(head.parameters()).dtype
    except StopIteration:
        return

    def _pre(mod, args):
        # policy_head(actions, hidden_states, states, is_pad)
        args = list(args)
        # actions
        if len(args) >= 1 and torch.is_tensor(args[0]) and args[0].dtype != head_dtype:
            args[0] = args[0].to(head_dtype)
        # hidden_states
        if len(args) >= 2 and torch.is_tensor(args[1]) and args[1].dtype != head_dtype:
            args[1] = args[1].to(head_dtype)
        # states
        if len(args) >= 3 and torch.is_tensor(args[2]) and args[2].dtype != head_dtype:
            args[2] = args[2].to(head_dtype)
        return tuple(args)

    head.register_forward_pre_hook(_pre)

def fix_scaledp_time_dtype(policy):
    if not hasattr(policy.policy, "policy_head"):
        return
    head = policy.policy.policy_head
    try:
        head_dtype = next(head.parameters()).dtype
    except StopIteration:
        return

    if hasattr(head, "t_embedder"):
        def _pre_time(mod, args):
            if not args: 
                return args
            x = args[0]
            if torch.is_tensor(x) and x.dtype != head_dtype:
                x = x.to(head_dtype)
            return (x,)
        head.t_embedder.register_forward_pre_hook(_pre_time)


def hook_block31(policy):
    """注册 Block31 的前向 hook，记录输出统计"""
    stats = {"b31_min": None, "b31_max": None, "b31_nan": None}

    # 若模型 block 少于 32 层，取最后一层
    blocks = policy.policy.visual.blocks
    idx = 31 if len(blocks) > 31 else len(blocks) - 1

    def _hook(_, __, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.detach().float()
        stats["b31_min"] = t.min().item()
        stats["b31_max"] = t.max().item()
        stats["b31_nan"] = torch.isnan(t).any().item()

    h = blocks[idx].register_forward_hook(_hook)
    return h, stats, idx

def sanitize_action_tensor(t: torch.Tensor):
    """动作 NaN/Inf 的兜底处理，防止下发到机械臂"""
    if torch.isnan(t).any() or not torch.isfinite(t).all():
        print("[WARN] action has NaN/Inf; zeroing")
        return torch.zeros_like(t)
    return t

def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp

def process_obs(obs, states, stats):
    """
    obs: 两路相机图像（RGB, 0-255）
    states: 机器人状态 (np.ndarray)
    stats: 训练数据统计
    返回: (images[B=1,T=2,C,H,W], states[B,7])
    """
    cur_webcam_1 = obs['webcam_1']
    cur_webcam_2 = obs['webcam_2']
    assert np.max(cur_webcam_1) > 1, "All images must be 0-255."

    traj_rgb_np = np.array([cur_webcam_1, cur_webcam_2])  # [2,H,W,3]
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)     # [2,1,H,W,3]
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))  # [1,2,3,H,W]

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)  # [1,7]

    print("cur_state min/max:", np.min(cur_state), np.max(cur_state))
    print('states.shape', states.shape)
    print('qpos_mean.shape', stats['qpos_mean'].shape)
    print('qpos_std.shape', stats['qpos_std'].shape)
    return traj_rgb_np, cur_state

def time_ms():
    return time.time_ns() // 1_000_000


# ==============================
# 策略封装
# ==============================
class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.multimodal_processor, self.img_rgb = imgs_arracontext_len = load_model_for_eval(
            model_path=model_path,
            model_base=model_base,
            policy_config=policy_config
        )
        # 建议：若训练未添加此 special token，就不要在推理时新增
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "image", "image": None},
                    {"type": "text", "text": f""},
                ],
            },
        ]
        messages[0]['content'][-1]['text'] = raw_lang
        return messages

    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):
        """
        curr_image: torch.Tensor [1,2,3,H,W] (0-255)
        robo_state: torch.Tensor [1,7]
        """
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)  # [2,3,H,W]

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        frames = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # 2 个 [1,3,H,W]
        image_list = []
        for each in frames:
            pil = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))  # PIL(RGB)
            # resize 到 (320,240) (W,H) —— 保持和训练一致
            pil = pil.resize((320, 240), resample=Image.BILINEAR)
            print("inference image shape:", pil.size, "min:", np.array(pil).min(), "max:", np.array(pil).max())
            image_list.append(pil)

        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.multimodal_processor(
            text=text,
            images=image_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v

        # —— 对齐到视觉塔的设备与精度 —— #
        vis_param = next(self.policy.visual.parameters())
        dev = vis_param.device
        vis_dtype = vis_param.dtype

        for k, v in data_dict.items():
            if torch.is_tensor(v):
                data_dict[k] = v.to(dev)

        # pixel_values 显式对齐到视觉塔 dtype（fp32_all -> fp32；bf16 模式 -> bf16）
        if "pixel_values" in data_dict:
            data_dict["pixel_values"] = data_dict["pixel_values"].to(vis_dtype)

        return data_dict




# ==============================
# 单次前向：做精度 A/B 测试
# ==============================
@torch.inference_mode()
def run_once(policy, env, stats, raw_lang):
    obs, states = env.get_obs()
    traj_rgb_np, robot_state = process_obs(obs, states, stats)
    robot_state = torch.from_numpy(robot_state).float().cuda()
    batch = policy.process_batch_to_qwen2_vla(torch.from_numpy(traj_rgb_np).float().cuda(), robot_state, raw_lang)

    h, b31, idx = hook_block31(policy)
    try:
        # 关键：在这里加上 AMP 守护，视觉里有 fp32 参数就禁用 autocast
        with amp_ctx_for_visual(policy):
            if policy.policy_config['tinyvla']:
                all_actions, outputs = policy.policy.evaluate_tinyvla(
                    **batch, is_eval=True, tokenizer=policy.tokenizer
                )
            else:
                all_actions, outputs = policy.policy.evaluate(
                    **batch, is_eval=True, tokenizer=policy.tokenizer
                )
    finally:
        h.remove()

    res = {
        "all_actions_nan": torch.isnan(all_actions).any().item(),
        "all_actions_min": float(all_actions.float().min().item()) if not torch.isnan(all_actions).any() else None,
        "all_actions_max": float(all_actions.float().max().item()) if not torch.isnan(all_actions).any() else None,
        "bxx_index": idx,
        **b31
    }
    print("[RUN] ", res)
    return res


from contextlib import nullcontext

def amp_ctx_for_visual(policy):
    # 只要视觉塔里出现任何 FP32 参数，就关掉 autocast，避免输入被降成 bf16
    any_visual_fp32 = False
    for p in policy.policy.visual.parameters():
        if p.dtype == torch.float32:
            any_visual_fp32 = True
            break
    # 需要禁用 AMP 时：enabled=False；否则用 bf16 AMP
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
            if any_visual_fp32 else
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True))

# ==============================
# 实际评估（含动作“止血”护栏）
# ==============================
def eval_bc(policy, deploy_env, policy_config, raw_lang=None, query_frequency=16, rand_crop_resize=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    policy.policy.eval()

    # 载入数据统计
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 动作后处理
    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'scale_dp_policy' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a

    from collections import deque
    action_queue = deque(maxlen=query_frequency)
    max_timesteps = int(1000 * 10)

    with torch.inference_mode():
        for t in range(max_timesteps):
            obs, states = deploy_env.get_obs()
            traj_rgb_np, robot_state = process_obs(obs, states, stats)
            robot_state = torch.from_numpy(robot_state).float().cuda()

            if t % query_frequency == 0:
                curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                if rand_crop_resize:
                    print('rand crop resize is used!')
                    original_size = curr_image.shape[-2:]
                    ratio = 0.95
                    curr_image = curr_image[..., 
                        int(original_size[0]*(1-ratio)/2):int(original_size[0]*(1+ratio)/2),
                        int(original_size[1]*(1-ratio)/2):int(original_size[1]*(1+ratio)/2)]
                    curr_image = curr_image.squeeze(0)
                    resize_transform = transforms.Resize(original_size, antialias=True)
                    curr_image = resize_transform(curr_image).unsqueeze(0)

                batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                with amp_ctx_for_visual(policy):
                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)

                # 止血：若 NaN，直接置零（也可尝试 FP32 头部重算，这里先兜底）
                all_actions = sanitize_action_tensor(all_actions)

                # 更新 action_queue
                while len(action_queue) > 0:
                    action_queue.popleft()
                action_queue.extend(torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

            raw_action = action_queue.popleft()
            raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
            action = post_process(raw_action)

            # 调试输出
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"batch[{k}] shape: {v.shape}, nan: {torch.isnan(v).any().item() if torch.is_floating_point(v) else 'N/A'}")
            print("all_actions shape:", all_actions.shape)
            print(f"after post_process action size: {action.shape}")
            print(f'step {t}, pred action: {outputs}{action}')

            # 下发动作（示例：已注释掉真实运动命令）
            if len(action.shape) == 2:
                action = action[0]
            deploy_env.step(action.tolist())

import torch

def apply_dtype_fixes(policy):
    if getattr(policy.policy, "_dtype_fixes_done", False):
        return

    def _fix_linear_in_dtype(mod, args):
        if not args:
            return args
        x = args[0]
        if torch.is_tensor(x) and x.dtype != mod.weight.dtype:
            return (x.to(mod.weight.dtype),)
        return args

    # -------- Visual tower --------
    vis = getattr(policy.policy, "visual", None)
    if vis is not None:
        for blk in getattr(vis, "blocks", []):
            if hasattr(blk, "attn"):
                if isinstance(getattr(blk.attn, "qkv", None), torch.nn.Linear):
                    blk.attn.qkv.register_forward_pre_hook(_fix_linear_in_dtype)
                if isinstance(getattr(blk.attn, "proj", None), torch.nn.Linear):
                    blk.attn.proj.register_forward_pre_hook(_fix_linear_in_dtype)
            if hasattr(blk, "mlp"):
                if isinstance(getattr(blk.mlp, "fc1", None), torch.nn.Linear):
                    blk.mlp.fc1.register_forward_pre_hook(_fix_linear_in_dtype)
                if isinstance(getattr(blk.mlp, "fc2", None), torch.nn.Linear):
                    blk.mlp.fc2.register_forward_pre_hook(_fix_linear_in_dtype)

        # visual_proj / visual_merger
        for mod_name in ("visual_proj", "visual_merger"):
            mod = getattr(policy.policy, mod_name, None)
            if mod is not None:
                for m in mod.modules():
                    if isinstance(m, torch.nn.Linear):
                        m.register_forward_pre_hook(_fix_linear_in_dtype)

    # -------- Policy head --------
    head = getattr(policy.policy, "policy_head", None)
    if head is not None:
        for m in head.modules():
            if isinstance(m, torch.nn.Linear):
                m.register_forward_pre_hook(_fix_linear_in_dtype)

        if hasattr(head, "t_embedder"):
            def _fix_t_emb_dtype(mod, args):
                if not args:
                    return args
                x = args[0]
                target = None
                if hasattr(mod, "mlp") and len(mod.mlp) > 0 and hasattr(mod.mlp[0], "weight"):
                    target = mod.mlp[0].weight.dtype
                else:
                    try:
                        target = next(head.parameters()).dtype
                    except StopIteration:
                        target = x.dtype
                if torch.is_tensor(x) and x.dtype != target:
                    x = x.to(target)
                return (x,)
            head.t_embedder.register_forward_pre_hook(_fix_t_emb_dtype)

    policy.policy._dtype_fixes_done = True
    print("[apply_dtype_fixes] dtype hooks installed once")




# 用法：
# 先 set_precision(policy, **cfg)，再调用：
# apply_dtype_fixes(policy)


def fix_t_embedder_dtype(policy):
    if not hasattr(policy.policy, "policy_head"):
        return
    head = policy.policy.policy_head

    if hasattr(head, "t_embedder"):
        t_emb = head.t_embedder
        def _pre_t_emb(mod, args):
            if not args:
                return args
            x = args[0]
            if torch.is_tensor(x) and x.dtype != mod.mlp[0].weight.dtype:
                x = x.to(mod.mlp[0].weight.dtype)
            return (x,)
        t_emb.register_forward_pre_hook(_pre_t_emb)

    def dbg_qkv_pre(mod, inp):
        x = inp[0]
        print("[QKV pre] in:", x.dtype, "w:", mod.weight.dtype)
    for i in [0, 1, len(policy.policy.visual.blocks)-1]:
        policy.policy.visual.blocks[i].attn.qkv.register_forward_pre_hook(dbg_qkv_pre)

    def _dbg_head(mod, args):
        ac = args[0] if len(args)>0 and torch.is_tensor(args[0]) else None
        hs = args[1] if len(args)>1 and torch.is_tensor(args[1]) else None
        st = args[2] if len(args)>2 and torch.is_tensor(args[2]) else None
        print("[HEAD] w:", next(mod.parameters()).dtype,
            "| actions:", getattr(ac,"dtype",None),
            "| hs:", getattr(hs,"dtype",None),
            "| st:", getattr(st,"dtype",None))
    policy.policy.policy_head.register_forward_pre_hook(_dbg_head)

def force_linears_in_head(policy):
    if not hasattr(policy.policy, "policy_head"):
        return
    head = policy.policy.policy_head

    def _pre_linear(mod, args):
        if not args:
            return args
        x = args[0]
        if torch.is_tensor(x) and x.dtype != mod.weight.dtype:
            return (x.to(mod.weight.dtype),)
        return args

    for m in head.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(_pre_linear)


# ==============================
# 机器人环境封装
# ==============================
class XArm6DexVLAEnv:
    def __init__(self, arm_ip, camera_ids, camera_names, crop_list, source_is_rgb=True, save_debug=True):
        self.arm = XArmAPI(arm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        print("Start reset")

        self.camera_ids = camera_ids
        self.camera_names = camera_names
        self.crop_list = crop_list
        self.source_is_rgb = source_is_rgb
        self.save_debug = save_debug

        # 启动相机接口
        self.cam_interfaces = {}
        for idx, camera_id in enumerate(self.camera_ids):
            cam_interface = CameraRedisSubInterface(
                camera_info={"camera_id": camera_id, "camera_name": self.camera_names[idx]},
                use_depth=False
            )
            cam_interface.start()
            self.cam_interfaces[camera_id] = cam_interface

    def reset(self):
        try:
            self.arm.reset(wait=True, timeout=30)
        except Exception as e:
            print(f"Reset failed: {e}")
        print("Reset to home position.")

    def step(self, action):
        # action: [px, py, pz, qx, qy, qz, gripper]
        target_action = np.zeros(7)
        target_action[:6] = action[:6]
        target_action[6] = action[6]
        target_action[:3] *= 1000.0  # m→mm
        print(f"Execute action: {target_action}")
        time.sleep(0.5)
        # self.arm.set_servo_cartesian_aa(target_action, speed=20, mvacc=200, is_radian=True)
        time.sleep(0.5)

    def get_obs(self):
        obs = {}
        x1, y1, x2, y2 = self.crop_list
        for idx, cam_id in enumerate(self.camera_ids):
            imgs_array = self.cam_interfaces[cam_id].get_img()
            img = imgs_array["color"]  # 假设来自 Redis 的是 RGB
            if not self.source_is_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img[y1:y2, x1:x2]
            if self.save_debug and idx == 0:
                cv2.imwrite("inference_sample.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            obs[self.camera_names[idx]] = img_rgb

        # 机器人末端位姿（轴角），单位转成米
        ee_state = np.array(self.arm.get_position_aa(is_radian=True)[1])
        ee_state[:3] /= 1000.0
        gripper_state = np.array([1])  # 如需换真实夹爪状态，这里替换
        states = np.concatenate([ee_state, gripper_state])

        return {
            'webcam_1': obs['webcam_1'],
            'webcam_2': obs['webcam_2'],
        }, states
    
import argparse


# ==============================
# 主程序
# ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fp32_all')
    args = parser.parse_args()

    modes = {
        "fp32_all": dict(vit_dtype=torch.float32,  head_fp32=True,  last2_fp32=False),
        "bf16_all": dict(vit_dtype=torch.bfloat16, head_fp32=False, last2_fp32=False),
        "bf16_head_fp32": dict(vit_dtype=torch.bfloat16, head_fp32=True,  last2_fp32=False),
        "bf16_last2_head32": dict(vit_dtype=torch.bfloat16, head_fp32=True,  last2_fp32=True),
    }
    cfg = modes[args.mode]

    # >>>>>>>>>>>>>>>> 超参数 <<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        "model_path": "/home/zekaijin/Dropbox/dexvla/output/qwen2_rebar_insertion_stage2/checkpoint-2000",
        "model_base": None,
        "enable_lora": False,
        "action_head": action_head,
        "tinyvla": False,
    }
    raw_lang = 'Insert the object into the target position'

    # 相机/机械臂配置
    arm_ip = "192.168.1.235"
    camera_ids = ['webcam_1', 'webcam_2']
    camera_names = ['webcam_1', 'webcam_2']
    crop_list = [700, 200, 1220, 900]  # 左、上、右、下
    source_is_rgb = True               # 如果相机给的是 BGR，改成 False

    # ====== 初始化环境与策略 ======
    xarm_bot = XArm6DexVLAEnv(arm_ip, camera_ids, camera_names, crop_list, source_is_rgb=source_is_rgb)
    policy = qwen2_vla_policy(policy_config)
    set_precision(policy, **cfg)
    apply_dtype_fixes(policy)
    policy.policy.eval()

    # 调试打印权重范围
    print("ViT block 31 fc1 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc1.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc1.weight.max().item())
    print("ViT block 31 fc2 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc2.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc2.weight.max().item())
    
    # 检查权重是否有 nan/inf 或极端大值
    with torch.no_grad():
        for name, param in policy.policy.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[ERROR] {name} contains nan or inf!")
            if param.abs().max() > 1e3:
                print(f"[WARN] {name} has very large value: max={param.abs().max().item()}")
                
    # 调试打印权重范围
    print("ViT block 31 fc1 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc1.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc1.weight.max().item())
    print("ViT block 31 fc2 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc2.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc2.weight.max().item())
    
    # 读取训练统计
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        _stats = pickle.load(f)

    # ====== 精度 A/B 测试（四种组合） ======
    modes = [
        ("fp32_all",          dict(vit_dtype=torch.float32,  head_fp32=True,  last2_fp32=False)),
        ("bf16_all",          dict(vit_dtype=torch.bfloat16, head_fp32=False, last2_fp32=False)),
        ("bf16_head_fp32",    dict(vit_dtype=torch.bfloat16, head_fp32=True,  last2_fp32=False)),
        ("bf16_last2_head32", dict(vit_dtype=torch.bfloat16, head_fp32=True,  last2_fp32=True)),
    ]

    results = {}
    for name, cfg in modes:
        print(f"\n==== Testing {name} ====")
        # 释放旧模型
        if 'policy' in locals():
            del policy
            torch.cuda.empty_cache()
            gc.collect()
        # 重新加载新模型
        policy = qwen2_vla_policy(policy_config)
        set_precision(policy, **cfg)
        apply_dtype_fixes(policy)
        try:
            print("b31 fc1 dtype:", policy.policy.visual.blocks[31].mlp.fc1.weight.dtype)
        except Exception:
            pass
        for mod_name, mod in policy.policy.named_modules():
            if "policy_head" in mod_name.lower() or "action_head" in mod_name.lower():
                try:
                    first_param = next(mod.parameters())
                    print("head dtype:", first_param.dtype, " @ ", mod_name)
                    break
                except StopIteration:
                    pass

        results[name] = run_once(policy, xarm_bot, _stats, raw_lang)
        torch.cuda.synchronize()

    print("\n==== SUMMARY ====")
    for k, v in results.items():
        print(k, "->", v)

    # ====== 若上面找到稳定的组合（例如 bf16_last2_head32 正常），可固定后再跑实际评估 ======
    # set_precision(policy, vit_dtype=torch.bfloat16, head_fp32=True, last2_fp32=True)
    # eval_bc(policy, xarm_bot, policy_config, raw_lang=raw_lang, query_frequency=query_frequency, rand_crop_resize=False)



