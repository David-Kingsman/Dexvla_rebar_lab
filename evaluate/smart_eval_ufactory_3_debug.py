from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
import time
from data_utils.utils import set_seed

from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *

import sys
sys.path.append('/home/zekaijin/DexVLA')

import numpy as np
from arm_control_base.cam_base.camera_redis_interface import CameraRedisSubInterface
from arm_control_base.utils.util_transform import convert_pos_axis_angle_singularity, convert_pose_rep, action_interpolation
import torch
from xarm.wrapper import XArmAPI
import cv2
from transformers import AutoConfig
from PIL import Image


import os, sys
LOG_PATH = "/home/zekaijin/DexVLA/inference log.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_fp = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.stdout, _log_fp)
sys.stderr = Tee(sys.stderr, _log_fp)


#  evaluate the Qwen2 VLA policy in the Agilex environment.
def pre_process(robot_state_value, key, stats, eps=1e-6, clip=10.0):
    mean = stats[key + '_mean']
    std  = stats[key + '_std']
    tmp = (robot_state_value - mean) / (std + eps)
    tmp = np.clip(tmp, -clip, clip)  # 避免极端值放大
    return tmp


def process_obs(obs, states, stats):
    """
    obs: 两路相机图像（RGB, uint8, 0-255）
    states: np.ndarray, 机器人状态
    返回: images[B=1,T=2,C,H,W], states[B,7]
    """
    cur_webcam_1 = obs['webcam_1']  # RGB
    cur_webcam_2 = obs['webcam_2']  # RGB
    assert np.max(cur_webcam_1) > 1, "All images must be 0-255."

    traj_rgb_np = np.array([cur_webcam_1, cur_webcam_2])   # [2,H,W,3]
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)      # [2,1,H,W,3]
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))  # [1,2,3,H,W]

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)       # [1,7]

    print("cur_state min/max:", np.min(cur_state), np.max(cur_state))
    print('states.shape', states.shape)
    print('qpos_mean.shape', stats['qpos_mean'].shape)
    print('qpos_std.shape', stats['qpos_std'].shape)
    return traj_rgb_np, cur_state


def time_ms():
    return time.time_ns() // 1_000_000

class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args


    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.multimodal_processor, self.img_rgb = imgs_arracontext_len = load_model_for_eval(
            model_path=model_path,
            model_base=model_base, 
            policy_config=policy_config
        )
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})
        
        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
        self.policy.policy_head.to(dtype=torch.float32)
        if hasattr(self.policy, 'input_action_proj'):
            self.policy.input_action_proj.to(dtype=torch.float32)
        if hasattr(self.policy, 'reasoning_action_proj'):
            self.policy.reasoning_action_proj.to(dtype=torch.float32)
        if hasattr(self.policy, 'reasoning_film'):
            self.policy.reasoning_film.to(dtype=torch.float32)

        # —— 统一把进入 Linear 的输入 cast 到该层权重的 dtype（防 dtype mismatch）——
        def _force_linear_input_dtype(mod, args):
            if not args:
                return args
            x = args[0]
            if torch.is_tensor(x) and hasattr(mod, "weight") and x.dtype != mod.weight.dtype:
                return (x.to(mod.weight.dtype),)
            return args

        def _install_linear_dtype_hooks(module):
            if module is None:
                return
            for m in module.modules():
                if isinstance(m, torch.nn.Linear):
                    m.register_forward_pre_hook(_force_linear_input_dtype)

        # 对所有可能用到的 FP32 模块安装 hook
        _install_linear_dtype_hooks(self.policy.policy_head)
        _install_linear_dtype_hooks(getattr(self.policy, "input_action_proj", None))
        _install_linear_dtype_hooks(getattr(self.policy, "reasoning_action_proj", None))
        _install_linear_dtype_hooks(getattr(self.policy, "reasoning_film", None))

        # 若 head 里有 t_embedder（时间步编码），也对输入做 dtype 对齐
        if hasattr(self.policy.policy_head, "t_embedder"):
            def _fix_t_emb_dtype(mod, args):
                if not args:
                    return args
                x = args[0]
                # 用 t_embedder 第一层 Linear 的权重 dtype 作为目标
                target = None
                if hasattr(mod, "mlp") and len(mod.mlp) > 0 and hasattr(mod.mlp[0], "weight"):
                    target = mod.mlp[0].weight.dtype
                else:
                    try:
                        target = next(self.policy.policy_head.parameters()).dtype
                    except StopIteration:
                        target = x.dtype
                if torch.is_tensor(x) and x.dtype != target:
                    x = x.to(target)
                return (x,)
            self.policy.policy_head.t_embedder.register_forward_pre_hook(_fix_t_emb_dtype)


    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {     # two image slot
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": None,
                    },
                    {
                        "type": "image",
                        "image": None,
                    },
                    # {
                    #     "type": "image",
                    #     "image": None,
                    # },
                    {"type": "text", "text": f""},
                ],
            },
            # {"role": "assistant", "content": f''},
        ]

        messages[0]['content'][-1]['text'] = raw_lang

        return messages
    

    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):
        """
        curr_image: torch.Tensor [1,2,3,H,W] (0-255, RGB)
        robo_state: torch.Tensor [1,7] (已在外面 .float().cuda())
        """
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)  # [2,3,H,W]

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        frames = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # 2 个 [1,3,H,W]

        image_list = []
        for each in frames:
            arr = each.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # HWC, RGB
            pil = Image.fromarray(arr, mode="RGB")
            pil = pil.resize((320, 240), resample=Image.BILINEAR)  # 训练尺寸 (W=320,H=240)
            print("inference image shape:", pil.size, "min:", np.array(pil).min(), "max:", np.array(pil).max())
            image_list.append(pil)  # ✅ 传 PIL

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

        # device / dtype 对齐视觉塔
        vis_param = next(self.policy.visual.parameters())
        dev = vis_param.device
        vis_dtype = vis_param.dtype

        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            if torch.is_tensor(v):
                v = v.to(dev)
                if k == "pixel_values":
                    v = v.to(vis_dtype)
                data_dict[k] = v
            else:
                data_dict[k] = v
        return data_dict


def eval_bc(policy, deploy_env, policy_config, raw_lang=None, query_frequency=16):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    rand_crop_resize = False   # ✅ 推理不做随机裁剪/缩放
    policy.policy.eval()

    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'scale_dp_policy' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a

    from collections import deque
    action_queue = deque(maxlen=query_frequency)
    max_timesteps = int(1000 * 10)

    clip = 10.0  # Define the clip value used for clamping
    with torch.inference_mode():
        for t in range(max_timesteps):
            obs, states = deploy_env.get_obs()
            traj_rgb_np, robot_state = process_obs(obs, states, stats)
            robot_state = torch.from_numpy(robot_state).float().cuda()
            robot_state = torch.nan_to_num(robot_state, posinf=clip, neginf=-clip)
            robot_state.clamp_(-clip, clip)

            if t % query_frequency == 0:
                curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                # ❌ 不再做随机增广
                batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                if policy_config['tinyvla']:
                    all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                else:
                    all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)

                while len(action_queue) > 0:
                    action_queue.popleft()
                action_queue.extend(torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

            raw_action = action_queue.popleft()
            raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
            action = post_process(raw_action)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"batch[{k}] shape: {v.shape}, nan: {torch.isnan(v).any().item() if torch.is_floating_point(v) else 'N/A'}")

            print("all_actions nan:", torch.isnan(all_actions).any().item())
            print("all_actions shape:", all_actions.shape)
            print(f"after post_process action size: {action.shape}")
            print(f'step {t}, pred action: {outputs}{action}')
            if len(action.shape) == 2:
                action = action[0]
            deploy_env.step(action.tolist())


# xarm wrapper
class XArm6DexVLAEnv:
    def __init__(self, arm_ip, camera_ids, camera_names, crop_list):
        self.arm = XArmAPI(arm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        print("Start reset")
        self.camera_ids = camera_ids
        self.camera_names = camera_names
        self.crop_list = crop_list
        # start camera interfaces
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
            self.arm.reset(wait=True, timeout=30)  # wait for the reset to complete
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
            img_rgb_raw = imgs_array["color"]
            img_rgb = img_rgb_raw[y1:y2, x1:x2]

            # 保存每个 webcam 的图片
            save_path = f"inference_sample_{self.camera_names[idx]}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            obs[self.camera_names[idx]] = img_rgb

        # 机器人末端位姿（轴角），单位转成米
        ee_state = np.array(self.arm.get_position_aa(is_radian=True)[1])
        ee_state[:3] /= 1000.0
        gripper_state = np.array([1])
        states = np.concatenate([ee_state, gripper_state])

        return {
            'webcam_1': obs['webcam_1'],
            'webcam_2': obs['webcam_2'],
        }, states


# ========== XArm6DexVLAEnv ==========
if __name__ == '__main__':
     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'scale_dp_policy'  # or 'unet_diffusion_policy'
    query_frequency = 16
    policy_config = {
        #### 1. Specify path to trained DexVLA(Required)###########################
        # "model_path": "/home/zekaijin/DexVLA/output/qwen2_lora_rebar_insertion_stage2/checkpoint-2000",
        #############################################################################
        "model_path": "/home/zekaijin/Dropbox/dexvla/output/qwen2_rebar_insertion_stage2/checkpoint-2000",
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": action_head,
        "tinyvla": False,
        # "pretrain_path": None,
    }
    raw_lang ='Insert the object into the target position'
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # configure your environment
    arm_ip = "192.168.1.235"  # your robot arm IP
    camera_ids = ['webcam_1', 'webcam_2']
    camera_names = ['webcam_1', 'webcam_2']
    crop_list = [700, 200, 1220, 900]  # your actual crop settings

    xarm_bot = XArm6DexVLAEnv(arm_ip, camera_ids, camera_names, crop_list)
    # xarm_bot.reset()

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    
    # print some policy weights for debugging
    print("ViT block 31 fc1 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc1.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc1.weight.max().item())
    print("ViT block 31 fc2 weight min/max:",
          policy.policy.visual.blocks[31].mlp.fc2.weight.min().item(),
          policy.policy.visual.blocks[31].mlp.fc2.weight.max().item())


    #######################################
    eval_bc(policy, xarm_bot, policy_config, raw_lang=raw_lang, query_frequency=query_frequency)
    policy.policy.to(dtype=torch.float32)
    # print("policy_head fc1 weight mean/std:",
    #     policy.policy.policy_head.blocks[0].mlp.fc1.weight.mean().item(),
    #     policy.policy.policy_head.blocks[0].mlp.fc1.weight.std().item())

    # print the weights from the file
    # import torch
    # weights = torch.load("/home/zekaijin/DexVLA/output/qwen2_lora_rebar_insertion_stage2/checkpoint-2000/non_lora_trainables.bin")
    # for k in weights:
    #     if "policy_head.blocks.0.mlp.fc1.weight" in k:
    #         print(weights[k].flatten()[:10])
    #         print("Loaded weight shape:", weights[k].shape)
    #         print("policy_head.blocks[0].mlp.fc1.weight nan:", torch.isnan(policy.policy.policy_head.blocks[0].mlp.fc1.weight).any())
import atexit
atexit.register(_log_fp.close)