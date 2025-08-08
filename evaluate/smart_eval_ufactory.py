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
import policy_heads.models.transformer_diffusion.modeling_scaledp
import policy_heads.models.unet_diffusion.modeling_unet_diffusion


#  evaluate the Qwen2 VLA policy in the Agilex environment.
def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


# process observations and states
def process_obs(obs, states, stats):
    """
    obs: three cameras' images
    states: Tensor, robot states
    stats: mean, std of robot states and actions
    This function is used to get observations(images and robot states) in your robot environment.
    """
    cur_webcam_1 = obs['webcam_1']
    cur_webcam_2 = obs['webcam_2']
    assert np.max(cur_webcam_1) > 1, "All images must be 0-255."
    traj_rgb_np = np.array([cur_webcam_1, cur_webcam_2]) # sequential must align with constants.py
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)

    print("cur_state min/max:", np.min(cur_state), np.max(cur_state))

    return traj_rgb_np, cur_state # images, states


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

        self.tokenizer, self.policy, self.multimodal_processor, self.img_rgb = imgs_arracontext_len = load_model_for_eval(model_path=model_path,
                                                                                                    model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})
        
        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
        

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

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0) 
        image_list = []
        for i, each in enumerate(image_data):
            ele = {}
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            ele['resized_height'] = 240
            ele['resized_width'] = 320

            image_list.append(torch.from_numpy(np.array(each)))
        # image_data = image_data / 255.0
        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict


def eval_bc(policy, deploy_env, policy_config, raw_lang=None, query_frequency=16):
        
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    rand_crop_resize = True

    policy.policy.eval()

    ## 4. load data stats(min,max,mean....) and define post_process####################################
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'scale_dp_policy' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    #############################################################################################################

    from collections import deque
    action_queue = deque(maxlen=query_frequency)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):

        rollout_id += 0

        image_list = []  # for visualization

        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):

                obs,states = deploy_env.get_obs()
                
                ### 5. Realize the function of get_obs###################
                traj_rgb_np, robot_state = process_obs(obs, states, stats)
                #########################################################
                image_list.append(traj_rgb_np)
                robot_state = torch.from_numpy(robot_state).float().cuda()
                if t % query_frequency == 0:
                    ### 6. Augment the images##############################################################################################
                    curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)
                    #######################################################################################################################

                    ###7. Process inputs and predict actions############################################################################################
                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)
                    ####################################################################################################################################
                    # clear previous actions
                    while len(action_queue) > 0:
                        action_queue.popleft()

                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])

                raw_action = action_queue.popleft()
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                ### 8. post process actions##########################################################
                action = post_process(raw_action)
                #####################################################################################
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"batch[{k}] shape: {v.shape}, nan: {torch.isnan(v).any().item() if torch.is_floating_point(v) else 'N/A'}")

                print("all_actions nan:", torch.isnan(all_actions).any())
                print("all_actions shape:", all_actions.shape)
                print(f"after post_process action size: {action.shape}")
                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                ##### Execute ######################################################################
                action_info = deploy_env.step(action.tolist())
                ####################################################################################

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
        for idx, cam_id in enumerate(self.camera_ids):
            imgs_array = self.cam_interfaces[cam_id].get_img()
            img_bgr_raw = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = self.crop_list
            img_bgr = img_bgr_raw[y1:y2, x1:x2]
            obs[self.camera_names[idx]] = img_bgr
        # robot arm state
        ee_state = np.array(self.arm.get_position_aa(is_radian=True)[1])
        ee_state[:3] /= 1000.0  # mm→m
        gripper_state = np.array([1])  # you can get the actual gripper state
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
    crop_list = [0, 0, 640, 480]  # your actual crop settings

    xarm_bot = XArm6DexVLAEnv(arm_ip, camera_ids, camera_names, crop_list)
    # xarm_bot.reset()

    #### 3. Load DexVLA####################
    policy = qwen2_vla_policy(policy_config)
    #######################################
    eval_bc(policy, xarm_bot, policy_config, raw_lang=raw_lang, query_frequency=query_frequency)
    
    # print the first 10 weights of the policy head
    # print(policy.policy.policy_head.blocks[0].mlp.fc1.weight.flatten()[:10])

    # print the weights from the file
    # import torch
    # weights = torch.load("/home/zekaijin/DexVLA/output/qwen2_lora_rebar_insertion_stage2/checkpoint-2000/non_lora_trainables.bin")
    # for k in weights:
    #     if "policy_head.blocks.0.mlp.fc1.weight" in k:
    #         print(weights[k].flatten()[:10])
    #         print("Loaded weight shape:", weights[k].shape)
    #         print("policy_head.blocks[0].mlp.fc1.weight nan:", torch.isnan(policy.policy.policy_head.blocks[0].mlp.fc1.weight).any())
    