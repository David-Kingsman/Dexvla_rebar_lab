import numpy as np
import os
from scipy.spatial.transform import Rotation
import h5py
import cv2
# import transform_utils
import copy
import sys
# sys.path.append("/mnt/sda1/act") 
# make the current NumPy’s core visible under the old name

"""
Convert the demos from Deoxys to hdf5 format used by ACT
Note that we use m as the unit rather than mm, so translation is divided by 1000
"""
# ----------------------- !!! Setting !!! -------------------- #
input_path = 'data/rebar_insertion' 
output_path = "data/rebar_insertion_hdf5" 
max_timesteps = 300 # # maximum timesteps for each episode
action_type = "abs" # "abs" for absolute position, "delta" for relative position
img_only = False # if we only use the image without the robot state, we need to set this to True
camera_names = ["webcam_1","webcam_2"]   # sequence can not be wrong
cam_idxes = ["1","2"] # "0","1","2"   sequence can not be wrong ["2","3","4"]
crop_wo_sam2_option = True  # if we crop the image, we need to set this to True
image_h = 700 # 900 # desired size 240 # 320 # 480 # 720 # 480 # 1080
image_w = 520 # 960 # desired size 320 # 480 # 640 # 1280 # 640 # 1920
crop_list = [[700, 200, 1220, 900]] # w1,h1,w2,h2

# ----------------------- !!! Advanced Setting !!! ----------------------- #
hdf5_start_idx = 0 # if we want to continue from the last hdf5 file, we can set this to the last hdf5 file index
# if we use sam2 to segment the image, we need to adjust the following parameters, we don't use them in the Ufactory implementation
sam2_option = False # whether to use SAM2 for segmentation
sam2_wo_bg = False # whether to use SAM2 without background
sam2_w_crop_option = False # whether to use SAM2 with cropping

# DexVLA specific settings
lang_intrs = "Insert the object into the target position"  
state_dim = 7  
action_dim = 7  

def get_target_id(eposide):
    # if eposide<=24:
    #     return np.array([0])
    # else:
    return np.array([0])

# def get_rebar_id(eposide_id):
#     if eposide_id <= 9 or 49>= eposide_id >= 22 or 104>= eposide_id >= 79 or 174>= eposide_id >= 154:
#         return "rebar1"
#     else:
#         return "rebar2"


os.makedirs(output_path, exist_ok=True)
# Search all subfolders starting with "run" and sort them
subfolders = sorted([folder for folder in os.listdir(input_path) if folder.startswith('run') and os.path.isdir(os.path.join(input_path, folder))])

# Initialize lists to store the merged data and episode end indices
all_ee_data = []
all_action_data = []
episode_ends = []
timestamps = []
current_index = 0

# To avoid singularity of axis angle representation
# Convert a position and axis-angle representation to a transformation matrix
def pos_axis2mat(pos_axis_angle):
    mat = np.eye(4)
    rot = Rotation.from_rotvec(pos_axis_angle[3:6])
    rot_mat = rot.as_matrix()
    mat[:3, :3] = rot_mat
    mat[:3, 3] = pos_axis_angle[:3]
    return  mat

# Convert a transformation matrix to a position and axis-angle representation
def mat2pos_axis(mat):
    pose_6D = np.zeros(6)
    mat = mat.reshape(4, 4, order='F')
    rot_mat = mat[:3, :3]
    trans_vec = mat[:3, 3]
    rot = Rotation.from_matrix(rot_mat)
    rot_vec = rot.as_rotvec()
    pose_6D[:3] = trans_vec
    pose_6D[3:] = rot_vec
    return pose_6D

# Convert a position and axis-angle representation to a transformation matrix
def avoid_singularity(Input, type = "mat"):
    A = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0.0],
         [0, 0, -1, 0.0],
         [0, 0, 0, 1.0]])
    if type == "mat":
        mat_new = A @ Input
        pos_angles_new = mat2pos_axis(mat_new)
        return pos_angles_new
    elif type == "pos_axis_angle":
        mat = pos_axis2mat(Input)
        mat_new = A @ mat  # A.T
        pos_angles_new = mat2pos_axis(mat_new)
        return pos_angles_new

    else:
        raise NotImplementedError(f"Type {type} is not implemented")

# repetitive paths can be caused by the incorrect data collection, so we can filter them
def filter_repetitives(rgb_paths): # if we incorrectly add repetitive paths, this can help
    filtered_rgb_paths = list(set(rgb_paths))
    filtered_rgb_paths.sort(key=rgb_paths.index)
    if len(rgb_paths) != len(filtered_rgb_paths): # delete competitive path
        print(f"Number of elements before filtering: {len(rgb_paths)}")
        print(f"Number of elements after filtering: {len(filtered_rgb_paths)}")
    return filtered_rgb_paths

# Convert absolute position and axis-angle representation to relative position and axis-angle representation
def abs2relative(ee_gripper_data, ref_pos, type="6D"):
    from scipy.spatial.transform import Rotation
    if type == "6D":
        # Take the first 6 values of the first row as the original pose
        initial_xyz = ref_pos[:3]
        initial_axis_angle = ref_pos[3:6]
        initial_quat = Rotation.from_rotvec(initial_axis_angle)
        # xyz
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
        # rx, ry, rz
        for i in range(relative_pose.shape[0] - 1):
            abs_axis_angle = relative_pose[i, 3:6]
            abs_quat = Rotation.from_rotvec(abs_axis_angle)

            quat_diff = abs_quat * initial_quat.inv()
            relative_pose[i, 3:6] = quat_diff.as_rotvec()
        # The last column (gripper state) remains unchanged
    elif type == "pos":
        initial_xyz = ref_pos[:3]
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
    else:
        raise NotImplementedError
    return relative_pose

# Convert absolute position and axis-angle representation to relative position and axis-angle representation
def abs2delta(ee_gripper_data, type="6D"):
    if type == "6D":
        # Take the first 6 values of the first row as the original pose
        initial_xyz = ee_gripper_data[0, :3]
        initial_axis_angle = ee_gripper_data[0, 3:6]
        initial_quat = Rotation.from_rotvec(initial_axis_angle)
        # xyz
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
        # rx, ry, rz
        for i in range(relative_pose.shape[0] - 1):
            abs_axis_angle = relative_pose[i, 3:6]
            abs_quat = Rotation.from_rotvec(abs_axis_angle)

            quat_diff = abs_quat * initial_quat.inv()
            relative_pose[i, 3:6] = quat_diff.as_rotvec()
        # The last column (gripper state) remains unchanged
    # if type == "pos":
    #     initial_xyz = ee_gripper_data[0, :3]
    #     relative_pose = ee_gripper_data.copy()
    #     relative_pose[:, :3] -= initial_xyz
    else:
        raise NotImplementedError
    return relative_pose

# Loop through each subfolder and process data 
for episode_idx, folder in enumerate(subfolders):
    print(f"Processing {folder}...")
    # Load data 
    ee_states_path = os.path.join(input_path, folder, 'demo_ee_states.npz')
    gripper_path = os.path.join(input_path, folder, 'demo_gripper_states.npz')
    FT_raw_path = os.path.join(input_path, folder, 'demo_FT_raw.npz')
    FT_processed_path = os.path.join(input_path, folder, 'demo_FT_processed.npz')
    action_path = os.path.join(input_path, folder, 'demo_action.npz')
    action_grasp_path = os.path.join(input_path, folder, 'demo_action_grasp.npz')
    action_hot = os.path.join(input_path, folder, 'demo_action_hot.npz')
    # Load camera data
    camera_paths = []
    npz_list = os.listdir(os.path.join(input_path, folder))
    npz_list.sort()
    for idx, file in enumerate(npz_list):
        if "camera" in file and 'npz' in file:
            cam_idx = file.split("_")[2].split(".")[0]
            if cam_idx in cam_idxes:
                camera_paths.append(os.path.join(input_path, folder, f'demo_camera_{cam_idx}.npz'))
    # Load end effector and gripper data
    ee_data_raw = np.load(ee_states_path, allow_pickle=True)['data'] # Ufactory: [x,y,z,rx,ry,rz], rotation is in rad 
    # Convert translation from mm → m 
    ee_data_raw[:, :3] /= 1000.0
    # Load gripper data
    gripper_data = np.load(gripper_path, allow_pickle=True)['data']
    episode_len = ee_data_raw.shape[0] 

    # To avoid singularity of axis angle representation, make ee rot same as the robot base rot
    if img_only:
        ee_gripper_data = None
    else:
        if action_type == "abs":
            ee_data_new = np.zeros((episode_len, 6))
            for i in range(episode_len):
                ee_data_new[i, :] = avoid_singularity(ee_data_raw[i], type="pos_axis_angle")

            ee_gripper_data = np.concatenate((ee_data_new, gripper_data.reshape(gripper_data.shape[0], 1)), axis=1)  #[length, 7]

        else:
            raise NotImplementedError

    # ----------------------- action states ----------------------- #
    # If we use relative action, we need to convert the absolute action to relative action
    if action_type == "abs":
        action_data = np.load(action_path, allow_pickle=True)['data']
        # Convert translation from mm → m
        action_data[:, :3] /= 1000.0
        action_data_new = np.zeros((episode_len, 6))
        for i in range(episode_len):
            action_data_new[i, :6] = avoid_singularity(action_data[i, :6],type="pos_axis_angle")
        
        action_grasp_data = np.load(action_grasp_path, allow_pickle=True)['data']
        action_data = np.concatenate((action_data_new, action_grasp_data.reshape(action_grasp_data.shape[0], 1)), axis=1) #[length, 7]
    else:
        raise NotImplementedError

    # ----------------------- force and torque ----------------------- #
    FT_raw_data = np.load(FT_raw_path, allow_pickle=True)['data']
    FT_processed_data = np.load(FT_processed_path, allow_pickle=True)['data']

    # ----------------------- images ----------------------- #
    assert len(camera_paths) == len(camera_names)
    all_image_data = {}

    for idx, cam_name in enumerate(camera_names):
        if sam2_option:
            pass
            # segmentor = Sam2()
            # segmentor.initialize(slot_id=f"{get_rebar_id(episode_idx)}_{cam_name}_slot5", single_class=False, manual=False)
        camera_data = np.load(camera_paths[idx], allow_pickle=True)
        rgb_paths = [os.path.join(input_path,"images", item['color_img_name'].split("/")[-2], item['color_img_name'].split("/")[-1] + ".jpg") for item in camera_data["data"]]
        
        # if we use sam2 to segment the image, we need to crop the image
        if sam2_option:
            rgb_data_crop_1 = []
            rgb_data_crop_2 = []
            rgb_data_crop_3 = []
        else:
            rgb_data_single_cam = []

        # rgb_paths = filter_repetitives(rgb_paths) # just for debugging

        # Check if the number of images matches the episode length
        if crop_wo_sam2_option:
            crop_region = crop_list[0]
            for i, image_path in enumerate(rgb_paths):
                img = cv2.imread(image_path)
                x1, y1, x2, y2 = crop_region
                img = img[y1:y2, x1:x2]
                # Ensure the cropped image matches the desired crop size
                if img.shape[:2] != (y2 - y1, x2 - x1):
                    raise ValueError(f"Unexpected crop size: {img.shape[:2]}, expected ({image_h}, {image_w})")
                rgb_data_single_cam.append(img)

        elif sam2_option:
            pass
            # for i, image_path in enumerate(rgb_paths):
            #     img = cv2.imread(image_path) # read as BGR
            #     if sam2_w_crop_option: # remove useless regions
            #         crop_region = crop_list[0]
            #         x1, y1, x2, y2 = crop_region
            #         img = img[y1:y2, x1:x2]
            #     img = img.transpose(2, 0, 1)  # Convert (H, W, C) -> (C, H, W)
            #     crop_images = segmentor.get_sub_images(img, slot_id=f"{get_rebar_id(episode_idx)}_{cam_name}_slot5",
            #                                                 wo_bg=sam2_wo_bg,
            #                                                bbox_height=320, bbox_width=480,
            #                                                h_shift=0,  w_shift=0, data_aug = True)   # list, img: (H, W, C)
            #     rgb_data_crop_1.append(crop_images[0])
            #     rgb_data_crop_2.append(crop_images[1])
            #     rgb_data_crop_3.append(crop_images[2])

        else:
            for i, image_path in enumerate(rgb_paths):
                img = cv2.imread(image_path)
                if img.shape[:2] != (image_h, image_w):
                    # Resize the image if dimensions do not match
                    img = cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
                rgb_data_single_cam.append(img) # img: (H, W, C)

        if sam2_option:
            pass

            # rgb_data_crop_1 = np.array(rgb_data_crop_1)
            # rgb_data_crop_2 = np.array(rgb_data_crop_2)
            # rgb_data_crop_3 = np.array(rgb_data_crop_3)
            # all_image_data[f"{cam_name}_rgb_crop1"] = rgb_data_crop_1
            # all_image_data[f"{cam_name}_rgb_crop2"] = rgb_data_crop_2
            # all_image_data[f"{cam_name}_rgb_crop3"] = rgb_data_crop_3
        else:
            rgb_data_single_cam = np.array(rgb_data_single_cam)
            all_image_data[f"{cam_name}_rgb"] = rgb_data_single_cam

        #  We didn't implement the depth map
        # if camera_data["data"][0]["depth_img_name"] != "":
        #     depth_paths = [os.path.join(input_path, images, item['depth_img_name'].split("/")[-2],
        #                               item['depth_img_name'].split("/")[-1] + ".png") for item in camera_data["data"]]
        #     depth_data_single_cam = []
        #     # depth_paths = filter_repetitives(depth_paths)  # just for debugging
        #     for i, image_path in enumerate(depth_paths):
        #         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #         depth_data_single_cam.append(img)
        #     depth_data_single_cam = np.array(depth_data_single_cam)
        #     all_image_data[f"{cam_name}_depth"] = depth_data_single_cam

    # ----------------------- target ID ----------------------- #
    target = get_target_id(episode_idx)

    # create hdf5
    if img_only:
        data_dict = {
            'action': [action_data],
            "target": [target]
        }
    else:
        data_dict = {
            'action':  [action_data],
            "target": [target],
            "observations/FT_raw": [FT_raw_data],
            "observations/FT_processed": [FT_processed_data],
            # "action_hot": [action_hot_data],
            'observations/qpos':  [ee_gripper_data],
            'observations/qvel': [np.zeros_like(ee_gripper_data)],  # 添加qvel，如果没有速度数据就用零填充
            'is_edited': [np.array([False])]  # 添加编辑标志
        }

    for idx, cam_name in enumerate(camera_names):
        if sam2_option:
            data_dict[f'/observations/images/{cam_name}_crop1'] = []
            data_dict[f'/observations/images/{cam_name}_crop1'].append(all_image_data[f"{cam_name}_rgb_crop1"])
            data_dict[f'/observations/images/{cam_name}_crop2'] = []
            data_dict[f'/observations/images/{cam_name}_crop2'].append(all_image_data[f"{cam_name}_rgb_crop2"])
            data_dict[f'/observations/images/{cam_name}_crop3'] = []
            data_dict[f'/observations/images/{cam_name}_crop3'].append(all_image_data[f"{cam_name}_rgb_crop3"])
        else:
            data_dict[f'/observations/images/{cam_name}'] = []
            data_dict[f'/observations/images/{cam_name}'].append(all_image_data[f"{cam_name}_rgb"])
            if "rs" in cam_name:
                data_dict[f'/observations/images/{cam_name}_depth'] = []
                data_dict[f'/observations/images/{cam_name}_depth'].append(all_image_data[f"{cam_name}_depth"])
    # import pdb;pdb.set_trace()
    assert action_data.shape[0] == ee_gripper_data.shape[0] == list(all_image_data.values())[0].shape[0], "Shape is incorrect"
    for idx in range(len(list(all_image_data.values()))):
        assert list(all_image_data.values())[0].shape[0] == list(all_image_data.values())[idx].shape[0], "Image numbers should be same"

    assert max_timesteps >= action_data.shape[0], f"Eposide length {action_data.shape[0]} is beyond max_timesteps {max_timesteps}"

    with h5py.File(os.path.join(output_path,f'episode_{episode_idx + hdf5_start_idx}.hdf5'), 'w') as root: # {episode_idx}
        root.attrs['sim'] = True 
        obs = root.create_group('observations')
        image = obs.create_group('images')

        for cam_name in camera_names:
            if sam2_option:
                _ = image.create_dataset(
                    f'{cam_name}_crop1',
                    (max_timesteps, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
                _ = image.create_dataset(
                    f'{cam_name}_crop2',
                    (max_timesteps, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
                _ = image.create_dataset(
                    f'{cam_name}_crop3',
                    (max_timesteps, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue = np.nan
                )
            if "rs" in cam_name:
                _ = image.create_dataset(
                    f"{cam_name}_depth",
                    (max_timesteps, image_h, image_w),  # Shape: (timesteps, height, width, channels)
                    dtype='uint16',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w),  # Chunk size for better performance
                    fillvalue = np.nan
                )

        qpos = obs.create_dataset('qpos', (max_timesteps, 7), fillvalue=np.nan) # 7-dimentional ee pose and gripper state
        qvel = obs.create_dataset('qvel', (max_timesteps, 7), fillvalue=np.nan)
        FT_raw = obs.create_dataset('FT_raw', (max_timesteps, 6), fillvalue=np.nan)
        FT_processed = obs.create_dataset('FT_processed', (max_timesteps, 6), fillvalue=np.nan)
        action = root.create_dataset('action', (max_timesteps, 7), fillvalue=np.nan)
        action_hot = root.create_dataset('action_hot', (max_timesteps, 1), fillvalue=np.nan)
        target_hot = root.create_dataset('target', (1), fillvalue=np.nan)
        is_edited = root.create_dataset('is_edited', (1), fillvalue=False)  # 添加编辑标志数据集
        
        # 添加语言指令数据集
        root.create_dataset("language_raw", data=[lang_intrs])
        # 如果有预处理的语言特征，也可以添加
        dummy_bert_features = np.zeros((1, 768))  # 假设使用768维的BERT特征
        root.create_dataset("distill_bert_lang", data=dummy_bert_features)

        for name, array in data_dict.items():
            dataset_path = name
            if name.startswith('/'):
                dataset_path = name[1:] 
            
            if dataset_path in root:
                try:
                    root[dataset_path][:array[0].shape[0], ...] = array[0]
                except Exception as e:
                    print(f"Error writing {name}: {e}")
                    raise Exception
            else:
                print(f"{dataset_path} not exist in root")
                raise NotImplementedError       

print("All data has been processed, merged, and saved.")