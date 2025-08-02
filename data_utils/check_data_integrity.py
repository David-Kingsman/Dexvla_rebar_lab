'''
- Traverse all task directories and collect all HDF5 data file paths.
- Read the /observations/qpos and /action data from each file sequentially and concatenate them into a large array.
- Calculate the mean, standard deviation, minimum, and maximum values of the actions and states, and perform extreme clipping to prevent normalization errors caused by zero standard deviation.
- Return these statistics for subsequent training and data preprocessing.
'''

from utils import find_all_hdf5, flatten_list
import os
import torch
import h5py
import numpy as np
from tqdm import tqdm


def get_norm_stats(dataset_path_list, rank0_print=print):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in tqdm(dataset_path_list):
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]

                # 🔥 过滤掉NaN值 - 只保留有效数据
                valid_mask = ~(np.isnan(qpos).any(axis=1) | np.isnan(action).any(axis=1))
                
                if valid_mask.sum() == 0:
                    rank0_print(f'Warning: No valid data in {dataset_path}')
                    continue
                
                # 只保留有效的数据点
                qpos_valid = qpos[valid_mask]
                action_valid = action[valid_mask]
                
                rank0_print(f'{os.path.basename(dataset_path)}: {len(qpos_valid)}/{len(qpos)} valid timesteps')
                
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_norm_stats')
            rank0_print(e)
            continue

        all_qpos_data.append(torch.from_numpy(qpos_valid))
        all_action_data.append(torch.from_numpy(action_valid))
        all_episode_len.append(len(qpos_valid))  # 记录实际的episode长度
    
    if not all_qpos_data:
        rank0_print("No valid data found!")
        return None, None

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # check data integrity
    rank0_print(f"Merged data shapes: qpos {all_qpos_data.shape}, action {all_action_data.shape}")
    rank0_print(f"Data contains NaN: qpos={torch.isnan(all_qpos_data).any()}, action={torch.isnan(all_action_data).any()}")

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": all_qpos_data[0].numpy()
    }

    return stats, all_episode_len


if __name__ == "__main__":
    path = "/home/zekaijin/DexVLA/data/rebar_insertion_hdf5"
    print(f"Checking dataset directory: {path}")

    # Collect all HDF5 files in the specified directory
    dataset_path_list = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith('.hdf5'):
                dataset_path_list.append(os.path.join(path, file))
        dataset_path_list.sort()
    
    if not dataset_path_list:
        print(f"No HDF5 files found in {path}")
        exit(1)

    print(f"Found {len(dataset_path_list)} HDF5 files:")
    # Print the first few files found
    for f in dataset_path_list[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(dataset_path_list) > 5:
        print(f"  ... and {len(dataset_path_list) - 5} more files")

    result = get_norm_stats(dataset_path_list)
    
    if result[0] is not None:
        stats, episode_lengths = result
        
        print(f"\n=== DATASET SUMMARY ===")
        print(f"Episodes: {len(episode_lengths)}")
        print(f"Total valid timesteps: {sum(episode_lengths)}")  # 🔥 显示有效时间步
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Min episode length: {min(episode_lengths)}")
        print(f"Max episode length: {max(episode_lengths)}")
        
        print(f"\n=== ACTION STATISTICS ===")
        for i in range(len(stats['action_mean'])):
            print(f"Dim {i}: mean={stats['action_mean'][i]:.4f}, "
                  f"std={stats['action_std'][i]:.4f}, "
                  f"range=[{stats['action_min'][i]:.4f}, {stats['action_max'][i]:.4f}]")
        
        print(f"\n=== QPOS STATISTICS ===")
        for i in range(len(stats['qpos_mean'])):
            print(f"Dim {i}: mean={stats['qpos_mean'][i]:.4f}, "
                  f"std={stats['qpos_std'][i]:.4f}")
        
        # save statistics to a file
        stats_file = os.path.join(path, "dataset_stats.npz")
        np.savez(stats_file, **stats, episode_lengths=episode_lengths)
        print(f"\nStatistics saved to: {stats_file}")
        
        print("\n✅ Data integrity check completed successfully!")
    else:
        print("❌ Data integrity check failed!")