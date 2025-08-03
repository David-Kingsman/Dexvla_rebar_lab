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
    
    # 🔥 添加训练字段检查
    missing_language_files = []
    missing_reasoning_files = []
    valid_language_files = 0
    valid_reasoning_files = 0

    for dataset_path in tqdm(dataset_path_list):
        try:
            with h5py.File(dataset_path, 'r') as root:
                # 检查现有的核心数据
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]

                # 🔥 检查训练必需的语言字段
                if 'language_raw' in root:
                    try:
                        raw_lang = root['language_raw'][0].decode('utf-8')
                        valid_language_files += 1
                        rank0_print(f'{os.path.basename(dataset_path)}: language_raw = "{raw_lang}"')
                    except Exception as e:
                        rank0_print(f'{os.path.basename(dataset_path)}: language_raw decode error: {e}')
                        missing_language_files.append(dataset_path)
                else:
                    rank0_print(f'{os.path.basename(dataset_path)}: ❌ MISSING language_raw field')
                    missing_language_files.append(dataset_path)

                # 🔥 检查可选的推理字段
                has_reasoning = False
                if 'substep_reasonings' in root.keys():
                    try:
                        # 检查第一个有效时间步的推理
                        reasoning = root['substep_reasonings'][0].decode('utf-8')
                        rank0_print(f'{os.path.basename(dataset_path)}: substep_reasonings = "{reasoning[:50]}..."')
                        has_reasoning = True
                        valid_reasoning_files += 1
                    except Exception as e:
                        rank0_print(f'{os.path.basename(dataset_path)}: substep_reasonings decode error: {e}')
                elif 'reasoning' in root.keys():
                    try:
                        reasoning = root['reasoning'][0].decode('utf-8')
                        rank0_print(f'{os.path.basename(dataset_path)}: reasoning = "{reasoning[:50]}..."')
                        has_reasoning = True
                        valid_reasoning_files += 1
                    except Exception as e:
                        rank0_print(f'{os.path.basename(dataset_path)}: reasoning decode error: {e}')
                
                if not has_reasoning:
                    missing_reasoning_files.append(dataset_path)

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

    # 🔥 返回训练字段检查结果
    training_check_results = {
        'missing_language_files': missing_language_files,
        'missing_reasoning_files': missing_reasoning_files,
        'valid_language_files': valid_language_files,
        'valid_reasoning_files': valid_reasoning_files,
        'total_files': len(dataset_path_list)
    }

    return stats, all_episode_len, training_check_results


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
        stats, episode_lengths, training_check = result
        
        print(f"\n=== DATASET SUMMARY ===")
        print(f"Episodes: {len(episode_lengths)}")
        print(f"Total valid timesteps: {sum(episode_lengths)}")  # 🔥 显示有效时间步
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Min episode length: {min(episode_lengths)}")
        print(f"Max episode length: {max(episode_lengths)}")

        # 🔥 训练字段检查报告
        print(f"\n=== TRAINING FIELDS CHECK ===")
        print(f"Language field (language_raw):")
        print(f"  ✅ Valid files: {training_check['valid_language_files']}/{training_check['total_files']}")
        if training_check['missing_language_files']:
            print(f"  ❌ Missing language_raw in {len(training_check['missing_language_files'])} files")
            print(f"     Need to add language_raw field in process_data.py")
        
        print(f"Reasoning field (optional):")
        print(f"  ✅ Valid files: {training_check['valid_reasoning_files']}/{training_check['total_files']}")
        if training_check['missing_reasoning_files']:
            print(f"  ⚠️  Missing reasoning in {len(training_check['missing_reasoning_files'])} files")
            print(f"     Can set use_reasoning = False in training config")
        
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

        # 🔥 生成修复建议
        print(f"\n=== RECOMMENDED ACTIONS ===")
        
        if training_check['missing_language_files']:
            print(f"🔧 FIX REQUIRED - Missing language_raw field:")
            print(f"   Add this to your process_data.py:")
            print(f"   ```python")
            print(f"   # Add language instruction")
            print(f"   root.create_dataset('language_raw', (1,), dtype=h5py.string_dtype())")
            print(f"   root['language_raw'][0] = 'insert the rebar into the hole'.encode('utf-8')")
            print(f"   ```")
        else:
            print(f"✅ Language field: Ready for training!")
            
        if training_check['missing_reasoning_files']:
            print(f"⚙️  OPTIONAL - Missing reasoning field:")
            print(f"   Set in training config:")
            print(f"   ```python")
            print(f"   use_reasoning = False  # Disable reasoning functionality")
            print(f"   ```")
        else:
            print(f"✅ Reasoning field: Available for enhanced training!")
        
        print("\n✅ Data integrity check completed successfully!")
    else:
        print("❌ Data integrity check failed!")