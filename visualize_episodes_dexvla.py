import os
import numpy as np
import cv2
import h5py
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
# from constants import DT
from data_preprocess_scripts.utils.transform_utils import axisangle2quat,quat_distance
import IPython
e = IPython.embed

# OSC control
STATE_NAMES = ["x", "y", "z", "axis angle x", "axis angle y", "axis angle z", "gripper", "quat diff"]

def load_hdf5(dataset_dir, dataset_name):
    """Load HDF5 dataset"""
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        FT_raw = root['/observations/FT_raw'][()]
        FT_processed = root['/observations/FT_processed'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, action, image_dict, FT_raw, FT_processed

def save_videos(video, dt, video_path=None):
    """Save video frames as video file"""
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # Swap B and R channels
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Video saved to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())

        # Save video for each camera separately
        for cam_name in cam_names:
            cam_video = video[cam_name]  # [T, H, W, 3]
            video_path1 = video_path.strip(".mp4") + f"_{cam_name}.mp4"
            
            n_frames, h, w, _ = cam_video.shape
            fps = int(1 / dt)
            out = cv2.VideoWriter(video_path1, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for t in range(n_frames):
                image = cam_video[t]
                image = image[:, :, [2, 1, 0]]  # Swap B and R channels
                out.write(image)
            out.release()
            print(f'Video saved to: {video_path1}')

        # If both cameras exist, create combined video
        if 'webcam_1' in cam_names and 'webcam_2' in cam_names:
            webcam1_video = video['webcam_1']  # [T, H, W, 3]
            webcam2_video = video['webcam_2']  # [T, H, W, 3]
            n_frames, h, w, _ = webcam1_video.shape
            combined_width = w * 2  # Horizontal concatenation
            combined_video_path = video_path.strip(".mp4") + "_combined.mp4"
            fps = int(1 / dt)
            out = cv2.VideoWriter(combined_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (combined_width, h))
            for t in range(n_frames):
                # Get frames from both cameras
                frame1 = webcam1_video[t][:, :, [2, 1, 0]]  # BGR
                frame2 = webcam2_video[t][:, :, [2, 1, 0]]  # BGR
                # Horizontal concatenation
                combined_frame = np.concatenate([frame1, frame2], axis=1)
                out.write(combined_frame)
            out.release()
            print(f'Combined video saved to: {combined_video_path}')
            
def save_first_frame(video, output_path=None):
    """Save first frame of each camera"""
    if isinstance(video, dict):
        cam_names = list(video.keys())
        first_frames = []
        
        for cam_name in cam_names:
            first_frame = video[cam_name][0]  # Get first frame of each camera. BGR format!
            first_frame = first_frame[:, :, [2, 1, 0]]  # Swap B and R channels
            first_frames.append(first_frame)
            
            # Save first frame for each camera
            output_path1 = output_path.strip(".jpg") + f"_{cam_name}.jpg"
            cv2.imwrite(output_path1, first_frame)

        # Save combined first frame
        combined_image = np.concatenate(first_frames, axis=1)  # Horizontal concatenation
        combined_path = output_path.strip(".jpg") + "_combined.jpg"
        cv2.imwrite(combined_path, combined_image)
        print(f'First frame saved to: {combined_path}')
    else:
        raise ValueError("Video input must be a dictionary or list of dictionaries")

def visualize_poses_quat(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    """Fix quaternion difference bug"""
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim + 1  # Add rotation quaternion difference
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # Plot states
    all_names = [name for name in STATE_NAMES]
    for dim_idx in range(num_dim + 1):
        if dim_idx < num_dim:
            ax = axs[dim_idx]
            ax.plot(qpos[:, dim_idx], label=label1)
            ax.set_title(f'State {dim_idx}: {all_names[dim_idx]}')
            ax.legend()
        else:
            # Fix quaternion difference bug
            action = command[:, 3:6]
            state = qpos[:, 3:6]
            nan_indices = np.where(np.isnan(state))
            
            # Check if there are NaN values in state array
            if len(nan_indices[0]) > 0:
                first_nan_index = nan_indices[0][0]
            else:
                # If no NaN values, use full length of state array
                first_nan_index = len(state)
            
            quat_diff = np.zeros(first_nan_index)
            for i in range(first_nan_index):
                if i == first_nan_index - 1:
                    quat_diff[i] = 0
                else:
                    try:
                        action_quat = axisangle2quat(action[i,:])
                        state_quat = axisangle2quat(state[i+1, :])
                        quat_diff_quat = quat_distance(action_quat, state_quat)
                        # Add last element to quat_diff
                        quat_diff[i] = 2 * np.arccos(np.clip(quat_diff_quat[-1], -1, 1))
                    except:
                        quat_diff[i] = 0

            ax = axs[dim_idx]
            ax.plot(quat_diff, label=label1)
            ax.set_title(f'State {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

    # Plot robot commands
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Pose plot saved to: {plot_path}')
    plt.close()

def visualize_FT(FT_list, plot_path=None, ylim=None, label_overwrite=None):
    """Visualize force-torque sensor data"""
    label = label_overwrite if label_overwrite else "State"

    FT = np.asarray(FT_list)  # Shape: (T, D)
    num_ts, num_dim = FT.shape
    h, w = 2, num_dim
    fig, axs = plt.subplots(num_dim, 1, figsize=(w, h * num_dim))

    # Ensure axs is iterable even when num_dim == 1
    if num_dim == 1:
        axs = [axs]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(FT[:, dim_idx], label=label)
        ax.set_title(f"Dimension {dim_idx}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Force-torque plot saved to: {plot_path}")
    plt.close()

def visualize_and_save_trajectory(action, save_path="trajectory.ply", marked_id=False):
    """
    Visualize and save 3D trajectory using Open3D
    
    Args:
        action (np.ndarray or list): Numpy array of shape (300, 7) or list of such arrays. 
                                     Each row is [x, y, z, rx, ry, rz, grasp].
                                     Any rows with NaN values will be ignored.
        save_path (str): Path to save trajectory as .ply file.
    """
    all_positions = []
    geometries = []

    # Ensure action is list of arrays
    if isinstance(action, np.ndarray):
        action = [action]

    # Process each trajectory
    for trajectory_idx, traj in enumerate(action):
        # Remove rows with NaN values
        valid_action = traj[~np.isnan(traj).any(axis=1)]
        if len(valid_action) == 0:
            continue
            
        # Extract positions and orientations
        positions = valid_action[:, :3]  # [x, y, z]
        axis_angles = valid_action[:, 3:6]  # [rx, ry, rz]

        # Add positions for saving
        all_positions.append(positions)
        # Create LineSet for trajectory
        if len(positions) > 1:
            trajectory_lines = [[i, i + 1] for i in range(len(positions) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(positions)
            line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)

            if marked_id and trajectory_idx == marked_id:
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in trajectory_lines])  # Red
            else:
                line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in trajectory_lines])  # Blue

            geometries.append(line_set)
            
        # Mark start point and orientation with red
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        start_sphere.paint_uniform_color([1, 0, 0])  # Red
        start_sphere.translate(positions[0])
        geometries.append(start_sphere)
        
        try:
            start_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=positions[0])
            start_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles[0])
            start_frame.rotate(start_rotation, center=positions[0])
            geometries.append(start_frame)
        except:
            pass
            
        # Mark end point and orientation with green
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        end_sphere.paint_uniform_color([0, 1, 0])  # Green
        end_sphere.translate(positions[-1])
        geometries.append(end_sphere)
        
        try:
            end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005, origin=positions[-1])
            end_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles[-1])
            end_frame.rotate(end_rotation, center=positions[-1])
            geometries.append(end_frame)
        except:
            pass

    # Flatten all positions and save as single point cloud
    if all_positions:
        all_positions_flattened = np.vstack(all_positions)
        o3d.io.write_point_cloud(save_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_positions_flattened)))

    # Add coordinate frame at origin with thinner axes
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    geometries.append(coordinate_frame)

    print(f'Trajectory saved to: {save_path}')

def visualize_language_info(language_raw, distill_bert_lang, plot_path=None):
    """Visualize language instruction information"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Display language instruction
    ax1.text(0.5, 0.5, f"Language Instruction:\n{language_raw[0].decode('utf-8')}", 
             ha='center', va='center', fontsize=14, wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("Language Instruction")

    # Display BERT feature heatmap
    if distill_bert_lang.shape[0] > 0:
        bert_features = distill_bert_lang[0].reshape(-1, 1)
        im = ax2.imshow(bert_features.T, cmap='viridis', aspect='auto')
        ax2.set_title("BERT Language Features Heatmap")
        ax2.set_xlabel("Feature Dimension")
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Language info saved to: {plot_path}')
    plt.close()

def visualize_action_state_comparison(qpos, action, plot_path=None):
    """Visualize comparison between action commands and actual states"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'Gripper']
    
    for i in range(7):
        ax = axes[i]
        ax.plot(qpos[:, i], label='Actual State', alpha=0.8, linewidth=2)
        ax.plot(action[:, i], label='Action Command', alpha=0.8, linewidth=2, linestyle='--')
        ax.set_title(f'{labels[i]} - Action vs State')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Calculate and plot position error
    ax = axes[7]
    error = np.linalg.norm(qpos[:, :3] - action[:, :3], axis=1)  # XYZ position error
    ax.plot(error, label='Position Error (XYZ)', color='red', linewidth=2)
    ax.set_title('Position Tracking Error')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Action-state comparison saved to: {plot_path}')
    plt.close()

def visualize_image_grid(image_dict, episode_name, output_dir, step_interval=10):
    """Create image grid showing key frames from episode, sampling every 10 steps"""
    for cam_name, images in image_dict.items():
        if len(images) > 0:
            n_images = len(images)
            
            # Modified: Sample every 10 steps until NaN or max value reached
            selected_indices = []
            for step in range(step_interval, min(n_images, 300), step_interval):  # 10, 20, 30, ..., 290
                if step < n_images:
                    # Check if image at this step is valid (not all zeros or NaN)
                    img = images[step]
                    if not (np.all(img == 0) or np.any(np.isnan(img))):
                        selected_indices.append(step)
                    else:
                        break  # Stop when invalid image encountered
            
            if not selected_indices:
                print(f"Warning: {episode_name} - {cam_name} no valid image frames found")
                continue
                
            # Limit to maximum 10 key frames
            selected_indices = selected_indices[:10]
            
            fig, axes = plt.subplots(1, len(selected_indices), figsize=(3*len(selected_indices), 4))
            if len(selected_indices) == 1:
                axes = [axes]
            
            for i, idx in enumerate(selected_indices):
                img = images[idx][:, :, [2, 1, 0]]  # BGR to RGB
                axes[i].imshow(img)
                axes[i].set_title(f'Step {idx}')
                axes[i].axis('off')
            
            plt.suptitle(f'{episode_name} - {cam_name} Key Frames (every {step_interval} steps)')
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f'{episode_name}_{cam_name}_grid.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'Image grid saved to: {output_path}')
            plt.close()

def main(args):
    if args['episode_idx'] is not False:
        # Single episode processing
        dataset_dir = args['dataset_dir']
        episode_idx = args['episode_idx']
        dataset_name = f'episode_{episode_idx}'

        qpos, action, image_dict, FT_raw, FT_processed = load_hdf5(dataset_dir, dataset_name)
        
        # Load language information
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            language_raw = root['language_raw'][()]
            distill_bert_lang = root['distill_bert_lang'][()]

        # Original visualizations
        if args['video']:
            DT = 0.1
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        if args['first_frame']:
            save_first_frame(image_dict, output_path=os.path.join(dataset_dir, dataset_name + '_1st_frame.jpg'))
        if args['traj_action']:
            visualize_and_save_trajectory(action, save_path=os.path.join(dataset_dir, dataset_name + '_traj_action.ply'))
        if args['traj_qpos']:
            visualize_and_save_trajectory(qpos, save_path=os.path.join(dataset_dir, dataset_name + '_traj_qpos.ply'))

        # Original plots
        visualize_poses_quat(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        visualize_FT(FT_raw, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_raw.png'))
        visualize_FT(FT_processed, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_processed.png'))
        
        # New visualizations (removed statistics plot)
        visualize_language_info(language_raw, distill_bert_lang, 
                               plot_path=os.path.join(dataset_dir, dataset_name + '_language.png'))
        # visualize_action_state_comparison(qpos, action, 
        #                                 plot_path=os.path.join(dataset_dir, dataset_name + '_action_comparison.png'))
        # Removed statistics plot generation
        # visualize_episode_statistics(qpos, action, FT_raw, FT_processed,
        #                             plot_path=os.path.join(dataset_dir, dataset_name + '_statistics.png'))
        
        if args.get('image_grid', False):
            visualize_image_grid(image_dict, dataset_name, dataset_dir)

    else:
        # Process all episodes
        dataset_dir = args['dataset_dir']
        
        # Get all hdf5 files
        file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
        file_list.sort()
        
        print(f"Found {len(file_list)} episode files")
        
        all_action_trajectories = []
        all_qpos_trajectories = []
        
        for file_name in file_list:
            dataset_name = file_name.replace('.hdf5', '')
            print(f"Processing {dataset_name}...")
            
            try:
                qpos, action, image_dict, FT_raw, FT_processed = load_hdf5(dataset_dir, dataset_name)
                
                # Load language information
                dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
                with h5py.File(dataset_path, 'r') as root:
                    language_raw = root['language_raw'][()]
                    distill_bert_lang = root['distill_bert_lang'][()]

                # Generate visualizations for each episode
                if args['video']:
                    DT = 0.1
                    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video'))
                
                if args['first_frame']:
                    save_first_frame(image_dict, output_path=os.path.join(dataset_dir, dataset_name + '_first_frame.jpg'))
                
                if args['traj_action']:
                    visualize_and_save_trajectory(action, save_path=os.path.join(dataset_dir, dataset_name + '_traj_action.ply'))
                    all_action_trajectories.append(action)
                
                if args['traj_qpos']:
                    visualize_and_save_trajectory(qpos, save_path=os.path.join(dataset_dir, dataset_name + '_traj_qpos.ply'))
                    all_qpos_trajectories.append(qpos)

                # Generate plots
                visualize_poses_quat(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
                visualize_FT(FT_raw, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_raw.png'))
                visualize_FT(FT_processed, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_processed.png'))
                
                # New visualizations (removed statistics plot)
                visualize_language_info(language_raw, distill_bert_lang, 
                                       plot_path=os.path.join(dataset_dir, dataset_name + '_language.png'))
                # visualize_action_state_comparison(qpos, action, 
                #                                 plot_path=os.path.join(dataset_dir, dataset_name + '_action_comparison.png'))
                # Removed statistics plot generation
                # visualize_episode_statistics(qpos, action, FT_raw, FT_processed,
                #                             plot_path=os.path.join(dataset_dir, dataset_name + '_statistics.png'))
                
                if args.get('image_grid', False):
                    visualize_image_grid(image_dict, dataset_name, dataset_dir)
                    
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue
        
        # Generate combined visualization of all trajectories
        # if args['traj_action'] and all_action_trajectories:
        #     print("Generating combined action trajectory plot...")
        #     visualize_and_save_trajectory(all_action_trajectories, 
        #                                 save_path=os.path.join(dataset_dir, 'all_episodes_traj_action.ply'))
        
        # if args['traj_qpos'] and all_qpos_trajectories:
        #     print("Generating combined state trajectory plot...")
        #     visualize_and_save_trajectory(all_qpos_trajectories, 
        #                                 save_path=os.path.join(dataset_dir, 'all_episodes_traj_qpos.ply'))
        
        print(f"✅ Completed visualization of all {len(file_list)} episodes!")

# Command line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset directory', required=True)
    parser.add_argument('--episode_idx', action='store', default=False, type=int, help='Episode index', required=False)
    parser.add_argument('--video', action='store_true', help='Generate video', required=False)
    parser.add_argument('--first_frame', action='store_true', help='Save first frame', required=False)
    parser.add_argument('--traj_action', action='store_true', help='Save action trajectory', required=False)
    parser.add_argument('--traj_qpos', action='store_true', help='Save position trajectory', required=False)
    parser.add_argument('--image_grid', action='store_true', help='Create image grid', required=False)
    main(vars(parser.parse_args()))