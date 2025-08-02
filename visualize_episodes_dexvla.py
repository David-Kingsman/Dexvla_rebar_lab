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
    """ Save video frames to a video file."""
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
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
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
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                out.write(image)
            out.release()
            print(f'Saved video to: {video_path1}')

        # combine webcam videos if both are present
        if 'webcam_1' in cam_names and 'webcam_2' in cam_names:
            webcam1_video = video['webcam_1']  # [T, H, W, 3]
            webcam2_video = video['webcam_2']  # [T, H, W, 3]
            n_frames, h, w, _ = webcam1_video.shape
            combined_width = w * 2  # horizontal stack
            combined_video_path = video_path.strip(".mp4") + "_combined.mp4"
            fps = int(1 / dt)
            out = cv2.VideoWriter(combined_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (combined_width, h))
            for t in range(n_frames):
                # get frames from both cameras
                frame1 = webcam1_video[t][:, :, [2, 1, 0]]  # BGR
                frame2 = webcam2_video[t][:, :, [2, 1, 0]]  # BGR
                # horizontal stack
                combined_frame = np.concatenate([frame1, frame2], axis=1)
                out.write(combined_frame)
            out.release()
            print(f'Saved combined video to: {combined_video_path}')
            
def save_first_frame(video, output_path=None):
    """ Save the first frame for each camera."""
    if isinstance(video, dict):
        cam_names = list(video.keys())
        first_frames = []
        
        for cam_name in cam_names:
            first_frame = video[cam_name][0]  # Get the first frame for each camera. BGR!
            first_frame = first_frame[:, :, [2, 1, 0]]  # Swap B and R channels.
            first_frames.append(first_frame)
            
            # every camera save first frame
            output_path1 = output_path.strip(".jpg") + f"_{cam_name}.jpg"
            cv2.imwrite(output_path1, first_frame)

        # save combined first frame
        combined_image = np.concatenate(first_frames, axis=1)  # Combine frames horizontally.
        combined_path = output_path.strip(".jpg") + "_combined.jpg"
        cv2.imwrite(combined_path, combined_image)
        print(f'Saved first frame to: {combined_path}')
    else:
        raise ValueError("Video input must be either a list of dictionaries or a dictionary.")

def visualize_poses_quat(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    """fix the bug of quat diff"""
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim + 1 # add rotation quat diff
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot state
    all_names = [name for name in STATE_NAMES]
    for dim_idx in range(num_dim + 1):
        if dim_idx < num_dim:
            ax = axs[dim_idx]
            ax.plot(qpos[:, dim_idx], label=label1)
            ax.set_title(f'State {dim_idx}: {all_names[dim_idx]}')
            ax.legend()
        else:
            # fix the bug of quat diff
            action = command[:, 3:6]
            state = qpos[:, 3:6]
            nan_indices = np.where(np.isnan(state))
            
            # check if there are NaN values in the state array
            if len(nan_indices[0]) > 0:
                first_nan_index = nan_indices[0][0]
            else:
                # if no NaN values, use the full length of the state array
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
                        # add the last element to quat_diff
                        quat_diff[i] = 2 * np.arccos(np.clip(quat_diff_quat[-1], -1, 1))
                    except:
                        quat_diff[i] = 0

            ax = axs[dim_idx]
            ax.plot(quat_diff, label=label1)
            ax.set_title(f'State {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_FT(FT_list, plot_path=None, ylim=None, label_overwrite=None):
    label = label_overwrite if label_overwrite else "State"

    FT = np.asarray(FT_list)           # shape: (T, D)
    num_ts, num_dim = FT.shape
    h, w = 2, num_dim
    fig, axs = plt.subplots(num_dim, 1, figsize=(w, h * num_dim))

    # Make sure axs is iterable even when num_dim == 1
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
    print(f"Saved FT plot to: {plot_path}")
    plt.close()

def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

def custom_draw_geometries_with_line_width(geometries, line_width=2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    # Access rendering options
    render_option = vis.get_render_option()
    render_option.line_width = line_width  # Set the line thickness
    vis.run()
    vis.destroy_window()

def visualize_and_save_trajectory(action, save_path="trajectory.ply", marked_id=False):
    """
    Visualizes and saves 3D trajectories using Open3D.

    Args:
        action (np.ndarray or list): A numpy array with shape (300, 7) or a list of such arrays. Each row is [x, y, z, rx, ry, rz, grasp].
                                     NaN values in any row will be ignored.
        save_path (str): Path to save the trajectories as a .ply file.
    """
    all_positions = []
    geometries = []

    # Ensure action is a list of arrays
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

        # Append positions for saving
        all_positions.append(positions)
        # Create a LineSet for the trajectory
        if len(positions) > 1:
            trajectory_lines = [[i, i + 1] for i in range(len(positions) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(positions)
            line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)

            if marked_id and trajectory_idx == marked_id:
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in trajectory_lines])  # Red color
            else:
                line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in trajectory_lines])  # Blue color

            geometries.append(line_set)
            
        # Mark the starting point in red with orientation
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        start_sphere.paint_uniform_color([1, 0, 0])  # Red color
        start_sphere.translate(positions[0])
        geometries.append(start_sphere)
        
        try:
            start_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=positions[0])
            start_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles[0])
            start_frame.rotate(start_rotation, center=positions[0])
            geometries.append(start_frame)
        except:
            pass
            
        # Mark the ending point in green with orientation
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        end_sphere.paint_uniform_color([0, 1, 0])  # Green color
        end_sphere.translate(positions[-1])
        geometries.append(end_sphere)
        
        try:
            end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005, origin=positions[-1])
            end_rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles[-1])
            end_frame.rotate(end_rotation, center=positions[-1])
            geometries.append(end_frame)
        except:
            pass

    # Flatten all positions and save as a single point cloud
    if all_positions:
        all_positions_flattened = np.vstack(all_positions)
        o3d.io.write_point_cloud(save_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_positions_flattened)))

    # Add a coordinate frame at the origin with thinner axes
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    geometries.append(coordinate_frame)

    # 注释掉GUI显示避免在服务器上出错
    # o3d.visualization.draw_geometries(geometries)
    print(f'Saved trajectories to: {save_path}')

def main(args):
    if args['episode_idx'] is not False:  # 修复：正确检查episode_idx
        dataset_dir = args['dataset_dir']
        episode_idx = args['episode_idx']
        dataset_name = f'episode_{episode_idx}'

        qpos, action, image_dict, FT_raw, FT_processed = load_hdf5(dataset_dir, dataset_name) 

        if args['video']:
            DT = 0.1 # TODO: support DT in npz
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        if args['first_frame']:
            save_first_frame(image_dict,
                             output_path=os.path.join(dataset_dir, dataset_name + '_1st_frame.jpg'))
        if args['traj_action']:
            visualize_and_save_trajectory(action, save_path=os.path.join(dataset_dir, dataset_name + '_traj_action.ply'))
        if args['traj_qpos']:
            visualize_and_save_trajectory(qpos, save_path=os.path.join(dataset_dir, dataset_name + '_traj_qpos.ply'))

        visualize_poses_quat(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        visualize_FT(FT_raw, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_raw.png'))
        visualize_FT(FT_processed, plot_path=os.path.join(dataset_dir, dataset_name + '_FT_processed.png'))
    else:
        file_list = os.listdir(args['dataset_dir'])
        file_list.sort()
        action_array = []
        qpos_array = []
        marked_id = None
        id = 0
        
        for entry in file_list:
            if "hdf5" in entry:
                print(f"Processing {entry}")
                qpos, action, image_dict, FT_raw, FT_processed = load_hdf5(args['dataset_dir'], entry[:-len(".hdf5")])
                
                if args['video']:
                    DT = 0.1  # TODO: support DT in npz
                    save_videos(image_dict, DT, video_path=os.path.join(args['dataset_dir'], entry[:-len(".hdf5")] + '_video.mp4'))
                if args['first_frame']:
                    save_first_frame(image_dict, output_path=os.path.join(args['dataset_dir'], entry[:-len(".hdf5")] + '_1st_frame.jpg'))
                if args['traj_action']:
                    if entry=="episode_1.hdf5":
                        marked_id = id
                    action_array.append(action)
                if args['traj_qpos']:
                    if entry == "episode_1.hdf5":
                        marked_id = id
                    qpos_array.append(qpos)
                
                id += 1
                visualize_poses_quat(qpos, action, plot_path=os.path.join(args['dataset_dir'], entry[:-len(".hdf5")] + '_qpos.png'))
                visualize_FT(FT_raw, plot_path=os.path.join(args['dataset_dir'], entry[:-len(".hdf5")] + '_FT_raw.png'))
                visualize_FT(FT_processed, plot_path=os.path.join(args['dataset_dir'], entry[:-len(".hdf5")] + '_FT_processed.png'))
        
        if args['traj_action'] and action_array:
            visualize_and_save_trajectory(action_array, save_path=os.path.join(args['dataset_dir'],'all_traj_action.ply'), marked_id=marked_id)
        if args['traj_qpos'] and qpos_array:
            visualize_and_save_trajectory(qpos_array, save_path=os.path.join(args['dataset_dir'], 'all_traj_qpos.ply'), marked_id=marked_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', default=False, type=int, help='Episode index.', required=False)
    parser.add_argument('--video', action='store_true', help='Generate video', required=False)
    parser.add_argument('--first_frame', action='store_true', help='Episode index.', required=False)
    parser.add_argument('--traj_action', action='store_true', help='Episode index.', required=False)
    parser.add_argument('--traj_qpos', action='store_true', help='Episode index.', required=False)
    main(vars(parser.parse_args()))