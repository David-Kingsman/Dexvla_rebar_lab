import cv2
import numpy as np
# import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation

def mat2pos_axis(mat):
    pose_6D = np.zeros(6)
    mat = mat.reshape(4, 4) # , order='F'
    rot_mat = mat[:3, :3]
    trans_vec = mat[:3, 3]
    rot = Rotation.from_matrix(rot_mat)
    rot_vec = rot.as_rotvec()
    pose_6D[:3] = trans_vec
    pose_6D[3:] = rot_vec
    return pose_6D

def pos_axis2mat(pos_axis_angle):
    mat = np.eye(4)
    rot = Rotation.from_rotvec(pos_axis_angle[3:6])
    rot_mat = rot.as_matrix()
    mat[:3, :3] = rot_mat
    mat[:3, 3] = pos_axis_angle[:3]
    return  mat


def mat4_to_6d(mat_4x4: np.ndarray) -> np.ndarray:
    # Extract translation
    translation = mat_4x4[0:3, 3]

    # Extract rotation matrix
    rotation_matrix = mat_4x4[0:3, 0:3]

    # Convert rotation matrix to rotvec
    rotation = Rotation.from_matrix(rotation_matrix)
    rotvec = rotation.as_rotvec()

    # Combine translation and rotation vector into a single 6D vector
    vec_6d = np.hstack([translation, rotvec])
    return vec_6d


def vect6d_to_mat4(vec_6d: np.ndarray) -> np.ndarray:
    # Split the vector into translation and rotvec
    translation = vec_6d[0:3]
    rotvec = vec_6d[3:6]

    # Convert rotvec to rotation matrix
    rotation_matrix = Rotation.from_rotvec(rotvec).as_matrix()

    # Form the 4x4 homogeneous transform
    mat_4x4 = np.eye(4)
    mat_4x4[0:3, 0:3] = rotation_matrix
    mat_4x4[0:3, 3] = translation

    return mat_4x4


def convert_pose_rep(poses, ref_pose, type="relative", backward=False):
    """
    poses,   # n x 6 array
    ref_pose, # 1 x 6 array
    type, # "relative"
    return: n x 6 array, relative xyz and 3 axis angels in terms of ref_pose
    """

    if not backward:
        if type == "relative":
            initial_xyz = ref_pose[:3]
            initial_axis_angle = ref_pose[3:6]
            initial_quat = Rotation.from_rotvec(initial_axis_angle)
            relative_pose = poses.copy()
            for i in range(poses.shape[0]):
                # xyz
                relative_pose[i, :3] = initial_quat.inv().apply(
                    (relative_pose[i, :3] - initial_xyz))  # TODO: Unify all the functions
                # axis angles
                rot_quat_i = Rotation.from_rotvec(relative_pose[i, 3:6])
                quat_diff = initial_quat.inv() * rot_quat_i
                relative_pose[i, 3:6] = quat_diff.as_rotvec()
        else:
            raise NotImplementedError
        return relative_pose
    else:
        if len(poses.shape) != 1:
            if type == "relative":
                initial_xyz = ref_pose[:3]
                initial_axis_angle = ref_pose[3:6]
                initial_quat = Rotation.from_rotvec(initial_axis_angle)

                abs_pose = np.zeros((poses.shape[0], 6))
                for i in range(poses.shape[0]):
                    # xyz
                    abs_pose[i, :3] = (initial_quat.apply(poses[i, :3])) + initial_xyz
                    # axis angles
                    rot_quat_i = Rotation.from_rotvec(poses[i, 3:6])
                    abs_quat = initial_quat * rot_quat_i
                    abs_pose[i, 3:6] = abs_quat.as_rotvec()
            else:
                raise NotImplementedError
        else:
            if type == "relative":
                initial_xyz = ref_pose[:3]
                initial_axis_angle = ref_pose[3:6]
                initial_quat = Rotation.from_rotvec(initial_axis_angle)

                abs_pose = poses.copy()
                # xyz
                abs_pose[:3] = (initial_quat.apply(poses[:3])) + initial_xyz
                # axis angles
                rot_quat_i = Rotation.from_rotvec(poses[3:6])
                abs_quat = initial_quat * rot_quat_i
                abs_pose[3:6] = abs_quat.as_rotvec()
            else:
                raise NotImplementedError
        return abs_pose

def convert_pose_mat_rep(poses, ref_pose, type="relative", backward=False):
    """
    poses,   # 4 x 4 array
    ref_pose, # 4 x 4 array
    type, # "relative"
    """
    if not backward:
        if type == "relative":
            # diff = np.zeros([4,4])

            diff = np.linalg.inv(ref_pose) @ poses # * is Elementwise Multiplication
        else:
            raise NotImplementedError
        return diff

    else:
        if type == "relative":
            diff = poses
            abs_pose = ref_pose @ diff
        else:
            raise NotImplementedError

        return abs_pose

# # Wrong implementation
# def convert_axis_angle_singularity(rot_vec):
#     A = np.array(
#         [[1, 0, 0, 0],
#          [0, -1, 0, 0.0],
#          [0, 0, -1, 0.0],
#          [0, 0, 0, 1.0]])
#     rot = Rotation.from_rotvec(rot_vec)
#     rot_mat = rot.as_matrix()
#     rot_mat_new = A[:3, :3] @ rot_mat  # A.T
#     rot_new = Rotation.from_matrix(rot_mat_new)
#     rot_vec_new = rot_new.as_rotvec()  # axis angle
#     return rot_vec_new

def convert_pos_axis_angle_singularity(pos_angles):
    assert pos_angles.shape[0] == 6
    A = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0.0],
         [0, 0, -1, 0.0],
         [0, 0, 0, 1.0]])
    mat = pos_axis2mat(pos_angles)
    mat_new = A @ mat  # A.T
    # return mat_new
    pos_angles_new = mat2pos_axis(mat_new)
    return pos_angles_new

def multi_A(b):
    b_mat = pos_axis2mat(b)
    # B= np.array(
    #     [[1, 0, 0, 0],
    #      [0, -1, 0, 0],
    #      [0, 0, -1, 0.0],
    #      [0, 0, 0, 1.0]])
    B= np.array(
        [[0.93019474,  0.31288531,  0.19193884, - 1.58713528],
         [0.31288531, - 0.40243336, - 0.86031981, - 3.64825393],
         [-0.19193884,  0.86031981, - 0.47223862,  1.054156],
        [0.,0.,0,1.]])
    b_mat = B @ b_mat

    return b_mat

if __name__ == "__main__":
    # a = np.array([1.0,2.0,3.0, 1.5,1.6,0.7]) # init
    # # b = np.array([-0.1, -0.2, -0.3, -0.1, 0.42, 0.1])
    # b = np.array([-0.1, -0.2, -0.3, -1.5, -1.6, 0.7])
    # # a_new = convert_pos_axis_angle_singularity(a)
    # # b_new = convert_pos_axis_angle_singularity(b)
    # # a_mat = pos_axis2mat(a_new)
    # # b_mat = pos_axis2mat(b_new)
    #
    # # a_mat = convert_pos_axis_angle_singularity(a)
    # # b_mat = convert_pos_axis_angle_singularity(b)
    #
    # # diff =  np.linalg.inv(a_mat) @ b_mat
    # # print(diff)
    #
    # c = multi_A(a)
    # d = multi_A(b)
    # diff1 =  np.linalg.inv(c) @ d
    # print(diff1)
    #
    # # e = convert_pos_axis_angle_singularity(a)
    # # f = convert_pos_axis_angle_singularity(b)
    #
    # e =  mat2pos_axis(c)
    # f =  mat2pos_axis(d)
    # h =  pos_axis2mat(e)
    # g =  pos_axis2mat(f)
    #
    # # e =  mat4_to_6d(c)
    # # f =  mat4_to_6d(d)
    # #
    # # h =  vect6d_to_mat4(e)
    # # g =  vect6d_to_mat4(f)
    #
    # diff2 =  np.linalg.inv(h) @ g
    # print(diff2)

    a = np.array([0.41538592,  0.30989957, - 0.33393304, - 1.5581526, - 0.05739241,  0.31106314])
    a_A = convert_pos_axis_angle_singularity(a)
    print(a_A)

def action_interpolation(actions: np.ndarray, target_T: int) -> np.ndarray:
    """
    Upsample a batch of actions from shape (B, T_old, 7)
    to shape (B, target_T, 7) using linear interpolation.

    Parameters
    ----------
    actions   : np.ndarray
        Shape (B, T_old, 7).  Order per action:
        [x, y, z, rx, ry, rz, grasp]
        rx,ry,rz are axis-angle (rad) *absolute* rotations.
    target_T  : int
        Desired number of timesteps after interpolation
        (e.g. 50 when you go from 10 Hz to 50 Hz).

    Returns
    -------
    np.ndarray
        Shape (B, target_T, 7)
    """
    B, T_old, D = actions.shape
    if D != 7:
        raise ValueError("actions must have last-dimension size 7")

    if T_old == target_T:               # nothing to do
        return actions.copy()

    # -- construct old/new time axes on [0, 1] --
    t_old = np.linspace(0.0, 1.0, T_old)
    t_new = np.linspace(0.0, 1.0, target_T)

    # allocate output
    out = np.empty((B, target_T, 7), dtype=actions.dtype)

    # ---- 1. xyz & rx,ry,rz: plain linear interpolation ----
    for dim in range(6):                # dims 0…5
        # np.interp is 1-D, so we loop over batch
        for b in range(B):
            out[b, :, dim] = np.interp(t_new, t_old, actions[b, :, dim])

    # ---- 2. grasp: forward-fill (step) so it stays 0/1 ----
    #   alternative: round(linear_interp) if you prefer a soft transition
    grasp_out = np.empty((B, target_T), dtype=actions.dtype)
    for b in range(B):
        # indices where grasp changes in original sequence
        step_func = actions[b, :, 6]
        # forward-fill by taking the value of the nearest earlier keyframe
        indices = np.searchsorted(t_old, t_new, side="right") - 1
        indices = np.clip(indices, 0, T_old - 1)
        grasp_out[b] = step_func[indices]
    out[:, :, 6] = grasp_out

    return out




    """
[[ 0.74917504  0.6619229   0.02438919 -1.1374613 ]
 [-0.59957197  0.66203596  0.44969082 -2.9674228 ]
 [ 0.28151413 -0.35152022  0.89285124 -2.61545095]
 [ 0.          0.          0.          1.        ]]
 
 [[ 0.90393801  0.06863831  0.42211948 -0.54606934]
 [-0.40395398  0.46111689  0.79005847 -4.05056093]
 [-0.14041815 -0.88468073  0.4445478  -0.48452497]
 [ 0.          0.          0.          1.        ]]
 
 [[ 0.74917504  0.6619229   0.02438919 -1.1374613 ]
 [-0.59957197  0.66203596  0.44969082 -2.9674228 ]
 [ 0.28151413 -0.35152022  0.89285124 -2.61545095]
 [ 0.          0.          0.          1.        ]]
    """
