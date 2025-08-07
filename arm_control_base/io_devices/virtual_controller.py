
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class VirtualController:
    def __init__(
        self,
        right_controller: bool = True,
        # pos_action_gain: float = 100, # 150
        # rot_action_gain: float = 25, # 5
        # gripper_action_gain: float = 3,
    ):
        # self.pos_action_gain = pos_action_gain
        # self.rot_action_gain = rot_action_gain
        # self.gripper_action_gain = gripper_action_gain
        self.controller_id = "r" if right_controller else "l"

        self._enabled = False
        self._stop_thread = False  # Flag to control the thread

        self._state = {
            "init_pose": None,
            "last_pose": None,
            "buttons": {"A": False, "B": False, "RG": False}, # TODO: name
            "movement_enabled": False,
            "controller_on": True,
        }
        self._robot_init_ee = None
        self._stop_record = False
        self._origin_controller_pose = {}

        # delta_poses we will conduct
        self.delta_pos = np.array([0.0, 0.0, 0.0])
        self.delta_axis_angle = np.array([0.0, 0.0, 0.0])

        self.thread = threading.Thread(target=self._update_internal_state)
        self.thread.daemon = True
        self.thread.start()
        self.initial_time = time.time()


    def start_control(self, init_ee_pose):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._robot_init_ee = init_ee_pose
        self._enabled = True

    def stop_control(self):
        """Stop the internal thread cleanly."""
        self._enabled = False
        self._stop_thread = True
        self.thread.join()  # Wait for the thread to finish
        # print("Thread has fully exited.")

    def interpolate_pose(self, start_pose, end_pose, alpha):
        # Extract translation and rotation from start and end poses
        start_translation = start_pose[:3, 3]
        end_translation = end_pose[:3, 3]

        # Interpolate translation linearly
        interpolated_translation = start_translation * (1 - alpha) + end_translation * alpha

        # Extract rotation matrices and convert to quaternions
        # start_rotation = R.from_matrix(start_pose[:3, :3])
        # end_rotation = R.from_matrix(end_pose[:3, :3])
        # start_quaternion = start_rotation.as_quat()
        # end_quaternion = end_rotation.as_quat()

        # # Interpolate rotation using SLERP
        # interpolated_quaternion = Slerp([0, 1], [start_quaternion, end_quaternion])([alpha])
        # interpolated_rotation = R.from_quat(interpolated_quaternion).as_matrix()

        # Combine interpolated rotation and translation into a single pose matrix
        interpolated_pose = np.eye(4)

        interpolated_pose[:3, :3] = start_pose[:3, :3]
        interpolated_pose[:3, 3] = interpolated_translation
        return interpolated_pose

    def get_transformations_and_buttons(self):
        # Command: move on some direction slowly
        poses_start = {"r": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])}
        poses_end = {"r": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.12], [0, 0, 0, 1]])} # x left-right,y up-down,z

        elapsed_time = time.time() - self.initial_time
        poses = {}
        if elapsed_time <= 6: # the action will be 3 seconds
            alpha = elapsed_time / 6  # Interpolation factor
            for key in poses_start:
                poses[key] = poses[key] = self.interpolate_pose(poses_start[key], poses_end[key], alpha)
        else:
            poses = {}  # Reset poses to empty after duration
        buttons = {"RTr": True, "A": False, "RG": False, "B": False}
        return poses, buttons

    def _update_internal_state(self, hz=50):

        last_read_time = time.time()

        while not self._stop_thread:

            if self._enabled:
            # Regulate Read Frequency #
                time.sleep(1 / hz)

                # Read Controller
                poses, buttons = self.get_transformations_and_buttons()

                if poses == {}:
                    continue
                # print(buttons)
                coord_rot = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                # coord_rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                curr_pose = coord_rot @ np.asarray(poses[self.controller_id])

                if self._state['init_pose'] is None:
                    self._state['init_pose'] = curr_pose

                self._state["last_pose"] = curr_pose
                self._state["buttons"] = buttons
                # print("1")

    def get_controller_state(self):
        target_pos = self._state["last_pose"][:3, 3] - self._state["init_pose"][:3, 3] + self._robot_init_ee[:3, 3]
        target_rot = self._state["last_pose"][:3, :3] @ self._state["init_pose"][:3, :3].T @ self._robot_init_ee[:3, :3]
        # A = self._state["init_pose"][:3, :3]
        # print(f"meta: {A}")
        # print(f"target_rot: {target_rot}")
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_pos
        target_pose[:3, :3] = target_rot

        if self._stop_record == False:
            self._stop_record = self._state["buttons"]["B"]

        return {
            "target_pose": target_pose,
            "grasp": self._state["buttons"]["RTr"],
            "reset": self._state["buttons"]["A"],
            "action_hot": self._state["buttons"]["RG"], # TODO: name RG RJ RThu  rightJS rightTrig  rightGrip
            "stop_record": self._stop_record,
        }