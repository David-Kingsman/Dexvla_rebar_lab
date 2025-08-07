import time
import threading
import numpy as np
from io_devices.oculus_reader.oculus_reader.reader import OculusReader

class Meta_quest2:
    def __init__(
        self,
        right_controller: bool = True,
        # pos_action_gain: float = 100, # 150
        # rot_action_gain: float = 25, # 5
        # gripper_action_gain: float = 3,
    ):
        self.oculus_reader = OculusReader()
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

    def _update_internal_state(self, hz=100):
        
        last_read_time = time.time()

        while not self._stop_thread:

            if self._enabled:
            # Regulate Read Frequency #
                time.sleep(1 / hz)

                # Read Controller
                poses, buttons = self.oculus_reader.get_transformations_and_buttons()
                # print(poses)

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
        # meta quest default unit m, xarm: mm
        # print(self._state["last_pose"][:3, 3])
        target_pos = 1000 * (self._state["last_pose"][:3, 3] - self._state["init_pose"][:3, 3]) + self._robot_init_ee[:3, 3]
        target_rot = self._state["last_pose"][:3, :3] @ self._state["init_pose"][:3, :3].T @ self._robot_init_ee[:3, :3]

        # import pdb;pdb.set_trace()
        # A = self._state["init_pose"][:3, :3]
        # print(f"meta: {A}")
        # print(f"target_rot: {target_rot}")
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_pos
        target_pose[:3, :3] = target_rot

        if self._stop_record == False:
            self._stop_record = self._state["buttons"]["B"]
        # print(target_pose)
        return {
            "target_pose": target_pose,
            "grasp": self._state["buttons"]["RTr"],
            "over": self._state["buttons"]["A"],
            "action_hot": self._state["buttons"]["RG"], # TODO: name RG RJ RThu  rightJS rightTrig  rightGrip
            "stop": self._stop_record,
        }