import time
import numpy as np
from xarm.wrapper import XArmAPI
import random

def main(robot: str = "xarm6",
         ip: str = "192.168.1.235",
         pose: list[float] = None):

    pose = np.array(pose, dtype=np.float32)

    print(f"Connecting to {robot} at {ip} ...")
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.clean_error()

    # manual mode first

    # arm.set_mode(2)
    # arm.set_state(0)
    
    # # ask operator for confirmation
    # resp = input("Reset the arm to this pose? (y to confirm): ").strip().lower()
    # if resp != "y":
    #     print("Aborted by user.")
    #     arm.disconnect()
    #     return

    # switch to position control (mode 0) and move
    print(f"current position: {np.asarray(arm.get_position_aa(is_radian=True)[1])}")
    # import pdb; pdb.set_trace()
    
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"Moving to pose: {pose} (units: mm, deg)")
    arm.set_position(*pose, wait=True) 

    arm.disconnect()
    print("Reset finished.")

# ==========================================================================
# Configure your robot here
# We will get in to the manual mode first, after the user confirms, we will switch to position control mode
# ==========================================================================
if __name__ == "__main__":
    ROBOT = "xarm6"                 # "xarm6" | "lite6" | "uf850"
    IP    = "192.168.1.235"                    # None → use default for ROBOT
    # randomized_pose = [420, -110, 310, -180, 0, 0]  # None → use robot’s default pose

    # central_pose = [420, -110, 410, -180, 0, 0]    # 35x30 mm rebar
    # central_pose = [420, -110, 360, -180, 0, 0]   # 35x25 mm rebar
    central_pose = [420, -110, 310, -180, 0, 0]     # 35x20 mm rebar
    # central_pose = [420, -110, 410, -180, 0, 0]   # 30x30 mm rebar

    # Custom random ranges for each element in the pose
    random_ranges = [
        (20, 20),  # For X-axis (420)
        (20, 20),  # For Y-axis (-90)
        (10, 10), # For Z-axis (300)
        (5, 5),  # For W-axis (-180)
        (0, 0),   # For 5th element (0)
        (5, 5)    # For 6th element (0)
    ]

    # Define a function to randomize the pose with custom ranges
    def randomize_pose(pose, ranges):
        return [
            p + random.randint(-r[0], r[1])  # Apply random variation within specified range
            for p, r in zip(pose, ranges)
        ]

    # Randomize the central pose with custom ranges
    randomized_pose = randomize_pose(central_pose, random_ranges)

    
    print(f"Randomized pose: {randomized_pose}")
    # import pdb; pdb.set_trace()
    main(robot=ROBOT, ip=IP, pose=randomized_pose)