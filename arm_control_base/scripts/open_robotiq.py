#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
from xarm.x3.code import APIState

# Connect to the robot
ip = '192.168.1.235'
arm = XArmAPI(ip)
arm.motion_enable(True)
arm.clean_error()
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

# 1. Read current position (gPO)
status = arm.robotiq_get_status()
current_pos = status['gPO'] if status and 'gPO' in status else 0
print(f"Current gripper position: {current_pos}")

# # # 2. Reset (deactivate) the gripper
# code, ret = arm.robotiq_reset()
# print('robotiq_reset, code={}, ret={}'.format(code, ret))
# # time.sleep(1)

# Activate the Robotiq gripper
code, ret = arm.robotiq_set_activate()
print('robotiq_set_activate, code={}, ret={}'.format(code, ret))
time.sleep(0.5)

# 3. Activate the gripper (no move yet)
code, ret = arm.robotiq_set_activate()
print('robotiq_set_activate, code={}, ret={}'.format(code, ret))
time.sleep(0.5)

# 4. Set position to current to ensure no motion
code, ret = arm.robotiq_set_position(current_pos)
print(f'robotiq_set_position({current_pos}), code={code}, ret={ret}')
time.sleep(0.5)

# 5. Open the gripper
# Wait for the gripper to be ready
code, ret = arm.robotiq_open()
print('robotiq_open, code={}, ret={}'.format(code, ret))


# Wait for 5 seconds
time.sleep(5)
# 5 seconds countdown
for i in range(5, 0, -1):
    print(f"Waiting for {i} seconds before setting width to 220mm...", end='\r', flush=True)
    time.sleep(1)
print()  

# Set the width to 220mm
code, ret = arm.robotiq_set_position(220)
print('robotiq_set_pos(220), code={}, ret={}'.format(code, ret))

if code == APIState.END_EFFECTOR_HAS_FAULT:
    print('robotiq fault code: {}'.format(arm.robotiq_status['gFLT']))

arm.disconnect()