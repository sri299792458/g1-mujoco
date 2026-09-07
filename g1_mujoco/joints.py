"""Array order: 29 Unitree body slots, seven left fingers, seven right fingers."""

BODY = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
# The executed retargeting-to-hardware mapping in Unitree xr_teleoperate and
# the lab's tabletop interface both use this order for BOTH hands.
FINGERS = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
JOINT_NAMES = tuple(f"{name}_joint" for name in BODY) + tuple(
    f"{side}_hand_{finger}_joint" for side in ("left", "right") for finger in FINGERS
)
LEGS = list(range(12))
UPPER_BODY = [12, *range(15, 29)]
LEFT_HAND = list(range(29, 36))
RIGHT_HAND = list(range(36, 43))
LOCKED = [13, 14]
ACTIVE = [i for i in range(43) if i not in LOCKED]
POLICY_JOINTS = LEGS + UPPER_BODY
