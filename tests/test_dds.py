from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__IMUState_,
)

from g1_mujoco.dds import DDS
from g1_mujoco.joints import UPPER_BODY
from g1_mujoco.sim import Simulator


def published_state(state):
    # Exercise real SDK2 message assignment without opening network channels.
    bridge = SimpleNamespace(
        body=unitree_hg_msg_dds__LowState_(),
        left=unitree_hg_msg_dds__HandState_(),
        right=unitree_hg_msg_dds__HandState_(),
        torso=unitree_hg_msg_dds__IMUState_(),
        wireless=None,
        publishers=[],
    )
    DDS.publish(bridge, state)
    return bridge.body.imu_state, bridge.torso


def test_dds_accelerometers_read_zero_in_freefall():
    sim = Simulator()
    sim.data.qpos[sim.base_qadr + 2] = 3
    sim.set_command(list(range(43)), kp=0, kd=0, tau=0)
    state = sim.step()
    np.testing.assert_allclose(state["base_acc"], [0, 0, -9.81], atol=1e-12)
    for imu in published_state(state):
        np.testing.assert_allclose(imu.accelerometer, 0, atol=1e-12)


@pytest.mark.parametrize(
    "pitch,yaw,pelvis_expected,torso_expected",
    [
        (0, 0, [0, 0, 9.81], [0, 0, 9.81]),
        (np.pi / 2, np.pi / 2, [-9.81, 0, 0], [0, 9.81, 0]),
    ],
)
def test_dds_accelerometers_use_each_sensor_frame(
    pitch, yaw, pelvis_expected, torso_expected
):
    sim = Simulator()
    sim.data.qpos[sim.base_qadr + 2] = 3
    sim.data.qpos[sim.base_qadr + 3 : sim.base_qadr + 7] = [
        np.cos(pitch / 2),
        0,
        np.sin(pitch / 2),
        0,
    ]
    sim.data.qpos[sim.model.joint("waist_yaw_joint").qposadr] = yaw
    # Prescribe zero motion; inverse dynamics supplies the required support.
    sim.data.qvel[:] = 0
    sim.data.qacc[:] = 0
    mujoco.mj_inverse(sim.model, sim.data)
    pelvis, torso = published_state(sim.state())
    np.testing.assert_allclose(pelvis.accelerometer, pelvis_expected, atol=1e-12)
    np.testing.assert_allclose(torso.accelerometer, torso_expected, atol=1e-12)


def test_arm_sdk_preserves_joint_ownership_and_accepts_passive_commands():
    sim = Simulator()
    bridge = SimpleNamespace(sim=sim)
    receive = DDS.receiver(bridge, UPPER_BODY, UPPER_BODY)
    command = unitree_hg_msg_dds__LowCmd_()
    for i, motor in enumerate(command.motor_cmd):
        motor.q, motor.kp, motor.kd = i / 100, 100, 0.5
    before = {key: value.copy() for key, value in sim.command.items()}
    receive(command)
    for key in before:
        # arm_sdk cannot take over legs, locked waist joints or hand motors.
        untouched = [*range(12), 13, 14, *range(29, 43)]
        np.testing.assert_array_equal(
            sim.command[key][untouched], before[key][untouched]
        )
    sim.step()
    for name, motor_id in (
        ("waist_yaw_joint", 12),
        ("left_shoulder_pitch_joint", 15),
        ("right_elbow_joint", 25),
    ):
        assert sim.data.ctrl[sim.model.actuator(name).id] == pytest.approx(motor_id)
    np.testing.assert_array_equal(sim.state()["q"][[13, 14]], 0)
    # G1Pilot releases the arms by explicitly zeroing their gains/feedforward.
    for motor in command.motor_cmd[15:29]:
        motor.kp = motor.kd = motor.tau = 0
    receive(command)
    sim.step()
    for side in ("left", "right"):
        for name in ("shoulder_pitch", "elbow", "wrist_yaw"):
            assert sim.data.ctrl[sim.model.actuator(f"{side}_{name}_joint").id] == 0
