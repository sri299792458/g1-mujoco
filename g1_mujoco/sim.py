"""MuJoCo state, motor commands and one explicit physics step."""

from pathlib import Path
import threading

import mujoco
import numpy as np

from .joints import ACTIVE, JOINT_NAMES, LEGS, UPPER_BODY
from .policy import LEG_Q, LEG_KP, LEG_KD


XML_PATH = Path(__file__).parent / "assets/g1.xml"
DT = 0.002
POLICY_EVERY = 10


def pd_torque(q, dq, target_q, target_dq, kp, kd, tau):
    return tau + kp * (target_q - q) + kd * (target_dq - dq)


class Simulator:
    def __init__(self, *, xml=XML_PATH, policy=None):
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml).expanduser()))
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self.policy = policy
        self.steps = 0
        self.slots = np.asarray(ACTIVE)
        names = [JOINT_NAMES[i] for i in ACTIVE]
        self.qadr = np.array([self.model.joint(name).qposadr[0] for name in names])
        self.vadr = np.array([self.model.joint(name).dofadr[0] for name in names])
        self.aids = np.array([self.model.actuator(name).id for name in names])
        base = self.model.joint("floating_base_joint")
        self.base_qadr, self.base_vadr = int(base.qposadr[0]), int(base.dofadr[0])
        self.torso = self.model.body("torso_link").id
        self.command = {key: np.zeros(43) for key in ("q", "dq", "kp", "kd", "tau")}
        self.command["q"][LEGS] = LEG_Q
        self.command["kp"][LEGS] = LEG_KP
        self.command["kd"][LEGS] = LEG_KD
        self.command["kp"][UPPER_BODY] = 100
        self.command["kd"][UPPER_BODY] = 0.5
        self.command["kp"][29:] = 1.5
        self.command["kd"][29:] = 0.1
        self.command_lock = threading.Lock()
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, slots, **fields):
        """Set selected motor fields; scalars broadcast. Other fields stay held.

        For example: set_command([18, 25], q=[0.25, 0.45], kp=100, kd=0.5).
        Use this method when a DDS thread and the physics thread run together.
        """
        with self.command_lock:
            for key, value in fields.items():
                self.command[key][slots] = value

    def step(self):
        """Apply PD, advance 2 ms, read state, then update the policy every 10 ticks."""
        with self.command_lock:
            if self.policy is not None:
                self.command["q"][LEGS] = self.policy.target
            c = self.command
            self.data.ctrl[self.aids] = pd_torque(
                self.data.qpos[self.qadr],
                self.data.qvel[self.vadr],
                c["q"][self.slots],
                c["dq"][self.slots],
                c["kp"][self.slots],
                c["kd"][self.slots],
                c["tau"][self.slots],
            )
        mujoco.mj_step(self.model, self.data)
        state = self.state()
        self.steps += 1
        if self.policy is not None and self.steps % POLICY_EVERY == 0:
            self.policy.update(state)
        return state

    def state(self):
        # mj_step integrates AFTER computing body kinematics. Refresh those
        # fields without overwriting the completed step's forces/accelerations.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)
        state = {key: np.zeros(43) for key in ("q", "dq", "ddq", "tau")}
        state["q"][self.slots] = self.data.qpos[self.qadr]
        state["dq"][self.slots] = self.data.qvel[self.vadr]
        state["ddq"][self.slots] = self.data.qacc[self.vadr]
        state["tau"][self.slots] = self.data.actuator_force[self.aids]
        bq, bv = self.base_qadr, self.base_vadr
        state["base_pos"] = self.data.qpos[bq : bq + 3].copy()
        state["base_quat"] = self.data.qpos[bq + 3 : bq + 7].copy()
        state["base_omega"] = self.data.qvel[bv + 3 : bv + 6].copy()
        state["base_acc"] = self.data.qacc[bv : bv + 3].copy()
        state["pelvis_accelerometer"] = self.data.sensor(
            "imu-pelvis-linear-acceleration"
        ).data.copy()
        state["torso_accelerometer"] = self.data.sensor(
            "imu-torso-linear-acceleration"
        ).data.copy()
        state["torso_quat"] = self.data.xquat[self.torso].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso, velocity, 1
        )
        state["torso_omega"] = velocity[:3].copy()
        state["time"] = float(self.data.time)
        return state
