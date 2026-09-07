"""OpenHomie's 456-input / 12-action ONNX adapter; all conventions are here."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .joints import POLICY_JOINTS


POLICY_PATH = Path.home() / ".cache/g1-mujoco/openhomie.onnx"
LEG_Q = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=np.float64)
LEG_KP = [100, 100, 100, 150, 40, 40] * 2
LEG_KD = [2, 2, 2, 4, 2, 2] * 2


def gravity_in_body(quaternion):
    """R(q).T @ [0, 0, -1], for a unit quaternion in w,x,y,z order."""
    w, x, y, z = quaternion
    return np.array(
        [2 * (w * y - x * z), -2 * (w * x + y * z), -(w * w - x * x - y * y + z * z)]
    )


def observation(state, action, velocity, height):
    """76 values, in the order used by OpenHomie's deployment code."""
    obs = np.empty(76, dtype=np.float32)
    obs[:3] = np.asarray(velocity) * [2, 2, 0.25]
    obs[3] = height
    obs[4:7] = state["base_omega"] * 0.25
    obs[7:10] = gravity_in_body(state["base_quat"])
    obs[10:37] = state["q"][POLICY_JOINTS]
    obs[10:22] -= LEG_Q.astype(np.float32)
    obs[37:64] = state["dq"][POLICY_JOINTS] * 0.05
    obs[64:] = action
    return obs


class OpenHomie:
    def __init__(self, path=POLICY_PATH, *, velocity=(0, 0, 0), height=0.74):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(path).expanduser()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.height = height
        self.history = np.zeros((6, 76), dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)
        self.target = LEG_Q.copy()

    def update(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = observation(state, self.action, self.velocity, self.height)
        self.action = self.session.run(
            None, {self.input_name: self.history.reshape(1, 456)}
        )[0].reshape(12)
        self.target = LEG_Q + 0.25 * self.action.astype(np.float64)
