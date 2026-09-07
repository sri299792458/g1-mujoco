"""Stand, then slowly move the elbows and fingers. Run as a Python module."""

import numpy as np

from g1_mujoco.joints import LEFT_HAND, RIGHT_HAND
from g1_mujoco.policy import OpenHomie
from g1_mujoco.sim import Simulator


def targets(time):
    fraction = np.clip((time - 5) / 3, 0, 1)
    elbows = fraction * np.array([0.25, 0.45])
    fingers = fraction * np.array([0, 0, 0, 0.45, 0.65, 0.20, 0.30])
    return elbows, -fingers, fingers


def command(sim):
    elbows, left, right = targets(sim.data.time)
    sim.set_command([18, 25], q=elbows)
    sim.set_command(LEFT_HAND, q=left)
    sim.set_command(RIGHT_HAND, q=right)


if __name__ == "__main__":
    sim = Simulator(policy=OpenHomie())
    for _ in range(10_000):
        command(sim)
        state = sim.step()
    print("Elbows:", state["q"][[18, 25]])
    print("Left hand:", state["q"][LEFT_HAND])
    print("Right hand:", state["q"][RIGHT_HAND])
