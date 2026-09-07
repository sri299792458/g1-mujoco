"""20-second SDK2 loopback check. --python checks direct in-process commands."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from examples.python_commands import command, targets
from g1_mujoco.policy import OpenHomie
from g1_mujoco.sim import Simulator, DT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sim = Simulator(policy=OpenHomie())
    policy_ticks = []
    update = sim.policy.update

    def observed_update(state):
        policy_ticks.append(sim.steps)
        update(state)

    sim.policy.update = observed_update
    transport = None
    channels, publishers, received, received_ticks = [], [], {}, []
    if not args.python:
        from g1_mujoco.dds import DDS
        from examples.dds_commands import messages, COMMAND_TOPICS
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowState_,
            HandState_,
            IMUState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

        transport = DDS(sim, domain=19, interface="lo")
    try:
        if transport:

            def receiver(key):
                def receive(msg):
                    received[key] = msg
                    if key == "body":
                        received_ticks.append(msg.tick)

                return receive

            for topic, kind, key in (
                ("rt/lowstate", LowState_, "body"),
                ("rt/dex3/left/state", HandState_, "left"),
                ("rt/dex3/right/state", HandState_, "right"),
                ("rt/secondary_imu", IMUState_, "torso"),
                ("rt/wirelesscontroller", WirelessController_, "wireless"),
            ):
                channel = ChannelSubscriber(topic, kind)
                channel.Init(receiver(key), 1)
                channels.append(channel)
            for topic, kind in COMMAND_TOPICS:
                channel = ChannelPublisher(topic, kind)
                channel.Init()
                channels.append(channel)
                publishers.append(channel)
            time.sleep(0.2)  # DDS discovery before the measured run.
        heights, tilts, elbow_errors, hand_errors = [], [], [], []
        start = time.monotonic()
        for tick in range(10000):
            if tick % 10 == 0:
                if transport:
                    for publisher, message in zip(publishers, messages(tick * DT)):
                        assert publisher.Write(message)
                else:
                    command(sim)
            state = sim.step()
            assert all(
                np.isfinite(state[key]).all()
                for key in (
                    "q",
                    "dq",
                    "tau",
                    "base_pos",
                    "base_quat",
                    "pelvis_accelerometer",
                    "torso_accelerometer",
                )
            )
            assert np.isfinite(sim.policy.action).all()
            if transport:
                transport.publish(state)
                time.sleep(max(0, start + (tick + 1) * DT - time.monotonic()))
            if tick >= 1000:
                heights.append(float(state["base_pos"][2]))
                w, x, y, z = state["base_quat"]
                tilts.append(
                    float(
                        np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
                    )
                )
            if tick >= 7500:
                elbows, left, right = targets(20)
                elbow_errors.append(
                    float(np.max(np.abs(state["q"][[18, 25]] - elbows)))
                )
                hand_errors.append(
                    float(np.max(np.abs(state["q"][29:] - np.r_[left, right])))
                )
        report = dict(
            transport="DDS loopback domain 19" if transport else "Python",
            physics_ticks=sim.steps,
            policy_updates=len(policy_ticks),
            simulation_seconds=state["time"],
            wall_seconds=time.monotonic() - start,
            base_height_range_m=[min(heights), max(heights)],
            max_tilt_degrees=max(tilts),
            max_elbow_error_final_5s_rad=max(elbow_errors),
            max_hand_error_final_5s_rad=max(hand_errors),
        )
        assert policy_ticks == list(range(10, 10001, 10))
        assert min(heights) > 0.65 and max(heights) < 0.9
        assert max(tilts) < 20 and max(elbow_errors) < 0.05 and max(hand_errors) < 0.12
        if transport:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                transport.publish(state)
                time.sleep(0.02)
                if len(received) == 5 and received["body"].tick == 20000:
                    break
            assert len(received) == 5 and received["body"].tick == 20000
            assert len(received_ticks) > 100
            assert all(a <= b for a, b in zip(received_ticks, received_ticks[1:]))
            for key, slots in (
                ("body", range(29)),
                ("left", range(29, 36)),
                ("right", range(36, 43)),
            ):
                actual = [m.q for m in received[key].motor_state][: len(slots)]
                np.testing.assert_allclose(actual, state["q"][list(slots)], atol=1e-6)
            np.testing.assert_allclose(
                received["torso"].quaternion, state["torso_quat"], atol=1e-6
            )
            np.testing.assert_allclose(
                received["torso"].gyroscope, state["torso_omega"], atol=1e-6
            )
            for imu, key in (
                (received["body"].imu_state, "pelvis_accelerometer"),
                (received["torso"], "torso_accelerometer"),
            ):
                np.testing.assert_allclose(imu.accelerometer, state[key], atol=1e-6)
            np.testing.assert_allclose(
                sim.command["q"][[18, 25]], [0.25, 0.45], atol=1e-6
            )
            np.testing.assert_allclose(
                sim.command["q"][29:], np.r_[left, right], atol=1e-6
            )
            report["received_body_states"] = len(received_ticks)
            report["last_tick_ms"] = received["body"].tick
            report["received_topics"] = sorted(received)
        report["passed"] = True
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
    finally:
        for channel in reversed(channels):
            channel.Close()
        if transport:
            transport.close()


if __name__ == "__main__":
    main()
