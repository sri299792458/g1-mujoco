"""Send the example elbow/finger targets to a simulator on loopback domain 1."""

import argparse
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__HandCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_

from examples.python_commands import targets
from g1_mujoco.joints import UPPER_BODY


COMMAND_TOPICS = (
    ("rt/arm_sdk", LowCmd_),
    ("rt/dex3/left/cmd", HandCmd_),
    ("rt/dex3/right/cmd", HandCmd_),
)


def messages(simulation_time):
    elbows, left, right = targets(simulation_time)
    body = unitree_hg_msg_dds__LowCmd_()
    for slot in UPPER_BODY:
        body.motor_cmd[slot].kp = 100
        body.motor_cmd[slot].kd = 0.5
    body.motor_cmd[18].q, body.motor_cmd[25].q = map(float, elbows)
    result = [body]
    for finger_targets in (left, right):
        hand = unitree_hg_msg_dds__HandCmd_()
        for i, value in enumerate(finger_targets):
            hand.motor_cmd[i].mode = i | 0x10
            hand.motor_cmd[i].q = float(value)
            hand.motor_cmd[i].kp = 1.5
            hand.motor_cmd[i].kd = 0.1
        result.append(hand)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=int, default=1)
    args = parser.parse_args()
    ChannelFactoryInitialize(args.domain, "lo")
    channels, publishers, latest = [], [], []

    def receive(message):
        latest[:] = [message]

    try:
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(receive, 1)
        channels.append(subscriber)
        for topic, kind in COMMAND_TOPICS:
            publisher = ChannelPublisher(topic, kind)
            publisher.Init()
            channels.append(publisher)
            publishers.append(publisher)
        deadline = time.monotonic() + 5
        while not latest and time.monotonic() < deadline:
            time.sleep(0.02)
        if not latest:
            raise SystemExit(
                "No simulator state received on loopback; run python -m g1_mujoco"
            )
        start = time.monotonic()
        while time.monotonic() - start < 15:
            for publisher, message in zip(
                publishers, messages(time.monotonic() - start)
            ):
                publisher.Write(message)
            time.sleep(0.02)
        print("Measured elbows:", [latest[0].motor_state[i].q for i in (18, 25)])
    except KeyboardInterrupt:
        pass
    finally:
        for channel in reversed(channels):
            channel.Close()


if __name__ == "__main__":
    main()
