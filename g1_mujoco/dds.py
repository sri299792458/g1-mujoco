"""Unitree SDK2 command/state transport. The physics loop calls publish(state)."""

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__IMUState_,
    unitree_go_msg_dds__WirelessController_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_,
    LowState_,
    HandCmd_,
    HandState_,
    IMUState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

from .joints import UPPER_BODY, LEFT_HAND, RIGHT_HAND


class DDS:
    def __init__(self, sim, *, domain=1, interface="lo"):
        ChannelFactoryInitialize(domain, interface)
        self.sim = sim
        self.channels = []
        self.body = unitree_hg_msg_dds__LowState_()
        self.left = unitree_hg_msg_dds__HandState_()
        self.right = unitree_hg_msg_dds__HandState_()
        self.torso = unitree_hg_msg_dds__IMUState_()
        self.wireless = unitree_go_msg_dds__WirelessController_()
        self.publishers = []
        for topic, kind in (
            ("rt/lowstate", LowState_),
            ("rt/dex3/left/state", HandState_),
            ("rt/dex3/right/state", HandState_),
            ("rt/secondary_imu", IMUState_),
            ("rt/wirelesscontroller", WirelessController_),
        ):
            channel = ChannelPublisher(topic, kind)
            channel.Init()
            self.publishers.append(channel)
            self.channels.append(channel)
        for topic, kind, slots, motor_ids in (
            ("rt/arm_sdk", LowCmd_, UPPER_BODY, UPPER_BODY),
            ("rt/dex3/left/cmd", HandCmd_, LEFT_HAND, range(7)),
            ("rt/dex3/right/cmd", HandCmd_, RIGHT_HAND, range(7)),
        ):
            channel = ChannelSubscriber(topic, kind)
            channel.Init(self.receiver(slots, motor_ids), 1)
            self.channels.append(channel)

    def receiver(self, slots, motor_ids):
        def receive(message):
            self.sim.set_command(
                slots,
                **{
                    key: [getattr(message.motor_cmd[i], key) for i in motor_ids]
                    for key in ("q", "dq", "kp", "kd", "tau")
                },
            )

        return receive

    def publish(self, state):
        for message, slots in (
            (self.body, range(29)),
            (self.left, LEFT_HAND),
            (self.right, RIGHT_HAND),
        ):
            for motor, slot in zip(message.motor_state, slots):
                motor.q = float(state["q"][slot])
                motor.dq = float(state["dq"][slot])
        for i in range(29):
            self.body.motor_state[i].ddq = float(state["ddq"][i])
            self.body.motor_state[i].tau_est = float(state["tau"][i])
        self.body.tick = int(round(state["time"] * 1000))
        self.body.imu_state.quaternion[:] = state["base_quat"]
        self.body.imu_state.gyroscope[:] = state["base_omega"]
        self.body.imu_state.accelerometer[:] = state["pelvis_accelerometer"]
        self.torso.quaternion[:] = state["torso_quat"]
        self.torso.gyroscope[:] = state["torso_omega"]
        self.torso.accelerometer[:] = state["torso_accelerometer"]
        for publisher, message in zip(
            self.publishers,
            (self.body, self.left, self.right, self.torso, self.wireless),
        ):
            publisher.Write(message)

    def close(self):
        for channel in reversed(self.channels):
            channel.Close()
