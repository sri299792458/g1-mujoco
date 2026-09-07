import mujoco
import numpy as np
import pytest

from g1_mujoco.policy import LEG_Q, OpenHomie, gravity_in_body, observation
from g1_mujoco.sim import Simulator, DT, pd_torque


@pytest.fixture
def sim():
    return Simulator()


def test_pd_with_hand_calculated_values_and_fresh_feedback(sim):
    # 0.7 + 10*(0.5-0.2) + 2*(0.2-(-0.1)) = 4.3 Nm.
    assert pd_torque(0.2, -0.1, 0.5, 0.2, 10, 2, 0.7) == pytest.approx(4.3)
    joint = sim.model.joint("left_elbow_joint")
    aid = sim.model.actuator("left_elbow_joint").id
    sim.data.qpos[joint.qposadr] = 0.2
    sim.data.qvel[joint.dofadr] = -0.1
    sim.set_command([18], q=0.5, dq=0.2, kp=10, kd=2, tau=0.7)
    sim.step()
    assert sim.data.ctrl[aid] == pytest.approx(4.3)
    expected = (
        0.7
        + 10 * (0.5 - sim.data.qpos[joint.qposadr[0]])
        + 2 * (0.2 - sim.data.qvel[joint.dofadr[0]])
    )
    sim.step()
    assert sim.data.ctrl[aid] == pytest.approx(expected)
    assert abs(expected - 4.3) > 1e-5


def test_freefall_has_no_hidden_base_support(sim):
    sim.set_command(list(range(43)), kp=0, kd=0, tau=0)
    sim.data.qpos[sim.base_qadr + 2] = 3
    initial_height = sim.state()["base_pos"][2]
    for _ in range(10):
        state = sim.step()
    # Semi-implicit Euler: z_n = z_0 - g*dt^2*n*(n+1)/2.
    assert state["base_pos"][2] == pytest.approx(
        initial_height - 9.81 * DT**2 * 55, abs=1e-10
    )
    assert sim.data.qvel[sim.base_vadr + 2] == pytest.approx(-9.81 * 10 * DT, abs=1e-10)


def test_each_finger_slot_drives_the_named_physical_joint(sim):
    for side, offset in (("left", 29), ("right", 36)):
        for motor, finger in enumerate(
            (
                "thumb_0",
                "thumb_1",
                "thumb_2",
                "middle_0",
                "middle_1",
                "index_0",
                "index_1",
            )
        ):
            sim.set_command(list(range(43)), kp=0, kd=0, tau=0)
            sim.set_command([offset + motor], tau=0.2)
            sim.step()
            expected = np.zeros(sim.model.nu)
            expected[sim.model.actuator(f"{side}_hand_{finger}_joint").id] = 0.2
            np.testing.assert_array_equal(sim.data.ctrl, expected)


def test_policy_updates_after_tenth_tick_and_applies_on_eleventh():
    class Policy:
        target = LEG_Q.copy()
        calls = 0

        def update(self, state):
            self.calls += 1
            assert state["time"] == pytest.approx(0.02 * self.calls)
            self.target = LEG_Q + 0.1

    policy = Policy()
    sim = Simulator(policy=policy)
    for _ in range(9):
        sim.step()
    assert policy.calls == 0
    sim.step()
    assert policy.calls == 1
    np.testing.assert_array_equal(sim.command["q"][:12], LEG_Q)
    sim.step()
    np.testing.assert_array_equal(sim.command["q"][:12], LEG_Q + 0.1)
    for _ in range(9):
        sim.step()
    assert policy.calls == 2


def test_projected_gravity_against_rotation_matrix():
    rng = np.random.default_rng(42)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, q)
        expected = matrix.reshape(3, 3).T @ [0, 0, -1]
        np.testing.assert_allclose(gravity_in_body(q), expected, atol=1e-14)


def test_observation_layout():
    state = dict(
        q=np.arange(43) / 100,
        dq=np.ones(43),
        base_omega=np.array([1, 2, 3]),
        base_quat=np.array([1, 0, 0, 0]),
    )
    obs = observation(state, np.arange(12), [0.1, 0.2, 0.3], 0.74)
    np.testing.assert_allclose(
        obs[:10], [0.2, 0.4, 0.075, 0.74, 0.25, 0.5, 0.75, 0, 0, -1]
    )
    expected_q = state["q"][[*range(13), *range(15, 29)]].copy()
    expected_q[:12] -= LEG_Q
    np.testing.assert_allclose(obs[10:37], expected_q, atol=2e-8)
    np.testing.assert_allclose(obs[37:64], 0.05)
    np.testing.assert_array_equal(obs[64:], np.arange(12))


def test_policy_history_is_oldest_first_with_previous_action(sim):
    from types import SimpleNamespace

    inputs = []

    def infer(_, feed):
        inputs.append(feed["obs"].copy())
        return [np.full((1, 12), len(inputs), dtype=np.float32)]

    policy = OpenHomie.__new__(OpenHomie)
    policy.session = SimpleNamespace(run=infer)
    policy.input_name = "obs"
    policy.velocity = np.zeros(3)
    policy.history = np.zeros((6, 76), dtype=np.float32)
    policy.action = np.zeros(12, dtype=np.float32)
    for height in range(1, 9):
        policy.height = height
        policy.update(sim.state())
    np.testing.assert_array_equal(inputs[0].reshape(6, 76)[:, 3], [0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(inputs[-1].reshape(6, 76)[:, 3], [3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(inputs[-1].reshape(6, 76)[-1, 64:], 7)
    np.testing.assert_allclose(policy.target, LEG_Q + 2)


def test_torso_state_at_current_time_preserves_dynamics(sim):
    sim.data.qpos[sim.base_qadr + 2] = 2
    sim.data.qvel[sim.base_vadr + 3 : sim.base_vadr + 6] = [0.4, 0.5, 0.6]
    sim.data.qvel[sim.model.joint("waist_yaw_joint").dofadr] = 1.2
    mujoco.mj_step(sim.model, sim.data)
    saved = {
        key: getattr(sim.data, key).copy()
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "actuator_force",
            "sensordata",
        )
    }
    reference = mujoco.MjData(sim.model)
    reference.qpos[:] = sim.data.qpos
    reference.qvel[:] = sim.data.qvel
    mujoco.mj_forward(sim.model, reference)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        sim.model, reference, mujoco.mjtObj.mjOBJ_BODY, sim.torso, velocity, 1
    )
    state = sim.state()
    np.testing.assert_allclose(
        state["torso_quat"], reference.xquat[sim.torso], atol=1e-12
    )
    np.testing.assert_allclose(state["torso_omega"], velocity[:3], atol=1e-12)
    for key, expected in saved.items():
        np.testing.assert_array_equal(getattr(sim.data, key), expected)
    np.testing.assert_array_equal(state["q"][[13, 14]], 0)
