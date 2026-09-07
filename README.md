# G1 MuJoCo

A Unitree SDK2-connected MuJoCo simulator for a floating-base G1 with two Dex3
hands, providing a foundation for building a digital twin. SDK2
applications send waist, arm and finger commands over DDS and receive simulated
robot state. OpenHomie supplies lower-body balance control. It runs on CPU, with
no ROS dependency.

The main path is **SDK2 application → DDS commands → MuJoCo robot → DDS state
feedback**. SDK2 is installed and DDS starts by default. The implemented topics
and fields are listed below; matching the physical robot's dynamics, timing and
factory-controller behavior requires further calibration and validation.

## Why OpenHomie

The goal is to experiment with arm and hand control while a simulated G1
maintains its own balance. Unitree's factory motion controller is not included
in its public MuJoCo/SDK2 stack: the official
[simulator](https://github.com/unitreerobotics/unitree_mujoco) provides physics
and low-level command/state interfaces, so a balance controller must be supplied
separately. This project uses OpenHomie for that role.

[HOMIE / OpenHomie](https://github.com/InternRobotics/OpenHomie) combines learned
lower-body control with direct upper-body joint commands. This fits experiments
that send arm and finger targets through Python or SDK2. NVIDIA's
[VIRAL](https://viral-humanoid.github.io/static/viral-v2.pdf) uses HOMIE as its
underlying whole-body controller, with velocity/height commands and upper-body
joint commands extended to include fingers. That is research precedent for this
design; this repository implements the simulation/control baseline, and its
validation covers unloaded standing with moderate arm and hand motion.

[SONIC](https://nvlabs.github.io/SONIC/) supports a broader motion-tracking
interface. Its [VLA integration](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)
uses a 64-dimensional latent motion token plus hand joint commands. Integrating
that representation would add work beyond the direct joint-command interface
needed here. OpenHomie was chosen for that simpler fit; no comparative controller
benchmark was performed. The GR00T robot assets used here do not require running
the SONIC controller.

## Run

Use Linux x86-64, Python 3.10 and [uv](https://docs.astral.sh/uv/). From this repo:

```bash
uv sync --locked
.venv/bin/python -m tools.fetch_policy
.venv/bin/python -m g1_mujoco
```

The first command installs the pinned environment, including SDK2. The second
downloads the hash-checked OpenHomie checkpoint to `~/.cache/g1-mujoco/openhomie.onnx`.
The third opens MuJoCo's viewer and starts DDS on loopback `lo`, domain 1.
Close the window or press Ctrl+C to stop.

In another terminal, send SDK2 arm and finger commands:

```bash
.venv/bin/python -m examples.dds_commands
```

For a finite headless run:

```bash
.venv/bin/python -m g1_mujoco --headless --seconds 20
```

Physics uses 2 ms steps; OpenHomie runs every ten steps. `--seconds` is simulation
time rounded up to a whole physics step. The default uses best-effort wall-clock
pacing for external SDK2 clients; `--no-realtime` runs as fast as possible.
Rendering is limited to 30 Hz independently of physics.
Use `--policy PATH` for another compatible ONNX checkpoint and `--xml PATH` for
an edited scene that retains the robot's joint, actuator and IMU sensor names.

## SDK2 / DDS

The simulator and SDK2 example use loopback `lo`, domain 1 by default.
The simulator also accepts `--domain` and `--interface`. On the supported Python 3.10/Linux x86-64 setup, the pinned
CycloneDDS wheel supplies its native library; no native DDS build is needed.

| Receive | SDK2 message | Consumed motor slots |
|---|---|---|
| `rt/arm_sdk` | `unitree_hg/LowCmd_` | Body 12 and 15–28 |
| `rt/dex3/left/cmd` | `unitree_hg/HandCmd_` | Hand 0–6 |
| `rt/dex3/right/cmd` | `unitree_hg/HandCmd_` | Hand 0–6 |

The simulator publishes `rt/lowstate`, both `rt/dex3/*/state` topics,
`rt/secondary_imu`, and a zero/default `rt/wirelesscontroller` message after each
step. `LowState.tick` is simulation time in milliseconds. Body states populate
`q/dq/ddq/tau_est`; hands populate `q/dq`; IMUs use the state conventions above.
Other message fields retain their defaults. DDS delivery depends on scheduling.

This simulator assumes well-formed experiment commands. It does not reproduce
Unitree's proprietary motion controller, authority blending, CRC checks,
watchdogs or motor fault behavior.

Client integration details:

- Use matching DDS domains/interfaces. High-level `LocoClient` RPCs such as
  `Move`, `BalanceStand` and FSM queries are not implemented by this simulator.
  G1Pilot's `backend=sim` arm path avoids its real-backend firmware-state gate.
- Initialize client targets from received joint state and generate any startup
  ramps in the client. Stopping command publication holds the last command;
  sending `kp=kd=tau=0` explicitly removes motor torque from those commands.
- `mode_pr`, `mode_machine`, motor mode bits and the arm authority weight in
  body slot 29 are not interpreted. `LowState.mode_machine` retains its default
  zero; the model itself fixes waist roll/pitch and allows waist yaw.
- The pinned G1Pilot hand bridge uses `thumb, index, middle` in its named-joint
  conversion. Reconcile that client mapping with the hardware ordering here
  (`thumb, middle, index`) before using named finger targets or ROS hand states.
  Keep OpenSoT's arm/collision model separate from the articulated MuJoCo hands.

## Read the code

| File | Responsibility |
|---|---|
| [sim.py](g1_mujoco/sim.py) | Model, motor commands, state, and one physics step |
| [policy.py](g1_mujoco/policy.py) | OpenHomie observation, history, inference, and leg targets |
| [joints.py](g1_mujoco/joints.py) | Explicit body and hand motor ordering |
| [dds.py](g1_mujoco/dds.py) | SDK2 command/state transport |
| [__main__.py](g1_mujoco/__main__.py) | Viewer and main loop |
| [build_model.py](tools/build_model.py) | Small, deterministic robot/scene adapter |

`Simulator.step()` computes
`tau = tau_ff + kp * (q_target - q) + kd * (dq_target - dq)`, advances MuJoCo
once, reads state, and updates OpenHomie after every tenth tick. Updated policy
targets take effect on the following tick. Torque is recomputed on **every**
physics tick. MuJoCo applies the model's actuator and joint effort limits.

## Python experiments

For experiments inside one Python process, use `Simulator` directly. This API
does not start a DDS bridge; the command-line simulator above does.

```python
from g1_mujoco.policy import OpenHomie
from g1_mujoco.sim import Simulator

sim = Simulator(policy=OpenHomie())
for step in range(10_000):
    if step >= 2_500:
        fraction = min(1, (step - 2_500) / 1_500)
        sim.set_command([18, 25], q=[0.25 * fraction, 0.45 * fraction])
    state = sim.step()
print(state["q"][[18, 25]])
```

`set_command(slots, q=..., dq=..., kp=..., kd=..., tau=...)` accepts arrays or
scalars. Unspecified fields retain their values. Commands remain active until
changed. Default upper-body gains are `kp=100`, `kd=0.5`; finger gains are
`kp=1.5`, `kd=0.1`. All upper-body/finger targets start at zero. OpenHomie owns
the leg targets. Units are radians, radians/s and Nm.

Arrays have 43 slots: body `0:29`, left hand `29:36`, right hand `36:43`.
Body ordering follows Unitree: six left leg joints, six right leg joints,
waist yaw/roll/pitch, seven left arm joints, seven right arm joints. Waist roll
and pitch (13 and 14) are fixed and report zero. Each hand is ordered
`thumb0, thumb1, thumb2, middle0, middle1, index0, index1`, for both sides.
Joint signs come from the model; there is no automatic left/right mirroring.

Run the elbow/finger example with:

```bash
.venv/bin/python -m examples.python_commands
```

`state` contains copied arrays `q`, `dq`, `ddq`, `tau`, `base_pos`, `base_quat`,
`base_omega`, `base_acc`, `torso_quat`, `torso_omega`, `pelvis_accelerometer`,
`torso_accelerometer`, and simulation `time`.
Quaternions use `w,x,y,z`; angular velocities are local to the body. Positions
and velocities describe the current integrated state. Accelerations and forces
come from the dynamics evaluation of the completed step. `base_acc` is world
translational acceleration. The two accelerometer arrays are specific force in
each IMU's local frame, read from the model's sensors and used for DDS IMU
messages: approximately zero in free fall and +9.81 m/s² vertically when held
upright at rest. These sensor values also come from the completed step's dynamics.

The MuJoCo `model` and `data` remain directly accessible for experiments.
`Simulator(policy=None)` lets you supply your own controller; it does not provide
balance. Policy parameters and the 500/50 Hz timing are ordinary Python constants
and fields, with no configuration framework.

## Check and extend

```bash
uv sync --locked --extra dev
.venv/bin/python -m tools.build_model --check
.venv/bin/pytest -q
.venv/bin/python -m tools.validate --output artifacts/dds.json
.venv/bin/python -m tools.validate --python --output artifacts/python.json
```

The DDS check uses loopback domain 19. The tests include independent equations
and Unitree hand kinematics; see [VALIDATION.md](VALIDATION.md) for exactly what
the results establish and the model-source differences.

Edit the scene adapter and run `python -m tools.build_model` to change the scene.
Keep pinned source assets unchanged. For a new controller, start at
`Simulator.step()`; for a planner, submit targets through `set_command()`.
Payloads, objects, walking and chair-supported manipulation require experiments
for those conditions; the included runtime check covers unloaded standing and
moderate arm/hand motion.

## Acknowledgements

This simulator builds on the following work:

- **[HOMIE / OpenHomie](https://github.com/InternRobotics/OpenHomie)** by Qingwei
  Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin and Jiangmiao Pang: the
  pretrained balance/locomotion policy and its deployment conventions.
- **[NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)**:
  the G1/Dex3 MJCF model and meshes used to build this scene.
- **[Unitree Robotics](https://github.com/unitreerobotics)**: SDK2 and the hand
  reference models/mappings; **[lnotspotl](https://github.com/lnotspotl/unitree_sdk2_python)**
  for the Python SDK2 fork used here.
- **[MuJoCo](https://github.com/google-deepmind/mujoco)**: the physics engine and
  interactive viewer.
- **[VIRAL](https://viral-humanoid.github.io/)** by NVIDIA and collaborators:
  research motivation for using HOMIE beneath a manipulation command interface.
- **[G1Pilot](https://github.com/sri299792458/g1pilot)**: prior implementation
  reference for the intended G1/OpenHomie behavior and upper-body gains.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned source revisions,
attribution and licenses, and [VALIDATION.md](VALIDATION.md) for measured results.
The pinned OpenHomie checkpoint is CC BY-NC 4.0.
