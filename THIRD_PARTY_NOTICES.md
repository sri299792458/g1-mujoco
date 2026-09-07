# Sources and licenses

## GR00T G1/Dex3 model

The original XML and 49 meshes are from
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/0e35637c34cb0296658b744735f63c5fdbd4ca62),
commit `0e35637c34cb0296658b744735f63c5fdbd4ca62`, model
`gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml`.
They are stored unmodified under `g1_mujoco/assets/sources/groot/`.
The upstream [license](g1_mujoco/assets/sources/groot/LICENSE) is included.
`sources.lock.json` records the hashes. The derived `g1.xml` fixes waist
roll/pitch, creates hinge torque actuators and adds a simple floor/light.

## OpenHomie

The ONNX checkpoint and observation/action convention come from
[InternRobotics/OpenHomie](https://github.com/InternRobotics/OpenHomie/tree/cefcd85fcf81f529e8be065795fb2a7273e69435),
commit `cefcd85fcf81f529e8be065795fb2a7273e69435`.
The download is `HomieDeploy/deploy.onnx`; its hash is in the asset manifest.
The [CC BY-NC 4.0 license](g1_mujoco/assets/sources/openhomie/LICENSE) is included.
The checkpoint is downloaded separately, not committed here.

The 76-value observation and 6-frame history follow `MujocoDeploy/`.
The 0.74 m standing command also appears in
`HomieDeploy/g1_gym_deploy/utils/cheetah_state_estimator.py`; the sample MuJoCo
YAML instead starts at 0.34 m. This simulator intentionally starts at 0.74 m.

## Unitree reference data and SDK2

The hand URDFs and retargeting mapping in `tests/reference_data/` come from
[unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate/tree/7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6),
commit `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`. They are test references, not
runtime dependencies. Their source paths/hashes are in
[sources.json](tests/reference_data/sources.json), with the upstream
[license](tests/reference_data/LICENSE).

Finger order follows the explicit API arrays in
[hand_retargeting.py](https://github.com/unitreerobotics/xr_teleoperate/blob/7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6/teleop/robot_control/hand_retargeting.py#L50),
which are used to order commands sent by motor ID. The older right-hand enum
labels in `robot_hand_unitree.py` differ; the executed API arrays put middle
before index for both hands. This also matches the lab's tabletop interface.

The SDK2 dependency is the same lnotspotl fork and revision as the
tabletop environment:
`7c661d27f4ae064ffd0dd633fd9d5b518ef0b508`. Python 3.10, NumPy 1.26.4,
CycloneDDS 0.10.2 and other shared package pins are recorded in `pyproject.toml`
and `uv.lock`. The wheel's Python version does not identify its bundled native
DDS library's version. Runtime dependencies retain their respective licenses.

## Prior work and implementation references

The choice of HOMIE beneath a direct manipulation command interface is informed
by [VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation](https://viral-humanoid.github.io/),
by Tairan He, Zi Wang, Haoru Xue, Qingwei Ben and collaborators. Section 2.1 of
the [paper](https://viral-humanoid.github.io/static/viral-v2.pdf) describes HOMIE
as its underlying whole-body controller. VIRAL is a research reference; this
repository does not include its teacher/student policies or training pipeline.

[SONIC](https://nvlabs.github.io/SONIC/) is related work considered when choosing
the controller interface. The model assets above come from its GR00T repository,
but no SONIC policy, latent-action decoder or deployment runtime is used here.

This is a new, small runtime implementation. Its intended G1/OpenHomie behavior
and upper-body gains were informed by
[G1Pilot](https://github.com/sri299792458/g1pilot/tree/2e3eac9eea48675b42ea10402bc52cbf42767200).
The prior BSD notice is retained in [LICENSE](LICENSE).

Step/state semantics were checked against the
[MuJoCo 3.3.7 simulation documentation](https://mujoco.readthedocs.io/en/3.3.7/programming/simulation.html).
The mathematical and runtime checks are described in `VALIDATION.md`.
