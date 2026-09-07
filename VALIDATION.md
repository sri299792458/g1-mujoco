# Validation

These checks establish specific software and simulation properties. They do
not establish hardware accuracy, learned-controller stability under arbitrary
conditions, or successful grasping.

## What is checked

| Check | Evidence |
|---|---|
| Source identity | Pinned GR00T XML, 49 meshes and license hashes; separate pinned Unitree test references |
| Model generation | Deterministic rebuild; 48 position coordinates, 47 velocities, 41 hinge motors; no base actuators |
| Physical data | Original body transforms, mass/inertias, joint axes/limits, damping/friction/armature retained; positive inertias |
| Hand geometry | Independent Rodrigues/URDF forward-kinematic calculation matches MuJoCo for both hands over 10 random configurations each |
| Hand slots | Matches Unitree's explicit retargeting-to-hardware arrays; individual torque commands select the named fingers |
| PD control | Hand-calculated 4.3 Nm case; next tick uses the newly measured joint position and velocity |
| Free base | Airborne zero-control motion matches semi-implicit Euler integration of gravity |
| Policy input | 76-value layout; projected gravity checked against rotation matrices for 100 random quaternions |
| History and timing | Six observations ordered oldest first; zero initialization and previous action; inference after ticks 10, 20, …, targets used from the following tick |
| State timing | Current torso orientation/angular velocity match a fresh forward calculation; reading state preserves completed-step dynamics and solver warmstart |
| SDK2 IMUs | Accelerometers read zero in free fall and the correctly rotated gravity reaction for prescribed stationary upright/tilted poses, including waist yaw |
| SDK2 command ownership | Arm commands preserve leg ownership and fixed waist slots; explicit passive arm commands produce zero arm motor torque |
| Actual ONNX/DDS | The 20-second checks below exercise the real checkpoint and, in DDS mode, actual SDK2 serialization/loopback transport |

The unit suite has **17 tests**. Some checks establish fidelity to declared
model data; the analytical checks independently test physical/control behavior.
Neither substitutes for measuring the real robot.

## Recorded result — 2026-09-07

Python 3.10.20, MuJoCo 3.3.7, NumPy 1.26.4, ONNX Runtime 1.22.1; SDK2 and
CycloneDDS from `uv.lock`. All 24 packages shared with the tabletop lock have
matching versions. SDK2 has the same fork/commit; tabletop installs it as a
submodule, while this simulator uses a pinned Git dependency.

Both runs stand for five seconds, ramp different left/right elbow targets and
different middle/index finger targets over three seconds, then hold until
20 seconds. Commands update at 50 Hz. DDS mode is paced in wall-clock time on
loopback domain 19.

| Measurement | Python | DDS |
|---|---:|---:|
| Physics ticks | 10,000 | 10,000 |
| Measured policy updates | 1,000 | 1,000 |
| Base height after 2 s | 0.7401–0.7480 m | 0.7401–0.7480 m |
| Maximum base tilt after 2 s | 4.846° | 4.846° |
| Maximum elbow error during final 5 s | 0.03218 rad | 0.03218 rad |
| Maximum finger error during final 5 s | 0.02754 rad | 0.02755 rad |

DDS received all five state topics. Final joint arrays and torso orientation/
angular velocity matched simulator state within 1e-6 after float serialization.
Both DDS accelerometers also matched their MuJoCo sensor values within 1e-6.
Body ticks were nondecreasing and reached 20,000 ms. The measured run received
10,001 body messages including a final repeated publication.

The acceptance bounds are base height 0.65–0.90 m, tilt below 20°, elbow error
below 0.05 rad, finger error below 0.12 rad, finite states/actions, and exactly
one policy update per ten ticks. These are regression bounds for this experiment,
not robot specifications or a certified operating envelope.

Checkpoint download from the pinned URL, finite headless CLI runs, and a
five-second viewer+DDS run also completed successfully. Viewer execution emitted
an environment-specific `NV-GLX` X11 warning but completed 2,500 ticks and exited.

A fresh default `uv sync --locked` installation includes SDK2 (28 installed
packages). The command-line simulator ran for 18 seconds on loopback domain 21
with no DDS opt-in flag, and the separate SDK2 example received state and moved
the elbows to within 0.05 rad of their targets. The default 20-second validation
command exercised all three command topics and all five state topics. The
Python-only comparison remains available through `tools.validate --python`.

## Explicit source differences and remaining limits

GR00T's thumb-1 range is `[-0.724312, 1.0472]` on the left and
`[-1.0472, 0.724312]` on the right. Unitree's reference hand URDF uses 0.920
instead of 1.0472 for those outer bounds. This simulator preserves GR00T's
declared limits and tests the discrepancy explicitly. It does not claim which
bound is correct for a particular physical hand.

Hand FK checks use the recorded wrist-to-palm mounting offset
`[0.0415, ±0.003, 0]` and compare the finger chains against Unitree's URDF.
The older right-hand enum labels in Unitree's controller differ from its actual
retargeting API arrays; this simulator follows the latter, also used by the lab
tabletop interface. See `THIRD_PARTY_NOTICES.md` for the exact reference files.

Walking, disturbances, payloads, object contacts/grasp retention, chair support,
and sim-to-real dynamics have not been validated. DDS implements the documented
command/state fields, not the proprietary firmware's complete behavior.

## Reproduce

```bash
uv sync --locked --extra dev
.venv/bin/python -m tools.build_model --check
.venv/bin/pytest -q
.venv/bin/python -m tools.validate --output artifacts/dds.json
.venv/bin/python -m tools.validate --python --output artifacts/python.json
```

The JSON files contain measured values and are local generated artifacts.
