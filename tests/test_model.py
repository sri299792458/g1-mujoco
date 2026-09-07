import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from g1_mujoco.joints import JOINT_NAMES
from g1_mujoco.sim import Simulator, XML_PATH
from tools.build_model import build

REFERENCES = Path(__file__).parent / "reference_data"


@pytest.fixture(scope="module")
def models():
    source = (
        XML_PATH.parent
        / "sources/groot/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml"
    )
    return mujoco.MjModel.from_xml_path(str(source)), Simulator().model


def test_source_hashes_and_generated_model(models):
    build(check=True)
    manifest = json.loads((REFERENCES / "sources.json").read_text())
    for name, metadata in manifest["files"].items():
        assert (
            hashlib.sha256((REFERENCES / name).read_bytes()).hexdigest()
            == metadata["sha256"]
        )
    _, model = models
    assert (model.nq, model.nv, model.nu) == (48, 47, 41)
    assert np.all(model.body_mass[1:] > 0)
    assert np.all(model.body_inertia[1:] > 0)
    assert np.all(
        2 * model.body_inertia[1:].max(axis=1) <= model.body_inertia[1:].sum(axis=1)
    )
    assert all(
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        for j in model.actuator_trnid[:, 0]
    )


def test_model_preserves_original_physical_parameters(models):
    source, model = models
    for body_id in range(1, source.nbody):
        before = source.body(body_id)
        after = model.body(before.name)
        for field in ("pos", "quat", "mass", "ipos", "iquat", "inertia"):
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
    for joint_id in range(1, source.njnt):
        before = source.joint(joint_id)
        if before.name in ("waist_roll_joint", "waist_pitch_joint"):
            continue
        after = model.joint(before.name)
        for field in ("pos", "axis", "range"):
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
        for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
            np.testing.assert_array_equal(
                getattr(model, field)[after.dofadr],
                getattr(source, field)[before.dofadr],
            )
        aid = model.actuator(before.name).id
        np.testing.assert_array_equal(
            model.actuator_ctrlrange[aid], source.jnt_actfrcrange[before.id]
        )


def test_hand_order_matches_unitree_hardware_mapping():
    tree = ast.parse((REFERENCES / "hand_retargeting.py.txt").read_text())
    for side, offset in (("left", 29), ("right", 36)):
        name = f"{side}_dex3_api_joint_names"
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == name for t in node.targets
            )
        ]
        expected = ast.literal_eval(assignments[0].value)
        assert list(JOINT_NAMES[offset : offset + 7]) == expected


def rotation(axis, angle):
    # Independent Rodrigues formula for the reference URDF joint chain.
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


@pytest.mark.parametrize(
    "side,offset,mount_y", [("left", 29, 0.003), ("right", 36, -0.003)]
)
def test_hand_forward_kinematics_against_unitree_urdf(models, side, offset, mount_y):
    _, model = models
    data = mujoco.MjData(model)
    reference = ET.parse(REFERENCES / f"unitree_dex3_{side}.urdf").getroot()
    rng = np.random.default_rng(3)
    for _ in range(10):
        for name in JOINT_NAMES[offset : offset + 7]:
            joint = model.joint(name)
            data.qpos[joint.qposadr] = rng.uniform(*joint.range)
        mujoco.mj_forward(model, data)
        wrist = data.body(f"{side}_wrist_yaw_link")
        palm = np.eye(4)
        palm[:3, :3] = wrist.xmat.reshape(3, 3)
        palm[:3, 3] = wrist.xpos + palm[:3, :3] @ [0.0415, mount_y, 0]
        poses = {f"{side}_hand_palm_link": palm}
        for joint_xml in reference.findall("joint"):
            if joint_xml.get("type") != "revolute":
                continue
            name = joint_xml.get("name")
            joint = model.joint(name)
            origin = joint_xml.find("origin")
            np.testing.assert_array_equal(np.fromstring(origin.get("rpy"), sep=" "), 0)
            axis = np.fromstring(joint_xml.find("axis").get("xyz"), sep=" ")
            np.testing.assert_array_equal(joint.axis, axis)
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            transform[:3, :3] = rotation(axis, data.qpos[joint.qposadr[0]])
            child = joint_xml.find("child").get("link")
            poses[child] = poses[joint_xml.find("parent").get("link")] @ transform
            np.testing.assert_allclose(
                data.body(child).xpos, poses[child][:3, 3], atol=1e-12
            )
            np.testing.assert_allclose(
                data.body(child).xmat.reshape(3, 3), poses[child][:3, :3], atol=1e-12
            )
            bounds = joint_xml.find("limit")
            expected_range = [float(bounds.get("lower")), float(bounds.get("upper"))]
            # Explicit source discrepancy: GR00T thumb-1 +/-1.0472; Unitree +/-0.920.
            if name.endswith("thumb_1_joint"):
                assert expected_range[1 if side == "left" else 0] == (
                    0.920 if side == "left" else -0.920
                )
                expected_range[1 if side == "left" else 0] = (
                    1.0472 if side == "left" else -1.0472
                )
            np.testing.assert_allclose(joint.range, expected_range, atol=5e-6)
            np.testing.assert_allclose(
                model.jnt_actfrcrange[joint.id],
                [-float(bounds.get("effort")), float(bounds.get("effort"))],
            )
