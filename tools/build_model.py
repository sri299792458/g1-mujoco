"""Derive the simulation scene from the pinned, unmodified GR00T robot XML."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

from g1_mujoco.joints import ACTIVE, JOINT_NAMES, LOCKED
from g1_mujoco.sim import XML_PATH


def build(check=False):
    assets = XML_PATH.parent
    source = json.loads((assets / "sources.lock.json").read_text())["sources"]["groot"]
    for name, digest in source["files"].items():
        assert (
            hashlib.sha256((assets / "sources/groot" / name).read_bytes()).hexdigest()
            == digest
        ), name
    root = ET.parse(assets / "sources/groot" / source["model"]).getroot()
    root.set("model", "g1_dex3")
    root.find("compiler").set(
        "meshdir", "sources/groot/gear_sonic/data/robots/g1/meshes"
    )
    # OpenHomie observes 27 body joints: waist roll/pitch stay fixed at zero.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "joint" and child.get("name") in [
                JOINT_NAMES[i] for i in LOCKED
            ]:
                parent.remove(child)
    # No virtual forces on the floating base. Each active hinge gets one motor.
    root.remove(root.find("actuator"))
    actuator = ET.SubElement(root, "actuator")
    for slot in ACTIVE:
        name = JOINT_NAMES[slot]
        joint = root.find(f".//joint[@name='{name}']")
        ET.SubElement(
            actuator,
            "motor",
            name=name,
            joint=name,
            ctrlrange=joint.get("actuatorfrcrange"),
        )
    root.append(
        ET.fromstring(
            '<option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>'
        )
    )
    root.append(ET.fromstring('<statistic center="0 0 0.7" extent="2"/>'))
    root.append(
        ET.fromstring("""<visual>
      <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    </visual>""")
    )
    root.find("asset").extend(
        ET.fromstring("""<asset>
      <texture name="floor_texture" type="2d" builtin="checker" width="512" height="512"
               rgb1="0.22 0.25 0.28" rgb2="0.3 0.33 0.36"/>
      <material name="floor_material" texture="floor_texture" texrepeat="4 4" texuniform="true"/>
    </asset>""")
    )
    world = root.find("worldbody")
    world.append(
        ET.fromstring(
            '<geom name="floor" type="plane" size="0 0 0.05" material="floor_material"/>'
        )
    )
    world.append(ET.fromstring('<light pos="0 0 3" dir="0 0 -1" directional="true"/>'))
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode") + "\n"
    if check:
        assert XML_PATH.read_text() == text, "Run python -m tools.build_model"
    else:
        XML_PATH.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    print(build(parser.parse_args().check))
