"""Download OpenHomie's pinned ONNX file once and check its SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from g1_mujoco.policy import POLICY_PATH
from g1_mujoco.sim import XML_PATH


def fetch(path=POLICY_PATH):
    path = Path(path).expanduser()
    source = json.loads((XML_PATH.parent / "sources.lock.json").read_text())["sources"][
        "openhomie"
    ]
    relative = "HomieDeploy/deploy.onnx"
    digest = source["files"][relative]
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return path
    repository = source["repository"].removeprefix("https://github.com/")
    url = (
        f"https://raw.githubusercontent.com/{repository}/{source['commit']}/{relative}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == digest, "Checkpoint checksum mismatch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=POLICY_PATH)
    print(fetch(parser.parse_args().output))
