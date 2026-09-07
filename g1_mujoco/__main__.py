"""Run from the repository with: python -m g1_mujoco."""

import argparse
from contextlib import nullcontext
import math
import time

from .policy import OpenHomie, POLICY_PATH
from .sim import Simulator, XML_PATH, DT
from .dds import DDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=XML_PATH)
    parser.add_argument("--policy", default=POLICY_PATH)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Simulation seconds; 0 runs until closed",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    end_step = math.ceil(args.seconds / DT) if args.seconds else None
    sim = Simulator(xml=args.xml, policy=OpenHomie(args.policy))
    transport = DDS(sim, domain=args.domain, interface=args.interface)
    try:
        if args.headless:
            viewer_context = nullcontext(None)
        else:
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(sim.model, sim.data)
        with viewer_context as viewer:
            if viewer:
                with viewer.lock():
                    viewer.cam.distance = 2.5
                    viewer.cam.azimuth = 120
                    viewer.cam.elevation = -25
                    viewer.cam.lookat[:] = [0, 0, 0.7]
            start = time.monotonic()
            next_frame = start
            state = sim.state()
            while (end_step is None or sim.steps < end_step) and (
                viewer is None or viewer.is_running()
            ):
                state = sim.step()
                transport.publish(state)
                now = time.monotonic()
                if viewer and now >= next_frame:
                    viewer.sync()
                    next_frame = now + 1 / 30
                if not args.no_realtime:
                    time.sleep(max(0, start + sim.data.time - time.monotonic()))
            print(
                f"Finished {sim.steps} physics ticks; base position: {state['base_pos']}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()


if __name__ == "__main__":
    main()
