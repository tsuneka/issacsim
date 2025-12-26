#!/usr/bin/env python3
"""Move Ridgeback + arm (Franka or UR5) by driving wheel joints.

Usage:
  python sample_ridgeback_arm.py ridgeback_franka
  python sample_ridgeback_arm.py ridgeback_ur5
"""

import math
import sys
from typing import List, Tuple

import numpy as np
from isaacsim.simulation_app import SimulationApp

# Start Kit before importing omni/isaacsim modules.
CONFIG = {"headless": False, "renderer": "RayTracedLighting"}
simulation_app = SimulationApp(CONFIG)

import omni
from pxr import Usd, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils import stage as stage_utils
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path


def _try_add_reference(usd_candidates: List[str], prim_path: str) -> str:
    last_err = None
    for usd_path in usd_candidates:
        try:
            stage_utils.add_reference_to_stage(usd_path, prim_path)
            print(f"[OK] referenced: {usd_path} -> {prim_path}")
            return usd_path
        except Exception as exc:
            last_err = exc
            print(f"[WARN] failed: {usd_path} ({type(exc).__name__}: {exc})")
    raise RuntimeError(f"Failed to add reference for {prim_path}. Last error: {last_err}")


def _usd_candidates(robot_kind: str, assets_root: str) -> List[str]:
    if robot_kind == "ridgeback_franka":
        return [
            f"{assets_root}/Isaac/Samples/ROS2/Robots/RidgebackFranka.usd",
            f"{assets_root}/Isaac/Samples/Robots/RidgebackFranka/ridgeback_franka.usd",
            f"{assets_root}/Isaac/Robots/RidgebackFranka/ridgeback_franka.usd",
        ]
    if robot_kind == "ridgeback_ur5":
        return [
            f"{assets_root}/Isaac/Samples/ROS2/Robots/RidgebackUR5.usd",
            f"{assets_root}/Isaac/Samples/Robots/RidgebackUR5/ridgeback_ur5.usd",
            f"{assets_root}/Isaac/Robots/RidgebackUR5/ridgeback_ur5.usd",
        ]
    raise ValueError("robot_kind must be 'ridgeback_franka' or 'ridgeback_ur5'")

def _find_articulation_root(stage, root_path: str) -> str | None:
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return None

    def _is_articulation(prim) -> bool:
        return (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
            or prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
        )

    if _is_articulation(root_prim):
        return str(root_prim.GetPath())

    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if _is_articulation(prim):
            return str(prim.GetPath())

    return None




def _find_articulation_root_any(stage, name_hint: str | None = None) -> tuple[str | None, list[str]]:
    roots: list[str] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(PhysxSchema.PhysxArticulationAPI) or prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            roots.append(str(prim.GetPath()))

    if not roots:
        return None, roots

    if name_hint:
        for p in roots:
            if name_hint.lower() in p.lower():
                return p, roots

    return roots[0], roots

def _find_wheel_groups(robot: SingleArticulation) -> Tuple[List[int], List[int]]:
    # Split wheel joints into left/right groups based on names.
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is None:
        dof_names = robot.get_dof_names()
    left: List[int] = []
    right: List[int] = []
    for i, name in enumerate(dof_names):
        lname = name.lower()
        if "wheel" not in lname:
            continue
        if "left" in lname:
            left.append(i)
        elif "right" in lname:
            right.append(i)
    if not left or not right:
        wheel_like = [i for i, n in enumerate(dof_names) if "wheel" in n.lower()]
        raise RuntimeError(
            "Could not split wheel joints into left/right groups. "
            f"wheel_like={wheel_like}, dof_names={dof_names}"
        )
    return left, right


def main() -> None:
    robot_kind = sys.argv[1] if len(sys.argv) > 1 else "ridgeback_franka"

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("get_assets_root_path() failed. Check assets setup.")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    prim_path = "/ridgeback"
    usd_candidates = _usd_candidates(robot_kind, assets_root)
    _try_add_reference(usd_candidates, prim_path)

    stage = omni.usd.get_context().get_stage()
    art_path = _find_articulation_root(stage, prim_path)
    if art_path is None:
        art_path, found = _find_articulation_root_any(stage, name_hint="ridgeback")
        if art_path is None:
            raise RuntimeError(
                f"No articulation root found under {prim_path}. "
                f"Found articulation roots: {found}. "
                "Check the USD or adjust the prim path candidates."
            )
        print(f"[INFO] Using articulation root: {art_path}")

    robot = SingleArticulation(prim_path=art_path, name=robot_kind)
    world.scene.add(robot)

    world.reset()
    robot.initialize()

    left_indices, right_indices = _find_wheel_groups(robot)
    controller = robot.get_articulation_controller()

    # Adjust these if the motion looks off for your model.
    wheel_radius = 0.165  # [m]
    wheel_base = 0.60     # [m] distance between left/right wheels

    t = 0.0
    dt = world.get_physics_dt()

    while simulation_app.is_running():
        world.step(render=True)

        v = 0.6  # linear velocity [m/s]
        w = 0.4 * math.sin(t * 0.6)  # angular velocity [rad/s]

        wl = (v - 0.5 * wheel_base * w) / wheel_radius
        wr = (v + 0.5 * wheel_base * w) / wheel_radius

        joint_indices = np.array(left_indices + right_indices, dtype=np.int32)
        joint_velocities = np.array(
            [wl] * len(left_indices) + [wr] * len(right_indices), dtype=np.float32
        )

        controller.apply_action(
            ArticulationAction(
                joint_velocities=joint_velocities,
                joint_indices=joint_indices,
            )
        )

        t += dt

    simulation_app.close()


if __name__ == "__main__":
    main()
