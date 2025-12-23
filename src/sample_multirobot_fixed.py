#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isaac Sim 4.5 standalone sample: spawn 10 Carter (with lidar in the USD) and drive them.
Fixes:
- Carter prim_path may NOT be an articulation root at "/World/envs/carter_i".
  We search under the referenced prim for the actual UsdPhysics ArticulationRoot and use that path.
- If you use *_ROS.usd, you must enable the ROS2 bridge extension; otherwise you'll see lots of
  "Could not find node type interface for isaacsim.ros2.bridge.*" warnings.
"""

import os
import numpy as np

from isaacsim import SimulationApp

# NOTE: RTX lidar/cameras are heavy. For multi-robot, start with non-RTX renderer.
# If you must use RTX sensors, switch renderer to "RayTracedLighting".
RENDERER = os.environ.get("ISAACSIM_RENDERER", "Default")  # "Default" or "RayTracedLighting"
HEADLESS = bool(int(os.environ.get("ISAACSIM_HEADLESS", "0")))

simulation_app = SimulationApp({"headless": HEADLESS, "renderer": RENDERER})

# Must import pxr AFTER SimulationApp
from pxr import Usd, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

# -----------------------
# Helpers
# -----------------------
def find_articulation_root_under(stage: Usd.Stage, root_path: str) -> str | None:
    """Return the first prim path under root_path that has UsdPhysics.ArticulationRootAPI applied."""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    # If the root itself is an articulation root:
    if root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return root_path

    # Search descendants
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    return None


def main():
    # If you use ROS USDs (Nova_Carter_ROS.usd etc.), enable ROS2 bridge to avoid missing node warnings.
    # Isaac Sim 4.5 uses "isaacsim.ros2.bridge" (older versions used "omni.isaac.ros2_bridge").
    if os.environ.get("ISAACSIM_ENABLE_ROS2", "0") == "1":
        enable_extension("isaacsim.ros2.bridge")
    print("enable ros2_bridge")

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )

    # Ground
    world.scene.add_default_ground_plane()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError(
            "Could not find Isaac Sim assets root (Nucleus). "
            "Make sure Nucleus is reachable or set up the assets path."
        )

    # Choose your Carter asset. If you don't need ROS graph + topics, prefer a non-ROS USD (lighter).
    # Example candidates (may differ by install):
    #   f"{assets_root_path}/Isaac/Robots/NVIDIA/Nova_Carter/Nova_Carter.usd"
    #   f"{assets_root_path}/Isaac/Robots/NVIDIA/Nova_Carter/Nova_Carter_ROS.usd"
    # In your original script you used:
    carter_usd_path = f"{assets_root_path}/Isaac/Robots/NVIDIA/Carter/carter_v2.usd"

    stage = get_current_stage()

    # Spawn 10 Carters in a grid
    num = 10
    spacing = 2.0
    carters: list[SingleArticulation] = []
    carter_art_paths: list[str] = []

    for i in range(num):
        x = (i % 5) * spacing
        y = (i // 5) * spacing

        env_path = f"/World/envs/carter_{i}"
        XFormPrim(prim_path=env_path, name=f"env_carter_{i}", position=np.array([x, y, 0.0]))

        # Reference the asset under env_path (this creates the prim, then references the USD)
        add_reference_to_stage(usd_path=carter_usd_path, prim_path=env_path)

    # Let USD references resolve a few frames (important when loading from Nucleus/remote)
    for _ in range(10):
        simulation_app.update()

    # Now find articulation roots under each env and register them in the scene
    for i in range(num):
        env_path = f"/World/envs/carter_{i}"
        art_path = find_articulation_root_under(stage, env_path)
        if art_path is None:
            # Fallback: show what exists so you can adjust quickly
            prim = stage.GetPrimAtPath(env_path)
            children = [c.GetName() for c in prim.GetChildren()] if prim.IsValid() else []
            raise RuntimeError(
                f"Could not find an articulation root under {env_path}. "
                f"Check the USD you referenced. Children at {env_path}: {children}"
            )

        carter_art_paths.append(art_path)
        carter = SingleArticulation(prim_path=art_path, name=f"carter_{i}")
        world.scene.add(carter)
        carters.append(carter)

    # Finalize physics
    world.reset()

    # Initialize handles
    for carter in carters:
        carter.initialize()

    # Drive each Carter: simple pattern (circle-ish) by commanding wheel DOFs
    # NOTE: DOF order depends on the Carter USD. We'll discover wheel DOFs by name.
    wheel_name_candidates = ("left", "right", "wheel", "caster")
    dof_names = carters[0].get_dof_names()
    wheel_dofs = [idx for idx, n in enumerate(dof_names) if any(k in n.lower() for k in wheel_name_candidates)]
    print("DOF names:", dof_names)
    print("Wheel-ish DOF indices:", wheel_dofs)

    # If your USD uses 2 wheel joints (diff-drive), you'll likely see 2 indices.
    # If it uses more, you may need to pick the correct ones.
    if len(wheel_dofs) < 2:
        print("WARNING: Could not auto-detect 2 wheel joints. "
              "Please inspect DOF names above and set wheel indices manually.")

    t = 0.0
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            t += world.get_physics_dt()

            for i, carter in enumerate(carters):
                # Basic per-robot speed profile
                v = 5.0 + 2.0 * np.sin(t + i * 0.3)
                w = 2.0 * np.cos(t * 0.7 + i * 0.2)

                # Convert (v,w) to left/right wheel angular velocities (very rough; tune for your model)
                left = v - w
                right = v + w

                if len(wheel_dofs) >= 2:
                    qd = np.zeros(carter.num_dof)
                    qd[wheel_dofs[0]] = left
                    qd[wheel_dofs[1]] = right
                    action = ArticulationAction(joint_velocities=qd)
                    carter.apply_action(action)

    simulation_app.close()


if __name__ == "__main__":
    main()
