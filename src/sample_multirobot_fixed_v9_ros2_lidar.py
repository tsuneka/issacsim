#!/usr/bin/env python3
"""
Multi-Carter spawn + RTX Lidar + ROS2 PointCloud publish (Isaac Sim 4.5.0, Python 3.10)

Key points for Isaac Sim 4.5:
- Avoid deprecated `omni.isaac.*` python modules; use `isaacsim.*` modules instead.
- Enable ROS2 via the extension `isaacsim.ros2.bridge` (do NOT enable `isaacsim.ros2.core`).
- Start SimulationApp BEFORE importing omni/pxr/isaacsim heavy modules to avoid startup warnings.

Tested for the user's environment pattern:
  isaacsim==4.5.0 (pip), Python 3.10, ROS 2 Humble (system rclpy)
"""

from isaacsim.simulation_app import SimulationApp

# -----------------------------------------------------------------------------
# Start Kit first (IMPORTANT: do this before importing omni/pxr modules)
# -----------------------------------------------------------------------------
CONFIG = {
    "headless": False,
    "renderer": "RayTracedLighting",  # safe default; change if you want
}
simulation_app = SimulationApp(CONFIG)

# -----------------------------------------------------------------------------
# Imports that touch omni/pxr AFTER SimulationApp has started
# -----------------------------------------------------------------------------
import os
import time
from typing import Tuple, Dict

import numpy as np
import omni
import carb
from pxr import Gf

import omni.replicator.core as rep

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim, is_prim_path_valid
from isaacsim.storage.native import get_assets_root_path

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _enable_extension(ext_name: str) -> None:
    """Enable a Kit extension if it exists in the local registries."""
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()

    # If the extension doesn't exist, do nothing (prevents hard failure).
    ext_id = ext_mgr.get_extension_id_by_module(ext_name)
    if ext_id is None:
        carb.log_warn(f"[ext] '{ext_name}' not found in extension registry (skipping).")
        return

    # "immediate" enables without requiring app restart.
    try:
        ext_mgr.set_extension_enabled_immediate(ext_id, True)
    except Exception:
        # Fallback for older Kit builds
        ext_mgr.set_extension_enabled(ext_id, True)
        app.update()


def _assets_carter_usd() -> str:
    """
    Return a Carter USD path.
    - Prefer local assets root from get_assets_root_path().
    - Fall back to the NVIDIA hosted HTTPS asset (may fail if your env blocks external HTTPS).
    """
    root = get_assets_root_path()
    if root:
        # Typical path layout (depends on your assets installation)
        # e.g. omniverse://localhost/NVIDIA/Assets/Isaac/4.5
        candidate = f"{root}/Isaac/Robots/Carter/carter_v2.usd"
        return candidate

    # fallback (requires external network access)
    return "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Carter/carter_v2.usd"


def _make_env_paths(i: int) -> Tuple[str, str]:
    env_path = f"/World/envs/env_{i}"
    carter_path = f"{env_path}/carter"
    return env_path, carter_path


def _spawn_carter_with_lidar(i: int, carter_usd: str, xy: Tuple[float, float]) -> Dict[str, str]:
    """
    Create:
      /World/envs/env_i         (Xform)
      /World/envs/env_i/carter  (USD reference)
      /World/envs/env_i/carter/rtx_lidar (RTX Lidar sensor)
    And connect RTX lidar to ROS2 PointCloud via replicator writer.
    """
    env_path, carter_path = _make_env_paths(i)

    # Ensure /World/envs exists
    if not is_prim_path_valid("/World/envs"):
        create_prim("/World/envs", "Xform")

    # Create env prim
    if not is_prim_path_valid(env_path):
        create_prim(env_path, "Xform")

    # Move env
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=f"{env_path}.xformOp:translate",
        value=Gf.Vec3d(float(xy[0]), float(xy[1]), 0.0),
        prev=None,
    )

    # Carter
    if not is_prim_path_valid(carter_path):
        create_prim(carter_path, "Xform")
        add_reference_to_stage(carter_usd, carter_path)

    # RTX Lidar
    lidar_path = f"{carter_path}/rtx_lidar"
    if not is_prim_path_valid(lidar_path):
        # Minimal RTX Lidar config; tune as needed
        lidar_cfg = {
            "minRange": 0.1,
            "maxRange": 50.0,
            "horizontalFov": 360.0,
            "verticalFov": 30.0,
            "horizontalResolution": 0.4,
            "verticalResolution": 1.0,
            "rotationRate": 10.0,
            "drawPoints": False,
            "drawLines": False,
        }

        # IMPORTANT: orientation must be quaternion (GfQuatd), not Euler (GfVec3d)
        omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lidar_path,
            parent=None,
            config=lidar_cfg,
            translation=Gf.Vec3d(0.0, 0.0, 0.35),
            orientation=Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)),
        )

    # Replicator render product for RTX sensor
    rp = rep.create.render_product(lidar_path, (1, 1))
    rp_path = str(rp)

    # ROS2 PointCloud publisher writer (provided by ROS2 Bridge)
    # Note: the writer name can vary across Isaac Sim builds; if you see "writer not found",
    # open Isaac Sim -> Replicator -> Writers list and adjust the name here.
    writer_name_candidates = [
        "RtxLidarROS2PublishPointCloud",
        "RtxLidarROS2PublishPointCloudWriter",
        "ROS2PublishPointCloud",  # fallback
    ]
    writer = None
    for wname in writer_name_candidates:
        try:
            writer = rep.WriterRegistry.get(wname)
            if writer is not None:
                writer.initialize(
                    # Namespace prefix like /carter_0
                    node_namespace=f"carter_{i}",
                    # topic name under the namespace
                    topic_name="pointcloud",
                    frame_id=f"carter_{i}_lidar",
                )
                writer.attach([rp])
                carb.log_info(f"[ROS2] Using writer='{wname}' for carter_{i}")
                break
        except Exception as e:
            carb.log_warn(f"[ROS2] writer '{wname}' not usable: {e}")

    if writer is None:
        carb.log_warn(
            "[ROS2] No suitable Replicator ROS2 writer found. "
            "LiDAR will exist, but PointCloud may not publish. "
            "Check installed ROS2 Bridge writers for your build."
        )

    return {"env": env_path, "carter": carter_path, "lidar": lidar_path, "render_product": rp_path}


def main() -> None:
    # Enable only what we actually need.
    # (Do NOT enable 'isaacsim.ros2.core' — it doesn't exist in Isaac Sim 4.5 registries.)
    _enable_extension("isaacsim.ros2.bridge")
    _enable_extension("isaacsim.sensors.rtx")

    # If you rely on system rclpy (ROS 2 Humble apt install), this helps the bridge pick it up.
    # In your shell, you can also set:
    #   export ISAACSIM_ENABLE_ROS2=1
    if os.environ.get("ISAACSIM_ENABLE_ROS2") != "1":
        carb.log_warn("ISAACSIM_ENABLE_ROS2 is not set to '1'. ROS2 bridge may not fully initialize.")

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    # Spawn
    carter_usd = _assets_carter_usd()
    carb.log_info(f"[assets] carter_usd = {carter_usd}")

    N = 6
    spacing = 2.5
    spawned = []
    for i in range(N):
        x = (i % 3) * spacing
        y = (i // 3) * spacing
        info = _spawn_carter_with_lidar(i, carter_usd, (x, y))
        spawned.append(info)
        print(f"[OK] carter[{i}] = {info['carter']}, lidar = {info['lidar']}, rp = {info['render_product']}")

    # Let Kit settle extensions and stage composition
    for _ in range(10):
        world.step(render=True)

    carb.log_info("Running... (Ctrl+C to stop)")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    finally:
        world.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
