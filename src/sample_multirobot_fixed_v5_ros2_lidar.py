#!/usr/bin/env python3
"""
Isaac Sim 4.5.0 (Python 3.10) – Multi-Carter + RTX Lidar + ROS2 PointCloud2 publish

What this does:
- Spawns N Carter robots under /World/envs/env_i/carter
- Adds one RTX Lidar under each Carter (child prim)
- Publishes each lidar as sensor_msgs/PointCloud2 on a unique ROS2 topic

Run (example):
  # Make sure ROS2 env is sourced in the same shell (Humble)
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

  # Enable ROS2 bridge (optional; script also enables extensions)
  export ISAACSIM_ENABLE_ROS2=1

  python sample_multirobot_fixed_v4_ros2_lidar.py

Then in another terminal:
  source /opt/ros/humble/setup.bash
  ros2 topic list
  ros2 topic echo /carter_0/pointcloud --once
"""

import math
import time
from typing import List

import numpy as np

from isaacsim import SimulationApp

# --------------------------------------------------------------------------------------
# App start
# --------------------------------------------------------------------------------------
simulation_app = SimulationApp({"headless": False})

# Enable ROS2 + RTX lidar extensions explicitly (prevents "node type interface not found")

ext_mgr = omni.kit.app.get_app().get_extension_manager()
for ext_name in [
    "isaacsim.ros2.core",
    "isaacsim.ros2.bridge",
    "isaacsim.sensors.rtx",
    "isaacsim.replicator.writers",
]:
    try:
        ext_mgr.set_extension_enabled_immediate(ext_name, True)
    except Exception:
        # Older Kit builds sometimes only have the non-immediate setter
        try:
            ext_mgr.set_extension_enabled(ext_name, True)
        except Exception as e:
            print(f"[WARN] Could not enable extension {ext_name}: {e}")

# Now import Isaac Sim APIs that depend on enabled extensions
from pxr import UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
NUM_ROBOTS = 10  # number of Carter robots to spawn
# Arrange env origins in a near-square grid so 10 robots don't overlap
GRID = math.ceil(math.sqrt(NUM_ROBOTS))
SPACING = 3.0

LIDAR_CONFIG = "Example_Rotary"   # built-in RTX lidar config name
LIDAR_REL_POS = (0.25, 0.0, 0.35) # relative to carter root
# IMPORTANT:
# IsaacSensorCreateRtxLidar expects a **quaternion** (pxr.Gf.Quatd) for orientation.
# If you pass Euler angles (Vec3), you'll get:
#   expected 'GfQuatd', got 'GfVec3d'
LIDAR_REL_RPY_DEG = (0.0, 0.0, 0.0)  # roll, pitch, yaw in degrees

# Publish topics:
TOPIC_FMT = "/carter_{i}/pointcloud"
FRAME_FMT = "carter_{i}/rtx_lidar"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _enable_ros2_domain_time():
    """Optional: ensure sim publishes /clock for ROS2 nodes that depend on it."""
    # Many ROS2 consumers like Nav2 expect /clock when using use_sim_time:=true.
    # If you need it, you can add an OmniGraph pipeline, but for PointCloud2 only it isn't mandatory.
    pass


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Gf.Quatd:
    """Convert roll/pitch/yaw (degrees) -> pxr.Gf.Quatd.

    IsaacSensorCreateRtxLidar requires a quaternion (GfQuatd) for orientation.
    """
    import math

    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    # Z (yaw) * Y (pitch) * X (roll)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    yq = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return Gf.Quatd(w, Gf.Vec3d(x, yq, z))


def _get_carter_usd() -> str:
    assets_root = get_assets_root_path()
    if assets_root:
        # Default Isaac asset location on Nucleus
        return assets_root + "/Isaac/Robots/Carter/carter_v2.usd"
    # Fallback (offline/local assets): user can set ISAACSIM_ASSETS_ROOT to a local path
    env_root = os.environ.get("ISAACSIM_ASSETS_ROOT", "")
    if env_root:
        return env_root.rstrip("/") + "/Isaac/Robots/Carter/carter_v2.usd"
    raise RuntimeError(
        "Could not resolve assets root. Start Isaac Sim with assets available, "
        "or set ISAACSIM_ASSETS_ROOT to your local Isaac assets directory."
    )

def _set_xform_translation(prim_path: str, xyz: np.ndarray):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    # Create a translate op if needed
    t_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
            break
    if t_op is None:
        t_op = xform.AddTranslateOp()
    t_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _spawn_carter_with_lidar(i: int, env_path: str, carter_usd: str):
    """
    Create:
      /World/envs/env_i           (Xform)
      /World/envs/env_i/carter    (Carter USD reference)
      /World/envs/env_i/carter/rtx_lidar  (RTX Lidar)
    And attach ROS2 writer for PointCloud2.
    """
    # Ensure env Xform exists
    if not is_prim_path_valid(env_path):
        create_prim(env_path, "Xform")

    carter_path = f"{env_path}/carter"
    if not is_prim_path_valid(carter_path):
        add_reference_to_stage(carter_usd, carter_path)

    # Create RTX lidar as a child prim of Carter root
    lidar_path = f"{carter_path}/rtx_lidar"
    if not is_prim_path_valid(lidar_path):
        omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=lidar_path,
            parent=None,
            config=LIDAR_CONFIG,
            translation=Gf.Vec3d(*LIDAR_REL_POS),
            orientation=_quat_from_rpy_deg(*LIDAR_REL_RPY_DEG),
        )

    # Replicator render product from the lidar prim
    stage = get_current_stage()
    lidar_prim = stage.GetPrimAtPath(lidar_path)
    if not lidar_prim.IsValid():
        raise RuntimeError(f"RTX Lidar prim not created: {lidar_path}")

    # 1x1 render product is enough for lidar data
    rp = rep.create.render_product(lidar_prim.GetPath(), [1, 1], name=f"carter_{i}_rtx_lidar")

    # ROS2 PointCloud2 writer
    writer = rep.WriterRegistry.get("RtxLidar" + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=FRAME_FMT.format(i=i),
        topicName=TOPIC_FMT.format(i=i),
        queueSize=10,
    )
    writer.attach([rp])

    return carter_path, lidar_path


def main():
    # World
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    carter_usd = _get_carter_usd()

    # Root group
    create_prim("/World/envs", "Xform")

    carter_paths: List[str] = []
    for i in range(NUM_ROBOTS):
        gx = i % GRID
        gy = i // GRID
        env_path = f"/World/envs/env_{i}"
        create_prim(env_path, "Xform")

        # Place each env
        x = gx * SPACING
        y = gy * SPACING
        _set_xform_translation(env_path, np.array([x, y, 0.0]))

        carter_path, lidar_path = _spawn_carter_with_lidar(i, env_path, carter_usd)
        carter_paths.append(carter_path)
        print(f"[OK] carter[{i}] = {carter_path}, lidar = {lidar_path}, topic = {TOPIC_FMT.format(i=i)}")

    # Important: let Isaac finalize & build physics/render graphs
    world.reset()

    # Simple demo motion: move env origins in a slow circle (kinematic) so lidar changes
    t0 = time.time()
    while simulation_app.is_running():
        t = time.time() - t0
        for i in range(NUM_ROBOTS):
            gx = i % GRID
            gy = i // GRID
            base = np.array([gx * SPACING, gy * SPACING, 0.0])
            wobble = 0.3 * np.array([math.cos(t + i), math.sin(t + i), 0.0])
            _set_xform_translation(f"/World/envs/env_{i}", base + wobble)

        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    import os
    main()
