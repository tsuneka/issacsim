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
from isaacsim.storage.native import get_assets_root_path # moved here in newer Isaac Sim stacks
from isaacsim.sensors.physx import _range_sensor  # PhysX ベースの LiDAR センサー IF

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()  # LiDAR センサー IF を取得

LIDAR_SCAN_FREQ = 180.0  # 回転周波数 [Hz]
LIDAR_H_RES = 2.0  # 水平分解能 [deg]
LIDAR_V_RES = 2.0  # 垂直分解能 [deg]

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
    root = get_assets_root_path()

    if root:
        return f"{root}/Isaac/Robots/Carter/nova_carter.usd"

    return "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Carter/nova_carter.usd"

def _make_env_paths(i: int) -> Tuple[str, str]:
    env_path = f"/World/envs/env_{i}"
    carter_path = f"{env_path}/carter"
    return env_path, carter_path

def _spawn_carter_with_lidar(self, i: int, carter_usd: str, xy: Tuple[float, float]) -> Dict[str, str]:
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
    lidar_path = f"{carter_path}/lidar"
    self.lidar_path = lidar_path
    if not is_prim_path_valid(lidar_path):

        # IMPORTANT: orientation must be quaternion (GfQuatd), not Euler (GfVec3d)
        omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=lidar_path,
            parent=None,
            min_range=0.05,  # 最小計測距離
            max_range=30.0,  # 最大計測距離
            draw_points=True,  # ポイント表示を有効化
            draw_lines=False,  # 線表示を無効化
            horizontal_fov=360.0,  # 水平視野角
            vertical_fov=180.0,  # 垂直視野角
            horizontal_resolution=LIDAR_H_RES,  # 水平分解能
            vertical_resolution=LIDAR_V_RES,  # 垂直分解能
            rotation_rate=LIDAR_SCAN_FREQ,  # 回転速度
            high_lod=True,  # 高 LOD を使用
            yaw_offset=0.0,  # ヨーオフセット
            enable_semantics=False,  # セマンティクス無効
        )

def get_head_lidar_pointcloud(self):  # LiDAR の点群を取得し前処理して返す
    points = lidarInterface.get_point_cloud_data(self.lidar_path)  # 点群データを取得
    depths = lidarInterface.get_linear_depth_data(self.lidar_path)  # 深度データを取得

    H = points.shape[0]  # 水平ステップ数
    V = points.shape[1]  # 垂直ステップ数

    # assume each vertical scan is a unique ID
    beam_ids = np.tile(np.arange(V), (H, 1))  # 垂直方向毎にビーム ID を割り振り

    # each vertical scan is a separate timestamp; assume timestamps are
    # evenly spaced across a timestep
    sweep_deg = (H - 1) * LIDAR_H_RES  # 1スキャンで回転する角度
    t = sweep_deg / 360.0 / LIDAR_SCAN_FREQ  # 1スキャンに要する時間
    timestamps = np.tile(
        np.linspace(0, t, H, endpoint=False)[:, np.newaxis],
        (1, V),
    )  # 各水平ステップに均等なタイムスタンプを付与

    # flatten to (H*V, dim)
    points = points.reshape(-1, 3)  # 点群を一次元に展開
    depths = depths.reshape(-1)  # 深度を展開
    beam_ids = beam_ids.reshape(-1)  # ビーム ID を展開
    timestamps = timestamps.reshape(-1)  # タイムスタンプを展開

    # remove points with no hits
    intens_mask = lidarInterface.get_intensity_data(self.lidar_path).reshape(-1)  # 反射強度を取得
    intens_mask = (intens_mask > 0).astype(np.bool_)  # ヒットがある点のみ True

    # remove points with zenith angle outside of [-pi/2, 0], since we used
    # a 180-degree vertical FOV
    zenith = lidarInterface.get_zenith_data(self.lidar_path)[np.newaxis, :]  # (1, V)  # 天頂角データを取得
    zenith_mask = np.tile((zenith >= -np.pi / 2) & (zenith <= 0), (H, 1))  # (H, V)  # 垂直視野に収まるマスク
    zenith_mask = zenith_mask.reshape(-1)  # Flatten to (H*V,)  # 一次元に展開

    mask = intens_mask & zenith_mask  # 強度と視野の両方を満たす点
    points = points[mask]  # マスクで点群をフィルタ
    depths = depths[mask]  # 深度をフィルタ
    beam_ids = beam_ids[mask]  # ビーム ID をフィルタ
    timestamps = timestamps[mask]  # タイムスタンプをフィルタ

    # assume intensity is inversely proportional to the square of depth
    intensities = 100 * 1 / (1 + depths**2)  # 距離に反比例する強度を仮定して計算

    pcl = np.column_stack((points, intensities, beam_ids, timestamps))  # 点群に強度・ID・時刻を付加

    return pcl  # PointCloud2 生成前の numpy 配列を返す


# def main() -> None:
#     # Enable only what we actually need.
#     # (Do NOT enable 'isaacsim.ros2.core' — it doesn't exist in Isaac Sim 4.5 registries.)
#     _enable_extension("isaacsim.ros2.bridge")
#     _enable_extension("isaacsim.sensors.rtx")

#     # If you rely on system rclpy (ROS 2 Humble apt install), this helps the bridge pick it up.
#     # In your shell, you can also set:
#     if os.environ.get("ISAACSIM_ENABLE_ROS2") != "1":
#         carb.log_warn("ISAACSIM_ENABLE_ROS2 is not set to '1'. ROS2 bridge may not fully initialize.")

#     world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
#     world.scene.add_default_ground_plane()

#     # Spawn
#     carter_usd = _assets_carter_usd()
#     carb.log_info(f"[assets] carter_usd = {carter_usd}")

#     N = 6
#     spacing = 2.5
#     spawned = []
#     for i in range(N):
#         x = (i % 3) * spacing
#         y = (i // 3) * spacing
#         info = _spawn_carter_with_lidar(i, carter_usd, (x, y))
#         spawned.append(info)
#         print(f"[OK] carter[{i}] = {info['carter']}, lidar = {info['lidar']}")

#     # Let Kit settle extensions and stage composition
#     for _ in range(10):
#         world.step(render=True)

#     carb.log_info("Running... (Ctrl+C to stop)")
#     try:
#         while simulation_app.is_running():
#             world.step(render=True)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         world.stop()
#         simulation_app.close()

# if __name__ == "__main__":
#     main()
