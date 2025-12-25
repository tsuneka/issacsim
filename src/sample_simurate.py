#!/usr/bin/env python3
"""Run Nova Carter with ROS2 cmd_vel, joint state + pointcloud publishing."""

from types import SimpleNamespace
from typing import List, Tuple

from isaacsim.simulation_app import SimulationApp

# -----------------------------------------------------------------------------
# Start Kit first (IMPORTANT: do this before importing omni/pxr modules)
# -----------------------------------------------------------------------------
CONFIG = {
    "headless": False,
    "renderer": "RayTracedLighting",  # safe default; change if you want
}
simulation_app = SimulationApp(CONFIG)

import rclpy
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

from ros2_sample import CarterWheelCmdNode, CarterPubNode
from sample_multirobot_ros2_lidar import _assets_carter_usd, _make_env_paths, _spawn_carter_with_lidar

def _enable_ros2_bridge() -> None:
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

def _find_wheel_indices(robot: SingleArticulation) -> Tuple[int, int]:
    wheel_name_candidates = [
        ("joint_wheel_left", "joint_wheel_right"),
        ("left_wheel_joint", "right_wheel_joint"),
        ("wheel_left_joint", "wheel_right_joint"),
    ]
    for left_name, right_name in wheel_name_candidates:
        try:
            li = robot.get_dof_index(left_name)
            ri = robot.get_dof_index(right_name)
            return li, ri
        except Exception:
            continue
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is None:
        try:
            dof_names = robot.get_dof_names()
        except Exception:
            dof_names = []
    wheel_like = [j for j, n in enumerate(dof_names) if "wheel" in n.lower()]
    raise RuntimeError(
        f"Could not find wheel DOFs. wheel-like indices={wheel_like}, dof_names={dof_names}"
    )

def main() -> None:
    _enable_ros2_bridge()
    rclpy.init()

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    carter_usd = _assets_carter_usd()

    # Spawn a single Carter + LiDAR (adjust N for multiple robots)
    N = 1
    carters: List[SingleArticulation] = []
    lidar_ctxs: List[SimpleNamespace] = []

    for i in range(N):
        env_path, carter_path = _make_env_paths(i)
        ctx = SimpleNamespace()
        _spawn_carter_with_lidar(ctx, i, carter_usd, (0.0, 0.0))
        robot = SingleArticulation(prim_path=carter_path, name=f"carter_{i}")
        world.scene.add(robot)
        carters.append(robot)
        lidar_ctxs.append(ctx)

    world.reset()
    for robot in carters:
        robot.initialize()

    controllers = [r.get_articulation_controller() for r in carters]
    wheel_indices = [_find_wheel_indices(r) for r in carters]

    cmd_nodes: List[CarterWheelCmdNode] = []
    pub_nodes: List[CarterPubNode] = []

    for i, (robot, ctrl, (li, ri), ctx) in enumerate(
        zip(carters, controllers, wheel_indices, lidar_ctxs)
    ):
        cmd_node = CarterWheelCmdNode(
            ctrl,
            (li, ri),
            topic="/cmd_vel" if N == 1 else f"/carter_{i}/cmd_vel",
            node_name=f"carter_cmd_node_{i}",
        )
        cmd_node.start()
        cmd_nodes.append(cmd_node)

        joint_names = robot.get_dof_names()
        pub_node = CarterPubNode(
            lidar_path=ctx.lidar_path,
            joint_names=joint_names,
            pointcloud_topic="utlidar/cloud" if N == 1 else f"/carter_{i}/pointcloud",
            frame_id=f"carter_{i}_lidar",
            node_name=f"carter_pub_node_{i}",
        )
        pub_nodes.append(pub_node)

    timeline = omni.timeline.get_timeline_interface()

    try:
        while simulation_app.is_running():
            world.step(render=True)
            sim_time_sec = timeline.get_current_time()

            for robot, cmd_node, pub_node in zip(carters, cmd_nodes, pub_nodes):
                cmd_node.apply()
                joint_pos = robot.get_joint_positions()
                joint_vel = robot.get_joint_velocities()
                pub_node.publish(sim_time_sec, joint_pos=joint_pos, joint_vel=joint_vel)
    finally:
        rclpy.shutdown()
        simulation_app.close()

if __name__ == "__main__":
    main()
