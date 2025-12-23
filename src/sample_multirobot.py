"""
Isaac Sim 4.5 Standalone sample:
- 10x Franka (manipulators)
- + 10x mobile robots (Nova Carter ROS, RTX Lidar already included in the USD)
- Move each mobile robot with differential drive wheel velocities

Run:
  (isaacsim) $ python send_sample.py

Notes:
- Import *only* SimulationApp before starting it. (Avoid omni/pxr imports before SimulationApp)
- Asset paths assume Nucleus asset root returned by get_assets_root_path().
"""
import math
import numpy as np
from isaacsim.simulation_app import SimulationApp
# You can set headless=True if you don't need GUI
simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils import stage as stage_utils
from isaacsim.core.utils.types import ArticulationAction


def _try_add_reference(usd_candidates, prim_path: str) -> str:
    """Try multiple USD paths; return the one that worked."""
    last_err = None
    for usd_path in usd_candidates:
        try:
            stage_utils.add_reference_to_stage(usd_path, prim_path)
            print(f"[OK] referenced: {usd_path} -> {prim_path}")
            return usd_path
        except Exception as e:
            last_err = e
            print(f"[WARN] failed to reference: {usd_path} ({type(e).__name__}: {e})")
    raise RuntimeError(f"Failed to add reference for {prim_path}. Last error: {last_err}")


def _grid_xy(i: int, cols: int, spacing: float, origin_xy=(0.0, 0.0)):
    x0, y0 = origin_xy
    r = i // cols
    c = i % cols
    return (x0 + c * spacing, y0 + r * spacing)


def main():
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("get_assets_root_path() failed. Check Nucleus connection / assets setup.")

    # --- World ---
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- 10x Franka ---
    franka_usd_candidates = [
        f"{assets_root}/Isaac/Robots/Franka/franka_alt_fingers.usd",
        f"{assets_root}/Isaac/Robots/Franka/franka.usd",
    ]
    frankas = []
    for i in range(10):
        prim_path = f"/World/envs/franka_{i}"
        _try_add_reference(franka_usd_candidates, prim_path)

        robot = SingleArticulation(prim_path=prim_path, name=f"franka_{i}")
        world.scene.add(robot)
        frankas.append(robot)

    # --- 10x Mobile robots with Lidar (Nova Carter ROS) ---
    # In Isaac Sim, this USD typically includes RTX Lidar prim(s) already.
    carter_usd_candidates = [
        f"{assets_root}/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd",
        f"{assets_root}/Isaac/Samples/ROS2/Robots/Nova_Carter.usd",
        f"{assets_root}/Isaac/Robots/Carter/carter_v2.usd",
        f"{assets_root}/Isaac/Robots/Carter/carter.usd",
    ]
    carters = []
    for i in range(10):
        prim_path = f"/World/envs/carter_{i}"
        _try_add_reference(carter_usd_candidates, prim_path)

        robot = SingleArticulation(prim_path=prim_path, name=f"carter_{i}")
        world.scene.add(robot)
        carters.append(robot)

    # Reset + initialize (important)
    world.reset()
    for r in frankas + carters:
        r.initialize()

    # --- Place robots to avoid collisions ---
    # Frankas: near origin
    for i, r in enumerate(frankas):
        x, y = _grid_xy(i, cols=5, spacing=1.8, origin_xy=(0.0, 0.0))
        r.set_world_pose(position=np.array([x, y, 0.0], dtype=np.float32))

    # Carters: shifted in +Y
    for i, r in enumerate(carters):
        x, y = _grid_xy(i, cols=5, spacing=2.0, origin_xy=(0.0, 12.0))
        r.set_world_pose(position=np.array([x, y, 0.0], dtype=np.float32))

    # --- Controllers ---
    franka_ctrls = [r.get_articulation_controller() for r in frankas]
    carter_ctrls = [r.get_articulation_controller() for r in carters]

    # --- Carter wheel DOF indices (common names for Nova Carter) ---
    # If your USD uses different names, print r.dof_names and update here.
    wheel_name_candidates = [
        ("joint_wheel_left", "joint_wheel_right"),
        ("left_wheel_joint", "right_wheel_joint"),
        ("wheel_left_joint", "wheel_right_joint"),
    ]
    carter_wheel_indices = []
    for i, r in enumerate(carters):
        found = False
        for left_name, right_name in wheel_name_candidates:
            try:
                li = r.get_dof_index(left_name)
                ri = r.get_dof_index(right_name)
                carter_wheel_indices.append((li, ri))
                found = True
                break
            except Exception:
                pass
        if not found:
            # Fallback: try to guess by substring "wheel"
            dof_names = getattr(r, "dof_names", None)
            if dof_names is None:
                try:
                    dof_names = r.get_dof_names()
                except Exception:
                    dof_names = []
            wheel_like = [j for j, n in enumerate(dof_names) if "wheel" in n.lower()]
            raise RuntimeError(
                f"Could not find wheel DOFs for carter_{i}. "
                f"Known candidates failed. wheel-like indices={wheel_like}, dof_names={dof_names}"
            )

    # Differential drive geometry (typical values for Carter)
    wheel_radius = 0.14     # [m]
    wheel_base = 0.413      # [m] distance between wheels (approx)

    # Loop
    t = 0.0
    dt = world.get_physics_dt()

    while simulation_app.is_running():
        world.step(render=True)

        # --- Franka: small sinusoidal motion on 2nd joint (9 DOF array) ---
        for i, ctrl in enumerate(franka_ctrls):
            q = np.zeros(9, dtype=np.float32)
            q[1] = 0.5 * math.sin(t + 0.3 * i)
            ctrl.apply_action(ArticulationAction(joint_positions=q))

        # --- Carters: move forward with a slight, phase-shifted turn ---
        for i, (r, ctrl, (li, ri)) in enumerate(zip(carters, carter_ctrls, carter_wheel_indices)):
            v = 0.6  # linear velocity [m/s]
            w = 0.5 * math.sin(t * 0.6 + 0.4 * i)  # angular velocity [rad/s]

            wl = (v - 0.5 * wheel_base * w) / wheel_radius  # left wheel rad/s
            wr = (v + 0.5 * wheel_base * w) / wheel_radius  # right wheel rad/s

            ctrl.apply_action(
                ArticulationAction(
                    joint_velocities=np.array([wl, wr], dtype=np.float32),
                    joint_indices=np.array([li, ri], dtype=np.int32),
                )
            )

        t += dt

    simulation_app.close()

if __name__ == "__main__":
    main()
