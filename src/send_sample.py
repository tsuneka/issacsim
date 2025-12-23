from isaacsim.simulation_app import SimulationApp

# 重要：SimulationAppより前に omni / pxr / isaacsim.* を大量importしない
simulation_app = SimulationApp({"headless": False})

import math
import numpy as np

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path  # 4.5系はこっち推奨（警告回避）


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Assets root not found. Nucleus/asset path settings are not available.")

    usd_path = assets_root + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

    robots: list[SingleArticulation] = []
    controllers = []

    # 10台作る（/World/envs/env_0/panda ... env_9/panda）
    for i in range(10):
        prim_path = f"/World/envs/env_{i}/panda"
        stage_utils.add_reference_to_stage(usd_path, prim_path)

        robot = SingleArticulation(prim_path=prim_path, name=f"franka_{i}")
        world.scene.add(robot)
        robots.append(robot)

    world.reset()

    # シミュ開始後に initialize（これで articulation view / controller が有効化される）
    for r in robots:
        r.initialize()

    # 位置を散らして衝突を避ける（5×2グリッド）
    for i, r in enumerate(robots):
        x = (i % 5) * 1.2
        y = (i // 5) * 1.2
        r.set_world_pose(position=np.array([x, y, 0.0]))

    # 各ロボットの controller を取る（SingleArticulationが暗黙に持ってる）
    for r in robots:
        controllers.append(r.get_articulation_controller())

    t = 0.0
    dt = world.get_physics_dt()

    # ループ
    while simulation_app.is_running():
        world.step(render=True)

        # 例：各ロボットの第2関節あたりを位相ずらしでサイン駆動（9DoF分の配列を送る）
        for i, ctrl in enumerate(controllers):
            q = np.zeros(9, dtype=np.float32)
            q[1] = 0.5 * math.sin(t + 0.3 * i)  # 2nd joint

            action = ArticulationAction(joint_positions=q)
            ctrl.apply_action(action)

        t += dt

    simulation_app.close()


if __name__ == "__main__":
    main()
