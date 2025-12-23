from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np

from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction


def main():
    world = World(stage_units_in_meters=1.0)

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find Isaac Sim assets folder (get_assets_root_path returned None).")

    franka_usd = assets_root + "/Isaac/Robots/Franka/franka.usd"

    robots = []
    controllers = []

    # 10台を /World/Franka_00 ... に配置
    for i in range(10):
        prim_path = f"/World/Franka_{i:02d}"
        add_reference_to_stage(usd_path=franka_usd, prim_path=prim_path)

        robot = SingleArticulation(prim_path=prim_path, name=f"franka_{i:02d}")
        robots.append(robot)

    # 重要：reset 後に articulation が初期化される
    world.reset()

    # コントローラ生成（各ロボットに紐づく）
    for r in robots:
        controllers.append(ArticulationController(r))

    # 各ロボットに違う目標を送る（例：joint0 をロボット番号でずらす）
    t = 0.0
    dt = 1.0 / 60.0

    while simulation_app.is_running():
        world.step(render=True)

        t += dt
        for i, r in enumerate(robots):
            dof = r.num_dof
            target = np.zeros(dof, dtype=np.float32)

            # 例：各ロボットで位相が違う sin 指令（joint0だけ動かす）
            target[0] = 0.8 * np.sin(t + 0.4 * i)

            action = ArticulationAction(joint_positions=target)
            controllers[i].apply_action(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
