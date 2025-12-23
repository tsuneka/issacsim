import isaacsim
isaacsim.initialize()
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import time

# ======================
# World 作成
# ======================
world = World(stage_units_in_meters=1.0)

ROBOT_USD = "/Isaac/Robots/Franka/franka.usd"
robots = []

# ======================
# ロボットを10台配置
# ======================
for i in range(10):
    prim_path = f"/World/Robot_{i}"
    add_reference_to_stage(ROBOT_USD, prim_path)

    robot = Articulation(
        prim_path=prim_path,
        name=f"robot_{i}",
        position=np.array([i * 1.2, 0.0, 0.0])  # 横に並べる
    )
    robots.append(robot)

world.reset()

# ======================
# メインループ
# ======================
for step in range(1000):
    world.step(render=False)

    for i, robot in enumerate(robots):
        # 各ロボットに「異なる指令」
        target_q = 0.2 * np.sin(0.01 * step + i) * np.ones(robot.num_dof)
        robot.set_joint_positions(target_q)

    time.sleep(0.01)

simulation_app.close()
