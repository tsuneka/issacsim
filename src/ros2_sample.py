import threading
from typing import Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from isaacsim.core.utils.types import ArticulationAction
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, JointState
from sensor_msgs_py import point_cloud2
from sample_multirobot_ros2_lidar import get_head_lidar_pointcloud

class CarterWheelCmdNode(Node):
    """Subscribe to Twist and drive Nova Carter wheels (differential drive)."""

    def __init__(
        self,
        controller,
        wheel_indices: Tuple[int, int],
        wheel_radius: float = 0.14,
        wheel_base: float = 0.413,
        topic: str = "/cmd_vel",
        node_name: str = "carter_cmd_node",
    ):
        super().__init__(node_name)
        self._controller = controller
        self._left_idx, self._right_idx = wheel_indices
        self._wheel_radius = wheel_radius
        self._wheel_base = wheel_base

        self._lock = threading.Lock()
        self._v = 0.0
        self._w = 0.0

        self._sub = self.create_subscription(Twist, topic, self._cmd_cb, 10)
        self._thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)

    def _cmd_cb(self, msg: Twist) -> None:
        # Standard diff-drive: use linear.x and angular.z
        with self._lock:
            self._v = msg.linear.x
            self._w = msg.angular.z

    def start(self) -> None:
        self._thread.start()

    def apply(self) -> None:
        # Call this every sim step to push the latest command.
        with self._lock:
            v = self._v
            w = self._w

        wl = (v - 0.5 * self._wheel_base * w) / self._wheel_radius
        wr = (v + 0.5 * self._wheel_base * w) / self._wheel_radius

        self._controller.apply_action(
            ArticulationAction(
                joint_velocities=np.array([wl, wr], dtype=np.float32),
                joint_indices=np.array([self._left_idx, self._right_idx], dtype=np.int32),
            )
        )

class CarterPubNode(Node):
    """Publish clock, joint state, and lidar pointcloud for Nova Carter."""

    def __init__(
        self,
        lidar_path: str,
        joint_names,
        clock_topic: str = "/clock",
        joint_state_topic: str = "/joint_states",
        pointcloud_topic: str = "utlidar/cloud",
        frame_id: str = "carter_lidar",
        node_name: str = "carter_pub_node",
    ):
        super().__init__(node_name)
        self.lidar_path = lidar_path
        self._joint_names = list(joint_names)
        self._frame_id = frame_id

        self._clock_pub = self.create_publisher(Clock, clock_topic, 10)
        self._joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self._pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, 10)

        self._clock_msg = None

    def publish(self, sim_time_sec: float, joint_pos=None, joint_vel=None, pointcloud=None) -> None:
        self._pub_clock(sim_time_sec)
        self._pub_joint_state(joint_pos, joint_vel)
        self._pub_pointcloud(pointcloud)

    def _pub_clock(self, sim_time_sec: float) -> None:
        msg = Clock()
        msg.clock = self.get_clock().now().to_msg()
        msg.clock.sec = int(sim_time_sec)
        msg.clock.nanosec = int((sim_time_sec - int(sim_time_sec)) * 1e9)
        self._clock_msg = msg
        self._clock_pub.publish(msg)

    def _pub_joint_state(self, joint_pos, joint_vel) -> None:
        msg = JointState()
        msg.name = self._joint_names

        if joint_pos is None:
            msg.position = []
        else:
            msg.position = [float(x) for x in np.asarray(joint_pos).reshape(-1)]

        if joint_vel is None:
            msg.velocity = []
        else:
            msg.velocity = [float(x) for x in np.asarray(joint_vel).reshape(-1)]

        if self._clock_msg is not None:
            msg.header.stamp.sec = self._clock_msg.clock.sec
            msg.header.stamp.nanosec = self._clock_msg.clock.nanosec
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        self._joint_state_pub.publish(msg)

    def _pub_pointcloud(self, pointcloud) -> None:
        if pointcloud is None:
            pointcloud = get_head_lidar_pointcloud(self)

        pcl = np.asarray(pointcloud).tolist()

        header = Header()
        header.frame_id = self._frame_id
        if self._clock_msg is not None:
            header.stamp.sec = self._clock_msg.clock.sec
            header.stamp.nanosec = self._clock_msg.clock.nanosec
        else:
            header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
            PointField(name="time", offset=24, datatype=PointField.FLOAT32, count=1),
        ]

        pcl_msg = point_cloud2.create_cloud(header, fields, pcl)
        self._pointcloud_pub.publish(pcl_msg)

