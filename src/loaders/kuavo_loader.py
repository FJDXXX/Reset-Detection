from __future__ import annotations

from pathlib import Path
from typing import Any

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from src.loader import Episode, GroupData
from .base_loader import BaseLoader


_BASE_TEXTS: dict[str, str] = {
    "kuavo_msgs/jointData": (
        "float64[] joint_q\n"
        "float64[] joint_v\n"
        "float64[] joint_vd\n"
        "float64[] joint_torque\n"
    ),
    "kuavo_msgs/imuData": (
        "geometry_msgs/Vector3 gyro\n"
        "geometry_msgs/Vector3 acc\n"
        "geometry_msgs/Vector3 free_acc\n"
        "geometry_msgs/Quaternion quat\n"
    ),
    "kuavo_msgs/endEffectorData": (
        "string[] name\n"
        "float64[] position\n"
        "float64[] velocity\n"
        "float64[] effort\n"
    ),
    "kuavo_msgs/FTsensorData": (
        "float64[] Fx\n"
        "float64[] Fy\n"
        "float64[] Fz\n"
        "float64[] Mx\n"
        "float64[] My\n"
        "float64[] Mz\n"
    ),
}

_SENSORSDATA_WITH_FT = (
    "std_msgs/Header header\n"
    "time sensor_time\n"
    "kuavo_msgs/jointData joint_data\n"
    "kuavo_msgs/imuData imu_data\n"
    "kuavo_msgs/endEffectorData end_effector_data\n"
    "kuavo_msgs/FTsensorData FTsensor_data\n"
)

_SENSORSDATA_NO_FT = (
    "std_msgs/Header header\n"
    "time sensor_time\n"
    "kuavo_msgs/jointData joint_data\n"
    "kuavo_msgs/imuData imu_data\n"
    "kuavo_msgs/endEffectorData end_effector_data\n"
)


def _has_ftsensor(bag_path: Path) -> bool:
    with Reader(bag_path) as reader:
        for conn in reader.connections:
            if conn.topic == "/sensors_data_raw":
                for mdef in conn.msgdef:
                    if "FTsensor" in str(mdef):
                        return True
    return False


def _build_typestore(has_ft: bool) -> Any:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    for name, text in _BASE_TEXTS.items():
        types = get_types_from_msg(text, name)
        typestore.register(types)

    sensors_text = _SENSORSDATA_WITH_FT if has_ft else _SENSORSDATA_NO_FT
    types = get_types_from_msg(sensors_text, "kuavo_msgs/sensorsData")
    typestore.register(types)
    return typestore


def _extract_bag(
    bag_path: Path,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    has_ft = _has_ftsensor(bag_path)
    typestore = _build_typestore(has_ft)

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}
        conn = connections.get("/sensors_data_raw")
        if conn is None:
            return frames

        for c, ts, rawdata in reader.messages(connections=[conn]):
            d = typestore.deserialize_ros1(rawdata, conn.msgtype)
            jd = d.joint_data
            imu = d.imu_data
            frames.append({
                "leg_joint_positions": list(jd.joint_q[0:12]),
                "arm_joint_positions": list(jd.joint_q[12:26]),
                "gripper_joint_positions": list(jd.joint_q[26:28]),
                "leg_joint_torques": list(jd.joint_torque[0:12]),
                "base_orientation": [
                    float(imu.quat.w),
                    float(imu.quat.x),
                    float(imu.quat.y),
                    float(imu.quat.z),
                ],
                "gravity_vector": [
                    float(imu.acc.x),
                    float(imu.acc.y),
                    float(imu.acc.z),
                ],
                "angular_velocity": [
                    float(imu.gyro.x),
                    float(imu.gyro.y),
                    float(imu.gyro.z),
                ],
            })
    return frames


class KuavoLoader(BaseLoader):
    def load_episode(self, episode_path: Path) -> Episode:
        episode_path = Path(episode_path)
        bag_path: Path

        if episode_path.is_dir():
            bags = sorted(episode_path.glob("*.bag"))
            if not bags:
                raise FileNotFoundError(f"No .bag files found in {episode_path}")
            bag_path = bags[0]
        else:
            bag_path = episode_path

        if not bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

        has_ft = _has_ftsensor(bag_path)
        frames = _extract_bag(bag_path)

        if not frames:
            raise ValueError(f"No /sensors_data_raw messages found in {bag_path}")

        meta = {
            "bag_file": bag_path.name,
            "has_ftsensor": has_ft,
            "num_frames": len(frames),
        }

        episode = Episode(path=bag_path, meta=meta)

        episode.groups["arm"] = GroupData(
            group_name="arm",
            frames=frames,
            fps=0,
        )

        return episode
