# Reset Detection Framework（复位检测框架）

---

## 1. 项目概述

Reset Detection Framework 是一个 Parameters base 的机器人复位检测框架，用于检测机器人（双臂机械臂、移动底盘、灵巧手、头部等）在执行完任务后是否准确回到 Home Position（复位位置/零位）。

### 解决的核心问题

- 机器人执行完一个 Episode（任务回合）后，末端/关节/夹爪/底盘是否回到预定义的 Home Position
- 支持两种检测方向：Initialization（初始化阶段首帧检测）与 Reset（任务结束后末帧检测）
- 支持多机器人异构数据源（yuanli、quanta_x1、kuavo）的统一加载、校准与检测
- 通过参数定义驱动行为，新增机器人时无需修改核心检测逻辑
- 输出标准化的 Episode-Level JSON Contract，便于下游评估系统集成

## 2. 核心概念

| 概念 | 说明 |
|---|---|
| **Episode（任务回合）** | 一次完整的任务执行过程，包含按时间序列采样的关节/末端数据帧 |
| **Home Position（复位位置）** | 机器人完成任务后应返回的参考位姿，作为检测基准 |
| **Initialization（初始化检测）** | 检测首帧——机器人在 Episode 开始时是否处于 Home Position |
| **Reset（复位检测）** | 检测末帧——机器人在 Episode 结束后是否成功回到 Home Position |
| **Calibration（校准）** | 从数据中自动统计计算 Home Position 和 Tolerance，写入 calibrated.yaml |

### 三种检测状态

| 状态 | 含义 |
|---|---|
| **PASS** | 所有检测参数均在容差范围内，Episode 正常复位 |
| **FAIL** | 存在检测参数超出容差，Episode 未正确复位 |
| **LOAD_ERROR** | 加载异常导致检测未能执行，既不是 PASS 也不是 FAIL |

> LOAD_ERROR 表示检测流程被阻断，原因包括文件缺失、格式错误、数据损坏等。它与 FAIL 有本质区别：FAIL 是检测执行后得出的负面结论，LOAD_ERROR 是检测根本未能执行。

---

## 3. 支持的机器人

| 机器人 | 数据格式 | Loader | 输出 Groups |
|---|---|---|---|
| **yuanli** | `episode_dir/meta.json` + `actions/*.jsonl` | `YuanliLoader` | `left_arm`, `right_arm` |
| **quanta_x1** | `episode_dir/*.json`（排除 `subtasks_*.json`） | `QuantaX1Loader` | `left_arm`, `right_arm`, `base`, `lifting`, `head` |
| **kuavo** | `*.bag`（ROS1 bag 格式） | `KuavoLoader` | `arm`（单一 group） |

---

## 4. 整体架构

### 架构分层

```
Config Layer  →  Discovery  →  Loader  →  Parameter Mapping  →  Calibration / Detection  →  Exporter
    │               │             │               │                       │                     │
   YAML           find         Factory         ParameterDef             Error/Tolerance       JSON
 分层加载        episode       Pattern         detect/calibrate         PASS/FAIL/LOAD_ERROR  Contract
                目录/文件
```

### 数据流

```
数据根目录
    │
    ▼
Discovery（按机器人类型发现 Episode 目录或文件）
    │
    ▼
Loader（按机器人类型加载 Episode 数据 → Episode + GroupData 结构）
    │
    ├──→ [校准流程] 收集首帧数据 → 统计量计算 → home_position + tolerances → 写入 calibrated.yaml
    │
    └──→ [检测流程] 计算首帧/末帧误差 → 容差判定 → PASS/FAIL/LOAD_ERROR 分类 → 导出 JSON Contract
```

### 模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | 加载合并配置文件（default.yaml + robot calibrated.yaml） |
| `parameters.py` | `ParameterDef`、`RobotParameters`、误差计算、统计量计算、容差计算 |
| `loader.py` | `Episode`、`GroupData` 基础数据结构、通用 JSONL 加载 |
| `loaders/*` | 机器人特定数据加载器（Loader Factory 模式） |
| `calibration.py` | 校准主流程 |
| `detector.py` | Episode 扫描、单 Episode 检测、触发导出 |
| `metrics.py` | 单 Group 指标计算 |
| `exporter.py` | Episode-Level JSON Contract 导出 |
| `loader_debug.py` | Loader 调试验证工具（开发用） |

---

## 5. 设计哲学

### 5.1 Parameter-Driven Design（参数驱动设计）

框架行为由参数定义驱动，而非机器人硬编码逻辑。

每个参数通过 `ParameterDef` 描述其角色：

```python
@dataclass
class ParameterDef:
    group: str            # 所属组：right_arm, left_arm, base, arm
    key: str              # 参数名：joint_positions, ee_position, gripper
    param_type: str       # 类型：scalar, vector, quaternion
    detect: bool          # 是否参与 PASS/FAIL 判定
    calibrate: bool       # 是否参与校准统计
```

- `detect` 和 `calibrate` 双标志构成核心抽象
- 新增机器人时只需定义参数 Schema，核心检测逻辑无需修改
- 框架不关心参数的具体物理含义，只按类型计算误差和容差

### 5.2 Robot-Agnostic Detection（机器人无关的检测核心）

检测核心分为两层：

| 层次 | 是否与机器人相关 | 包含模块 |
|---|---|---|
| Robot-Specific | 是 | Loader、Discovery 策略 |
| Robot-Agnostic | 否 | Calibration、Detection、Metrics、Exporter、Parameters |

- Loader 负责将异构数据统一转换为 `Episode` + `GroupData` 内部表示
- 检测核心只操作 `ParameterDef` 定义的抽象参数路径，不感知机器人类型
- 添加新机器人时，Loader 和 Discovery 是主要工作点，核心检测逻辑无需修改

---

## 6. 配置体系

### 6.1 文件层级与合并规则

```
configs/default.yaml        ← 全局入口，指定 robot、data_path
         │
         └── robots/{robot}/
                 └── calibrated.yaml    ← 唯一机器人配置文件（校准生成或手工编辑）
```

### 6.2 全局配置 default.yaml

```yaml
robot: kuavo               # 机器人标识

data:
  raw_path: /path/to/datasets          # 检测数据路径

calibration:
  data_path: /path/to/datasets         # 校准数据路径
  exclude_episodes: []                 # 排除的 Episode 名称列表
  tolerance_factor: 3.0                # 容差因子（sigma 倍数）
```

### 6.3 参数定义 Schema

```yaml
robot:
  name: yuanli
  parameters:
    right_arm:
      joint_positions:
        type: vector
        calibrate: true
        detect: false
      ee_position:
        type: vector
        calibrate: true
        detect: true
      ee_orientation:
        type: quaternion
        calibrate: true
        detect: true
      gripper:
        type: scalar
        calibrate: true
        detect: true
```

完整类型支持：`scalar`、`vector`、`quaternion`、`angle`、`enum`。

### 6.4 calibrate / detect 双轨制

| 标志 | 含义 | 典型用法 |
|---|---|---|
| `calibrate: true` | 参与校准统计（计算 home/tolerance） | 默认值，适用于大多数参数 |
| `calibrate: false` | 跳过校准，使用预设值 | 已知标准零位、无需统计的参数 |
| `detect: true` | 参与 PASS/FAIL 判定 | 核心检测参数：ee_position、gripper、base_orientation |
| `detect: false` | 仅记录误差，不影响判定 | 辅助参数：关节位置、腿部扭矩、重力向量 |

**典型配置差异**：

| 机器人 | detect=true | detect=false |
|---|---|---|
| yuanli | ee_position, ee_orientation, gripper | joint_positions |
| quanta_x1 | position, rotation, gripper, car_pose, lifting, head | joint_pos |
| kuavo | base_orientation, left_dexhand_positions, right_dexhand_positions | arm/leg/leg_torque/gripper/gravity/angular_velocity |

---

## 7. Calibration（校准）

校准流程自动从 Episode 数据中统计计算 `home_position` 和 `tolerances`。

### 7.1 校准流程

```
1. 发现 Episode 目录/文件（支持 exclude_episodes 过滤）
2. 遍历每个 Episode，收集 calibrate=true 参数的首帧值
3. 对每个参数计算统计量（mean, std, min, max, count）
4. home_position = 参数的 mean 值
5. tolerance = max(sigma_factor * std, tolerance_floor)
6. 检查异常大的 tolerance（>1.0 时报警，提示可能存在多峰分布）
7. 写入 configs/robots/{robot}/calibrated.yaml
```

### 7.2 统计量计算

| 类型 | 统计量 | 说明 |
|---|---|---|
| **scalar** | `{mean, std, min, max, count}` | 直接计算 |
| **vector** | `{mean[], std[], min[], max[], count}` | 逐维度独立计算 |
| **quaternion** | `{mean[4], std_angle, min_angle, max_angle, count}` | 平均四元数归一化后，计算各样本与该均值的角度误差分布 |

### 7.3 Home Position 生成

对每个 `calibrate=true` 的参数，取其统计量的 `mean` 作为 Home Position。

### 7.4 Tolerance 生成

核心函数 `compute_tolerance()`：

```
tolerance = max(sigma_factor * std, tolerance_floor)
```

- **scalar** 和 **quaternion**：`max(sigma_factor * std, floor)`，结果为单个 float
- **vector**：逐维度 `max(sigma_factor * std[i], floor[i])`，结果为 list

### 7.5 tolerance_floor 优先级与默认值

`tolerance_floor` 用于防止数据过于集中时 tolerance 过小导致误报。

**优先级链（由高到低）**：

```
1. param_key 在配置 tolerance_floor 中（如 joint_positions: 0.02）
2. 最终 fallback: 0.0
```

### 7.6 校准输出配置

校准结果写入 `configs/robots/{robot}/calibrated.yaml`：

---

## 8. Detection（检测）

### 8.1 检测流程

```
1. 扫描数据根目录，发现所有 Episode（按机器人类型选择 Discovery 策略）
2. 对每个 Episode：
   a. 使用对应 Loader 加载 Episode 数据
   b. 对每个 Group，对每个 detect=true 的参数：
      - 计算首帧误差 vs home_position
      - 计算末帧误差 vs home_position
      - 检查误差是否超过 tolerance
      - 记录 fail_reasons
   c. 聚合判定：任意 detect=true 参数超限 → FAIL；所有参数均在容差内 → PASS
   d. 导出 Episode-Level JSON Contract
```

### 8.2 误差计算方法

| 类型 | 指标 | 定义 |
|---|---|---|
| **scalar** | 绝对差值 | `abs(current - home)` ，单位 m |
| **vector** | 逐维度误差 + 模长 | `per_dim[i] = abs(current[i] - home[i])`，`magnitude = norm(per_dim)` ，单位 m |
| **quaternion** | 角度距离 | `2 * acos(abs(dot(q1_normalized, q2_normalized)))`，单位 rad |

### 8.3 容差判定逻辑

| 类型 | 失败条件 |
|---|---|
| **scalar** | `error > tolerance` |
| **vector** | `any(per_dim_error[i] > tolerance[i])` |
| **quaternion** | `angle_error > tolerance` |

### 8.4 Initialization 与 Reset 双检测

每个 Episode 产出两次独立评估：

- **Initialization**：基于首帧数据的检测，判断机器人初始化时是否已在 Home Position
- **Reset**：基于末帧数据的检测，判断机器人任务结束后是否回到 Home Position

两次评估使用相同的 `home_position` 和 `tolerances`，但判断依据不同：Initialization 检查 `first_frame_home`，Reset 检查 `last_frame_home`。

---

## 9. 输出契约

### 9.1 输出路径

```
detection_summary/{dataset_name}/{episode_name}_detection_output.json
```

每个 Episode 产出一份 JSON，包含 Initialization 和 Reset 两个 Contract（JSON 数组），通过 `"name"` 字段区分。

---

## 10. 数据加载器

### 10.1 Loader Factory 架构

Loader 采用注册表模式，通过 `LoaderFactory` 统一管理：

```python
loader = LoaderFactory.get_loader("yuanli")       # → YuanliLoader 实例
loader = LoaderFactory.get_loader("quanta_x1")    # → QuantaX1Loader 实例
loader = LoaderFactory.get_loader("kuavo")        # → KuavoLoader 实例
```

### 10.2 BaseLoader 抽象接口

```python
class BaseLoader(ABC):
    @abstractmethod
    def load_episode(self, episode_path: Path) -> Episode:
        ...
```

所有 Loader 返回统一的 `Episode` 数据结构，下游模块无需感知数据源差异。

### 10.3 YuanliLoader

- 读取 `episode_dir/meta.json` 获取元信息
- 根据 `meta.json` 中定义的 action schemas 读取 `actions/*.jsonl`
- 字段别名归一化：`joint_position` → `joint_positions`，`ee_pose` → `ee_positions` 等
- 将 `ee_positions[7]` 拆分为 `ee_position[3]`（位置）+ `ee_orientation[4]`（朝向四元数）
- 输出 Groups：`left_arm`、`right_arm`

### 10.4 QuantaX1Loader

- 读取 `episode_dir` 下的 `*.json` 文件（排除 `subtasks_*.json`）
- 通过 `_GROUP_SOURCE_MAP` 字段映射表将 JSON 字段映射到标准 Group/Key
- 忽略 `master_left`/`master_right` 数据源
- 输出 Groups：`left_arm`、`right_arm`、`base`、`lifting`、`head`

### 10.5 KuavoLoader

- 读取 `*.bag`（ROS1 bag 格式），依赖 `rosbags` 库
- 自动检查 bag 中是否包含 FTsensor 数据，按需注册 message 类型
- 解析 `/sensors_data_raw` topic，提取：
  - `arm_joint_positions[14]`（双臂 + 头部关节）
  - `leg_joint_positions[12]`
  - `leg_joint_torques[12]`
  - `base_orientation[4]`（四元数）
  - `gravity_vector[3]`
  - `angular_velocity[3]`
  - `gripper_joint_positions[2]`
- 解析 `/dexhand/state` topic，提取：
  - `left_dexhand_positions[6]`（左手灵巧手：thumb, thumb_aux, index, middle, ring, pinky）
  - `right_dexhand_positions[6]`（右手灵巧手：thumb, thumb_aux, index, middle, ring, pinky）
- `/dexhand/state` 与 `/sensors_data_raw` 按时间戳同步（Zero-Order Hold + 前向填充）
- 若 bag 中包含 `/dexhand/state` topic 但消息为空，发出 `UserWarning`
- 若 bag 中不包含 `/dexhand/state` topic，灵巧手字段不写入 frame（向后兼容）
- 输出 Group：`arm`（单一 group，包含所有字段）

---

## 11. 开发工具

### 11.1 Loader Debug（loader_debug.py）

`loader_debug.py` 是一个**开发调试工具**，用于验证 Loader 能否正确加载 Episode 数据、检查字段覆盖情况、预览首末帧数值。不属于生产检测主流程。

**用途**：

- 新机器人适配时验证 Loader 加载逻辑
- 检查数据源字段名是否映射正确
- 查看 Episode 的数据完整性
- 快速定位加载失败的原因

### 11.2 使用方式

```bash
# 通过 main.py 进入 debug 模式
python main.py --debug-loader \
  --robot yuanli \
  --data-path /path/to/episodes \
  --sample 5

# 或直接调用模块
python -m src.loader_debug \
  --robot kuavo \
  --data-path /path/to/bags \
  --sample-size 10
```

| 参数 | 说明 |
|---|---|
| `--robot` | 机器人名称 |
| `--data-path` | 数据根目录 |
| `--sample` | 随机采样 N 个 Episode（不指定则全量） |

### 11.3 输出解读

- **控制台输出**：每个 Episode 的 groups、字段列表、首帧/末帧数值、验证结果
- **文件输出**：`detection_summary/{robot}/loader_debug_report.txt`，包含完整报告

每条 Episode 的报告包含：

```
Episode: episode_good
Detected Groups:
  right_arm
  left_arm
Start Pose — right_arm:
  ee_positions: [0.200, 0.000, ...]
  gripper: 0.000
  joint_positions: [0.000, 0.000, ...]
  ...
Fields: 5
  ee_positions: list
  gripper: float
  joint_positions: list
  ...
Result: PASS
```

---

## 12. 扩展新机器人

### 12.1 实现 Loader

创建新的 Loader 类，继承 `BaseLoader`：

```python
from pathlib import Path
from src.loader import Episode
from src.loaders.base_loader import BaseLoader

class MyRobotLoader(BaseLoader):
    def load_episode(self, episode_path: Path) -> Episode:
        # 1. 读取数据源（JSONL / JSON / bag / 其他格式）
        # 2. 构造 Episode 对象
        # 3. 填充 groups，每个 group 为 GroupData(frames=[...])
        # 4. 返回 Episode
        ...
```

### 12.2 注册到 Factory

```python
from src.loaders.loader_factory import LoaderFactory
LoaderFactory.register("my_robot", MyRobotLoader)
```

### 12.3 添加配置

创建机器人配置文件：

```
configs/robots/my_robot/calibrated.yaml
```

配置中定义参数 Schema，指定 `type`、`calibrate`、`detect`，以及 `home_position` 和 `tolerances`。

- **自动校准**：运行 `python -m src.calibration` 自动生成 `home_position` 和 `tolerances`
- **手工零位**：直接编辑 `calibrated.yaml` 中的 `home_position` 和 `tolerances`

### 12.4 新增参数类型

如果现有类型（scalar、vector、quaternion）不能满足需求，扩展 `_PARAMETER_TYPES` 并实现：

- `compute_<type>_error()` — 计算当前值与 Home 的误差
- `is_<type>_fail()` — 判定误差是否超出容差
- `compute_<type>_statistics()` — 计算统计量
- `compute_tolerance()` 中增加对应类型分支

---

## 13. CLI 使用说明

### 13.1 检测命令

```bash
# 使用默认配置（读取 configs/default.yaml）
python main.py

# 指定数据路径、配置、输出目录
python main.py \
  --data-path /path/to/episodes \
  --config configs/default.yaml \
  --output-dir detection_summary

# 指定机器人（覆盖配置）
python main.py --robot quanta_x1

# 标签映射：将目录名映射为显示标签
python main.py \
  --label episode_001 "Task A" \
  --label episode_002 "Task B"
```

### 13.2 校准命令

```bash
# 使用默认配置
python -m src.calibration

# 指定数据路径、机器人、容差因子
python -m src.calibration \
  --data-path /path/to/episodes \
  --robot yuanli \
  --tolerance-factor 3.0

# 干运行（仅打印结果，不写入配置文件）
python -m src.calibration --dry-run
```

### 13.3 Loader Debug 命令

```bash
# 通过 main.py 入口
python main.py --debug-loader \
  --robot kuavo \
  --data-path /path/to/bags \
  --sample 5

# 直接调用模块
python -m src.loader_debug \
  --robot yuanli \
  --data-path /path/to/episodes \
  --sample-size 10
```

---

## 14. 测试

```bash
pytest tests/ -v
```

### 测试覆盖范围

| 测试文件 | 覆盖内容 |
|---|---|
| `test_loader.py` | Episode 加载、首末帧获取、字段验证、Loader Factory 注册、异常场景 |
| `test_metrics.py` | 误差计算（scalar/vector/quaternion）、容差判定、统计量计算、检测指标 |
| `test_detector.py` | PASS / FAIL 场景分类、fail_reasons 生成 |

---

## 15. 故障排查

| 现象 | 常见原因 | 解决办法 |
|---|---|---|
| `LOAD_ERROR` | 数据文件缺失或格式不兼容 | 先运行 `--debug-loader` 检查数据完整性 |
| 所有 Episode 均为 PASS | Tolerance 过大，漏检 | 降低 `tolerance_factor` 或降低 `tolerance_floor` |
| 所有 Episode 均为 FAIL | Tolerance 过小，误报 | 增大 `tolerance_factor` 或提高 `tolerance_floor` |
| Tolerance > 1.0 报警 | 数据存在多峰分布（multimodal） | 通过 `exclude_episodes` 排除离群 Episode |
| 未发现任何 Episode | 数据路径错误或 Discovery 策略不匹配 | 检查 `data_path` 配置和 robot_name 是否正确 |
| 校准结果不合理 | 数据中包含异常 Episode | 使用 `exclude_episodes` 排除异常数据，或调整 `tolerance_floor` |
