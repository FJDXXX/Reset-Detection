# Reset Detection Framework

检测机器人在 Episode 结束后是否准确回到 Home Position（复位位置/零位）。

**输入**：机器人的 Episode 数据（关节/末端/底盘/灵巧手等）  
**输出**：每个 Episode 一份 JSON，包含 Initialization 和 Reset 两个 Contract（score + passed + reasons）  
**支持机器人**：yuanli、quanta_x1、kuavo

---

## 1. Installation

**要求**：Python >= 3.10

```bash
pip install -e .
```

核心依赖（自动安装）：numpy, pyyaml, pandas

Kuavo 额外依赖（bag 文件解析）：

```bash
pip install rosbags
```

开发验证：

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 2. Quick Start

### Step 1: 编辑配置

```yaml
# configs/default.yaml
robot: kuavo

output:
  dir: /path/to/output

data:
  raw_path: /path/to/episodes

calibration:
  data_path: /path/to/episodes
  exclude_episodes: []
  tolerance1_factor: 2.0
  tolerance2_factor: 3.0
```

### Step 2: 运行校准

从 Episode 数据中自动统计 home_position 和 tolerance，写入 calibrated.yaml：

```bash
python -m src.calibration --config configs/default.yaml
```

### Step 3: 运行检测

逐 Episode 检测复位状态，输出 JSON Contract：

```bash
python main.py --config configs/default.yaml
```

### Step 4: 查看结果

```bash
# 终端输出汇总
[Initialization] Total: 10 | PASS: 8 | FAIL: 1 | LOAD_ERROR: 1 | Avg Score: 87.3
[Reset] Total: 10 | PASS: 7 | FAIL: 2 | LOAD_ERROR: 1 | Avg Score: 82.1
Output -> /path/to/output/dataset_detection_output

# 查看具体 Episode 结果
cat /path/to/output/dataset_detection_output/episode_001_detection_output.json
```

---

## 3. Configuration

### 3.1 configs/default.yaml

```yaml
robot: kuavo                        # 机器人标识：yuanli | quanta_x1 | kuavo

output:
  dir: /path/to/output              # 检测结果输出目录

data:
  raw_path: /path/to/episodes       # 检测数据路径

calibration:
  data_path: /path/to/episodes      # 校准数据路径
  exclude_episodes: []              # 排除的 Episode 名称列表
  tolerance1_factor: 2.0            # 紧阈值 sigma 倍数（默认 3.0）
  tolerance2_factor: 3.0            # 宽阈值 sigma 倍数（默认 6.0）
```

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `robot` | 机器人标识 | `yuanli` |
| `output.dir` | 检测结果输出目录 | `detection_summary` |
| `data.raw_path` | 检测数据根目录 | — |
| `calibration.data_path` | 校准数据根目录 | — |
| `calibration.exclude_episodes` | 校准时排除的 Episode | `[]` |
| `calibration.tolerance1_factor` | 紧阈值 sigma 倍数 | `3.0` |
| `calibration.tolerance2_factor` | 宽阈值 sigma 倍数 | `6.0` |

### 3.2 calibrated.yaml

校准后自动写入 `configs/robots/{robot}/calibrated.yaml`，包含三部分：

```yaml
robot:
  name: kuavo
  parameters:
    arm:
      left_arm_joint_positions:
        type: vector
        calibrate: true
        detect: true
      gripper_joint_positions:
        type: vector
        calibrate: true
        detect: false
    base:
      base_orientation:
        type: quaternion
        calibrate: true
        detect: true
    # ...

home_position:       # 复位参考位姿（校准统计均值）
  arm:
    left_arm_joint_positions: [...]
    # ...

tolerance1:          # 紧阈值（100 分边界）
  arm:
    left_arm_joint_positions: [...]
    # ...

tolerance2:          # 宽阈值（0 分边界）
  arm:
    left_arm_joint_positions: [...]
    # ...
```

**参数属性**：

| 属性 | 说明 |
|------|------|
| `type` | 参数类型：scalar, vector, quaternion |
| `calibrate` | 是否参与校准统计（计算 home/tolerance） |
| `detect` | 是否参与评分判定（影响 score 和 passed） |

---

## 4. Calibration

从多个 Episode 的首帧数据中统计计算 home_position 和 tolerance。

**输入**：Episode 数据目录 + calibrated.yaml 中的参数定义  
**输出**：更新 `configs/robots/{robot}/calibrated.yaml`

```
Episode 首帧数据
      │
      ▼
统计量计算（mean, std, min, max, count）
      │
      ├──→ home_position = mean
      ├──→ tolerance1 = tolerance1_factor × std
      └──→ tolerance2 = tolerance2_factor × std
```

### CLI

```bash
# 使用默认配置
python -m src.calibration

# 指定配置和数据路径
python -m src.calibration \
  --config configs/default.yaml \
  --data-path /path/to/episodes

# 指定容差因子
python -m src.calibration \
  --tolerance1-factor 2.0 \
  --tolerance2-factor 4.0

# 干运行（仅打印，不写入文件）
python -m src.calibration --dry-run

# 覆盖机器人
python -m src.calibration --robot kuavo
```

| 参数 | 说明 |
|------|------|
| `--config, -c` | 配置文件路径 | 
| `--data-path, -d` | 覆盖校准数据路径 |
| `--robot, -r` | 覆盖机器人标识 |
| `--tolerance1-factor` | 覆盖紧阈值 sigma 倍数 |
| `--tolerance2-factor` | 覆盖宽阈值 sigma 倍数 |
| `--dry-run` | 仅打印结果，不写入配置文件 |

---

## 5. Detection

对每个 Episode 产出两次独立评估：

- **Initialization**：首帧检测 — 机器人开始时是否在 Home Position
- **Reset**：末帧检测 — 机器人结束后是否回到 Home Position

两次评估使用相同的 home_position 和 tolerance，但分别基于首帧和末帧数据。

### CLI

```bash
# 使用默认配置
python main.py

# 指定数据路径和输出目录
python main.py \
  --data-path /path/to/episodes \
  --config configs/default.yaml \
  --output-dir /path/to/output

# 覆盖机器人
python main.py --robot quanta_x1

# 标签映射
python main.py \
  --label episode_001 "Task A" \
  --label episode_002 "Task B"
```

| 参数 | 说明 |
|------|------|
| `--config, -c` | 配置文件路径 |
| `--data-path, -d` | 覆盖检测数据路径 |
| `--output-dir, -o` | 覆盖输出目录 |
| `--robot, -r` | 覆盖机器人标识 |
| `--label, -l` | 目录名 → 显示标签映射 |

---

## 6. Score V2

### Parameter Score

每个 detect=true 的参数，基于双阈值线性插值计算 0-100 连续分数：

```
error ≤ tolerance1
        ↓
     100 分

tolerance1 < error < tolerance2
        ↓
     线性下降

error ≥ tolerance2
        ↓
       0 分
```

公式：

```
score = 100 × (tolerance2 - error) / (tolerance2 - tolerance1)
```

### Assessment Score

一个 Episode 的 Assessment Score 由所有 detect=true 参数的 Parameter Score 聚合：

- **任意 Parameter Score = 0** → Assessment Score = 0，passed = false
- **否则** → Assessment Score = 所有 Parameter Score 的平均值

### Passed

`passed = (Assessment Score > 0)`

### Reasons

未满分的参数会生成自然语言原因描述：

```json
[
  "左臂末端位置未完全回到初始状态（72分）",
  "机器人基座姿态严重偏离初始状态（0分）"
]
```

---

## 7. Output

### 目录结构

```
{output.dir}/
└── {dataset}_detection_output/
    ├── episode_001_detection_output.json
    ├── episode_002_detection_output.json
    └── ...
```

其中 `dataset` 为数据根目录的文件夹名。

### JSON 示例

每个 Episode 产出一份 JSON，包含 Initialization 和 Reset 两个 Contract：

```json
[
  {
    "module": "reset_detection",
    "name": "Initialization",
    "score": 92.5,
    "grade": null,
    "passed": true,
    "verdict": null,
    "reasons": [
      "左臂末端位置未完全回到初始状态（72分）"
    ],
    "attribution": [],
    "sub_indicators": [],
    "threshold_profile": {}
  },
  {
    "module": "reset_detection",
    "name": "Reset",
    "score": 100.0,
    "grade": null,
    "passed": true,
    "verdict": null,
    "reasons": [],
    "attribution": [],
    "sub_indicators": [],
    "threshold_profile": {}
  }
]
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `name` | 评估类型：`Initialization` 或 `Reset` |
| `score` | Assessment Score（0-100） |
| `passed` | 是否通过（score > 0） |
| `reasons` | 未满分参数的自然语言原因列表 |

---

# Part 2: Architecture

## 8. Core Concepts

| 概念 | 说明 |
|------|------|
| **Episode** | 一次完整的任务执行，包含时间序列采样的关节/末端数据帧 |
| **Home Position** | 复位参考位姿，由校准从数据中统计生成 |
| **Initialization** | 首帧检测 — 机器人开始时是否在 Home Position |
| **Reset** | 末帧检测 — 机器人结束后是否回到 Home Position |

---

## 9. Supported Robots

| 机器人 | 数据格式 | Groups |
|--------|---------|--------|
| **yuanli** | `episode_dir/meta.json` + `actions/*.jsonl` | `left_arm`, `right_arm` |
| **quanta_x1** | `episode_dir/*.json` | `left_arm`, `right_arm`, `base`, `lifting`, `head` |
| **kuavo** | `*.bag`（ROS1 bag） | `arm`, `leg`, `base`, `imu`, `dexhand` |

---

## 10. Extending a New Robot

1. **实现 Loader**：继承 `BaseLoader`，实现 `load_episode()` 返回 `Episode` 对象

2. **注册到 Factory**：在 `LoaderFactory._registry` 中注册

3. **创建配置**：在 `configs/robots/{name}/calibrated.yaml` 中定义参数 Schema

4. **运行校准**：`python -m src.calibration --robot {name}` 自动生成 home_position + tolerance1 + tolerance2
