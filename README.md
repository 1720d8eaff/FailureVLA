# FailureVLA

面向机器人操作的数据处理与动作执行研究：本仓库提供 G1-D 抓取轨迹、状态历史预测基线和状态对齐的异步动作缓冲组件，为后续扩展多任务、失败与恢复数据提供基础。

## 数据

首批公开 70 段轨迹，共 43,187 帧，约 1,438 秒，来自 G1-D 左臂与 RH56H1 六指手的抓取示教。相机数据体积较大，不便于传输，本仓库公开轨迹数据。文件位于 `data/traj/`，HDF5 中包括：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `t` | T | 相对时间，秒 |
| `joints` | T × 7 | 左臂关节位置，弧度 |
| `fingers` | T × 6 | 手指位置，设备原始量纲 |
| `eef_left` | T × 10 | 左侧末端位置、旋转矩阵前两列和归一化手部开合度 |
| `eef_right` | T × 10 | 右侧占位通道 |
| `instruction` | 标量 | 任务指令 |

轨迹中的末端状态由关节正运动学计算；手部开合度由六指位置的均值换算。数据保留原始时间戳，使用时请以 `t` 为准。

## 安装和数据检查

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m failurevla.inspect
python -m unittest discover -s tests
```

检查程序验证每条轨迹的字段形状、有限值和时间顺序，并汇总帧数及时间。Windows 可用 `.venv\Scripts\activate`。

## 状态历史预测基线

```bash
python -m failurevla.train --data data/traj --history 4 --stride 5 --epochs 10 --seed 42 --output runs/baseline
```

程序以连续状态历史预测下一帧的关节及手指位置增量。手指数值按 2000 缩放；输入归一化参数仅由训练集计算。数据按整条轨迹划分训练集和测试集，输出模型、划分清单、预测误差及零增量基线误差到 `runs/baseline/`。这是一项轨迹预测基线，误差指标不等同于机器人任务成功率。

## 异步动作缓冲

`failurevla.action_buffer.ActionBuffer` 接收带请求序号、观测时间和观测状态的动作块。它根据时间差与状态偏差筛选结果，并拒绝晚到的旧请求；通过 `pop()` 消费动作。时间应来自同一单调时钟，状态误差阈值应按状态单位和实际控制系统设置。该组件不直接控制机器人；连接实际系统时需由调用方处理动作尺度、限位、碰撞检查和紧急停止。

```python
from failurevla.action_buffer import ActionBuffer

buffer = ActionBuffer(max_age_s=0.2, max_state_error=0.1)
buffer.replace(1, 10.0, 10.1, [0.0, 0.0], [0.01, 0.0], [[0.1, 0.2]])
action = buffer.pop()
```
