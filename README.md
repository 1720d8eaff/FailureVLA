# FailureVLA

面向机器人操作的数据处理与动作执行研究：本仓库提供 G1-D 抓取轨迹、状态历史预测基线和状态对齐的异步动作缓冲组件，为后续扩展多任务、失败与恢复数据提供基础。

配套工程：[unitree-g1d-grasp](https://github.com/1720d8eaff/unitree-g1d-grasp) 提供 AnyGrasp、MoveIt 和机器人执行链路；独立容器操作项目见 [ContainerLoop-VLA](https://github.com/1720d8eaff/ContainerLoop-VLA)。

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

## 策略、恢复与评测接口

`failurevla.failure_aware.FailureAwareRefiner` 在状态历史的失败概率和进度条件下，对策略动作块生成有界残差；残差头零初始化，训练之前保留基础动作。`FailureAwareSmolVLA` 接收 SmolVLA 策略和观测、历史、动作转换回调，输出动作块与失败概率、进度、恢复决策；使用 `predict_with_diagnostics` 后由上层处理恢复或终止。`predict` 可接入现有执行器，遇到需要恢复的决策会停止并要求调用方处理。

使用按 episode 预先划分的 NPZ 训练：`python -m failurevla.train_failure_aware --data train.npz --output runs/failure_aware.pt`。数组为 `observation`、`state`、`action`（均为 `[batch, history, dim]`），`base_actions` 和 `target_actions`（均为 `[batch, horizon, action_dim]`），以及一维 `failure`、`progress`、`stage` 标签。动作必须使用同一归一化空间；故障编码参照 `representation.FAILURES`。运行时由调用方加载 checkpoint、创建对应尺寸的模型并设置 eval 模式，按具体 LeRobot 版本准备策略输入和历史。测试使用模拟策略，不提供真实训练结果。

`failurevla.policy.SmolVLAAdapter` 接受已加载的 LeRobot 策略、观测编码和动作解码回调，调用 `predict_action_chunk`；参考 [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla)。`MockPolicy` 仅用于接口测试。`failurevla.representation` 提供历史 GRU、失败类别、任务阶段和进度预测头及监督损失；`configs/training.json` 给出损失权重和重采样参数。需要使用有标签数据训练并保存权重。

对按完整轨迹划分好的训练集，可使用 `python -m failurevla.train_representation --data train.npz --output runs/failure_head.pt`。NPZ 字段是 `observation`、`state`、`action` 三组 `[样本, 历史长度, 维度]` 数组，`target_action`、`failure`、`progress`、`stage` 是对应监督标签。请先按轨迹划分数据集，再仅对训练集重采样。

`failurevla.recovery` 提供有预算的无恢复、固定重试、规则与可注入学习策略；`robot_backends/unitree_grasp_backend.py` 对接机器人仓库的回调式 bridge。AnyGrasp 辅助恢复通过该 bridge 获取候选并交由碰撞、可达和规划回调筛选。两个仓库之间的消息使用 `failurevla/contracts.py` 中的数据类；硬件集成时需配置实际的观测、规划和已审阅执行回调。

`failurevla.execution.Executor` 支持同步、朴素异步和状态对齐异步；异步推理由 `InferenceWorker` 执行，记录推理与等待时间，缓冲区支持提前重规划和过期块筛选。实验方法 B0–B7 的配置位于 `configs/`。运行时需提供实际环境的 episode 函数，其返回包含 `episode_id` 与实测指标的字典：

```bash
python -m failurevla.runner --method B6 --runner my_experiment:run_episode --log runs/episodes.jsonl
python -m failurevla.evaluation runs/episodes.jsonl
```

JSONL 支持 `task_success`、`grasp_success`、`failure_tp/fp/fn`、`recovery_attempts`、`recovery_successes`、`inference_blocking_s`、`task_time_s`、`stale_rejected`、`chunks_received`、`human_intervention`；无测量值保持缺失，不当作成功或失败。统计按方法分组，分别保留完整任务与抓取指标。接口测试、真实模型运行与真机实验的证据范围见 [简历证据索引](docs/RESUME_EVIDENCE.md)。
