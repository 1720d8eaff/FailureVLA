# artifacts/replay

`python -m failurevla.replay_runtime` 在录制轨迹上的本机回放测量，用于核对 30 Hz deadline 调度、过期块筛选与拒绝逻辑。延迟由人工注入（40 ms），不是真实模型推理耗时，也不含整机控制链路。

## summary.json 字段范围

| 字段 | 含义 |
| --- | --- |
| `target_hz` / `ticks` | 目标频率与回放周期数 |
| `lateness_p50_s` / `lateness_p95_s` | 相对 deadline 的迟到时间分位数 |
| `work_p95_s` | 单周期处理耗时 p95 |
| `inference_count` | 该模式下启动的推理次数 |
| `chunks_rejected` | 被拒绝的动作块数 |

本目录不含 `task_time_s` 与 `success`。连续任务时长、推理阻塞与扰动任务成功率来自整机实验记录，未随本仓库发布；逐周期数据见同目录 `synchronous.csv`、`naive_async.csv`、`aligned_async.csv`。
