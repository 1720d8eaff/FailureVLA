# 简历技术证据索引

| 技术点 | 代码 / 配置 | 测试 | 日志和验证范围 |
| --- | --- | --- | --- |
| SmolVLA | `failurevla/policy.py`；注入 LeRobot policy 和观测/动作转换 | `tests/test_components.py` | Mock 接口测试；真实模型权重和端到端运行记录需另行提供 |
| 历史、失败和进度 | `failurevla/representation.py`, `configs/training.json` | `tests/test_components.py` | 合成张量验证梯度；有标签训练及 checkpoint 需另行提供 |
| 恢复与困难样本 | `failurevla/recovery.py`, `failurevla/representation.py` | `tests/test_components.py` | 决策接口及采样权重；恢复轨迹和训练日志需另行提供 |
| 异步与状态对齐 | `failurevla/worker.py`, `execution.py`, `action_buffer.py` | `tests/test_execution.py`, `test_pipeline.py` | Mock 执行；真实机器人延迟日志需另行提供 |
| B0–B7 | `configs/B0.json`–`B7.json`, `failurevla/runner.py`, `evaluation.py` | `tests/test_components.py` | 配置和日志统计接口；成功率、恢复率和延迟由运行日志计算，简历数值尚需对应日志 |
| 轨迹 | `data/traj/`, `failurevla/data.py` | `tests/test_pipeline.py` | 70 条 HDF5；`python -m failurevla.inspect` |
