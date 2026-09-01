# 复现训练模型

[English](reproducible-model-bundles.md)

`glr.model-bundle.v1` 将训练产物与检查、复跑所需的软件输入放在同一个包中：

- 环境与协议身份；
- 算法、框架和框架版本；
- Learner/Environment 随机种子；
- 训练配置、Runtime 配置、奖励安全策略、示范数据策略、源码快照与依赖锁文件；
- 模型产物与聚合指标；
- 每个文件的相对路径、字节数和 SHA-256。

对于 Agent-first run，应把准确的 goal、research bundle、trial plan、训练/安全配置、
选中的 transition manifest 和 capture manifest 一起作为 inputs。也可以为同一
environment/protocol 的新实例加入 `glr.spatial-knowledge.v1` 导出文件；使用
`glr knowledge import` 单独导入后，这些位置和路线会先降级为 advisory，直到新运行时
再次观察确认。

训练器可直接创建模型包：

```python
from game_learning_runtime import build_model_bundle

manifest = build_model_bundle(
    "dist/model-bundle",
    environment_id="example.environment-v1",
    protocol_version="1.0",
    algorithm="ppo",
    framework="pytorch",
    framework_version="2.8.0",
    seeds=(7,),
    inputs={
        "training.json": "training.json",
        "reward-safety.json": "reward-safety.json",
        "demonstration-policy.json": "demonstration-policy.json",
        "runtime-integration.json": "runtime-integration.json",
        "uv.lock": "uv.lock",
    },
    artifacts={"weights/model.pt": "runs/model.pt"},
)
```

缺失、符号链接、大小变化或内容篡改都会导致校验失败：

```python
from game_learning_runtime import verify_model_bundle

manifest = verify_model_bundle("dist/model-bundle")
```

Builder 不会覆盖非空目录，也不会把来源机器的绝对路径写进清单。允许再分发时，
应同时提供许可证和 Model Card。完整性与输入快照可以提高复现能力，但不能保证
不同 GPU kernel、驱动、实时游戏状态完全一致，也不能证明模型质量。
