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

## Checkpoint 合同迁移

长期训练可以把 checkpoint 绑定到精确的 learner 合同，而不把框架代码放进 GLR。
先以 `confirm=False` 调用 `migrate_checkpoint_manifest` 获取字段级差异；只有
reward/knowledge 摘要变化允许通用迁移，协议、observation 或 action 变化会 fail-closed。
传入 `confirm=True` 后才会创建相邻备份并原子重写 manifest。需要框架反序列化/重保存时，
可以传入 `saver(source, destination, contract)`；未传入时仅逐字节复制并重新校验哈希，
不会在未明确确认时修改 checkpoint。

独立 CLI 提供同一套机器可读的预检：

```shell
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json
```

合同未变化时退出码为 `0`；可迁移差异在没有 `--force` 时以退出码 `3` 报告，
不兼容的 observation/action/protocol 变化以退出码 `4` fail-closed。成功迁移保持
checkpoint 字节不变，写入相邻 `.bak` 备份并重新校验 manifest。

原生适配器的 C#、C++ SDK 也提供相同的
`glr.checkpoint-contract.v1` 与 `glr.checkpoint-manifest.v1` 字段校验类型。
它们只负责契约校验；迁移仍必须由 Python/CLI 显式执行，SDK 不会静默改写
learner 状态。
