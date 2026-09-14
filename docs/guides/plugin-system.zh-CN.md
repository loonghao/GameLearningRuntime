# GLR 插件系统

GLR 的插件层是一个小而严格的声明式控制面，借鉴了
DeepSeek Harness（DSH）的 bundle/profile 工作流：
https://deepseek.com/harness/en/。它为项目提供可审查的扩展清单，同时避免把安装
插件变成隐式执行代码。

## Bundle 契约

Bundle 是一个本地目录，包含 glr-plugin.json 和负载文件。id 与 version 决定项目内
不可变存储路径 .glr/plugins/<id>/<version>。entrypoint 只是元数据；inspect、install、
health 和 profile 解析都不会导入或执行它。

~~~json
{
  "schema_version": "glr.plugin.v1",
  "id": "torchrl-learner",
  "version": "0.1.0",
  "kind": "learner",
  "name": "TorchRL learner",
  "description": "A project-owned TorchRL learner adapter.",
  "entrypoint": "torchrl_plugin:create",
  "capabilities": ["learner.ppo", "collector.process"],
  "requires": {"glr": ">=0.17.0,<1.0.0", "torchrl": ">=0.13.0,<0.14.0"},
  "platforms": ["windows", "linux", "macos"],
  "isolation": "process",
  "permissions": ["read:environment", "write:checkpoint"]
}
~~~

## Profile

Profile 保存于 .glr/profiles/<name>.json。Profile 中授予的权限必须是 manifest 声明
权限的子集。依赖按满足语义版本范围的最高版本选择，并以稳定的依赖优先顺序输出；
循环、冲突、不支持的平台、不兼容的 GLR 版本和缺失 Bundle 都会失败关闭。如果依赖项
同时在 Profile 中显式列出，显式授予的权限和配置会保留，不会被依赖解析静默丢弃；不兼容
的重复请求会失败关闭。

Rust CLI 会用自身编译版本校验 `requires.glr`；Python `PluginManager` 在需要绑定具体
runtime 时传入 `glr_version`，不传则适合没有 runtime 绑定的源代码清单检查。
其它 `requires` 键（例如 `torchrl` 或 `sample-factory`）在当前控制面版本中只是保留的
声明元数据，由未来的 host runner 检查实际可用性；`health` 的 ready 仅表示静态 Bundle/
Profile 就绪，不代表所有外部框架已经安装。

配置会参与 profile digest 的规范化计算。当前支持有限 JSON 数字、布尔值、字符串、数组
和对象，非有限值会被拒绝。Rust 启用正确舍入的浮点解析，使 Python 与 Rust 在小数边界
保留一致的 digest 字节。
文本字段的长度限制按 UTF-8 字节计算，因此 Python SDK 与 Rust CLI 对非 ASCII 元数据使用
一致的边界。

## CLI 流程

独立 Rust CLI 是正式入口；Python SDK 提供同名开发接口：

~~~powershell
glr --project . --json plugin inspect --source plugins/torchrl-learner
glr --project . --json plugin install --source plugins/torchrl-learner --sha256 <digest>
glr --project . --json plugin list
glr --project . --json plugin profile enable training torchrl-learner --grant read:environment
glr --project . --json plugin profile resolve training
glr --project . --json plugin health --profile training
~~~

plugin inspect 不要求目标目录已经是完整 GLR 项目；其它命令只把 --project 当作
.glr/plugins 与 .glr/profiles 的所有者，不会启动 runtime role。输出使用稳定的
glr.cli-output.v1 envelope。

## 信任与执行边界

第一版只接受本地目录。它会严格校验 JSON、ASCII 可移植相对路径、文件大小、符号链接/
reparse point 策略和 SHA-256 清单，然后原子复制并再次校验暂存结果。当前没有 Git、
npm、HTTP 拉取、签名验证、import hook、shell 命令、build hook 或自动启动进程。未来
registry 或 host runner 必须在此契约之上增加显式授权与签名溯源，不能在安装时静默
执行 entrypoint。

## TorchRL 与 Sample Factory

插件槽位保持框架中立。需要现有可选 TorchRL 0.13 集成及其 CI 契约时，优先使用
TorchRL learner plugin；需要吞吐导向执行模型（尤其 Linux）时，再独立加入 Sample
Factory plugin。两者都消费同一套 GameEnvironment、transition 和 checkpoint 契约，
不能把游戏语义搬进 learner 层。

仓库当前交付的是契约与生命周期，不包含打包好的 TorchRL/Sample Factory 负载。项目
应把自有适配器放入经过审查的 Bundle，并在部署文档或 model bundle 中固定 digest。
