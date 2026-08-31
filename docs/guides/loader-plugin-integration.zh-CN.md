# 接入获授权的 BepInEx 与 UE4SS 运行时

[English](loader-plugin-integration.md)

Loader Plugin 是 GLR 的中间接入层：代码运行在获授权的游戏进程内，但
Adapter 并不拥有游戏源码工程。它可以读取语义状态并在游戏主线程执行动作，
同时必须如实声明为实时 `attach`，不能冒充物理 `reset`。

## 第一批模板

| Loader | 生成内容 | 必须重新确认的上游条件 |
| --- | --- | --- |
| BepInEx | 面向 BepInEx 5 LTS 的 Unity Mono C# 插件 | 准确游戏、运行时与正式版本兼容性 |
| UE4SS | 面向 UE4SS 3.x 的 Unreal Lua Mod | 准确 Unreal/游戏版本兼容性 |

GLR 不内置两个 Loader 的二进制。当前也不生成 BepInEx IL2CPP 或 UE4SS
C++ 模板，Agent 不得静默替换成这些变体。

## 生成接入工程

Unity Mono：

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_bepinex `
  --package example_bepinex `
  --environment-id example.bepinex-v1 `
  --engine unity `
  --access loader `
  --loader bepinex `
  --loader-version v5.4.23.5
```

Unreal：

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_ue4ss `
  --package example_ue4ss `
  --environment-id example.ue4ss-v1 `
  --engine unreal `
  --access loader `
  --loader ue4ss `
  --loader-version v3.0.1
```

这些版本只是可复现示例，不是对所有游戏的兼容承诺。生成前应从官方上游
重新确认版本，并验证目标游戏的 Mod、许可和反作弊策略。

## Agent 操作面

生成的 `agent-interface.json` 只声明 `describe`、`attach`、`step` 和
`close`，未知操作默认拒绝。`step` 必须携带当前 episode 和预期
step 身份；`action_vocabulary` 初始为空。`AGENTS.md` 会告诉新 Agent 应读取
哪些契约、执行哪些命令以及禁止扩展哪些权限。

开发者只应添加小而明确的游戏语义动作表。不得暴露反射/对象搜索与 dump、
任意 Lua/C# 执行、通用函数调用或原始输入入口。BepInEx 模板提供固定队列
上限和 `TryApply` 后置状态契约；UE4SS 模板通过 `ExecuteInGameThread`
调度，在没有受审 Lua handler 时拒绝全部动作。

## 构建和部署暂存

BepInEx 项目使用操作者提供的引用目录构建，这些目录不得提交到 Git：

```powershell
dotnet build runtime/bepinex/GlrBridge.csproj -c Release `
  -p:BEPINEX_ROOT="$env:BEPINEX_ROOT" `
  -p:GAME_MANAGED_ROOT="$env:GAME_MANAGED_ROOT"
```

UE4SS Lua 不需要编译。两种 Loader 都使用相同命令生成部署暂存包：

```powershell
vx run package-runtime
```

`.glr-dist/loader-package` 只包含 `payload/` 和校验和清单。打包脚本不会
扫描游戏，也不会执行安装；操作者必须单独选择并批准准确的授权目标。

## 训练和复现

```powershell
vx run check
vx run train
vx run reproduce
```

初始训练器是确定性的合成 BC 冒烟测试，会产出自包含的
`glr.model-bundle.v1`。替换成真实 PPO、IMPALA 或 BC 后仍应保留相同的
模型包校验门。详见[可复现模型包](reproducible-model-bundles.zh-CN.md)。

合成测试不能证明 Loader 成功启动或真实游戏行为。真实验收必须在单独的、
有界的授权运行时中完成，公开材料只包含聚合 conformance 结果。

官方资料：

- [BepInEx](https://github.com/BepInEx/BepInEx)
- [BepInEx 插件开发](https://docs.bepinex.dev/master/articles/dev_guide/plugin_tutorial/index.html)
- [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS)
- [UE4SS Lua Mod](https://docs.ue4ss.com/dev/guides/creating-a-lua-mod.html)
- [UE4SS 游戏线程调度](https://docs.ue4ss.com/dev/lua-api/global-functions/executeingamethread.html)
