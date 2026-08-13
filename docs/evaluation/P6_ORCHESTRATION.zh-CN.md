# P6 资源感知运行编排

## 1. 范围与声明边界

P6 是 Benchmark 的实验执行层。它让一个已配置实验能够恢复、受资源边界约束、按内容寻址并可审计。P6 不新增科学指标，不重新计算 Atomic Result，不聚合排行榜，不跨硬件归一化性能，也不会让任何模型、数据集、运行或指标自动进入 Official Results。

现有 `run` 命令在 P8 迁移窗口内仍是旧的直接适配器管线。新的 P6 执行使用明确的 `benchmark` 命令组。

## 2. 七阶段计划

`benchmark run` 会解析并执行以下有序 DAG：

1. `prepare`：解析实验配置、适配器、Dataset Spec、源码身份和就绪证据；
2. `train`：在用户请求时执行所选适配器的训练动作；
3. `sample`：在用户请求时生成解码后的合成表；
4. `validate`：确认生成产物是可读取的普通文件，记录校验和与行数，但不修复数据；
5. `evaluate`：在用户请求时执行已配置的适配器评测动作；
6. `aggregate`：索引已完成的运行阶段产物，但不进行 P7 的科学或排行榜聚合；
7. `report`：生成运行报告，但不重新计算科学数值。

未启用的阶段以 `not_requested` 记录为 `skipped`。依赖失败时，其消费者以 `dependency_failed` 跳过。即使前面失败，`aggregate` 和 `report` 仍可运行，以保留已完成产物和失败证据。

每个启用动作都在独立进程组中执行。私有 worker 会把动作本身的时间与进程启动、适配器导入开销分开记录。请求 CUDA 且运行时可用时，会在加速器计时边界执行同步。

## 3. 命令接口

安装执行依赖：

```powershell
python -m pip install ".[orchestration]"
```

运行一个配置：

```powershell
std-tabular-diffusion benchmark run `
  --config configs/my-experiment.json `
  --hardware-profile-id local-windows-rtx5080
```

可以显式指定资源边界与重试策略：

```powershell
std-tabular-diffusion benchmark run `
  --config configs/my-experiment.json `
  --timeout-seconds 3600 `
  --memory-gib 48 `
  --max-retries 1 `
  --cache-dir D:/std-tabular-cache
```

查看并验证最近一次执行：

```powershell
std-tabular-diffusion benchmark status --output-dir artifacts/model/dataset/run-001
std-tabular-diffusion benchmark validate --output-dir artifacts/model/dataset/run-001
std-tabular-diffusion benchmark hardware-profile --profile-id local-windows-rtx5080
```

`--no-cache` 强制重新执行所有启用阶段。`--no-resume` 要求输出目录中不存在已有编排状态；它不会静默隐藏旧尝试。

## 4. 状态与产物分离

执行诊断统一保存在 `<output_dir>/.orchestration/`：

```text
.orchestration/
  run.json
  profiles/hardware-<fingerprint>.json
  profiles/software-<fingerprint>.json
  invocations/<invocation-id>.json
  attempts/<stage-id>.json
  logs/<stage-id>.jsonl
  worker-results/<stage-id>.json
  cache/
```

科学结果和适配器产物保持原有位置。运行阶段摘要位于 `artifacts/execution/`。`pipeline_result.json` 和 `benchmark_report.json` 会明确标记为运行信息，不是排行榜事实来源。

时间戳、主机诊断、进程标识和资源观测不会写入科学产物。日志采用结构化 JSON Lines。常见凭证、Bearer token、GitHub token、用户主目录、仓库根目录、运行根目录和缓存根目录都会脱敏。实验配置中疑似内嵌凭证的字段会在执行前报错；凭证必须通过外部、基于环境的机制提供。

## 5. 身份与内容寻址缓存

每个阶段身份绑定以下内容：

- 阶段名称与版本；
- 规范化科学配置身份；
- 规范化科学配置指纹，以及数据输入、外部合成表、上游配置、输入 checkpoint 和源码锁文件（如存在）的校验和；
- 依赖阶段的状态、身份和输出指纹；
- 仓库提交与未提交补丁身份；
- 已安装包和关键环境身份；
- 硬件比较键。

缓存项只能由这套完整身份寻址。每个声明输出都保存为 SHA-256 blob，复用前同时验证摘要与字节数。缺失、格式错误或损坏的缓存会被标记为 `invalid`，明确拒绝并重新执行。普通损坏 blob 只能用新执行后验证通过的输出修复；符号链接或非普通缓存对象会关闭失败。

缓存命中会建立新的阶段尝试并链接原始尝试。它记录 `measurement_mode=cache-reuse` 且 `efficiency_eligible=false`；命中时的耗时只是缓存验证成本，不是模型效率。无法完整捕获到运行根目录中的阶段产物不会进入缓存，避免只有元数据却缺少 checkpoint、生成表或评测产物的错误复用。

## 6. 恢复、重试与失败语义

每次调用和阶段尝试都有新标识。恢复调用会记录前一次调用；重试阶段会记录所有相同身份的祖先尝试。尝试文件和日志按身份保留：成功重试不会删除或改写失败祖先。

如果同一输出目录的配置指纹发生变化，恢复会关闭失败。`--max-retries` 默认只用于明确的超时和实现失败类别。超时、进程树内存耗尽、依赖/导入失败、实现失败和用户中断保持为不同结构化类别。

如果必需阶段全部成功而可选阶段失败，总体状态为 `partial`。已经完成的输出继续由校验和绑定，并可被明确允许失败依赖的下游阶段使用。缺失或失败观测不会转成 0。

## 7. 资源测量

全新执行在可靠时记录：

- 完整子进程 wall time；
- worker 报告的动作 wall time；
- 被排除的 setup 时间；
- 进程树 CPU 时间；
- 进程树峰值常驻内存；
- 加速器进程峰值内存和框架报告峰值；
- 请求与实际行数；
- 每动作秒生成行数；
- warm-up 政策；
- 测量可靠性代码。

进程树 RAM/CPU 强制边界使用 `psutil`。请求内存上限但依赖不可用时，阶段会在启动前失败。NVIDIA 显存通过 `nvidia-smi` 的计算进程观测采样；请求 CUDA 的 worker 在 PyTorch 可用时也报告框架峰值。不支持的测量会以 `null` 和原因呈现，不会编造数值。

依赖安装和数据下载不计入阶段执行时间。完整子进程时间和动作本身时间分别保留。当前标准适配器计划将采样标记为 `cold-single-run-diagnostic`，不声称已经实现协议未来要求的三次 warm generation 中位数。

## 8. 硬件配置与可比性

每次调用都会捕获操作系统、Python、CPU 型号、逻辑/允许线程数、RAM、加速器清单、驱动/CUDA 运行时以及关键运行设置。这些字段形成 SHA-256 硬件比较键。

仓库提供的兼容性检查会拒绝合并不同比较键的效率记录。系统不会使用 FLOPS 换算或跨设备归一化。用户提供的 profile 标签只命名一次观测配置。捕获或命名 profile 始终保持 `official_efficiency_eligible=false`；正式准入仍是独立的 P7/发布决策。

## 9. 可选后端与部分成功

公开的 `StageSpec` 和 `execute_plan` API 支持分别命名的可选评测阶段，例如 `evaluate.metric-a`。每个阶段都有独立进程、时间/内存边界、日志、输出、缓存身份和失败状态。下游运行聚合可以显式允许失败依赖。在 P8 把中央 P2–P5 表格评测接入该引擎前，CLI 的标准七阶段适配器计划仍使用一个评测阶段。

## 10. 验证与当前状态

P6 为硬件 profile、软件 profile、编排阶段尝试和运行 manifest 提供 JSON Schema。`benchmark validate` 会重新计算 profile 身份与当前输出校验和、验证依赖引用、检查缓存命中证据、拒绝把缓存复用当作效率结果、核对日志身份与密钥脱敏，并根据当前尝试重新计算终态。

P6 退出门验证器和 CI 覆盖：

- 强制超时和进程树内存耗尽；
- 模拟中断和依赖阶段取消；
- 失败后重试及祖先链；
- 精确缓存命中、身份变化 miss、损坏缓存拒绝与修复；
- 可选后端部分成功，同时保留已完成结果；
- 同一命名观测硬件下固定工作负载的重复时间/RAM 观测；
- 拒绝跨硬件 profile 的效率比较。

这些都只是工程和运行声明。真实模型质量、Official Results 准入、排行榜聚合和发布支持仍是独立关卡。

完成证据（2026-08-13）：完整退出验证器已在 Windows/AMD64 与 Python 3.11.15 上针对兼容性提交 `da47011` 通过。[机器可读证据](../evidence/evaluation/p6-windows-py311-da47011.json)已留存，SHA-256 为 `67b5f40889c2f3e2b2da853303afaba99f3a866829dd020cb1e30e147f6887c2`。该证据锁定实现、Schema、测试、CLI、打包检查以及 P4/P6 的 CI 依赖边界；它不声明模型质量、排行榜、跨硬件归一化或 Official Results 准入。初始实现提交 `0fb5d07` 的证据仍作为已被取代的历史记录保留。
