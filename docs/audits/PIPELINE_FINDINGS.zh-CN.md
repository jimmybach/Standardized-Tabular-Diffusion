# 流水线真实功能问题清单

英文原文：[PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md)

- 状态：持续更新
- 上级方案：[跨 Baseline 流水线真实功能审计](PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md)
- 最近同步：2026-08-14

## 规则

本文件是跨 baseline 问题的人工可读索引。它只记录已经确认的事实，不记录猜测。大型日志和生成产物继续留在 Git 之外；存在精简复现、测试或保留证据时，每一行都应链接到它们。

严重程度与实施路线中的阶段编号相互独立：

- **S0 严重：** 凭证/数据暴露、越过运行边界的破坏行为，或无效科学证据被发布。
- **S1 高：** 无声科学错误、泄漏、种子/配置/数据错误、误用旧结果或假成功。
- **S2 中：** 明确运行失败、不支持的平台边缘问题，或不存在无声科学损坏的恢复缺陷。
- **S3 低：** 能安全失败的诊断、文档或易用性问题。

允许的状态为 `confirmed`、`fix-in-progress`、`fixed`、`verified`、`deferred` 和 `accepted-blocker`。只有回归测试及适用的真实功能探针均通过后，问题才能标记为 `verified`。

每个新条目必须包括：问题 ID、任务、严重程度、受影响适配器、观察行为、期望行为、复现/证据、根因、解决方案、回归覆盖、状态和结论边界。

## 当前问题

| ID | 任务 | 严重程度 | 受影响适配器 | 问题与解决方案 | 证据/回归 | 状态 |
|---|---|---:|---|---|---|---|
| RF-TABDIFF-001 | T01 | S1 | `tabdiff` | 上游入口没有暴露仓库请求的配置路径和训练/生成种子。现在使用受源码校验和保护的运行时 overlay 传递二者，不修改纳入版本控制的上游文件。 | [`TABDIFF_VALIDATION.md`](../TABDIFF_VALIDATION.md)；适配器/验证测试 | verified |
| RF-TABDIFF-002 | T03 | S1 | `tabdiff` | 多个生成种子此前依赖上游默认结果位置。适配器现在把每个解码样本复制到种子专属运行产物，并记录哈希。 | [Adult 真实功能证据](../evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json) | verified |
| RF-TABDIFF-003 | T02 | S1 | `tabdiff` | 官方 Adult 配置在逆去量化为 `none` 时，会在声明的整数列中生成小数。标准化输出契约现在要求使用官方 `round` 路径，否则拒绝该样本。 | [Adult 真实功能证据](../evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json)；整数恢复测试 | verified |
| RF-TABDIFF-004 | T04 | S2 | `tabdiff` | 当前 PyTorch 和 Windows 运行需要范围很小的 scheduler、诊断绘图、编码和非 ASCII 路径处理。每个兼容桥都会封闭失败，并受源码校验和保护。 | [`TABDIFF_VALIDATION.md`](../TABDIFF_VALIDATION.md)；运行时 overlay manifest | verified |
| RF-TABDIFF-005 | T04 | S2 | `tabdiff` | 运行元数据构建此前会在父进程无条件导入可选依赖 PyTorch，导致轻量适配器契约测试失败。现在 PyTorch 不存在时会明确记录“父环境不可检查”；真实模型执行仍要求声明的模型依赖。 | 可选依赖回归测试；完整核心测试套件 | verified |
| RF-CORE-001 | T01 | S1 | `arf`、`bn`、`ctab-gan`、`ctab-gan-plus`、`ctgan`、`great`、`nflow`、`nrgboost`、`smote`、`tabddpm`、`tabdiff`、`tabebm`、`tabsds`、`tabsyn`、`tabula`、`tvae` | 未知的顶层动作参数没有被统一拒绝，因此拼写错误的选项可能静默使用默认值。第二阶段必须增加严格白名单，且不改变模型数学逻辑。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-002 | T02 | S1 | 全部 21 个适配器的直接 `run`/`run-action` 路径 | 共享预检仅记录数据路径和是否存在，没有记录文件大小或 SHA-256，所以无法把直接运行与注册数据集身份按内容绑定。第二阶段必须增加共享内容绑定。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-003 | T03 | S1 | 全部 21 个适配器的直接 `run`/`run-action` 路径 | 共享输出边界会接受已存在的目录，但没有通用运行/种子身份或非空目录策略；后一次生成可能覆盖 `samples.csv`。第二阶段必须隔离输出所有权，或拒绝冲突。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-004 | T04 | S2 | `great` 训练、`tabula` 训练、`nrgboost`、`smote`、`tabddpm`、`tabsds` | 这些路径会忽略请求的设备，或把选择交给自动机制但不封闭失败地记录结果。第二阶段必须明确 CPU-only 拒绝规则和 CPU/CUDA 选择记录。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-INTERNAL-DATA-001 | T02 | S1 | `codi`、`stasy`、`tabddpm`、`tabdiff`、`tabsyn` | 原生运行时读取内部布局或 TOML 路径，而共享边界未证明其内容与内嵌的标准 `DatasetSpec` 一致。第二阶段必须用校验和绑定或确定性生成原生视图。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-TABDDPM-001 | T01 | S1 | `tabddpm` | 普通适配器只传递 TOML 路径和动作标志。受控模拟表明，改变种子、设备、请求行数和输出目录后，上游命令仍完全不变。第二阶段必须显式绑定这些参数。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-TABDDPM-002 | T03 | S1 | `tabddpm` | 输出所有权仍由 TOML 决定，且生成 bundle 没有暴露 `generated_sample_path`，因此顶层路由无法自动把表交给中央评测。第二阶段必须明确生成样本的所有权。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-GOGGLE-001 | T01 | S1 | `goggle` | 生成阶段使用训练运行配置重新设置种子，而不是独立的生成 `RunSpec.seed`。第二阶段必须传递并应用生成种子。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CTGAN-FAMILY-001 | T01 | S1 | `ctgan`、`tvae` | 训练会设置官方模型的随机状态，但生成重新加载检查点后直接调用 `sample()`，没有用请求的生成种子重置状态。第二阶段必须在生成前恢复官方包的随机状态控制。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-UPSTREAM-WORKSPACE-001 | T03 | S1 | `tabddpm`、`tabdiff`、`tabsyn` | 如果没有模型专属的重定向/复制，可变检查点或结果可能保留在上游工作树中。第二阶段必须把可变产物归入声明的运行目录，同时不修改权威实现的数学逻辑。 | [第一阶段报告](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)；[不可变证据](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |

这些已解决的 TabDiff 条目说明，仅做 V1 模拟还不够。第一阶段确认的 10 个条目都只是边界明确的 V0/V1 发现；它们不表示 V2 已经失败，只有回归覆盖和适用的真实功能探针都通过后，才能关闭。

## 新问题模板

~~~text
ID: RF-<MODEL-OR-CORE>-NNN
任务: T01-T06
严重程度: S0-S3
受影响适配器:
观察行为:
期望行为:
复现/证据:
根因:
解决方案:
回归覆盖:
状态:
结论边界:
~~~
