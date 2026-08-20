# 流水线真实功能问题清单

英文原文：[PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md)

- 状态：持续更新
- 上级方案：[跨 Baseline 流水线真实功能审计](PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md)
- 最近同步：2026-08-20

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
| RF-CORE-001 | T01 | S1 | `arf`、`bn`、`ctab-gan`、`ctab-gan-plus`、`ctgan`、`great`、`nflow`、`nrgboost`、`smote`、`tabddpm`、`tabdiff`、`tabebm`、`tabsds`、`tabsyn`、`tabula`、`tvae` | 明确的逐动作白名单现在会在执行前拒绝未知顶层参数；过期 GReaT 配置已迁移到当前接口。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；[回归证据](../evidence/audits/pipeline-phase2-remediation-20260814.json) | fixed |
| RF-CORE-002 | T02 | S1 | 全部 21 个适配器的直接 `run`/`run-action` 路径 | 共享预检和公共 `RunSpec` 现在绑定常规文件状态、字节数和 SHA-256，并拒绝构建后发生的内容变化。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；内容变更回归 | fixed |
| RF-CORE-003 | T03 | S1 | 全部 21 个适配器的直接 `run`/`run-action` 路径 | 直接输出目录现在携带共享且不可变的数据集身份以及逐动作运行身份；冲突数据、种子或配置会被拒绝，兼容动作可合并到同一声明运行。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；输出身份回归 | fixed |
| RF-CORE-004 | T04 | S2 | `great` 训练、`tabula` 训练、`nrgboost`、`smote`、`tabddpm`、`tabsds` | CPU-only 适配器拒绝 CUDA；trainer 适配器明确选择并观察设备；TabDDPM 在运行 TOML 中绑定设备并拒绝不可用 CUDA。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；设备回归 | fixed |
| RF-CORE-005 | T02 | S2 | 公开 `DatasetSpec` 消费者；在 `tabula` Windows 采样和 `show-dataset` 中实际发现 | `DatasetSpec.extra` 内的嵌套 `Path` 会原样跨过公开 JSON 边界，导致规范 Adult 采样在隔离子进程启动前失败。现在公开接口 payload 会递归序列化映射和序列中的路径。 | [TabuLa Windows V2 证据](../evidence/tabula/windows-v2-real-function-8d72ee8.json)；递归接口序列化回归 | verified |
| RF-CORE-006 | T05 | S2 | 各模型专属运行环境中的 Pipeline V2 最终化；在 `tabularargn` 中实际发现 | 第一次 TabularARGN 模型探针已通过，但模型环境没有声明 `jsonschema`，所以中央 P3 最终化明确失败。现在模型执行与中央评测使用各自独立的冻结环境，并且最终化会在读取样本前核验中央依赖锁。 | [保留的失败记录](../evidence/tabularargn/windows-v2-finalization-dependency-failure-ec53f98.json)；[干净重跑成功证据](../evidence/tabularargn/windows-v2-real-function-6f9e065.json)；V2 环境锁回归 | verified |
| RF-CORE-007 | T05 | S2 | 全部适配器的纯评测动作；在独立中央评测环境中由 `arf` 实际发现 | ARF 的训练与两次生成都已通过，但中央 P3 最终化误导入 ARF 适配器；评测环境按设计不安装模型专属的 `sklearn`，因此明确失败。现在纯评测的动作、上下文和流水线路由只从注册表读取来源信息，不再导入模型运行时；随后独立评测器成功最终化了 ARF 的干净重跑。 | [保留的失败记录](../evidence/arf/windows-v2-finalization-adapter-import-failure-c84a869.json)；[干净重跑成功证据](../evidence/arf/windows-v2-real-function-eb37290.json)；禁止模型导入的回归测试 | verified |
| RF-CORE-008 | T04 | S2 | 干净的 Pipeline V2 模型运行环境；在 Windows `nrgboost` 验证环境中实际发现 | 第一次 NRGBoost 探针在训练前停止，因为其冻结运行环境漏掉了 Pipeline V2 校验依赖身份所需的 `packaging`。现在 NRGBoost extra 与独立 Windows V2 锁都明确声明精确版本，Linux 等价环境锁保持不变；干净重跑已通过模型执行与独立中央最终化。 | [保留的失败记录](../evidence/nrgboost/windows-v2-probe-dependency-failure-3271298.json)；[源码构建来源](../evidence/nrgboost/windows-source-build-provenance-20260820.json)；[干净重跑成功证据](../evidence/nrgboost/windows-v2-real-function-64eec7d.json)；锁文件回归测试 | verified |
| RF-INTERNAL-DATA-001 | T02 | S1 | `codi`、`stasy`、`tabddpm`、`tabdiff`、`tabsyn` | 模型原生视图现在按校验和绑定到规范 `DatasetSpec`，或在运行所有权下确定性物化。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；绑定回归 | fixed |
| RF-TABDDPM-001 | T01 | S1 | `tabddpm` | 经过语义往返校验的运行 TOML 现在绑定训练/变换/采样种子、设备、请求行数、数据和输出，且不修改源 TOML。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；生效 TOML 受控模拟 | fixed |
| RF-TABDDPM-002 | T03 | S1 | `tabddpm` | 适配器现在校验运行所有检查点、解码按种子隔离的规范表、保留原始数组并公开 `generated_sample_path`。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；TabDDPM 解码回归 | fixed |
| RF-GOGGLE-001 | T01 | S1 | `goggle` | 独立采样种子会传给启动器，并在上游构造函数用训练种子重置 RNG 之后、实际采样之前，重新应用于 Python、NumPy 和 PyTorch。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；启动器 RNG 回归；Windows V2 真实功能探针 | fixed |
| RF-CTGAN-FAMILY-001 | T01 | S1 | `ctgan`、`tvae` | 加载后的官方合成器现在会在生成前使用请求的采样种子重置随机状态。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；CTGAN 系列随机状态回归 | fixed |
| RF-UPSTREAM-WORKSPACE-001 | T03 | S1 | `tabddpm`、`tabdiff`、`tabsyn` | 所有可变检查点和结果均重定向到模型专用的运行所有目录；权威源码树保持不变。 | [第二阶段报告](PHASE_2_REMEDIATION_REPORT.zh-CN.md)；运行路径回归 | fixed |

第一阶段的 10 个发现已完成修复和 V1 回归，且所有未阻塞的受影响适配器现已通过 V2。声明范围还包含外部阻塞 TabEBM 的条目仍保守保持 `fixed`；该阻塞不会被当作探针通过。

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
