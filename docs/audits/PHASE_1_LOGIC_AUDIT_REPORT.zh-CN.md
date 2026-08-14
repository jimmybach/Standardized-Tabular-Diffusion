# 第一阶段跨 Baseline 逻辑审计报告

英文原文：[PHASE_1_LOGIC_AUDIT_REPORT.md](PHASE_1_LOGIC_AUDIT_REPORT.md)

- 状态：已完成，存在已确认问题
- 审计日期：2026-08-14
- 被审计提交：`e80296340f6a083eead55fded24a5ea6027d5c93`
- 范围：T01–T04，V0 静态审阅与 V1 受控模拟，覆盖全部 21 个已注册适配器
- 证据：[`pipeline-phase1-logic-audit-20260814.json`](../evidence/audits/pipeline-phase1-logic-audit-20260814.json)

## 1. 结果

第一阶段完成了全部 **84 个任务/模型单元**：21 个适配器分别完成四类低依赖审计。所有适配器的配置投影均通过：训练种子、独立生成种子、请求设备、行数、动作专属参数以及嵌入的 `DatasetSpec` 都能正确进入 `RunSpec`。

继续向下审阅后确认了 **10 个待解决问题**：9 个 S1 高严重程度契约缺口和 1 个 S2 设备边界缺口。没有发现 S0 级凭证泄露、破坏性删除或敏感数据暴露。因为本轮审计是在另一条轴上检查更广泛的公开流水线契约，所以这些问题不会自动推翻既有的上游原生等价结果。

第一阶段没有训练真实模型，也没有修改适配器行为。修复属于第二阶段；只有先修正共享 S1 边界，才开始 V2 原生 Windows 最小真实运行。

## 2. 检查内容

对每个适配器都检查了：

- 配置优先级以及训练/生成动作专属参数；
- 训练种子和生成种子是否被下游真正使用；
- 标准数据集路径与上游原生数据布局的关系；
- 解码后的行数、列、缺失值和类型后置条件；
- 检查点、样本、元数据和结果的归属；
- 非空、重复使用和符号链接输出目录的行为；
- 请求 CPU/CUDA 后的行为以及意外设备回退；
- Windows 安全的子进程参数、路径和环境处理。

审计用 SHA-256 绑定了相关的仓库适配器、启动器、共享配置/runner 文件和直接涉及的上游入口。受控模拟还直接证明了 TabDDPM 参数不变缺陷，以及公共直接预检缺少数据摘要的问题。

## 3. 已确认问题

### 3.1 T01：种子与配置

`RF-TABDDPM-001` 是优先级最高的模型专属问题。普通 TabDDPM 适配器只传入 `--config <path> --train/--sample`；改变 `RunSpec.seed`、`RunSpec.device`、`RunSpec.num_samples` 或 `RunSpec.output_dir` 都不会改变上游调用。受控模拟同时改变这四项，仍观察到完全相同的命令。

`RF-GOGGLE-001` 表明 Goggle 生成时使用训练阶段运行配置中的种子。独立生成种子虽然进入 `RunSpec`，但没有进入启动器配置。

`RF-CTGAN-FAMILY-001` 影响 CTGAN 和 TVAE。训练时使用了 `spec.seed`，但生成阶段加载检查点后直接调用 `model.sample()`，没有使用请求的生成种子重置官方包随机状态。

`RF-CORE-001` 影响 16 个适配器。它们的动作参数解析不会拒绝未知顶层 extra，因此拼错的调参字段可能无声退回默认值。CoDi、Goggle、REaLTabFormer、STaSy 和 TabularARGN 已经具有严格的顶层参数白名单。

### 3.2 T02：数据与输出契约

`RF-CORE-002` 影响全部 21 个适配器的直接公开 runner。预检会记录数据集路径和是否存在，但不记录文件大小或 SHA-256。部分适配器稍后会自己绑定数据，P6 也具有更强的内容寻址编排；但公共直接 `run`/`run-action` 边界仍未绑定文件内容。

`RF-INTERNAL-DATA-001` 影响 CoDi、STaSy、TabDDPM、TabDiff 和 TabSyn。它们的权威运行时读取上游内部布局或 TOML 路径，而不是嵌入的标准 `DatasetSpec` 路径。当前预检没有证明这些内部文件与注册数据集身份内容一致，因此可能选中陈旧的原生布局副本。

### 3.3 T03：检查点与产物

`RF-CORE-003` 影响全部 21 个适配器的直接运行。公共输出边界会直接创建或复用已有目录，没有统一的运行/种子身份，也没有统一非空策略。生成通常写入 `samples.csv`；在同一输出目录运行后一个种子时可能替换前一个结果。只有通过 P6 内容寻址路径运行时，P6 才能缓解这个问题。

`RF-TABDDPM-002` 表明 TabDDPM 的产物归属仍由 TOML 决定。它的 sample bundle 没有提供 `generated_sample_path`，因此普通顶层流水线无法自动把生成表交给中央评测。

`RF-UPSTREAM-WORKSPACE-001` 影响 TabDDPM、TabDiff 和 TabSyn。除非通过模型专属运行配置重定向或复制，它们的原生训练检查点或结果布局仍位于上游工作树内。这与新审计政策冲突：受版本控制的源码树应是只读运行时，所有可变产物都应属于声明的运行目录。

### 3.4 T04：Windows、GPU 与依赖

`RF-CORE-004` 影响 GReaT 训练、Tabula 训练、NRGBoost、SMOTE、TabDDPM 和 TabSDS。这些路径会忽略 `RunSpec.device`，或把设备选择交给自动机制且没有记录封闭式观察。CPU-only 算法必须明确拒绝 CUDA 请求；自动 trainer 必须尊重明确的 CPU 请求并记录实际设备。

静态审阅没有确认其他非 ASCII 参数引用问题。但这只是 V1 结论：每个未阻塞适配器仍需从当前仓库路径完成 V2 真实运行，才能获得原生 Windows 资格。

## 4. 逐模型结果矩阵

空白表示 V0/V1 没有为该任务分配模型专属或共享问题，不表示 V2 已通过。

| 模型 | T01 | T02 | T03 | T04 |
|---|---|---|---|---|
| ARF | CORE-001 | CORE-002 | CORE-003 | — |
| BN | CORE-001 | CORE-002 | CORE-003 | — |
| CoDi | — | CORE-002、INTERNAL-DATA-001 | CORE-003 | — |
| CTAB-GAN | CORE-001 | CORE-002 | CORE-003 | — |
| CTAB-GAN+ | CORE-001 | CORE-002 | CORE-003 | — |
| CTGAN | CORE-001、CTGAN-FAMILY-001 | CORE-002 | CORE-003 | — |
| Goggle | GOGGLE-001 | CORE-002 | CORE-003 | — |
| GReaT | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| NFlow | CORE-001 | CORE-002 | CORE-003 | — |
| NRGBoost | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| REaLTabFormer | — | CORE-002 | CORE-003 | — |
| SMOTE | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| STaSy | — | CORE-002、INTERNAL-DATA-001 | CORE-003 | — |
| TabDDPM | CORE-001、TABDDPM-001 | CORE-002、INTERNAL-DATA-001 | CORE-003、TABDDPM-002、UPSTREAM-WORKSPACE-001 | CORE-004 |
| TabDiff | CORE-001 | CORE-002、INTERNAL-DATA-001 | CORE-003、UPSTREAM-WORKSPACE-001 | — |
| TabEBM | CORE-001 | CORE-002 | CORE-003 | — |
| TabSDS | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| TabSyn | CORE-001 | CORE-002、INTERNAL-DATA-001 | CORE-003、UPSTREAM-WORKSPACE-001 | — |
| Tabula | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| TabularARGN | — | CORE-002 | CORE-003 | — |
| TVAE | CORE-001、CTGAN-FAMILY-001 | CORE-002 | CORE-003 | — |

## 5. 第二阶段修复顺序

建议顺序如下：

1. 在共享代码中增加数据内容绑定，以及运行/种子输出隔离。
2. 修复普通 TabDDPM 适配器，使种子、设备、行数、数据身份、输出归属和生成样本路径都明确可控。
3. 修复 CTGAN、TVAE 和 Goggle 的独立生成种子传递。
4. 在不改变上游数学逻辑的前提下，移动或安全重定向 TabDiff/TabSyn 可变工作区；如确实必须修改上游源码，先与你讨论。
5. 增加严格动作参数白名单和封闭式设备策略。
6. 重跑第一阶段全部测试，然后按已经认可的由快到重批次开始 V2。

第二阶段必须保留现有源码锁和原生等价证据。修复仓库适配层边界，并不等于获得修改权威实现的授权。
