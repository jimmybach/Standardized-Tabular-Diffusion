# 架构

## 公共流程

~~~text
数据集 + 已审阅 Profile
        |
        v
官方包/源码 <- 适配器训练与采样 -> 原始解码表格
                                      |
                                      v
                                 中央 P2-P5 评测
                                      |
                                      v
                              最终 Result Bundle
                                      |
                                      v
                         P7 诊断/Official 发布门
~~~

适配器可以转换配置、数据布局、调用方式和产物，但不拥有指标公式。公开的 `run-action evaluate`、顶层 `run` 和 P6 `benchmark run` 都会把解码样本交给中央评测器。

## 主要组件

- `registry.py` 分开记录源码权威性、分发形式、复现目标、修改状态、适配器验证、榜单轨道、支持级别、许可和证据。
- `models/` 包含围绕官方包或校验和锁定源码的训练/采样适配器。
- `evaluation/` 管理版本化请求、Profile、结构门、指标、Atomic Result、bundle、旧格式导入和快照。
- `orchestration/` 在隔离且受资源约束的进程中执行七阶段流程。
- `resources/` 与 `schemas/` 保存机器可读契约和可再分发的小型夹具。
- `validation/` 生成工程证据，但不能单独授予科学准入。

## 证据类型

`artifacts.json` 是适配器元数据；`evaluation-result/manifest.json` 标识包含 Atomic Result 的新 Result Bundle；P7 快照是另一种发布产物，三者不能混用。

旧 `standardized_summary.json` 不含 Atomic Result，也缺少 Official 准入证据。P8 只允许 `import-legacy-summary` 原样保存其字节，并由 `legacy_import_record.json` 明确声明“未转换”。

## 信任与状态边界

表格、Profile、配置、下载源码、checkpoint 和 bundle 在校验前都不可信。系统拒绝符号链接、路径穿越、重复键、非有限 JSON、覆盖最终证据和嵌入凭证。

适配器验证、榜单轨道、发布支持、指标生命周期、数据集准入、运行准入和发布类别互相独立。快速开始或 CI 成功不能自动提升其他状态。
