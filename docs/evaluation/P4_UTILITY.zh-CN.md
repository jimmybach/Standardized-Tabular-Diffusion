# P4 Local 与 Global Utility

## 状态与声明边界

P4 已实现，其有限范围诊断门已在 [Linux/Python 3.11](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31053624769) 上通过，并已留存[机器可读证据](../evidence/evaluation/p4-utility-run-31053624769.json)，SHA-256 为 `bb2b5f3d48647122b1036f8ce010eeecee948a0dfb4a0bfc247ab7100439cd59`。它仍属于**诊断性 pilot**：尚未协议冻结、尚未达到 release-supported，也不能进入 Official Results。

当前实现已经建立从三张不可变解码表，到 Local/Global Utility Atomic Results，再到可自校验最终 bundle 的完整审计路径。但目前不声称与完整 TabEval 预测器运行时达到可执行等价。该声明还需要在 Linux/Python 3.11 上运行锁定的 AutoGluon、XGBoost、KNN 与 TabPFN 栈，审阅运行预算并留存证据。

## 必需输入与数据泄露边界

P4 必须提供三张由校验和绑定的表：

- real train：唯一允许拟合特征变换和 TRTR 预测器的真实表；
- synthetic train：唯一允许拟合 TSTR 预测器的表；
- held-out real test：只允许评测，绝不能参与拟合。

三张表都必须通过严格的 canonical model-view gate。缺失的模型输入、非有限数值、有损类型转换、模式差异以及错误的合成行数都会被拒绝。生成值绝不会被修复。

Evaluation Request 分别保存 real train、real test 和 synthetic 的校验和。Bundle 校验会验证这些身份，并要求每个 Local run 证明使用了同一个测试视图。

## Local Utility

每个已审阅数据集声明一个主要任务。Adult 预测 `income`，Sick 预测 `Class`。Pilot profile 使用：

- 分类主指标：Macro-F1；
- 回归主指标：RMSE；
- 分类辅助指标：适用时的 Balanced Accuracy、ROC-AUC 和 PR-AUC；
- 回归辅助指标：适用时的 MAE 和 R-squared；
- 三类评测器：线性模型、随机森林和直方图梯度提升。

具体包类和参数保存在 `p4-utility-pilot-v1.json`。分类使用 Logistic Regression、Random Forest 和 Histogram Gradient Boosting；回归使用 Ridge、Random Forest 和 Histogram Gradient Boosting。随机模型接收 Evaluation Request 中的种子，并在支持时只使用一个 worker。

类别独热编码和数值缩放状态只在 real-train 特征上拟合一次，再把同一个冻结变换应用于 real train、synthetic train 和 real test。目标列和 held-out test 值都不会参与变换拟合。

对每个评测器和种子，P4 都保留主指标及适用辅助指标的三条原始结果：

- Dummy：在 real train 上拟合的多数类分类器或均值回归器；
- TRTR：在 real train 上训练，在 real test 上评测；
- TSTR：使用相同预测器配置在 synthetic train 上训练，在同一个 real test 上评测。

单独标识的、本基准派生的 Local Utility retention 为：

~~~text
越大越好：(TSTR - Dummy) / (TRTR - Dummy)
越小越好：(Dummy - TSTR) / (Dummy - TRTR)
~~~

Retention 不截断。如果 TRTR 相对 Dummy 的改善不超过 `1e-12`，retention 为 `mathematically_undefined`。只有所有请求的评测器/种子 retention 都可计算时，严格汇总才给出数值。

## Global Utility

Global Utility 轮流把每个纳入的 canonical model-view 列作为目标。Dataset Profile 必须说明每一列：要么纳入，要么给出稳定的排除原因。标识符、ignored 字段和 audit-only 字段不能纳入。在 datetime 和 string 目标策略冻结前，这两类目标必须明确排除。

经审阅的来源公式是 TabStruct 公式 4：

~~~text
类别目标：balanced_accuracy(TSTR) / balanced_accuracy(TRTR)
数值目标：RMSE(TRTR) / RMSE(TSTR)
Global Utility：先对目标等权平均，再对种子等权平均
~~~

大于 1 的比率有效，绝不截断。分母为零或非有限值时明确标记 `mathematically_undefined`。只要任一请求的目标/种子比率不可用，诊断性 `global_utility` 就为 null；绝不会静默地重新分配已计算目标的权重。

锁定的低成本预测器身份是 TabEval revision `dba19a4ee7aa391621cbeb464609285fd515dece`、时间戳 `2025-08-09` 的 `UtilityPerFeature`，通过 AutoGluon 配置 XGB、KNN 和 TabPFN。P4 会记录后端实际训练的模型名称和逐模型分数。一个目标只有在 TRTR 与 TSTR 使用相同预测器集合时才能计算比率。禁止未记录地替换模型或缩减 profile。

锁定的 TabPFN 实现最多支持十个类别。对更高基数的类别目标，AutoGluon 可能按来源行为跳过 TabPFN；P4 会记录 `source-predictor-set-reduced`，并且仅当两条 arm 暴露相同预测器集合时接受比率。这一行为仍需在来源运行时 pilot 中审阅。

本库拒绝 TabEval 对恒定合成目标赋值 1 的有利回退。合成目标缺少类别时标记 `insufficient_support`，保留可见，并使严格 Global Utility 汇总不可用。

## Atomic Results 与 bundle 证据

P4 写入：

- 每个指标、arm、评测器和种子一条 Local 原始 Atomic Result；
- 每个评测器和种子一条 Local retention Atomic Result；
- 每个目标、arm 和种子一条 Global 原始 Atomic Result；
- 每个目标和种子一条 Global target-ratio Atomic Result；
- `artifacts/utility-details.json`：保存原始 arm 映射、目标支持情况、精确预测器集合、逐预测器分数和测试边界；
- `metrics.parquet`、`summary.json`、`metadata.json`、阶段记录、artifact inventory 与最终校验和。

原始 arm 的聚合权重为零。Local retention 对评测器和种子等权；Global target ratio 对目标和种子等权。Bundle finalization 会从 Atomic Results 独立重建两个公式，并拒绝被修改的汇总、缺失原始 arm、不等权重、不一致预测器集合、被改变的分母或缺失的 real-test 身份。

## 失败状态

- `insufficient_support`：合成目标缺少类别，或 real train 无法支持 real-test 标签；
- `mathematically_undefined`：Local 分母过弱、Global 分母为零，或辅助指标无数学定义；
- `implementation_failure`：声明的预测器不能满足结果契约；
- `resource_failure`：权威可选 Global 后端、权重、内存或时间预算不可用；
- `not_applicable`：保留给明确审阅过的适用性决定。

失败绝不会被删除。Bundle 可以以 `partial` 状态完成，但 partial 汇总不能作为榜单分数。

## 命令

使用以下命令安装已经冻结的 Local Utility、契约、表格处理和 bundle 依赖：

~~~bash
python -m pip install -e ".[utility]"
~~~

该可选依赖组有意不包含仍待来源运行验证的 AutoGluon/XGBoost/TabPFN Global 全栈。如果没有单独审阅过的 Global 环境，命令仍会计算 Local Utility，并为 Global 记录明确的 `resource_failure`；不会换用其他预测器。

~~~bash
std-tabular-diffusion evaluate-table \
  --protocol p4-utility \
  --reference real_train.csv \
  --real-test real_test.csv \
  --synthetic synthetic_train.csv \
  --dataset-profile configs/datasets/adult-uci-2-v1.json \
  --output artifacts/p4/adult/run-001
~~~

P4 默认使用种子 `0,1,2,3,4`。诊断运行可以传入 `--evaluator-seeds 23` 或其他逗号分隔列表。不同种子集合会被记录，不能自动视为榜单兼容。

## P4 剩余出口工作

P4 在提升为非诊断用途前，还需要：

1. 在 Linux/Python 3.11 上运行精确的可选 Global 预测器栈；
2. 留存来源运行时、依赖、模型集合、数值、稳定性和资源证据；
3. 审阅 Adult 与 Sick pilot 行为，包括高基数目标；
4. 冻结预测器版本、参数、种子策略和运行预算；
5. 单独作出协议冻结与 Official Results 准入决定。
