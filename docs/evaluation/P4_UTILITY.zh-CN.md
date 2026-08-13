# P4 Local 与 Global Utility

## 状态与声明边界

P4 只有一个生成结果的实现：已冻结的 `p4-utility@1.0.0`，绑定未改变的 `p4-utility-stable@0.2.0` 评测器 profile。这组指标只有在数据集、模型、comparison track、具体运行和精确执行环境分别通过独立准入时，才可以作为 Official Results 组件使用。P4 尚未获得仓库发布支持；本次冻结没有准入 Adult、Sick 或任何生成模型结果。

TabEval 在提交 `dba19a4ee7aa391621cbeb464609285fd515dece` 的原始源码仍通过校验和锁定，用于来源追踪和内部源码等价测试。它不是第二个 CLI 选项、指标 profile 或榜单实现。面向用户的评测只使用下文的稳定适配器。

## 必需输入与数据泄露边界

P4 需要三张绑定校验和的规范模型视图表：

- 真实训练集：只用于拟合特征变换、Dummy 模型和 TRTR 预测器；
- 合成训练集：只用于拟合 TSTR 预测器；
- 留出的真实测试集：只用于打分，绝不作为任何拟合操作的输入。

结构门会拒绝模式漂移、模型输入缺失、非有限数值、丢失信息的逻辑类型转换和不符合要求的合成行数。系统不会修复生成值。Bundle finalization 会分别核验三张表，并记录 `real_test_used_for_fit: false`。

## Local Utility

每个数据集声明一个经过审阅的主要预测任务。Adult 预测 `income`，Sick 预测 `Class`。分类的主要指标为 Macro-F1，并在有定义时计算 Balanced Accuracy、ROC-AUC 和 PR-AUC；回归的主要指标为 RMSE，并在有定义时计算 MAE 和 R-squared。

冻结的 Local 预测器面板包含三个家族：

- Logistic Regression 或 Ridge；
- Random Forest；
- Histogram Gradient Boosting。

具体的 scikit-learn 官方类和参数保存在 `p4-utility-stable-v1.json`。独热编码和数值缩放仅在真实训练集特征上拟合，随后冻结并应用于真实训练集、合成训练集和真实测试集。

对每个评测种子，P4 都保留 Dummy、TRTR 和 TSTR 原始结果。仓库定义的 retention 为：

~~~text
越大越好：(TSTR - Dummy) / (TRTR - Dummy)
越小越好：(Dummy - TSTR) / (Dummy - TRTR)
~~~

Retention 不做裁剪。当 TRTR 相对 Dummy 的改善未超过 `1e-12` 容差时，结果为 `mathematically_undefined`。

## Global Utility

Global Utility 依次将规范模型视图中的每个纳入列作为目标。每一列都必须被纳入，或给出稳定的排除理由。计算公式遵循 TabStruct 公式 4：

~~~text
类别目标：balanced_accuracy(TSTR) / balanced_accuracy(TRTR)
数值目标：RMSE(TRTR) / RMSE(TSTR)
全局效用：先对目标等权平均，再对种子等权平均
~~~

比值不裁剪。类别支持不足、分母为零、预测器失败和资源失败都必须显式保留；不可用目标绝不能被静默移出分母。

预测器面板沿用锁定的 TabEval `UtilityPerFeature` 配置：通过 AutoGluon 运行 XGB、KNN 和 TabPFN，并关闭加权集成。只有类别数超过锁定上限 10 的分类目标可以省略 TabPFN；XGB 和 KNN 仍为必需，且 TRTR/TSTR 必须拥有相同的实际训练模型集合。

### 稳定拟合协议

TabEval 向 AutoGluon 传入 `tuning_data=None`。AutoGluon 随后根据输入行的位置生成隐藏验证集。历史 identity-surrogate 运行证明：仅仅打乱同一批数据的行顺序，就会导致相同的行多重集合获得不同的拟合边界。

唯一的公开适配器在不修改任何上游模型源码的前提下修复该协议问题：

1. 将三张输入表转换为经过审阅的规范模型视图；
2. 使用 `lexicographic-all-model-columns-v1`，按所有模型视图列（包括目标列）对每个训练 arm 做字典序排序；
3. 计算 AutoGluon 声明的默认 holdout 比例；
4. 调用 `autogluon.core.utils.utils.generate_train_test_split`，使用 Evaluation Request 中的评测种子，并在分类任务中执行适用的分层划分；
5. 将得到的拟合集和调参集显式传给 AutoGluon；
6. 将种子传给 AutoGluon learner，同时保证留出的真实测试集不属于上述任何一张表。

每个 arm 都会在 `utility-details.json` 中记录排序身份、划分实现、种子、任务与 problem type、holdout 比例、行数、规范输入/拟合集/调参集的内容指纹，以及 `real_test_used_for_fit: false`。证据不会嵌入任何原始行内容。

这一变化保留了官方 XGB/KNN/TabPFN 实现、TabStruct 公式、目标覆盖范围和五种子设计；它只修正此前会受到任意输入行顺序影响的仓库调用边界。

## Atomic Results 与 bundle 证据

P4 会写入逐指标/arm/评测器/种子的 Local 原始结果、逐评测器/种子的 Local retention、逐目标/arm/种子的 Global 原始结果，以及逐目标/种子的 Global 比值。最终 bundle 包含 `metrics.parquet`、`summary.json`、`metadata.json`、`artifacts/utility-details.json`、阶段记录、制品清单和最终校验和。

Bundle 校验会从原始 arm 重新构造派生值，并拒绝 arm 缺失、摘要被修改、权重不等、预测器集合不一致、分母改变或真实测试集来源缺失。部分 bundle 可以留作诊断，但不能成为榜单分数。

## 源码等价与正式结果计算的区别

内部 `_source_exact_global_scorer` 只在锁定源码验证程序中重建 TabEval 原始的隐式划分调用，用于证明我们正确理解了锁定源码、预测器 wrapper、依赖和检查点。它没有注册，也不能通过 `evaluate-table` 使用。

稳定适配器不声称逐行源码等价或数值等价，因为显式划分是一个有意且有记录的调用变化。其科学来源由不变的公式、预测器、打分指标和对官方包的直接调用保证。

## 历史证据与后继身份

历史结果保持不可变：

- Linux 有限范围源码运行时证据保留在 [p4-global-source-runtime-run-31057073762.json](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json)；
- Windows GPU 精确源码运行时证据保留在 [p4-global-source-windows-gpu-c0e6e72.json](../evidence/evaluation/p4-global-source-windows-gpu-c0e6e72.json)；
- 完整的历史 Windows 数据集规模结果保留在 [p4-dataset-scale-windows-gpu-a754ca1.json](../evidence/evaluation/p4-dataset-scale-windows-gpu-a754ca1.json)。

最后一项证据完成了全部 67 个任务和 134 个 arm，并通过执行与资源门，但 Adult `native-country`、Sick `referral-source` 和 Sick `tsh` 未通过原定的 `0.05` 稳定性门。该结果仍然是失败结果，不能用新实现重新解释。

后继验证使用新身份 `p4-dataset-scale-windows-gpu-stable-candidate@0.3.0`。它保留相同的数据集、目标、五个种子、整行置换 identity surrogate、预测器策略、`0.05` 门限和资源限制，只绑定新的评测器与数据集 profile 版本。完整验证只允许在声明的原生 Windows 11、Python 3.11 和 RTX 5080 环境运行。GitHub 托管 runner 只校验契约，不能代替所需的 GPU 实验。

提交 `6dc485f` 上的完整后继运行通过了全部 9 个 shard、67 个任务和 134 个 arm。Adult 的 `income`、`native-country`、`fnlwgt` 与 Sick 的 `class`、`referral-source`、`tsh` 的每一个五种子 identity ratio 都精确为 `1.0`，因此所有范围和最大绝对偏差均为 `0.0`。单个 arm 的最长时间为 `9.50134` 秒，进程树 RSS 峰值为 `2.03462` GiB，CUDA 分配增量峰值为 `3.64762` GiB，全部位于预注册限制内。留存的[机器可读证据](../evidence/evaluation/p4-dataset-scale-windows-gpu-stable-6dc485f.json) SHA-256 为 `19d55b260eaa5a3d1e522d1a1cabeadf4e952698a833e529576fd64b71797721`。

## 命令

安装标准 Utility 依赖：

~~~powershell
python -m pip install -e ".[utility]"
~~~

可选的 Global 运行时还必须满足 `requirements-p4-windows-gpu-validation.txt`。如果缺少该环境，Local Utility 仍可计算，Global 失败会被明确记录，系统不会替换成其他预测器。

~~~powershell
std-tabular-diffusion evaluate-table `
  --protocol p4-utility `
  --reference real_train.csv `
  --real-test real_test.csv `
  --synthetic synthetic_train.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --output artifacts/p4/adult/run-001
~~~

P4 默认使用评测种子 `0,1,2,3,4`。任何覆盖值都会写入证据，且不会自动与榜单结果兼容。

## 冻结决定与剩余准入工作

单独的[冻结决定](../evidence/evaluation/p4-protocol-freeze-decision-2026-08-12.json)审阅了不可变证据，并在不改变科学身份的前提下完成冻结。P4 的精确资格仅限原生 Windows 11、Python 3.11 和 NVIDIA GeForce RTX 5080。其他硬件和操作系统在分别验证前仍属于兼容性诊断；不能把 RTX 5080 证据外推到其他 GPU。

发布具体 Official Result 前仍必须：

1. 准入数据集及其精确版本、视图和划分；
2. 准入模型适配器、comparison track 和源码来源；
3. 在合格环境上验证该具体结果 bundle；
4. 获得仓库发布支持，并在后续 P7 通过榜单快照准入。

identity surrogate 验证的是评测器稳定性，不是生成器质量。历史失败候选仍保持失败证据。
