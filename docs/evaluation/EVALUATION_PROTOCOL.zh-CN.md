# 评测协议

英文规范：[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)

- 状态：设计基线
- 协议族：Standardized Tabular Diffusion Benchmark
- 文档版本：0.1.0
- 最后更新：2026-08-03

本文件是英文规范的对应中文译文。若两者存在歧义，以英文规范为准。

## 1. 目的

本规范定义基准评测什么，以及如何生成可比较的评测结果。其覆盖统计保真度、下游任务效用、数据有效性、经验性隐私风险和效率。

本文件是评测语义的规范性来源。[仓库质量标准](../QUALITY_STANDARD.md)仍是仓库质量、实现身份、法律审查、安全和发布门槛的上位依据。本规范存在并不表示其中每项要求均已实现。

## 2. 规范性用语

MUST、MUST NOT、SHOULD、SHOULD NOT 和 MAY 的含义与仓库质量标准中的定义一致。

标为“经 pilot 冻结”的要求，须在阈值或实现通过有记录的 pilot 决策确定后成为强制要求。在此之前，依赖未决事项的结果不得进入已冻结的官方协议。

## 3. 评测范围

### 3.1 支持的问题类型

初始协议面向静态单表合成数据。规范公共接口可以接受 CSV、Parquet 或内存中的 DataFrame，但每次官方运行必须将输入解析为相同的、带版本的规范表和 schema。

主要发布环境为 Linux 与 Python 3.11。依赖硬件的结果还必须标识兼容的硬件 profile。

### 3.2 评测对象

协议支持两类评测对象：

1. 通过已登记模型适配器完成训练和采样的模型运行；
2. 不执行模型、由用户提供的外部合成表。

外部合成表可以获得完整诊断报告。但如果其训练输入、模型身份、实现来源、最终配置、随机种子和生成过程无法被独立验证，就不得进入官方模型榜单。

### 3.3 数据边界

每个数据集必须提供冻结的 train、validation 和 test 划分。

- Train 用于拟合数据预处理器和生成模型。
- Validation 仅用于允许的提前停止、checkpoint 选择和超参数选择。
- Test 仅用于最终评测。

Validation 不合并回最终训练数据。除非带版本的协议明确规定其他样本规模 profile，官方合成样本数为训练行数的一倍。

凡是从数据中学习得到的预处理参数、编码器、填补器、缩放器、离散器和类别词表，都必须只在 train 上拟合。禁止依赖 test 推断 schema、调参、配置指标或修复数据。

### 3.4 评测表示

模型可以使用特定于方法的内部表示。评测必须在适配器解码后的原始语义列空间中进行。

原始解码合成输出必须保持不可变。如果评测器为了继续运行而需要修复或规范化视图，该视图必须单独保存，只能使用在 train 上拟合的规则生成，并标记 evaluation_repair_applied。修复不得抹去或改变原始输出的有效性结果。

## 4. 协议身份与可比性

官方结果必须标识：

- 协议版本；
- 数据集标识、版本、view、划分标识和校验和；
- 模型标识、上游版本、适配器版本和配置哈希；
- 比较 Track；
- 生成随机种子；
- 评测器和指标版本；
- 请求与实际合成行数；
- 环境与硬件 profile。

只有协议定义的所有兼容字段均匹配时，结果才可比较。划分、预处理契约、指标实现、评测器、样本数、调参规则、资源制度、聚合规则或失败策略发生变化时，必须产生新的兼容协议版本，或进入明确分离的结果组。

## 5. 执行生命周期

评测工作流由可独立审计的阶段组成：

1. prepare；
2. train；
3. sample；
4. validate；
5. evaluate；
6. aggregate；
7. report。

每个阶段必须记录输入、输出、状态、耗时和完整性哈希。只要早期阶段的内容寻址输入没有变化，后续阶段失败不得迫使已经成功的早期阶段重新运行。缓存复用和恢复行为由[结果规范](RESULT_SPECIFICATION.zh-CN.md)规定。

## 6. 统计保真度

统计保真度评估合成数据是否表达目标总体的分布结构。它不证明效用、有效性、隐私或因果正确性。

### 6.1 Shape

初始来源等价候选采用锁定版本 SDMetrics `ColumnShapes` 的行为。每个可评测列都会得到一个原子 Shape 分数。

- 数值列使用 KSComplement，即一减二样本 Kolmogorov-Smirnov 统计量。
- 类别列和布尔列使用 TVComplement，即一减总变差距离。
- 日期时间列转换为该实现的数值时间戳表示，再使用 KSComplement。

来源实现会在计算 KSComplement 和 TVComplement 前删除空值；KS 输入为空时返回未定义值。正式数据集经过声明的预处理后不含缺失值，但等价性测试仍必须覆盖这些来源边界行为。必须保留逐列 Shape 分数。来源定义的 property 分数是所有列等权平均。按类型平均、最差列列表和支持量属于本基准派生的报告视图，不得归因于 SDMetrics。

月份、星期和小时等日历成分只能作为带独立标识的基准诊断加入，不属于来源等价的 `ColumnShapes`。

### 6.2 Trend

初始来源等价候选采用锁定版本 SDMetrics `ColumnPairTrends` 的行为。在该实现中：

- 数值—数值列对默认使用 Pearson 相关相似度，即 `1 - abs(correlation_real - correlation_synthetic) / 2`；
- 类别—类别列对使用联合列联表相似度；
- 数值—类别列对使用列联表相似度；其 report 会分别用 NumPy 直方图边界对真实数值列和合成数值列独立离散化；
- 布尔列按类别列处理。

列联表相似度是在真实与合成联合单元并集上计算的 `1 - 0.5 × L1(P_real, P_synthetic)`。当两个关联阈值均明确固定为零时，来源 report 会评测全部合格列对。

SDMetrics 底层指标支持 Spearman，但它不是 `ColumnPairTrends` property 的默认行为。底层 `ContingencySimilarity` 也可以使用从真实数据拟合并共同应用的十个 bins，但这不同于 report 对混合列对采用的独立 bins。Spearman、共同 train-fitted bins 或其他科学上更合适的适配必须使用不同指标标识，不得声称与该 report 来源等价。

只有在确定性、带版本的基准适配中记录启用阈值和采样种子后，才可以引入超宽表列对采样策略。

必须保留每个列对的结果。报告必须包含整体与列对类型汇总，并识别表现最差的列对。

### 6.3 高阶结构

GReaT 的来源定义判别器实验使用经过调参的随机森林、真实与合成等量的评测样本、test accuracy，并以 0.5 为目标值。该原始行为使用类似 `great_rf_discriminator_accuracy` 的来源专用标识保存；它不是 AUROC 指标。

初始基准高阶候选为基于 AUROC 的分类器二样本检验，即 C2ST。它遵循同一可区分性原理，但属于本基准派生的定义，而不是 GReaT 公式。

评测器必须：

- 平衡真实类与合成类；
- 使用在 train 上拟合且带版本的特征变换；
- 训练固定且带版本的判别器；
- 在未用于拟合判别器的数据上评测；
- 报告原始 AUROC 和不确定性。

令调整后的 AUROC 为 AUROC 与一减 AUROC 中较大的值。保真度补分为：

~~~text
c2st_fidelity = 2 × (1 - adjusted_AUROC)
~~~

当可区分性等同随机猜测时取一，完全可区分时取零。

原始 AUROC、调整后 AUROC 和派生 complement 必须使用不同字段。只有在判别器和转换规则经协议冻结后，派生值才进入 `high_order_score`。不得把 GReaT accuracy 诊断与 AUROC 候选平均成仿佛相互独立的构念。

Alaa 等人定义的积分 Alpha-Precision 和 Beta-Recall 为：

~~~text
integrated_alpha_precision = 1 - 2 × integral_0^1 |P_alpha - alpha| d alpha
integrated_beta_recall     = 1 - 2 × integral_0^1 |R_beta - beta| d beta
~~~

锁定的作者代码在 `[0, 1]` 上使用 30 点网格，并在其给定嵌入空间中以欧氏距离近似两个积分。在混合表嵌入、数值积分规则、边界情况和跨数据集行为通过来源与协议验证前，这两个指标仍属于实验性诊断。DCR 和 Authenticity 在本基准中不属于保真度指标。

### 6.4 参考比较

正式保真度比较为合成数据对 held-out 真实 test。报告还必须包含：

- 合成数据对真实 train，作为过拟合诊断；
- 带版本的 real-versus-real 参考，用于估计有限样本波动。

在协议冻结前，必须固定 real-versus-real 的构造、平衡和重采样策略。它不得向生成模型暴露 test 信息。

### 6.5 保真度聚合

数据集级组成分数为：

~~~text
shape_score
trend_score
high_order_score
~~~

三者均成功计算时，正式数据集保真度分数为：

~~~text
fidelity_score =
    (shape_score + trend_score + high_order_score) / 3
~~~

原子结果、组成分数、最差情况、不确定性和参考比较必须保持可见。实验性指标不影响正式保真度分数。

## 7. 下游任务效用

下游任务效用评估合成数据是否能够支持可泛化到 held-out 真实数据的预测关系。

### 7.1 评测分支

每个适用任务都使用：

- TRTR：在真实 train 上训练预测器，在真实 test 上测试；
- TSTR：在合成 train 上训练相同预测器，在同一个真实 test 上测试；
- Dummy：训练与任务匹配的朴素预测器，在真实 test 上测试。

三个分支的特征定义、任务定义、评测器族、评测器超参数、转换规则和测试集必须一致。

### 7.2 Local Utility

每个数据集在其 profile 中声明一个主要目标。可选次要目标单独报告，不改变主榜权重。

主要任务指标为：

- 二分类和多分类使用 Macro-F1；
- 回归使用 RMSE。

在数学上适用时，Balanced Accuracy、ROC-AUC、PR-AUC、MAE 和 R-squared 为辅助指标。

评测器套件至少代表线性模型、随机森林和梯度提升树。具体实现和冻结超参数经 pilot 冻结。评测器选择只能使用 train 和允许的 validation 数据，并且必须独立于正在评分的合成数据方法。

已实现的 P4 诊断候选通过 `p4-utility-pilot@0.1.0` 绑定 scikit-learn Logistic Regression/Ridge、Random Forest 和 Histogram Gradient Boosting，并默认使用五个评测种子。其有限范围工程门已经通过，并留存 [Linux/Python 3.11 证据](../evidence/evaluation/p4-utility-run-31053624769.json)。这是已实现的 pilot 身份，不是已冻结的 Official Results profile。

这一以 Macro-F1/RMSE 为主的面板属于本基准契约。它不得称为 GReaT 的精确复现——GReaT 使用分类 accuracy、回归 MSE，以及线性或逻辑回归、决策树和随机森林；也不得称为 TabStruct 的精确复现——TabStruct 的分类效用使用 Balanced Accuracy。

### 7.3 Global Utility

Global Utility 遵循 TabStruct 公式 4，轮换每个可评测的非标识符列作为预测目标：

- 数值目标为回归任务；
- 类别和布尔目标为分类任务；
- 排除目标必须在数据集 profile 中给出明确原因。

对目标 `j`，令 `D` 为合成训练表，`D_ref` 为真实训练参照。两个预测器分支都在同一个 held-out 真实 test 上评测。来源定义的逐目标效用为：

~~~text
类别目标：utility_j(D) = balanced_accuracy_j(D) / balanced_accuracy_j(D_ref)
数值目标：utility_j(D) = RMSE_j(D_ref) / RMSE_j(D)
global_utility(D) = mean_j utility_j(D)
~~~

所有目标等权。大于一的值有效，不得截断。分母为零或非有限值时结果在数学上未定义，除非未来指标版本声明并验证其他规则。

预测器配置属于指标身份的一部分。TabStruct 的 Full-tuned profile 使用九个调参预测器，Tiny-default profile 使用三个未调参预测器，并被论文实证推荐为成本更低的 Global Utility 选项。锁定的 TabEval 快照实现了三预测器形式。这些 profile 必须使用不同标识，不能混入同一榜单兼容组。

已实现的 P4 诊断候选绑定 TabEval revision `dba19a4ee7aa391621cbeb464609285fd515dece` 中锁定的 `UtilityPerFeature` 来源配置 `XGB + KNN + TabPFN`。缺失依赖成为 `resource_failure`；TRTR 与 TSTR 预测器集合不一致会使该目标 ratio 无效。初始工程证据记录 `global_source_runtime.executed: false`；后续单独的[有限范围来源运行时证据](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json)已真实执行三个模型族，并严格通过分类/回归聚合等价。首次预注册的[数据集规模准入决定](../evidence/evaluation/p4-dataset-scale-admission-decision-run-31060416318.json)未通过：Adult 执行不完整，且 Sick 的两个完整稳定性哨兵超过固定门限。该 profile 仍为诊断状态，不得进入 Official Results。

Local Utility 和 Global Utility 是两个独立的正式子榜单。初始协议不将二者合并为单个 Utility 分数。

### 7.4 本基准派生的 Local Utility 保留率

必须保留原始 Dummy、TSTR 和 TRTR 数值。

下面的基线调整 Local Utility 转换由本基准定义，不是 GReaT 或 TabStruct 的效用公式。

对于越大越好的指标：

~~~text
retention = (TSTR - Dummy) / (TRTR - Dummy)
~~~

对于越小越好的误差：

~~~text
retention = (Dummy - TSTR) / (Dummy - TRTR)
~~~

原始 retention 可以小于零或大于一。如果 pilot 审查后允许把截断到 `[0, 1]` 的值用于榜单聚合，它必须使用不同的转换标识；原始值必须继续作为主要可审计结果。截断值不得标为 TabStruct utility。

如果 TRTR 相对 Dummy 的改善未超过经 pilot 冻结的数值容差，则 retention 在数学上未定义，不得伪造。对应原子任务仍须连同原始结果和状态保持可见。

### 7.5 目标支持失败

如果合成数据遗漏真实目标类别，结果状态为 insufficient_support，原因代码为 insufficient_target_support。能够计算的预测器继续针对真实 test 中的全部类别评测。如果合成目标恒定导致预测器无法训练，该任务在正式聚合中获得协议定义的最低效用贡献，并保持可见；不得将其删除或赋予有利默认值。

## 8. 数据有效性

数据有效性评估解码后的合成记录是否满足声明的表语义。

PAFT 提供了显式检查领域规则与函数依赖的动机，但本节的公式和聚合规则是本基准原生定义。它们必须引用并版本化为本仓库的契约，而不得归因于 PAFT。

### 8.1 结构门槛

评测器必须验证请求行数、必需列和额外列、列名唯一性、序列化可读性与安全类型转换。

可以按列名重排列，也可以规范化已经证明无损的转换。缺列、意外多列、输出不可读、行数错误或有损转换属于结构性失败。结构性失败阻止该次运行进入后续正式指标计算。

### 8.2 列有效性

带版本 schema 可以定义可空性、有限性、整数约束、领域边界、类别词表、字符串格式和日期时间范围。

对每列：

~~~text
valid_cell_rate =
    满足全部硬列规则的单元格数 / 单元格总数
~~~

column_validity_score 为所有列等权平均。必须保留逐列比率和违规原因。

### 8.3 跨列约束

只有得到权威数据字典、数据集文档或有记录人工审查支持的约束才是硬约束。例如时间顺序、互斥、条件领域规则和总量关系。

每项约束获得等权满足率，其平均值为 constraint_validity_score。如果没有适用的经审查跨列约束，该组成项为 not_applicable，而不是伪造的满分。

观察到的 train 最小值、最大值、稀有组合和推断依赖均属于软诊断，除非其被独立确立为领域规则。

### 8.4 有效性聚合

如果存在经审查的适用跨列约束：

~~~text
validity_score =
    0.5 × column_validity_score
    + 0.5 × constraint_validity_score
~~~

否则 validity_score 等于 column_validity_score。

fully_valid_row_rate 单独报告，不进入汇总，因为它的严苛程度会随表宽机械增长。

内容违规不会自动阻止其他诊断维度运行。任何仅供评测使用的修复必须显式标记，且不得改善或覆盖原始有效性分数。

## 9. 经验性隐私风险

隐私评测衡量声明威胁模型下的证据，不构成正式隐私保证。

### 9.1 威胁模型族

协议区分：

1. 直接披露与复制；
2. 成员推断；
3. 属性推断。

报告必须标识攻击者知识、可访问产物、成员与非成员采样、距离或特征表示、攻击模型和评测指标。

### 9.2 Exact Collision

基准报告：

- 与 train 行完全相同的合成行比例；
- 被匹配的不同 train 行数量；
- 任一被匹配 train 行的最大重复次数；
- 作为独立多样性诊断的合成数据内部重复率。

相等性规范化可以统一无损语义表示，但不得使用会制造碰撞的舍入或有损强制转换。

### 9.3 校准 DCR

锁定的混合表距离实现采用锁定版本 SDMetrics `calculate_dcr` 的行为。对每个候选—参照行对，数值与日期时间距离为 `min(abs(x - r) / reference_range, 1)`，类别与布尔距离在匹配时为零、不匹配时为一，行距离为所有列的等权平均。零范围数值列在相等时贡献零，否则贡献一。双方均为空值贡献零，仅一方为空值贡献一。DCR 是对全部参照行计算的精确最小行距离。

以真实 train 作为参照数据集，基准同时报告：

~~~text
DCR(synthetic, train)
DCR(test, train)
~~~

比较这两个分布遵循 GReaT 的原则：synthetic-to-train DCR 应非零，并与 held-out-real-to-train DCR 相近。1% 和 5% 参照阈值是本基准派生的低尾诊断，不是 SDMetrics 的 `DCRBaselineProtection` 分数，也不是 GReaT 论文给出的公式。

DCR 是分布型指标，不具有单调方向。更大的值不得自动标为更安全，因为极端距离可能意味着合成数据不可用。官方结果要求精确最近邻计算，除非另有经过单独验证的近似方法声明。

### 9.4 Authenticity

对每个合成样本 `g_j`，Alaa 等人将 `d_g,j` 定义为它到最近真实训练样本 `r_i*` 的距离，将 `d_r,i*` 定义为 `r_i*` 到另一个最近真实训练样本的距离。论文分类器在 `d_g,j > d_r,i*` 时将该合成样本判为 authentic；Authenticity 是这些样本指示量的平均值。

锁定的作者仓库没有逐字执行这一索引：它为每个真实行寻找最近合成样本，然后使用合成索引访问 real-to-real 距离向量。这可能与论文结果不同，并可能在样本数不同时失败。因此，`authenticity_alaa2022_paper` 与 `authenticity_alaa2022_repo` 是两个独立变体。在差异得到裁决、复现目标得到声明且等价性测试通过之前，两者均不得进入 Official Results。

任何混合表表示都属于本基准适配，必须只在 train 上拟合、带版本，并与论文嵌入空间方法使用不同标识。Authenticity 是复制与泛化诊断，不是隐私证明。

### 9.5 成员推断与属性推断

在正式隐私风险套件冻结前，至少需要一种通过来源验证的成员推断攻击。它报告攻击 AUROC、攻击 advantage、低假阳性率下的行为和不确定性。

只有数据集 profile 声明了适合的敏感属性和准标识符时才运行属性推断。它必须与不使用合成数据的攻击基线比较。

不支持的威胁模型为 not_applicable，样本支持不足为 insufficient_support。两种状态都不代表零风险。

### 9.6 隐私展示

不产生统一 Privacy Score，也不评选隐私冠军。结果与 Fidelity、Utility 一起以质量—风险权衡和 Pareto 视图展示。

差分隐私模型还必须报告 epsilon、delta、会计方法、裁剪、噪声和组合假设。经验攻击不能替代正式差分隐私会计。

锁定的 SynthCity `DeltaPresence` 实现在真实非敏感特征上用 `k` 属于 `{2, 5, 10, 15}` 拟合 KMeans，跳过不可行的 `k`，忽略合成数据中缺失的真实聚类，对共有聚类计算 `real_count / (synthetic_count + 1e-8)`，最后返回最大值。该原始值无上界，不能逐字解释为概率。在威胁模型、方向、空列表行为和科学适用性解决之前，它继续排除在正式评分之外。

## 10. 效率

效率只能在兼容硬件 profile 内衡量。

下面的计时、吞吐量、RAM 和 VRAM 定义是本基准原生的运行测量，不是六篇已审论文中的公式。

### 10.1 计时阶段

基准分别记录：

- 数据准备；
- 模型拟合；
- checkpoint 保存与加载；
- 适配器预处理与后处理；
- 冷启动生成；
- 热生成；
- 端到端执行。

依赖安装与数据集下载不计入计时执行。模型必需的数据转换计入端到端时间，并同时单独报告。GPU 计时必须在计时边界同步设备。

### 10.2 资源与吞吐量

每次官方运行记录 wall-clock 时间、请求与实际训练轮数、提前停止、峰值 RAM、适用时的峰值 VRAM、checkpoint 大小、请求与实际行数、每秒行数、超时、内存不足状态和数值失败状态。

热生成重复三次并取中位数汇总。主要工作负载生成训练行数的一倍。可选固定规模曲线可以单独发布。

### 10.3 硬件 profile

官方硬件 profile 固定操作系统、Python、CPU 型号、CPU 线程上限、RAM、GPU 型号与数量、CUDA 软件栈和重要运行设置。

不兼容硬件 profile 的结果不得混合。纯 CPU、官方单 GPU和社区硬件结果分开标识。基准不使用理论 FLOPS 换算设备间观测时间。

### 10.4 效率展示

Training Time、Sampling Throughput 和 Peak Resource Usage 为独立子榜单。不产生统一 Efficiency Score。

Native 比较报告最终官方训练成本。Standardized Tuning 比较还要报告全部搜索试验、失败、总计算时间和最终重训练成本。超时和内存不足结果保持可见。

## 11. 统计协议

初始协议使用一个冻结的官方数据划分和五个生成随机种子。评测器随机性固定并记录。后续 split robustness 研究属于独立协议，不得静默并入初始榜单。

聚合顺序为：

1. 保留原子指标值；
2. 按规定在一次运行内部聚合评测器重复或目标；
3. 在一个数据集内聚合五个生成随机种子；
4. 对数据集汇总做数据集等权宏平均。

数据集绝不按行数加权。置信区间采用带版本的分层或配对 bootstrap 程序。排名使用未舍入数值。在经 pilot 冻结的等价容差下，实践上和统计上不可区分的结果标记为并列。

## 12. 结果状态与分母完整性

每次指标调用必须且只能返回一个状态：

- computed；
- mathematically_undefined；
- insufficient_support；
- not_applicable；
- implementation_failure；
- resource_failure。

未计算结果没有数值，并且必须包含原因和适用计数。禁止裸 NaN、静默删除行和有利的回退分数。

失败、跳过、不适用和不支持的结果必须在聚合分母中保持可区分。完整官方榜单需要达到[榜单政策](LEADERBOARD_POLICY.zh-CN.md)规定的覆盖率。

## 13. 经 pilot 冻结的事项

首个正式协议冻结前，以下细节需要经验性 pilot 证据：

- 已通过来源运行时验证的 P4 预测器候选在协议冻结前，还需一次新预注册且完整通过数据集规模稳定性、高基数省略和资源预算门的重跑；
- real-versus-real 保真度重采样；
- 混合表 C2ST 变换与判别器；
- 来源忠实与适配版 Column Pair Trends 离散化的选择；
- Alpha-Precision 与 Beta-Recall 的准入；
- 论文与仓库 Authenticity 的差异；
- 成员推断攻击选择；
- 效用归一化和统计并列的数值容差；
- Validity 准入门槛；
- CPU、GPU、时间、内存和调参预算；
- 可选规模 profile。

每项决策必须记录测试过的替代方案、代表性数据集、代表性模型、证据和对协议版本的影响。

## 14. 相关规范

- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [指标来源审阅](METRIC_SOURCE_REVIEW.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [数据集 Profile 规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
- [仓库质量标准](../QUALITY_STANDARD.md)
