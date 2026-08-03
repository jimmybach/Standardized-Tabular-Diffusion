# 指标来源审阅

英文原文：[Metric Source Review](METRIC_SOURCE_REVIEW.md)

## 1. 状态与目的

审阅日期：2026-08-03

审阅状态：定义已审阅；实现等价性尚未认证

本记录把已审论文中的评测定义与锁定上游源码树的可执行行为进行核对，防止将论文公式、上游实现和本基准派生转换表述成同一个指标。

本次审阅确立后续实现目标。任何指标在影响 Official Results 前，仍必须具备实现、golden fixtures、直接来源等价性测试、协议冻结和发布批准。

## 2. 已审证据

### 2.1 论文

| 标识 | 相关证据 |
|---|---|
| [P01 GReaT](../../research_inputs/evaluation/papers/P01_GReaT_ICLR2023.pdf) | 第 4.1-4.2 节和第 7-9 页的 TSTR 效用、随机森林判别器与 DCR |
| [P02 PAFT](../../research_inputs/evaluation/papers/P02_PAFT_Why_LLMs_Are_Bad_2025.pdf) | SDMetrics Shape/Trend 用法与领域规则违反 |
| [P03 GraDe](../../research_inputs/evaluation/papers/P03_GraDe_2025.pdf) | TSTR、相关误差和方向解释矛盾的 DCR |
| [P04 TabStruct](../../research_inputs/evaluation/papers/P04_TabStruct_ICLR2026.pdf) | 效用公式 4、Global Utility、预测器 profile、实验设置和指标比较 |
| [P05 Alaa 等](../../research_inputs/evaluation/papers/P05_Alaa_Faithful_Synthetic_Data_ICML2022.pdf) | Alpha-Precision、Beta-Recall、积分分数与 Authenticity 分类器 |
| [P06 SynthCity](../../research_inputs/evaluation/papers/P06_Synthcity_2023.pdf) | SynthCity 评测器的框架背景；可执行细节以锁定源码为准 |

### 2.2 锁定的可执行来源

不可变版本记录在 [source_lock.json](../../research_inputs/evaluation/provenance/source_lock.json) 中。相关快照为：

| 来源 | Revision | 审阅目标 |
|---|---|---|
| SDMetrics | `ba8842f2ba04ce914f698cc1cf746ca12338ab0e` | Column Shapes、Column Pair Trends 与混合表 DCR |
| TabStruct | `501f48942f890c92a796edf236e670cdc270ad9f` | 论文 pipeline 与评测配置 |
| TabEval | `dba19a4ee7aa391621cbeb464609285fd515dece` | `UtilityPerFeature` 可执行行为 |
| evaluating-generative-models | `093910e487d07959db7d87b54698da60aaeb50c0` | 作者的 Alpha-Precision、Beta-Recall 与 Authenticity 代码 |
| SynthCity | `23f322fe381326ed01c41b13d469a06e38cce545` | `DeltaPresence` 可执行行为 |
| GReaT / be_great | `7d30fcd4b811b49173e54c04e3d68a6c5cc2d483` | GReaT 相关评测工具 |

资料包完整性与许可证检查记录在[资料包 QA](../../research_inputs/evaluation/provenance/PACKAGE_QA.md)中。

## 3. 来源发现与基准决定

### 3.1 Column Shapes

锁定版本 SDMetrics `ColumnShapes` property 使用：

~~~text
数值或日期时间：KSComplement = 1 - 二样本 KS 统计量
类别或布尔：TVComplement = 1 - 0.5 × sum_k |p_real(k) - p_synthetic(k)|
property 分数：所有已计算列等权平均
~~~

实现会在原子指标前删除空值；KS 输入为空时返回未定义值。日期时间值转换为实现的数值表示。按类型汇总和最差列列表属于本基准报告视图，而非来源定义指标。

决定：把锁定版本的精确行为作为初始 Shape 来源等价目标。

### 3.2 Column Pair Trends

锁定的 `ColumnPairTrends` property 默认对连续—连续列对使用 Pearson 相关：

~~~text
correlation_similarity = 1 - |correlation_real - correlation_synthetic| / 2
~~~

类别—类别和混合列对使用列联相似度：

~~~text
contingency_similarity = 1 - 0.5 × L1(P_real_joint, P_synthetic_joint)
~~~

report 会分别为真实与合成连续列生成直方图边界来预处理混合列对。这与直接调用底层 `ContingencySimilarity` 不同；后者可以从真实数据生成十个 bins，再把同一边界应用于两套数据。底层相关指标支持 Spearman，但它不是 report 默认值。

决定：report 行为是初始来源等价目标。Spearman 和共同 train-fitted bins 属于不同实验变体，即使后续审查认为它们在科学上更合适，也不能混名。

### 3.3 判别器与 C2ST

GReaT 训练经过调参的随机森林判别器，在真实与生成样本等量的测试集上评测 accuracy，并把 50% 视为理想不可区分状态。它没有定义 AUROC complement。

本基准的 C2ST 候选记录 AUROC，并定义：

~~~text
adjusted_AUROC = max(AUROC, 1 - AUROC)
c2st_fidelity = 2 × (1 - adjusted_AUROC)
~~~

决定：把 GReaT 判别器 accuracy 作为来源专用诊断保留；把 AUROC complement 作为本基准派生指标，并使用不同标识冻结。不得把二者平均成同一个高阶组成项。

### 3.4 积分 Alpha-Precision 与 Beta-Recall

Alaa 等人定义：

~~~text
integrated_alpha_precision = 1 - 2 × integral_0^1 |P_alpha - alpha| d alpha
integrated_beta_recall     = 1 - 2 × integral_0^1 |R_beta - beta| d beta
~~~

作者代码在 `[0, 1]` 上使用 30 点网格，并在欧氏嵌入空间中用离散求和计算。support 构造和嵌入都是方法的重要组成部分。

决定：在混合表嵌入及其拟合边界复现作者方法，且离散积分通过等价性 fixture 前，两个指标均保持实验性。

### 3.5 TSTR 与 Local Utility

GReaT 在合成数据上训练预测器，并在 held-out 真实数据上测试。其主要面板使用线性或逻辑回归、决策树和随机森林；分类报告 accuracy，回归报告 MSE，并对五个种子平均。

本基准的 Local Utility 面板改为分类以 Macro-F1 为主、回归以 RMSE 为主，并加入其他模型族和辅助端点。其基线调整转换为：

~~~text
越大越好：(TSTR - Dummy) / (TRTR - Dummy)
越小越好：(Dummy - TSTR) / (Dummy - TRTR)
~~~

决定：当评测器 profile 与来源一致时，原始 TSTR/TRTR 结果可与来源比较。基线调整 retention 是本基准派生指标，必须保留原始项，且不得称为 GReaT 或 TabStruct utility。

### 3.6 TabStruct Local 与 Global Utility

TabStruct 公式 4 对类别目标使用 Balanced Accuracy，对数值目标使用 RMSE。令 `D` 为合成训练表，`D_ref` 为真实参照训练表：

~~~text
类别目标：Utility_j(D) = BalancedAccuracy_j(D) / BalancedAccuracy_j(D_ref)
数值目标：Utility_j(D) = RMSE_j(D_ref) / RMSE_j(D)
GlobalUtility(D) = mean_j Utility_j(D)
~~~

所有目标等权。大于一表示与真实参照分支相当或更好，不得截断。

论文的 Full-tuned profile 集成九个调参预测器：Logistic Regression、KNN、MLP、Random Forest、Extra Trees、LightGBM、CatBoost、XGBoost 和 TabPFN。Tiny-default profile 使用三个未调参预测器，论文支持将其作为成本更低的 Global Utility profile。锁定的 TabEval `UtilityPerFeature` 快照实现了三预测器配置，并在合成目标恒定时赋予有利的 `[1]` 分类值。

决定：不同预测器 profile 具有不同指标身份。本基准不接受恒定目标的有利回退；该情况成为显式支持失败，因此在这一边界情况上不能声称与代码完全等价。TabStruct 公式仍是 Global Utility 的目标。

### 3.7 DCR

GReaT 使用 L1 风格的混合距离定义最近训练记录距离，并认为良好结果应具有非零 synthetic DCR，且其分布与 held-out-real-to-train DCR 相近。GraDe 的公式和文字表示普通最近距离，但其方向和解释与该公式矛盾；本基准拒绝该矛盾方向。

锁定的 SDMetrics 混合距离对每一列计算：

~~~text
数值或日期时间：min(|x - r| / reference_range, 1)
类别或布尔：相等为 0，否则为 1
行距离：所有列等权平均
DCR(x)：对所有参照行计算的精确最小行距离
~~~

零范围数值列相等时贡献零，不相等时贡献一。双方均为空值贡献零，恰有一方为空值贡献一。

决定：使用精确 SDMetrics 距离作为实现身份，以真实 train 为参照。比较完整 synthetic-to-train 与 held-out-real-to-train 分布。低尾参照阈值属于本基准派生诊断。不得使用或改名 SDMetrics `DCRBaselineProtection`，也不得按原始 DCR 单调排名。

### 3.8 Authenticity

对每个合成样本 `g_j`，Alaa 论文找到其最近真实样本 `r_i*`，比较 `d_g,j = d(g_j, r_i*)` 与 `r_i*` 到另一个最近真实样本的距离 `d_r,i*`，并在 `d_g,j > d_r,i*` 时判定 `g_j` authentic。

锁定的作者代码反向执行最近邻查询：它为每个真实行寻找最近合成样本，再使用得到的合成索引访问 real-to-real 距离向量。这不是同一个样本级计算，而且可能在真实与合成样本数不同时失败。

决定：分别注册 `authenticity_alaa2022_paper` 与 `authenticity_alaa2022_repo`。在差异得到裁决前，两者均不准入 Official Results。任何混合表编码器都是另一项本基准适配，必须单独标识。

### 3.9 Delta Presence

锁定的 SynthCity 实现会移除已声明敏感特征，在真实数据上对 `k` 属于 `{2, 5, 10, 15}` 拟合 KMeans，预测合成聚类，跳过合成数据中缺失的真实聚类，对共有聚类计算 `real_count / (synthetic_count + 1e-8)`，并返回最大 ratio。该返回值无上界，因此尽管来源 docstring 如此描述，也不能逐字当作概率。

决定：只把精确原始行为作为带名称的诊断保留。在威胁模型、方向、缺失聚类处理和空结果行为得到科学决定前，将其排除在正式评分外。

### 3.10 Validity 与 Efficiency

PAFT 支持报告规则违反的科学必要性，但没有定义本基准的列有效性、跨列约束或聚合公式。同样，已审论文没有定义本基准的计时阶段、吞吐量、RAM 或 VRAM 契约。

决定：把 Validity 与 Efficiency 指标注册为本基准原生指标，直接验证其公式，不声称论文等价性。

## 4. 审阅后的准入矩阵

| 指标或指标族 | 当前定义决定 | 最早允许的发布角色 |
|---|---|---|
| SDMetrics Column Shapes | 锁定版本精确行为 | 通过等价性与发布门槛后的正式候选 |
| SDMetrics Column Pair Trends | 锁定 report 精确行为 | 通过等价性与发布门槛后的正式候选 |
| C2ST AUROC complement | 本基准派生 | pilot 冻结与验证前为诊断 |
| 积分 Alpha-Precision/Beta-Recall | 来源公式；混合表适配未解决 | 实验性诊断 |
| 基线调整 Local Utility retention | 本基准派生 | pilot 冻结与验证前为诊断 |
| TabStruct Global Utility | 论文公式 4；预测器 profile 仍属于指标身份 | profile 冻结并通过等价性验证后的正式候选 |
| SDMetrics DCR 加 held-out 校准 | 来源距离加本基准派生校准 | 隐私风险诊断候选 |
| Alaa Authenticity | 论文与仓库冲突 | 排除在 Official Results 外 |
| SynthCity Delta Presence | 精确代码行为的语义未解决 | 排除在正式评分外 |
| Validity 与 Efficiency | 本基准原生 | 直接验证并通过发布门槛后的正式候选 |

## 5. 必需的等价性 fixtures

一个来源指标在达到 `source-parity-validated` 前，测试必须至少覆盖：

- 可手算的普通情况和精确上游输出；
- 适用时的恒定、空、单行、缺失值、未知类别和零范围情况；
- 来源支持的数值、类别、布尔、日期时间和混合列对；
- report 级与原子指标级的聚合和预处理；
- 真实与合成样本数不同的情况；
- 本基准归一化或截断前的原始输出；
- 确定性种子、容差、警告和结果状态；
- 锁定的来源 revision 与解析参数集。

对随机预测器和判别器，等价性必须使用预声明统计容差或确定性小 fixture，不能在看到结果后选择容差。

## 6. 阻止发布的来源问题

以下事项有意保持未决，并阻止受影响指标进入 Official Results：

- 正式 Trend 保持来源忠实的混合列对独立 bins，还是采用单独命名的共同 train-fitted-bin 变体；
- C2ST 的具体判别器、预处理、划分和不确定性过程；
- Alpha-Precision 与 Beta-Recall 的混合表嵌入；
- Local Utility 预测器 profile 和任何截断 retention 贡献；
- Global Utility 预测器 profile；
- Authenticity 的论文与仓库差异；
- 成员推断实现；
- SynthCity Delta Presence 的科学适用性。

## 7. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [数据集 Profile 规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
