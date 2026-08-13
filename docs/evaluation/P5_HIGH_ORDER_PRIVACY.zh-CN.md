# P5 高阶保真度与经验隐私

## 状态与声明边界

`p5-high-order-privacy@1.0.0` 是已预注册的冻结候选。其科学身份已经通过单元测试、结果包测试、Windows 11/Python 3.11 Adult/Sick identity-surrogate 留存验证和一次探索性真实生成器试验。Adult/Sick 字段角色及有边界的黑盒成员推断威胁模型已经审阅。候选协议尚未冻结、尚未获得发布支持，也不能进入 Official Results；未解决指标仍是明确的 excluded 记录。

高阶保真度与经验隐私风险是两个独立维度。P5 不生成总 Fidelity 分数，也不生成总 Privacy 分数。经验攻击和距离诊断不能证明差分隐私，也不能证明其他形式化隐私保证。

## 输入表与拟合边界

P5 需要由校验和绑定的真实训练集、留出真实测试集和合成表，三者必须采用已审阅的规范模型视图。统一结构门会拒绝缺失值、非有限数值、模式漂移、有损类型转换和不符合要求的合成行数。系统不会修复合成输出。

- C2ST 的混合表转换只在真实训练集上拟合；真实测试集和合成数据都不会参与转换拟合。
- DCR 只把真实训练集作为最近记录参照。
- DOMIAS 将真实测试集拆成互不重叠的非成员与辅助真实参照；混合表转换、PCA 和参照密度只在辅助参照上拟合。

## 分类器二样本检验

主要比较对象是留出真实测试集与合成数据。五个评测种子分别采用等量类别、确定性的规范排序与抽样、分层 70/30 判别器划分，以及固定的 scikit-learn 随机森林：200 棵树、`min_samples_leaf=2`、单线程。数值列使用只在真实训练集上拟合的标准化；类别和布尔列使用只在真实训练集上拟合的 one-hot 编码，并忽略未见类别。

协议分别记录：

- `std-c2st-rf-auroc`：判别器原始 AUROC，目标值为 `0.5`；
- 调整后 AUROC：`max(AUROC, 1 - AUROC)`，保存在详细证据中；
- `std-c2st-fidelity`：`2 * (1 - adjusted_AUROC)`，随机不可区分时为一，完全可区分时为零。

原始 AUROC 与保真度补分始终是不同指标。系统保留五种子结果和确定性的 1,000 次分层 bootstrap 95% AUROC 区间。AUROC 及其补分是本基准对分类器二样本检验思想的扩展，不会被描述成原论文以 accuracy 为主的统计量。

GReaT 随机森林判别 accuracy 是明确排除于本 P5 请求之外的来源专用诊断，不会与 C2ST 混合平均。

## 精确复制与内部重复

每行会被转换为带类型、无损且不做舍入的语义 token。P5 分开报告：

- 与任一真实训练行完全一致的合成行比例与数量；
- 被匹配的不同真实训练行数量；
- 任一被匹配训练行在合成表中的最大重复次数；
- 合成表内部重复率与重复数量。

合成表内部重复不一定是训练数据复制；训练碰撞也不会被隐藏到多样性指标中。

## SDMetrics 最近记录距离

P5 直接调用 SDMetrics `0.28.3.dev0` 中的 `sdmetrics.single_table.privacy.dcr_utils.calculate_dcr`。安装的源码树必须与提交 `ba8842f2ba04ce914f698cc1cf746ca12338ab0e` 的校验值一致。系统保留以下每一个距离值：

- `DCR(synthetic, real_train)`；
- `DCR(heldout_real_test, real_train)`。

详细证据包含最小值、多个分位数、均值、总体标准差、零距离比例、留出数据的 1%/5% 阈值，以及合成距离落在这些阈值以内的比例。`std-dcr-heldout-wasserstein` 比较两个抽样后的完整分布。

DCR 是分布型指标，不能按“越大越安全”进行单调排序。远离训练集但完全无用的合成表不能被视为隐私成功。

## DOMIAS 成员推断

威胁模型是黑盒攻击：攻击者只获得公开的合成表和一份互不重叠的辅助真实参照，不获得生成器源码、参数、梯度或内部状态。

- 成员来自真实训练集的种子抽样；
- 非成员来自留出真实测试集；
- 辅助参照来自另一份与非成员不重叠的真实测试子集；
- 成员与非成员数量相等。

P5 保留 DOMIAS 密度比分数 `p_G(x) / (p_R(x) + 1e-10)`、官方合并分数中位数严格阈值的攻击 accuracy，以及攻击 AUROC；另外报告最大 attack advantage 与 FPR 不超过 1% 时的 TPR。AUROC 不确定性使用与 C2ST 相同的确定性分层 bootstrap。

DOMIAS 的密度比公式保持不变，但混合表表示是本仓库的适配：只在辅助参照上拟合标准化/one-hot、PCA 和高斯 KDE。PCA 不是可逆变换，因此该适配器不声称端到端源码精确等价。密度估计奇异或样本支持不足时会产生显式未计算状态，不会偷偷替换估计器。

## 明确排除的指标

以下指标在 Metric Registry 中标为 excluded，P5 协议无法选择它们：

- GReaT RF discriminator accuracy：来源专用诊断，不是 C2ST AUROC；
- integrated Alpha-Precision/Beta-Recall：混合表支持嵌入尚未解决；
- Alaa 论文与仓库两个 Authenticity 变体：论文/代码语义差异尚未裁决；
- SynthCity Delta Presence：威胁模型、方向和失败语义尚未解决；
- attribute inference：Adult 与 Sick 的字段角色已经审阅，但 P5 v1 尚无单独获批的属性推断实现与威胁模型。

## Bundle 与命令

`evaluate-table` 会写入 Atomic Results、原始 DCR 数组、C2ST 逐种子结果与区间、精确行证据、完整 DOMIAS 威胁模型与攻击结果、来源信息、阶段记录、制品清单和最终校验和。Bundle finalization 会拒绝被修改的汇总、缺失的指标/种子作用域、非零总分权重、不完整威胁模型、错误拟合边界，以及不一致的精确行或 DCR 证据。

~~~powershell
std-tabular-diffusion evaluate-table `
  --protocol p5-high-order-privacy `
  --reference real_train.csv `
  --real-test real_test.csv `
  --synthetic synthetic_train.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --output artifacts/p5/adult/run-001
~~~

P5 固定要求评测种子 `0,1,2,3,4`。已留存的 [Windows Adult/Sick identity-surrogate 证据](../evidence/evaluation/p5-windows-py311-identity-c66fa23.json)验证了完整执行和结果边界，但明确没有评估生成模型质量。已留存的 [TabDDPM/Adult 三生成种子证据](../evidence/evaluation/p5-tabddpm-adult-windows-py311-a2e4f27.json)关闭了首个探索性非 identity 试验。独立的生成种子 `3,4,5` 确认性试验及其通过/失败门槛已经写入[确认性预注册](../evidence/evaluation/p5-confirmatory-preregistration-2026-08-13.json)。这些记录都不能单独让结果进入 Official Results。

## 已完成的首个生成器试验

首个非 identity 试验已在有意限定的小范围内通过：在经过审阅的 Adult 数据集上运行一个经过校验和锁定的 TabDDPM 模型。试验使用官方 Adult `ddpm_cb_best` 配置训练一个检查点，再以生成种子 `0,1,2` 对同一个检查点进行采样，每张合成表严格生成 32,561 行。每张表仍分别使用未改动的 P5 评测种子 `0,1,2,3,4` 进行评测。

虽然 TabDDPM 的原生加载器要求提供验证数组，但生成器训练不会使用该数组。因此，本试验仅在加载接口提供一行真实训练数据的镜像，同时仍以完整的 32,561 行官方训练集作为唯一拟合输入。适配器将上游分类索引映射为经过审阅的标签，并在解码 CSV 时将声明为整数的字段转换到最近整数（中点取偶数）；原始上游数组、哈希以及转换行数均保存在被 Git 忽略的实验制品中。P5 评测器不会修复合成数据。

本试验属于探索性验证。其通过结果不会冻结 P5，不会让 TabDDPM 或 Adult 自动进入 Official Results，也不构成形式化隐私保证。属性推断仍排除在 P5 v1 之外；已审阅字段角色不能替代单独获批的属性推断实现与威胁模型。

本试验还覆盖了小型 TabDDPM 等价性样例未触发的一处旧依赖边界：上游将数学上为整数的 `1e9` 以浮点类型传给 `QuantileTransformer.subsample`，而支持 Python 3.11 的 scikit-learn 会在拟合前拒绝该类型。因此，适配器启动桥仅把整数值浮点数转换为完全相等的整数；它不修改上游源码，也不改变任何非整数的估计器参数。

## 已预注册的确认性试验

冻结候选试验使用训练种子 `0` 重新训练一个检查点，再用新的生成种子 `3,4,5` 对同一份校验和一致的检查点采样。每张 32,561 行合成表均使用未改变的五评测种子面板。冻结判断只依据预注册的完整性、结构、环境和来源门槛；指标数值的好坏不能导致接受、拒绝、替换种子或选择性重跑。行级数据仅保存在被 Git 忽略的本地制品中。成功的确认性试验只能支持另行作出的协议冻结决定，不能准入数据集、模型、赛道、具体运行或其他环境，也不能证明隐私或法规合规。
