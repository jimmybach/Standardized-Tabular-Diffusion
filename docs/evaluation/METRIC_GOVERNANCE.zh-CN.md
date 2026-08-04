# 指标治理

英文规范：[METRIC_GOVERNANCE.md](METRIC_GOVERNANCE.md)

- 状态：设计基线
- 治理版本：0.1.0
- 最后更新：2026-08-03

本文件是英文规范的对应中文译文。若两者存在歧义，以英文规范为准。

## 1. 目的

本规范治理评测指标如何被识别、溯源、实现、验证、版本化，以及如何获得进入官方基准评分的资格。

其核心规则是：论文定义、上游软件行为和本基准的报告契约是三个独立层次。三者之间的一致性必须得到证明，不能被假定。

## 2. 范围

本政策适用于所有可能影响下列事项的计算：

- 官方或诊断分数；
- 指标适用性或结果状态；
- 归一化或方向；
- 聚合或榜单排序；
- 不确定性或显著性；
- 隐私风险表述；
- 性能测量。

只有在显示格式不可能改变所存储数值含义时，简单显示格式才不属于本规范范围。

## 3. 三层来源模型

### 3.1 方法层

方法层记录科学定义：

- 论文或技术报告；
- 公式、算法或文字定义；
- 声明的方向和解释；
- 假设与威胁模型；
- 已知局限；
- 官方补充材料。

当来源含糊时，仅有引用还不够。实际可行时，Registry 必须标识相关章节、公式、算法或来源产物。

### 3.2 实现层

实现层记录可执行行为：

- 官方包或仓库；
- 精确包版本或不可变 commit；
- 函数或源码路径；
- 默认参数和显式覆盖参数；
- 随机行为；
- 边界情况行为；
- 依赖软件栈；
- 许可证与再分发要求。

论文未定义的软件行为不得归因于论文。

### 3.3 基准契约层

基准契约定义：

- 规范输入表示；
- 适用性；
- 原始输出记录；
- 状态映射；
- 归一化；
- 聚合；
- 不确定性；
- 报告；
- 准入状态；
- 兼容版本。

基准转换必须保留上游原始结果。如果转换改变了科学含义，就不得将其表述为原始指标。

## 4. Metric Registry

每个指标在实现工作被视为完成前，都必须拥有机器可读的 Registry 条目。

### 4.1 必需身份字段

- metric_id；
- metric_version；
- display_name；
- dimension；
- subdimension；
- description；
- definition origin；
- planned leaderboard role；
- owner；
- lifecycle status；
- review record。

`definition_origin` 取 `source-defined`、`source-parameterized`、`benchmark-derived` 或 `benchmark-native` 之一。source-parameterized 指标保留来源公式，同时冻结来源允许的选择；benchmark-derived 指标对来源输出做转换或扩展；benchmark-native 指标不声称存在上游公式。

指标标识必须区分不等价变体。例如，使用不同 Authenticity 索引规则、Column Pair Trends 离散化、Utility 归一化或预测器 profile 的实现需要不同标识。

### 4.2 必需来源字段

- 方法引用；
- 上游仓库或包；
- 精确版本；
- 实现符号或源码路径；
- 来源权威性；
- 实现模式；
- 获取日期；
- 适用时的完整性哈希；
- 许可证；
- 必需署名；
- 已知偏差。

### 4.3 必需语义字段

- 支持的表和列类型；
- 必需的数据集 profile 元数据；
- 输入 view；
- 拟合所用划分；
- 输出形状和单位；
- 原始范围；
- 原始方向；
- 适用时的目标值；
- 未定义条件；
- 最小支持量；
- 随机性；
- 默认参数和覆盖参数；
- 归一化函数与版本；
- 聚合规则；
- 不确定性程序；
- 失败贡献策略。

### 4.4 必需验证字段

- 单元测试证据；
- 边界测试证据；
- 来源等价性证据；
- 数值容差；
- 参考 fixture；
- 支持环境；
- 未解决局限；
- 发布决策。

### 4.5 方向词表

raw_direction 使用下列之一：

- maximize；
- minimize；
- target；
- distributional；
- descriptive。

Target 指标必须声明目标值和距离规则。诸如校准 DCR 的 distributional 指标必须声明参考分布和比较输出。未经单独审查的转换，不得将其强制改为 maximize 或 minimize。

### 4.6 示例条目

~~~yaml
metric_id: calibrated_mixed_dcr
metric_version: 0.1.0
dimension: empirical_privacy_risk
subdimension: direct_disclosure
definition_origin: benchmark-derived
planned_leaderboard_role: diagnostic
lifecycle_status: registered

method:
  references:
    - identifier: pending-citation-record

implementation:
  authority: official-upstream
  mode: package-wrapper
  package: pending
  package_version: pending
  source_symbol: pending
  license: pending-review

semantics:
  input_view: decoded_semantic_table
  fitted_on: train
  raw_direction: distributional
  outputs:
    - synthetic_to_train_dcr_distribution
    - test_to_train_dcr_distribution
  requires_reference_calibration: true
  lower_tail_quantiles: [0.01, 0.05]

validation:
  status: pending
  source_parity: pending
~~~

示例条目不构成实现或准入声明。

## 5. 实现来源层级

优先实现顺序为：

1. 包装锁定的官方包；
2. 包装锁定的官方源码版本；
3. 对锁定官方源码应用经审查补丁；
4. 本地兼容性重实现。

### 5.1 包包装器

包包装器必须在运行时验证已安装版本。版本不匹配必须明确失败，不得静默导入任意可用包版本。

只要公共上游 API 保持所需语义，包装器就应调用公共 API。如果必须使用私有符号，则必须记录依赖和兼容性风险。

### 5.2 源码包装器

当不存在合适的包，或所需 commit 无法通过包分发获得时，可以固定官方源码。只要最小化、许可合规并通过完整性检查的依赖机制足够，项目就应避免复制整个仓库。

### 5.3 经批准的源码补丁

补丁是最后手段。它必须：

- 标识精确上游版本；
- 解释为什么包装不足；
- 保持最小且隔离；
- 区分兼容性影响与语义影响；
- 保留许可证与通知；
- 包含修改前后测试；
- 对每个等价行为声明通过来源等价性审查。

语义补丁会产生独立指标身份，除非且直到它成为权威上游发布的一部分。

### 5.4 本地重实现

本地重实现默认是实验性的。在普通输入上匹配公式并不足够。正式准入要求：

- 独立定义审查；
- 全面的边界测试；
- 与权威可执行行为直接比较；
- 在代表性 fixture 和数据集上于声明容差内一致；
- 等价的结果状态行为；
- 显式批准。

如果权威代码无法运行，该实现必须继续明确标识为本地方法，并且不得声明来源等价。

## 6. 指标生命周期

生命周期状态是累积的：

1. registered：记录身份、目的、来源、许可证状态、预期角色和已知未决问题。

2. definition-reviewed：已审查论文、上游行为、方向、假设、差异和拟议基准契约。

3. implementation-complete：包装器或实现以及结构化结果契约已经存在。

4. unit-validated：确定性单元测试、边界测试、负向测试和状态测试通过。

5. source-parity-validated：直接权威调用与基准路径调用在批准的等价性协议下保持一致。

6. protocol-frozen：公式、来源版本、输入表示、方向、归一化、聚合、不确定性、适用性和失败策略已对某一协议版本固定。

7. release-supported：指标具有负责人、支持环境、依赖锁、文档、测试、兼容策略和发布批准。

只有 protocol-frozen 且 release-supported 的指标可以影响 Official Results。更早生命周期阶段只能出现在明确标识的诊断输出中。

## 7. 原始值与派生值

每项转换都保留上游或方法层原始输出。

一个指标结果可以包含：

- raw_value；
- normalized_value；
- aggregate_contribution；
- reference_value 或 distribution；
- direction；
- transformation identifier 和 version；
- clipping indicator。

原始值不得被归一化值覆盖。截断、取反、缩放、校准、基线调整或阈值化必须显式记录并独立测试。

示例包括：

- 作为来源定义诊断的原始 GReaT 风格判别器 accuracy，与原始 C2ST AUROC 及其本基准派生 fidelity complement；
- 原始 Dummy、TSTR 和 TRTR、本基准派生的 Local Utility retention，以及不同的 TabStruct performance ratio；
- 原始 DCR 分布和校准低尾诊断；
- 原始 wall-clock 测量和派生每秒行数吞吐量。

## 8. 结果状态

每次指标调用必须且只能返回一个状态：

- computed；
- mathematically_undefined；
- insufficient_support；
- not_applicable；
- implementation_failure；
- resource_failure。

### 8.1 状态规则

- computed 要求有限数值或有效的结构化分布结果。
- mathematically_undefined 用于所请求量在观测输入下不存在数学值的情况。
- insufficient_support 用于量本身有意义，但样本、类别、群体或邻域支持低于声明要求的情况。
- not_applicable 用于数据集 profile 或任务不满足指标声明领域的情况。
- implementation_failure 用于缺陷、意外异常、不兼容依赖行为或无效内部输出。
- resource_failure 用于超时、内存不足、磁盘耗尽或其他声明资源限制。

非 computed 状态必须具有空数值、原因代码、人类可读详情和适用计数。序列化官方结果中禁止裸 NaN 和无穷值。

### 8.2 聚合影响

每个 Registry 条目必须预先声明每种状态的聚合影响。不得在观察某项结果有利还是不利于模型后重新分类状态。

## 9. 验证要求

### 9.1 核心边界套件

每个正式指标在适用时必须测试：

- 空输入；
- 单行；
- 单列；
- 恒定数值列；
- 零数值范围；
- 单类别目标；
- 合成数据缺少目标类别；
- 合成数据出现未知类别；
- 缺失值；
- 正无穷和负无穷；
- 群体支持不足；
- 完全相同的真实与合成表；
- 刻意分离的分布；
- 列顺序改变；
- 行数不等；
- 确定性重复执行；
- 畸形元数据。

预期行为同时包括数值结果和结构化状态。

### 9.2 Golden fixture

Golden fixture 必须小型、允许纳入仓库、确定且可由手工或权威输出解释。它们必须保存：

- 输入校验和；
- 权威来源版本；
- 最终参数；
- 预期原始输出；
- 预期状态；
- 数值容差；
- fixture 生成证据。

大型真实数据集不能替代易理解的边界 fixture。

### 9.3 来源等价性

等价性测试在完全相同的输入和最终参数上执行：

~~~text
直接权威实现
基准包装器或适配器
~~~

等价性协议必须比较：

- 原始值；
- 形状与标签；
- 随机种子；
- 状态行为；
- 影响解释的警告；
- 确定性预处理；
- 聚合输入；
- 支持的数据类型组合。

对于预期确定的离散输出，要求完全相等。浮点和随机输出使用预先声明的数值或统计容差。

### 9.4 科学审查

来源等价性本身不能证明指标在科学上适当。审阅者还必须确认：

- 威胁模型或质量构念与基准声明匹配；
- 方向解释正确；
- 所需参考数据可在不泄漏的情况下获得；
- 聚合不会隐藏不支持情况；
- 指标不会以误导方式重复；
- 局限对用户可见。

## 10. 差异政策

当论文文字、公式、作者代码或第三方代码不一致时：

1. 分别记录每种行为；
2. 标识权威复现目标；
3. 不静默修复或合并定义；
4. 为重要变体分配不同指标标识；
5. 说明哪个变体属于正式、诊断或排除；
6. 保留决策证据。

相互矛盾的 DCR 方向表述、论文与仓库不同的 Authenticity 索引、report 与直接指标不同的混合列对离散化，以及有利的恒定目标回退，都属于需要这种处理的例子。

## 11. 依赖与执行隔离

指标依赖必须锁定版本。评测器必须：

- 在 preflight 验证依赖版本；
- 必要时隔离不兼容指标环境；
- 记录运行后端和可选加速；
- 控制随机种子与线程设置；
- 在支持并发评测时避免可变全局导入状态；
- 所需实现不可用时明确失败。

只有 Registry 声明了独立标识的 fallback 指标时才允许回退。Fallback 结果不等价于不可用实现的来源等价性。

## 12. 许可证与署名

复制代码或资产前，项目必须审查来源许可证、文件级通知、再分发条款和引用义务。

实际可行时，优先使用锁定依赖加包装器，而不是 vendoring。Vendored 代码必须保留版权、许可证、NOTICE、修改注释和不可变来源身份。

仓库顶层许可证不会重新许可第三方指标代码。

## 13. 指标版本

当下列任一变化改变数值含义或结果状态时，需要新指标版本：

- 论文或方法目标；
- 上游版本；
- 输入编码或距离表示；
- 参数默认值或覆盖值；
- 随机性；
- 公式；
- 方向；
- 归一化或截断；
- 聚合；
- 适用性；
- 未定义或失败处理；
- 数值容差；
- bug 修复。

无法改变解释的纯文档变更可以保留版本。

历史结果继续绑定原始指标版本。重新计算产生新的不可变结果，而不是覆盖历史。

## 14. 初始指标角色计划

本表记录来源审查决定和设计意图，而非当前实现状态。

| 指标族 | 定义来源与锁定行为 | 初始角色 |
|---|---|---|
| SDMetrics Column Shapes | 锁定 commit 的来源定义：数值/日期时间使用 KSComplement，类别/布尔使用 TVComplement，所有列等权平均 | 正式 Shape 候选 |
| SDMetrics Column Pair Trends | 锁定 commit 的来源定义：连续列对使用 Pearson，其他列对使用 contingency similarity；混合列对采用 report 专属的真实/合成独立离散化 | 正式 Trend 候选 |
| Spearman 或共同 train-fitted-bin Trend | 使用不同标识的本基准派生变体 | 科学比较完成前为实验性 |
| GReaT 随机森林判别器 accuracy | 来源定义的 accuracy，目标值为 0.5 | 来源等价诊断 |
| C2ST AUROC fidelity complement | 本基准派生的 AUROC 转换 | pilot 冻结后的正式高阶候选 |
| 积分 Alpha-Precision 与 Beta-Recall | 来源定义公式；锁定代码使用 30 点网格和嵌入空间欧氏距离 | 嵌入与积分验证前为实验性 |
| 原始 TRTR 与 TSTR | 带来源参数化的 TSTR 评测；任务指标和预测器由 profile 决定 | 正式 Utility 证据 |
| 基线调整 Local Utility retention | 由 Dummy、TSTR 与 TRTR 派生，不是 TabStruct 公式 | pilot 冻结后的正式 Local Utility 候选 |
| TabStruct Local 与 Global Utility | 来源定义：类别目标使用 Balanced-Accuracy TSTR/TRTR ratio，数值目标使用 RMSE TRTR/TSTR ratio，Global 对所有目标等权平均 | 正式 Global Utility 候选；来源可比 Local 诊断 |
| 列与跨列有效性 | 本基准原生；PAFT 提供动机而非公式 | 正式 Validity 候选 |
| 训练集精确碰撞 | 本基准原生 | 正式隐私风险诊断候选 |
| SDMetrics 混合 DCR 加 held-out 校准 | 来源定义的距离实现；依据 GReaT 参照原则增加本基准派生的分布和低尾比较 | 正式分布型诊断候选 |
| Alaa Authenticity | 论文定义与仓库执行不一致，需要不同标识 | 裁决与等价性验证前排除在 Official Results 外 |
| 成员推断 | 尚未选择来源实现 | 隐私套件冻结前必须具备 |
| 属性推断 | 尚未选择来源实现与数据集威胁模型 | 依赖数据集 profile 的候选 |
| SynthCity Delta Presence | 来源定义的可执行行为返回最大聚类计数比，而非有界概率 | 科学问题解决前排除在正式评分外 |
| 计时、吞吐量、RAM 和 VRAM | 本基准原生的运行定义 | 硬件 profile 内的正式 Efficiency 候选 |

表中任何一行都不能绕过生命周期或发布门槛。

## 15. 审阅与所有权

每个 release-supported 指标需要：

- 实现负责人；
- 科学审阅者；
- 适用时独立的兼容性与依赖审阅者；
- 有日期的准入决策；
- 已链接证据；
- 定期上游审查节奏；
- 弃用或替换计划。

当审阅者是本地实现或待验证基准声明的作者时，必须披露重大利益冲突。语义补丁和隐私声明需要独立审查。

## 16. 正式准入清单

在满足下列条件前，指标不得影响 Official Results：

- Registry 字段完整；
- 许可证已澄清；
- 定义审查获批；
- 实现身份不可变；
- 原始值与派生值分离；
- 结构化状态已实现；
- 边界与负向测试通过；
- 所声明的来源等价性通过；
- 科学适用性获批；
- 协议语义已冻结；
- 结果 schema 兼容性通过；
- 文档说明局限；
- 已记录 release-supported 决策。

## 17. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [指标来源审阅](METRIC_SOURCE_REVIEW.zh-CN.md)
- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [数据集 Profile 规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
- [仓库质量标准](../QUALITY_STANDARD.md)
