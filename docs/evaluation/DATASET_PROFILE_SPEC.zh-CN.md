# 数据集 Profile 规范

英文规范：[DATASET_PROFILE_SPEC.md](DATASET_PROFILE_SPEC.md)

- 状态：设计基线
- Profile schema 版本：0.1.0
- 最后更新：2026-08-03

本文件是英文规范的对应中文译文。若两者存在歧义，以英文规范为准。

## 1. 目的

本规范定义机器可读 Dataset Profile，用于绑定数据集身份、权利、schema、view、划分、预处理、预测任务、有效性规则、隐私威胁模型和指标适用性。

Dataset Profile 是评测协议的一部分。它防止在观察模型结果后临时决定目标选择、约束、敏感角色和指标排除。

## 2. 治理原则

每个官方数据集必须：

- 可追溯到具体来源和版本；
- 已针对项目预期用途与分发方式完成法律审查；
- 通过校验和进行内容寻址；
- 由显式语义 schema 表示；
- 分配冻结的数据划分；
- 通过只在 train 上拟合的转换处理；
- 通过预先声明的任务和约束评测；
- 通过有记录审查进入带版本数据集套件。

本地文件或上游脚本的存在并不能证明再分发权、科学适用性或官方资格。

## 3. Profile 身份

每个 profile 声明：

- profile_schema_version；
- dataset_profile_version；
- dataset_id；
- display_name；
- dataset_version；
- dataset_view；
- dataset_family；
- status；
- owners；
- review record；
- change log。

dataset_id 标识概念数据集，dataset_version 标识来源内容，dataset_view 标识该版本的确定性表示。

稳定标识符应使用小写 ASCII，并可安全用于可移植路径。显示名称和翻译单独存储。

## 4. 数据集套件

suite_membership 使用下列一个或多个值：

- universal-core；
- extended-catalog；
- diagnostic。

### 4.1 Universal Core Suite

Universal Core Suite 是完整 Official Results 的强制数据集集合。成员资格要求：

- 针对预期工作流的访问和发布权利已澄清；
- 检索与校验和稳定；
- schema 与划分完整；
- 任务和数据类型覆盖具有代表性；
- 在预期 Core Model Set 上通过能力测试；
- 在官方资源 profile 下运行时间可接受；
- 必需评测元数据完整。

Core 成员由协议版本冻结。

### 4.2 Extended Catalog

Extended Catalog 包含有效但不强制纳入完整 Core 覆盖的基准目标。原因可能包括算法能力、运行时间、数据权利、特殊 schema、规模或待完成验证。

### 4.3 Diagnostic Suite

Diagnostic 数据集或 fixture 面向具体边界情况、回归或失败模式。CI 只使用小型、合成且可再分发的 fixture。Diagnostic 结果不改变 Core 排名。

## 5. 来源、权利与溯源

必需来源字段包括：

- 规范来源 URL 或访问机制；
- 来源发布者；
- 来源版本或获取日期；
- 可用时的不可变 archive 标识；
- 原始文件清单与校验和；
- 数据集引用；
- 许可证或条款；
- 访问限制；
- 允许用途；
- 再分发状态；
- 修改与署名要求；
- 包含审阅者和日期的权利审查决策。

redistribution_status 为下列之一：

- permitted；
- metadata-only；
- download-script-only；
- restricted；
- prohibited；
- unknown。

Unknown 或 prohibited 数据不得提交、纳入 release 或作为基准产物发布。只有条款允许时，才可以提供可复现注册或下载程序。

源码许可证不决定数据集权利。

## 6. 规范表契约

Profile 声明：

- 规范序列化方式；
- 编码；
- 适用时的分隔符；
- 行数预期；
- 唯一列名；
- 规范列顺序；
- 重复行策略；
- 主键或标识符策略；
- 目标存在性；
- 缺失值清单；
- 支持的输入形式；
- 规范表校验和。

公共接口可以接受 CSV、Parquet 或 DataFrame。官方评测将所有形式解析为相同规范逻辑表和校验和。

## 7. 列 schema

每个列声明：

- name；
- 稳定 column_id；
- semantic_type；
- storage_type；
- 一个或多个 role；
- 原始数据是否可空；
- 规范模型输入是否可空；
- 有效领域；
- 类别词表来源；
- 转换策略；
- 逆转换策略；
- 描述与单位；
- 敏感性元数据；
- 约束引用。

### 7.1 语义类型

初始语义类型为：

- continuous；
- integer；
- categorical；
- boolean；
- datetime；
- string。

自由文本和复杂嵌套值不属于初始静态单表基准，除非后续协议明确准入。

### 7.2 列角色

一个列可以声明适用角色：

- feature；
- primary_target；
- secondary_target；
- identifier；
- sensitive_attribute；
- quasi_identifier；
- group_attribute；
- ignored；
- audit_only。

标识符和纯行索引默认排除在模型特征和 Global Utility 外。例外需要明确科学理由。

敏感属性和准标识符角色必须基于数据集文档和审查，不能从列名猜测。

### 7.3 类别词表

允许类别来自权威数据字典或经审查 schema 声明。如果词表从数据中学习，就必须只从 train 学习，并明确这一局限。

根据权威领域定义有效但未出现在 train 中的类别，可以继续保持 schema-valid。预处理和未知类别行为必须另行声明。

### 7.4 数值领域

硬数值边界需要权威领域来源或经审查科学理由。观测到的 train 最小值和最大值默认属于软支持诊断。

整数列必须定义无损转换规则。带小数部分的值不得被静默截断。

## 8. 缺失值与预处理

原始数据集可以包含缺失值。初始模型输入契约不将缺失值传给生成模型。

每个具有原始缺失的数据集 profile 必须调用集中式预处理模块，并声明：

- 缺失标记；
- 各划分受影响列与比例；
- 各语义类型的填补策略；
- 是否增加缺失指示器；
- 拟合划分；
- 适用时的随机种子；
- 预处理配置版本；
- 预处理实现版本；
- 学习产物校验和；
- 转换后 schema 校验和；
- 逆变换或解码表示。

所有学习得到的填补和转换状态只在 train 上拟合。Validation 和 test 在不重新拟合的情况下转换。

填补政策具有科学实质性。政策变化会产生新的预处理和 dataset-view 身份，并需要结果兼容性审查。

## 9. Dataset view

Dataset view 是某个来源数据集版本的确定性、带名称且带版本的转换。

示例包括：

- 规范混合类型 view；
- 权威上游方法要求的 view；
- 无类别 view；
- 隐私距离分析 view；
- 填补 view。

nocat 或 dcr 等后缀形式属于 view，而不是独立来源数据集。

每个 view 必须声明：

- 父数据集与版本；
- 转换图；
- 实现与配置版本；
- 拟合边界；
- 输出 schema；
- 校验和；
- 信息损失；
- 预期模型和指标；
- 可比性局限；
- 可用时的逆转换。

官方跨模型比较应使用同一个规范语义 view。方法专属内部编码位于适配器内部，并且必须解码回规范语义 view 后评测。

## 10. 划分契约

每个官方 profile 声明一个冻结的 split 标识以及：

- 划分生成方法；
- 划分种子；
- train、validation 与 test 成员产物或确定性规则；
- 行数与类别或目标汇总；
- 分层或分组规则；
- 适用时的时间顺序；
- 校验和；
- 泄漏检查；
- 局限。

Train 用于拟合，Validation 仅用于允许的模型选择，Test 仅用于最终评测。

划分验证器必须检查：

- 存在身份时的行身份互斥；
- 按数据集政策定义的精确和近重复泄漏；
- 分组数据的群组泄漏；
- 时间数据的时间泄漏；
- 目标分布异常；
- 最小类别支持；
- 确定性重构。

划分成员或逻辑变化会产生新 split 和协议兼容组。

## 11. Local Utility profile

每个官方预测数据集必须且只能声明一个主要 Local Utility 目标：

- 目标列；
- 任务类型；
- 二分类正类定义；
- 标签映射；
- 主指标；
- 辅助指标；
- 评测器 profile；
- 最小支持量；
- 当本基准派生 retention 适用时所采用的 Dummy 策略；
- 归一化指标标识及版本；
- 已知局限。

可选次要目标使用同样完整的声明。它们单独报告，不改变主要 Local Utility 榜单权重。

目标必须反映有文档的数据集任务或经审查科学目的。不得在比较生成器结果后选择。

## 12. Global Utility profile

Global Utility 默认包括每个可评测的非标识符列，并轮换作为目标。

Profile 声明：

- 包含目标；
- 排除目标与稳定原因代码；
- 每个目标的任务类型；
- 评测器 profile；
- Global Utility 指标与预测器 profile 标识；
- 目标专属支持阈值；
- 高基数处理；
- 日期时间处理；
- 聚合权重。

对 TabStruct 兼容 profile，每个类别目标使用 Balanced-Accuracy TSTR/TRTR ratio，每个数值目标使用 RMSE TRTR/TSTR ratio，所有目标等权。不同归一化或预测器 profile 会形成不同的指标身份和兼容组。禁止因为模型表现差而排除目标。

## 13. Validity profile

### 13.1 硬列规则

硬规则可以定义：

- 可空性；
- 有限数值；
- 整数要求；
- 权威边界；
- 允许类别；
- 字符串格式或长度；
- 日期时间解析与范围；
- 唯一性；
- 标识符格式。

每条硬规则需要一个来源：

- authoritative-data-dictionary；
- dataset-documentation；
- legal-or-policy-requirement；
- recorded-human-review。

### 13.2 跨列约束

跨列规则可以定义：

- 时间顺序；
- 条件领域；
- 互斥；
- 求和或总量关系；
- 函数依赖；
- 逻辑不可能组合；
- 数据集专属业务规则。

每项约束声明标识、表达式或可执行实现、适用性、缺失值行为、证据来源、严重级别、测试 fixture 与版本。

### 13.3 软诊断

软诊断可以包括：

- 超出 train 支持的比例；
- 未见类别组合；
- 稀有群体行为；
- 超出观测范围；
- 推断依赖；
- 异常分数。

除非通过经审查证据和新 profile 版本提升，软诊断不决定硬有效性。

## 14. 隐私风险 profile

隐私评测需要预先声明角色和威胁模型。

Profile 可以声明：

- 敏感属性；
- 准标识符；
- 允许的攻击者知识；
- 成员与非成员总体；
- 群体定义；
- 精确碰撞规范化；
- DCR 特征表示；
- 成员推断适用性；
- 属性推断目标与已知特征；
- 最小群体和类别支持；
- 正式差分隐私元数据预期；
- 逐行产物发布限制。

如果没有可辩护的敏感属性或准标识符集合，对应攻击为 not_applicable，而不是报告为零风险。

只有声明群体属性且支持充分时才执行公平性或分组分析。阈值由协议冻结，群体级结果不得暴露敏感个体。

## 15. 指标适用性

Profile 将每个指标或指标族映射到：

- required；
- optional；
- experimental；
- not_applicable；
- prohibited。

每个非 required 决策都需要稳定原因和证据。适用性在模型评测前固定。

示例包括：

- 回归指标禁止用于类别目标；
- 没有经审查敏感目标时属性推断不适用；
- Global Utility 排除纯标识符；
- 日期时间诊断在协议冻结前为可选；
- 当攻击产物违反数据条款时禁止该隐私攻击。

## 16. 预声明的评测修复

Profile 定义是否允许对内容无效的合成输出做仅供评测使用的修复。

允许的修复必须：

- 保持原始合成数据不可变；
- 只使用在 train 上拟合的参数；
- 在记录种子下保持确定性；
- 记录每个受影响列和行数；
- 将未知类别映射到显式未知级别；
- 保留原始 Validity 惩罚；
- 设置 evaluation_repair_applied。

结构性失败不能被修复成官方运行。

## 17. Profile 示例

~~~yaml
profile_schema_version: 0.1.0
dataset_profile_version: 0.1.0

identity:
  dataset_id: example-income
  display_name: Example Income Dataset
  dataset_version: source-version
  dataset_view: canonical-imputed-v1
  status: registered

suite_membership:
  - extended-catalog

source:
  publisher: pending
  url: pending
  retrieved_at: 2026-08-03
  raw_checksums: {}
  citation: pending
  license_or_terms: pending-review
  redistribution_status: unknown

canonical_table:
  formats: [csv, parquet, dataframe]
  expected_columns: [age, occupation, income]
  canonical_checksum: pending

columns:
  - column_id: age
    name: age
    semantic_type: integer
    storage_type: int64
    roles: [feature]
    raw_nullable: true
    model_input_nullable: false
    hard_domain:
      minimum: 0
      source: recorded-human-review

  - column_id: occupation
    name: occupation
    semantic_type: categorical
    storage_type: string
    roles: [feature, quasi_identifier]
    raw_nullable: true
    model_input_nullable: false
    categories:
      source: pending
      values: []

  - column_id: income
    name: income
    semantic_type: boolean
    storage_type: string
    roles: [primary_target, sensitive_attribute]
    raw_nullable: false
    model_input_nullable: false
    categories:
      source: pending
      values: [low, high]

preprocessing:
  fitted_on: train
  missing_policy:
    implementation: centralized
    configuration_version: pending
  learned_artifact_checksum: pending

split:
  split_id: official-v1
  method: pending
  seed: pending
  checksums: {}

local_utility:
  primary_target: income
  task_type: binary_classification
  primary_metric: macro_f1
  secondary_metrics: [balanced_accuracy, roc_auc, pr_auc]
  evaluator_profile: pending
  dummy_strategy: most_frequent
  normalization_metric: baseline_adjusted_local_utility_retention@pending

global_utility:
  default_include_non_identifiers: true
  exclusions: []
  metric: tabstruct_global_utility@pending
  predictor_profile: pending

validity:
  hard_constraints: []
  soft_diagnostics:
    - outside_train_support_rate

privacy:
  sensitive_attributes: [income]
  quasi_identifiers: [occupation]
  threat_models:
    membership_inference: pending-review
    attribute_inference: pending-review

metric_applicability: {}

review:
  owners: []
  reviewers: []
  decision: pending
~~~

本示例有意保持不完整，不得作为真实数据集 profile 使用。

## 18. Profile 生命周期

建议的 profile 生命周期：

1. registered：已记录身份和候选来源。

2. source-and-rights-reviewed：已审查来源、引用、访问、许可证与再分发状态。

3. schema-reviewed：已审查列、语义类型、角色、领域与缺失情况。

4. split-and-preprocessing-validated：划分边界、只在 train 上拟合、校验和与转换后 schema 通过验证。

5. evaluation-profile-complete：任务、约束、隐私角色、适用性与评测器要求完整。

6. benchmark-eligible：针对声明套件通过能力、运行时间、法律、科学和协议门槛。

7. release-supported：针对发布持续维护检索、验证、文档、测试、所有权与兼容性。

只有 benchmark-eligible profile 才能进入官方套件。Universal Core 成员还需要协议级批准。

## 19. 接入与审查

数据集接入必须包括：

1. 来源与权利清单；
2. 原始完整性验证；
3. schema 与语义角色审查；
4. 缺失情况与预处理提案；
5. 划分设计与泄漏审计；
6. 目标与评测器审查；
7. 硬约束证据审查；
8. 隐私角色与威胁模型审查；
9. 指标适用性审查；
10. 代表性模型能力与运行时间测试；
11. profile 验证；
12. 套件准入决策。

批准权利的审阅者应独立于仅仅下载数据的人。数据集有需要时，敏感角色和领域约束决策需要领域专家审查。

## 20. 变更控制

当变化影响下列任一项时，需要新的 dataset profile 或 view 版本：

- 来源内容；
- 数据权利；
- 行或列；
- 语义类型或角色；
- 目标定义；
- 类别词表；
- 硬约束；
- 划分；
- 预处理或填补；
- 隐私角色或威胁模型；
- 指标适用性；
- 评测器 profile；
- 聚合权重。

历史结果继续绑定其原始 profile、split 和预处理校验和。

## 21. 初始目录工作

预期的 21 个数据集目录需要逐一审查。任何数据集都不会仅仅因为出现在现有脚本或研究资料包中就获得准入。

对每个候选数据集，项目将确定：

- 精确来源与权利；
- 规范版本；
- 任务与目标；
- schema 与缺失情况；
- Core、Extended 或 Diagnostic 角色；
- 约束与隐私元数据；
- 跨候选模型兼容性；
- pilot 运行时间。

Universal Core 成员只有在预期 Core Model Set 上完成模型能力测试后才会选定。

## 22. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
- [仓库质量标准](../QUALITY_STANDARD.md)
