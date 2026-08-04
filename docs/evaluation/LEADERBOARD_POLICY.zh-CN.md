# 榜单政策

英文规范：[LEADERBOARD_POLICY.md](LEADERBOARD_POLICY.md)

- 状态：设计基线
- 政策版本：0.1.0
- 最后更新：2026-08-03

本文件是英文规范的对应中文译文。若两者存在歧义，以英文规范为准。

## 1. 目的

本政策定义结果如何获得发布资格、哪些结果可以比较、分数如何聚合，以及如何展示官方、部分、诊断和社区结果。

它从属于[仓库质量标准](../QUALITY_STANDARD.md)，并采用[评测协议](EVALUATION_PROTOCOL.zh-CN.md)中的指标语义。发布在名为“榜单”的页面上，本身并不会让结果成为官方结果。

## 2. 术语与相互独立的分类

基准有意分离经常被混淆的属性。

### 2.1 Eligibility Track

Eligibility Track 描述科学性和来源方面的准入资格：

- official；
- experimental；
- excluded。

这些值由仓库质量标准定义。一个模型不会仅仅因为成功运行或由本仓库维护就成为 official。

### 2.2 Comparison Track

Comparison Track 描述配置和调参条件：

- native；
- standardized-tuning。

同一个模型可以通过不同运行同时出现在两个 Comparison Track 中。不同 Comparison Track 的结果不得混合。

### 2.3 Support Level

Support Level 描述维护承诺：

- unsupported；
- experimental；
- release-supported。

Support Level 独立于准入资格和 Comparison Track。一个 release-supported 的实验实现仍然没有官方排名资格。

### 2.4 Publication Class

已发布结果使用下列类别之一：

- Official Results；
- Partial/Diagnostic Results；
- Community Results。

Publication Class 不会弱化上述分类所要求的来源、验证或支持要求。

## 3. Native Comparison Track

Native Track 回答：

> 权威实现在按照官方或文档所表达的预期配置使用时表现如何？

### 3.1 配置优先级

Native 配置按下列顺序选择：

1. 官方仓库中明确推荐用于可比基准测试的配置；
2. 方法论文中发布的配置；
3. 官方包默认配置；
4. 当前述选项无法运行时使用最小兼容配置。

必须引用配置来源，并在 test 评测前冻结最终配置。最小兼容配置需要有记录的理由，不得通过查看 test 结果来选择。

### 3.2 允许的适配

Native Track 允许：

- 经批准的适配器层输入输出转换；
- 保持上游语义的配置映射；
- 确定性产物发现和元数据记录；
- 已通过原生等价性验证的批准兼容性补丁。

它不允许未经批准的语义补丁、本地算法替代、由 test 驱动的参数更改，或静默回退到不同实现。

### 3.3 提前停止

权威提前停止程序可以只查看 validation。必须记录监控量、patience、checkpoint 规则和所选训练轮数。Test 指标不得影响 checkpoint 选择。

## 4. Standardized Tuning Comparison Track

Standardized Tuning Track 回答：

> 当每个权威实现获得声明且可比较的超参数搜索机会后，其表现如何？

### 4.1 搜索空间

每个模型拥有经审阅的模型专属搜索空间。不同算法不需要暴露相同参数名或范围，但所有搜索空间必须：

- 由上游文档或方法行为支持；
- 在搜索开始前冻结；
- 排除会改变所声明方法身份的参数；
- 说明参数分布和条件依赖；
- 带有版本与完整性哈希。

观察到 test 表现后不得扩大搜索空间。

### 4.2 搜索方法与预算

初始搜索方法为固定种子的随机搜索。调参 profile 同时施加：

- 最大试验次数；
- 最大计算时间或资源预算。

达到任一限制即停止搜索。数据规模 profile 可以为 small、medium 和 large 定义不同预算，但规模划分规则和全部阈值必须在执行前冻结。

贝叶斯优化或其他搜索方法需要独立版本化的调参 profile，不得静默混入随机搜索结果。

### 4.3 选择数据和目标

调参试验在 train 上训练，在 validation 上选择。Test 在最佳配置冻结前始终不可访问。

初始选择原则为：

1. 先满足经 pilot 冻结的最低 Validity 门槛；
2. 再最大化 validation Fidelity 与本基准派生、经基线调整的 validation Local Utility retention 的调和平均。

该目标防止一个很强的组成项完全掩盖另一个已经崩坏的组成项。隐私攻击不作为调参目标；效率通过资源预算约束，而不是加入质量目标。

具体组成项、未定义 retention 的数值处理、可能采用的截断转换、Validity 阈值和并列规则均为经 pilot 冻结的事项。该调参目标不得描述为 TabStruct 效用公式。

### 4.4 最终运行与成本核算

选择完成后：

1. 冻结最佳最终配置及其哈希；
2. 在同一个 train 划分上重新训练模型；
3. 执行五个正式生成随机种子；
4. 每个最终种子执行一次 test 评测；
5. 保留每次试验、失败、剪枝决策和资源测量。

已发布结果必须同时报告总搜索成本和最终重训练成本。禁止只报告最快或最佳的一次试验。

## 5. Official Results 准入

只有满足全部下列条件，结果才能进入 Official Results：

- 模型已依据仓库质量标准获得 official 准入决策；
- 来源权威性、复现目标、上游版本、修改状态和等价性证据完整；
- 模型针对声明的协议和数据集套件具有 benchmark eligibility；
- 所有计分指标均已 protocol-frozen 且 release-supported；
- 数据集 profile、数据权利、划分、预处理和校验和已批准；
- 所有要求的 Universal Core Suite 数据集和五种子运行完整；
- 未发生禁止的 test 访问；
- result bundle 通过 schema、完整性、来源和兼容性验证；
- 不存在强制项失败或缺失分母；
- 审阅者批准准入记录。

被宣传为公共 Core Model Set 成员的模型必须具有 release support。只有发布计划和仓库质量标准明确允许时，协议才可以为其他受维护模型发布 official 证据。

## 6. Partial 和 Diagnostic Results

当一次运行具有科学价值但未达到完整官方覆盖时，属于 Partial/Diagnostic Results。例如：

- 只完成部分 Universal Core 数据集；
- 少于五个生成随机种子；
- 某个必需指标仍为实验性；
- 资源限制阻止完成；
- 实现仍在等价性审查中；
- 运行使用非默认诊断样本规模 profile。

Partial 结果必须：

- 说明缺失要求；
- 展示已完成和失败覆盖；
- 在视觉上与 Official Results 分离；
- 不给出官方汇总排名。

部分发布不得成为隐藏困难数据集的手段。

## 7. Community Results

Community Results 接受在官方执行环境之外生成的有效 result bundle。

它们必须标识：

- 提交者和仓库版本；
- 模型与实现来源；
- 协议和指标版本；
- 数据集校验和；
- 最终配置和随机种子；
- 环境与硬件；
- 全部验证警告。

即使数值可以复现，Community Results 也不得标为官方。晋升需要官方准入审查，并在需要时由维护者在官方环境中重新运行。

没有可验证生成来源的外部合成表保持为 Community 或 Diagnostic Results。

## 8. 数据集套件与覆盖率

数据集属于带版本的套件：

- Universal Core Suite；
- Extended Catalog；
- Diagnostic Suite。

套件成员由协议版本固定。带后缀形式、无类别形式或面向 DCR 的特殊形式属于带版本 dataset view，而不是静默作为独立数据集。

完整 Official Results 要求 Universal Core Suite 百分之百覆盖，以及全部强制五种子运行。Extended 和 Diagnostic 结果单独展示，不改变 Core 覆盖率。

覆盖率报告必须区分：

- 适用且已计算；
- 数学上未定义；
- 支持不足；
- 不适用；
- 实现失败；
- 资源失败。

## 9. 聚合

### 9.1 聚合顺序

要求的层级为：

1. 保留原子值；
2. 按指标定义聚合评测器重复、列、列对、预测器或目标；
3. 在每个数据集内聚合生成随机种子；
4. 计算数据集汇总；
5. 对数据集汇总做数据集等权宏平均。

行数、目标数、预测器数量或随机种子数量不得让某个数据集获得更大的跨数据集权重。

### 9.2 维度输出

榜单发布：

- Fidelity 组成项和等权组成的 Fidelity 分数；
- 原始 Local Utility 结果，以及带独立标识的本基准派生 Local Utility retention；
- 来源定义的 TabStruct Global Utility ratio，包括每个逐目标 ratio；
- Validity 组成项和 Validity 分数；
- 分离的经验性隐私风险诊断；
- 分离的 Training Time、Sampling Throughput 和 Peak Resource Usage 视图。

不存在跨维度总分，不存在统一 Privacy Score 或 Efficiency Score。初始协议不合并 Local Utility 和 Global Utility。Local retention 与 TabStruct Global Utility ratio 是不等价的归一化方式，不得相互改名。

### 9.3 未定义项和失败项的贡献

指标定义必须声明一个未定义原子值应当：

- 使父级汇总未定义；
- 作为真正不适用项在显示分母的前提下排除；
- 因为代表模型失败而获得协议定义的最差贡献。

这些选择必须在查看结果前完成。使训练无法进行的恒定合成目标属于支持失败，并获得规定的最低效用贡献；不得静默排除。

官方汇总结果要求全部强制贡献。无法满足要求的结果进入 Partial/Diagnostic Results，而不是改变分母。

## 10. 排名与不确定性

### 10.1 排序

排名使用未舍入的完整精度汇总值。显示舍入不得改变排名顺序。每个名次都展示分数、不确定性区间、数据集覆盖、种子覆盖和协议身份。

### 10.2 置信区间

正式协议使用尊重数据集与种子结构的带版本分层或配对 bootstrap。必须记录重采样单位、重复次数、区间方法和随机种子。

### 10.3 并列

在实践上和统计上不可区分的模型标记为并列。等价边界和统计决策规则由 pilot 证据冻结。

点估计可以继续排序以方便浏览，但界面不得暗示并列组内部存在具有科学意义的先后次序。

### 10.4 两两结论

当比较大量模型时，两两优越性结论应使用配对的数据集—种子比较和合适的多重比较策略。视觉排名本身不是普遍优越性的证据。

## 11. 效率比较

只有硬件 profile、软件 profile、线程上限、设备数量、样本规模 profile 和计时定义相同时，效率结果才可比较。

官方效率视图必须展示：

- 整个套件总时间；
- 每个数据集时间；
- 吞吐量；
- 峰值 RAM 和 VRAM；
- checkpoint 大小；
- 超时与内存不足覆盖；
- Native 或 Standardized Tuning 成本范围。

项目不使用理论性能比率对不同硬件的 wall-clock 时间做归一化。

## 12. 榜单视图

初始发布生成静态 HTML 和 Markdown 视图，并提供可下载的结构化数据。

用户必须能够按以下条件筛选：

- Comparison Track；
- Official、Partial/Diagnostic 或 Community Publication Class；
- 数据集或套件；
- 任务类型；
- 模型家族；
- 协议和指标版本；
- 硬件 profile；
- 支持与验证状态。

界面必须支持从模型汇总下钻到数据集、种子和适用的原子结果。图表必须链接或附带用于渲染的数据。

推荐视图包括数据集热力图、覆盖矩阵、配对比较、不确定性图、Fidelity–Utility 视图、Utility–Privacy Pareto 视图和 Quality–Cost Pareto 视图。雷达图可以汇总维度，但不得描述为总分。

## 13. 提交与发布工作流

初始官方流程为：

~~~text
提交 adapter 或配置
→ 验证身份、来源与准入资格
→ 执行接口和 smoke 检查
→ 在官方环境中运行
→ 验证 result bundle
→ 审查准入证据
→ 通过 Pull Request 发布
~~~

榜单更新必须作为仓库变更接受审查，并且必须标识新增、删除、被替代和失效的结果。

第一版不需要在线评测服务或数据库。在提交量足以证明额外运维复杂度合理之前，优先采用静态发布流水线。

## 14. 版本、纠正与失效

榜单快照绑定：

- 仓库发布版本；
- 协议版本；
- Metric Registry 版本；
- 数据集套件版本；
- 结果 schema 版本；
- 适用时的官方硬件 profile。

影响数值含义的变化会产生新快照。历史结果不得被静默覆盖。

如果发现错误、泄漏事件、许可证问题、来源缺陷或无效指标：

1. 将受影响结果标记为失效或撤回；
2. 记录原因和范围；
3. 重新生成下游汇总；
4. 纠正结果获得新的不可变标识；
5. 在公开变更日志中说明科学结论是否改变。

## 15. 发布安全

榜单默认发布指标、最终配置、环境元数据、来源、汇总日志、校验和与审阅证据。

它不会自动发布：

- 受限真实数据；
- 可能暴露训练记录的合成记录；
- 包含敏感内容的逐行攻击输出；
- 大型 checkpoint；
- 密钥、本地绝对路径或私人基础设施标识；
- 未获得再分发许可的产物。

数据和产物发布与结果准入分别决策。

## 16. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [数据集 Profile 规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
- [仓库质量标准](../QUALITY_STANDARD.md)
