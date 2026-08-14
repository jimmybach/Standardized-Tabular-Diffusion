# 指标卡

Metric Registry JSON 是权威来源；本页只提供便于阅读的索引。生命周期与 Official 准入互相独立。

| 协议 | 主要维度 | 输入表 | 当前发布边界 |
|---|---|---:|---|
| `p2-shape-trend@0.2.0` | SDMetrics Column Shapes、Column Pair Trends | 真实训练 + 合成 | 诊断；上游来源已验证，但不自动获得 Official 准入 |
| `p3-validity@0.3.0` | 逐列与已审阅跨列硬规则有效性 | 真实训练 + 合成 | 诊断；评测器不修复数据 |
| `p4-utility@1.0.0` | Local Utility 保留率与 TabStruct 公式 4 Global Utility | 真实训练 + 真实测试 + 合成 | 条件性协议冻结；数据集、模型、运行和发布门仍独立 |
| `p5-high-order-privacy@1.0.0` | C2ST、重复/碰撞、DCR、限定 DOMIAS 攻击 | 真实训练 + 真实测试 + 合成 | 条件性协议冻结；不产生总分或正式隐私保证 |

每个指标以 Atomic Result 保存精确身份、作用域、状态、方向、分母、原始值、适用时的归一化值、聚合贡献、reason code 和证据。缺失、数学未定义、支持不足、实现失败、资源失败和不适用均不得静默变成 0 或被删除。

旧 `tabstruct-aligned-v1` 只能作为 `legacy-diagnostic`，缺少当前聚合器要求的 Atomic Result，不能进入 Official Results。

公式、来源、状态和准入字段见[指标治理](evaluation/METRIC_GOVERNANCE.md)、[评测协议](evaluation/EVALUATION_PROTOCOL.md)及包内 Metric Registry JSON。
