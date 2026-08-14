# 数据集卡

Dataset Profile JSON 是权威来源。源码构建清单、模型输入 Dataset Profile 和物化表格是三类不同产物。

## Adult（UCI 数据集 2）

- 使用固定官方训练/测试划分，是数值与类别混合的分类表格。
- 部分类别列含缺失标记，模型使用前只能用训练集众数填补。
- Dataset Profile：`configs/datasets/adult-uci-2-v1.json`。
- 来源与权利：UCI、CC BY 4.0；署名和精确校验和记录在 Profile 与来源清单中。
- 当前仅属诊断套件，不自动进入 Official Results。

## Sick / Thyroid Disease（UCI 数据集 102）

- 已审阅的混合类型分类视图，有独立的获取、划分和预处理声明。
- Dataset Profile：`configs/datasets/sick-uci-102-v1.json`。
- 来源与权利决定以 Profile 为准；公开可访问不等于允许任意使用。
- 当前仅属诊断套件，不自动进入 Official Results。

## 快速开始人工数据集

- 24 行完全人工数据，不对应真人或真实事件。
- Apache-2.0，仅用于验证安装 → 官方 SMOTE 适配器 → P3 Result Bundle → 诊断快照。
- 它不是科学 benchmark 数据集，不得支持质量主张。

新增数据集必须记录规范来源/版本/校验和、权利、视图与排除列、列语义、任务、冻结划分、缺失与训练集预处理、约束、隐私角色、指标适用性及独立准入。具体见[数据治理](DATA_GOVERNANCE.md)和 Dataset Profile schema。
