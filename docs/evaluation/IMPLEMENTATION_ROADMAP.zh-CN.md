# 评测与榜单实施路线图

英文原文：[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)

- 状态：P1 至 P3 已通过各自适用的退出门；权威 Linux/Python 3.11 证据均已留存
- 路线图版本：0.3.0
- 最后更新：2026-08-05
- 主要发布环境：Linux 与 Python 3.11

## 1. 目的

本路线图把已经批准的评测规范转化为适用于当前仓库的可执行工程顺序。它与现有代码直接对应：记录已经存在的内容、可以复用的内容、必须替换的内容、兼容性如何保留，以及某个指标或榜单在被称为“正式”之前必须具备哪些证据。

本文件本身不会批准任何指标、模型、数据集、结果或发布版本。规范语义仍以相关规范文件为准。只有通过退出门槛并具备所需证据，某个阶段才算完成。

## 2. 已锁定的实施原则

实现必须遵守以下决定：

1. 评测是独立子系统。适配器生成的样本和外部提交的合成表使用同一个评测引擎。
2. 模型适配器负责训练、采样和解码，不负责指标公式、结果聚合或榜单资格。
3. 对来源定义的指标，应尽可能包装不可变的官方软件包或锁定的官方源码。本地重实现绝不能在不说明的情况下替代来源等价指标。
4. 上游模型和指标源码树是只读证据或运行时。任何不可避免的源码修改都必须先讨论，并具备隔离补丁、来源记录、许可证审阅和等价性验证。
5. 必须提供机器可读的 Dataset Profile、Metric Registry 记录、协议配置和结果 schema。仅在文档中描述默认值是不够的。
6. 签入仓库的 JSON Schema 是线格式契约的规范校验器。内部 Python 模型可以在不改变线格式契约的情况下演进，但不能绕过已签入 schema 和版本规则。
7. 指标调用返回结构化状态。异常、`NaN`、无穷值、被遗漏的目标和静默缩小的分母都不是有效结果表示。
8. 来源原始值、基准库派生变换和聚合贡献必须是不同字段。
9. 在首个正式协议中，缺失值绝不能进入生成模型。原始数据含缺失值的数据集必须调用显式、版本化、仅在 train 上拟合的预处理模块；注册过程不得静默删除或填补行。
10. 初始发布不设置总榜单分数。Fidelity、Local Utility、Global Utility、Validity、Privacy Risk 和 Efficiency 保持为可见维度或独立子榜单。
11. 只有同时达到 `protocol-frozen` 和 `release-supported` 的指标才能影响 Official Results。
12. 第一个端到端实现切片是结构校验加来源等价的 Column Shapes 与 Column Pair Trends。只有该切片能够生成有效的 finalized result bundle 后，才扩展指标广度。

TabStruct 仅作为指标审阅与公式来源追踪的研究参考材料。其论文和提取出的参考代码不定义新评测子系统的运行时架构。P1 之前的 `evaluation/tabstruct.py` 路径只作为旧版诊断兼容路径保留；未经独立的生命周期审阅，不得将它升级、改名或包装为正式指标。

## 3. 当前实现审计

本节取代 2026-08-03 的 P0 之前审计。历史失败仍可作为迁移证据，但已经不再描述当前代码树。

### 3.1 当前执行路径

目前有四条刻意隔离的路径：

~~~text
旧版兼容路径
ExperimentConfig -> model_adapter.evaluate -> evaluation.tabstruct.normalize_*
                 -> standardized_summary.json -> comparison.compare_summaries

P1 评测基础
EvaluationRequest -> 严格契约/schema 校验
                  -> registry/profile 身份解析
                  -> IncompleteRunBundleWriter
                  -> manifest/metadata/config/environment/summary/event-log 校验

P2 独立表评测
EvaluationRequest + Dataset Profile + 参考表/合成表
                  -> 规范结构门禁
                  -> 精确锁定的 SDMetrics Shape/Trend 后端
                  -> 逐列/逐列对 Atomic Result
                  -> 校验和完整的 finalized Run Result bundle

P3 独立有效性评测
EvaluationRequest + 已审阅有效性契约 + 参考表/合成表
                  -> 保留内容违规的 P3 结构门禁
                  -> 逐列/逐约束硬规则评测
                  -> 不可变输出证据和 benchmark-native 聚合
                  -> 校验和完整的 finalized Run Result bundle
~~~

旧版路径始终只用于诊断。P1 契约路径继续支持 incomplete bundle。独立 P2 和 P3 路径分别完成来源等价 Fidelity 与 benchmark-native Validity bundle，均不经过 `evaluation/tabstruct.py`。

### 3.2 当前代码处置

| 当前位置 | 当前作用 | 当前决定 |
|---|---|---|
| [`evaluation/tabstruct.py`](../../standardized_tabular_diffusion/evaluation/tabstruct.py) | P1 之前的兼容评测器 | 冻结为旧版诊断路径；TabStruct 是研究参考，不是新引擎或指标权威来源 |
| [`evaluation/contracts.py`](../../standardized_tabular_diffusion/evaluation/contracts.py) | Evaluation Request、Atomic Result、阶段与生命周期契约 | P1 活跃路径；严格执行有限值、状态、支持计数、聚合、时间戳、身份和路径不变量 |
| [`evaluation/serialization.py`](../../standardized_tabular_diffusion/evaluation/serialization.py) | Canonical JSON、安全结构化加载、哈希和原子替换 | P1 活跃路径；JSON/YAML 重复键和非有限值均采用 fail-closed |
| [`evaluation/registry.py`](../../standardized_tabular_diffusion/evaluation/registry.py) | 数据驱动 Metric Registry 与累积生命周期校验 | P1 活跃路径；P1 之前的八条记录均明确为 `legacy-diagnostic` 且非正式 |
| [`evaluation/profiles.py`](../../standardized_tabular_diffusion/evaluation/profiles.py) | 数据集/协议加载、精确身份和旧元数据导入 | P1 活跃路径；重复身份和不一致的 profile 引用均采用 fail-closed |
| [`evaluation/bundle.py`](../../standardized_tabular_diffusion/evaluation/bundle.py) | 事务化 Run Result writer、finalizer 与跨文件校验器 | 保留 P1 incomplete bundle；P2/P3 重算科学汇总，并以最后一个原子提交标记发布 finalized 状态 |
| [`evaluation/table.py`](../../standardized_tabular_diffusion/evaluation/table.py) | CSV/Parquet/DataFrame 规范解析器与协议专用结构门禁 | P2 保留严格来源兼容检查；P3 保留可安全表示的内容违规用于 Validity 评分 |
| [`evaluation/backends/sdmetrics.py`](../../standardized_tabular_diffusion/evaluation/backends/sdmetrics.py) | 隔离的权威 Shape/Trend 后端 | 要求 SDMetrics `0.28.3.dev0` 以及 commit `ba8842f2...` 的完整 121 文件源码树哈希 |
| [`evaluation/shape_trend.py`](../../standardized_tabular_diffusion/evaluation/shape_trend.py) 与 [`evaluation/evaluate_table.py`](../../standardized_tabular_diffusion/evaluation/evaluate_table.py) | Atomic Result 映射与端到端表评测器 | P2 活跃路径；重建来源聚合且不生成合并 Fidelity 分数 |
| [`evaluation/validity.py`](../../standardized_tabular_diffusion/evaluation/validity.py) | 封闭硬规则语言、逐列/逐约束 Atomic Result 与 Validity 聚合 | P3 活跃诊断路径；禁止任意代码和推断硬规则，不修复原始输出 |
| [`evaluation/utility.py`](../../standardized_tabular_diffusion/evaluation/utility.py) | held-out-test Local/Global Utility、原始 arms、支持状态与严格 ratio 聚合 | P4 诊断工程门与有限范围真实 TabEval XGB/KNN/TabPFN 来源运行时等价 pilot 已在 Linux/Python 3.11 上通过；完整数据集准入仍待完成 |
| [`preprocessing.py`](../../standardized_tabular_diffusion/preprocessing.py) | 集中式均值/众数缺失值边界 | 只在真实 train 上拟合；禁止目标/合成数据修复；状态、schema、配置、输入与输出均有指纹 |
| [`schemas/evaluation/`](../../standardized_tabular_diffusion/schemas/evaluation) | 十个 Draft 2020-12 线格式 schema | P1 规范线格式校验器，并随 wheel 打包 |
| [`resources/evaluation/`](../../standardized_tabular_diffusion/resources/evaluation) | 版本化指标、协议、评测器与来源身份资源 | 八个旧记录、两个 P2、两个 P3 和十一个 P4 记录均非正式 |
| [`configs/datasets/`](../../configs/datasets) | 已审阅的 Adult 与 Sick Dataset Profile | 仅属于诊断集合；当前均不具备正式资格 |
| [`cli.py`](../../standardized_tabular_diffusion/cli.py) | Registry/profile/result 检查、可选协议的 `evaluate-table` 与旧版命令 | P2 保持默认；P3/P4 显式选择，P4 必须提供 `--real-test` |
| [`pyproject.toml`](../../pyproject.toml) 与 [`core-ci.yml`](../../.github/workflows/core-ci.yml) | Python 3.11 打包、依赖组、测试边界、lint、类型检查和构建 | P0 在 Linux 上启用并通过；参考代码树不进入默认发现或分发包 |
| [`tests/evaluation/`](../../tests/evaluation) | 契约、结构、来源等价、Atomic Result、中断、bundle 与 CLI 测试 | P1 回归测试和 P2 直接权威测试按依赖与 marker 边界隔离 |

### 3.3 P4 实现后仍存在的缺口

- P2 已通过，并留存[权威 Linux/Python 3.11 证据](../evidence/evaluation/p2-shape-trend-run-31025796906.json)；后续门槛不得夸大这一诊断性声明。
- 两个 P2 指标仅为来源等价候选；均未达到 protocol-frozen、release-supported 或 Official Results 准入。
- P3 已通过，并留存[权威 Linux/Python 3.11 证据](../evidence/evaluation/p3-validity-run-31036844043.json)，但在协议冻结和发布审批前仍为诊断用途。
- P4 Local/Global Utility 已通过有限范围诊断工作流，并留存[工程证据](../evidence/evaluation/p4-utility-run-31053624769.json)。单独的[真实来源运行时 pilot](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json) 使用校验和锁定的 TabPFN 检查点，直接执行精确锁定的 TabEval 文件与真实 AutoGluon XGB/KNN/TabPFN 模型；分类和回归聚合等价均严格通过。Adult/Sick 目标覆盖、多种子稳定性、资源预算与准入仍待完成。
- 经批准的高阶 fidelity/privacy、效率、不确定性、兼容聚合和榜单发布仍未实现。
- Adult 与 Sick 是已审阅的诊断 profile，不是已冻结的 Universal Core Dataset Suite。
- Evaluator 与 hardware profile、兼容性分组、resume/cache 执行、不确定性和榜单发布仍属于后续阶段。
- 模型等价性证据本身不会授予 benchmark eligibility 或 release support。

## 4. 目标架构

具体文件名可以经审阅后调整，但职责边界必须保持稳定：

~~~text
standardized_tabular_diffusion/evaluation/
  contracts.py          # requests、contexts、atomic results、enums
  registry.py           # 指标定义和生命周期记录
  profiles.py           # 协议和 evaluator profile 加载
  engine.py             # 具备依赖关系的指标执行
  validation.py         # 表与契约校验
  bundle.py             # 原子写入、校验和、finalization、resume
  aggregation.py        # 只聚合兼容的 run/dataset
  metrics/
    fidelity/
    utility/
    validity/
    privacy/
    efficiency/
  backends/             # 经批准的权威指标来源隔离 wrapper

schemas/evaluation/     # 签入仓库的 JSON Schema
configs/evaluation/
  protocols/
  metrics/
  evaluators/
  hardware/
configs/datasets/       # 已审 Dataset Profile
tests/evaluation/
  fixtures/             # 小型、合成、可再分发表
  golden/               # 版本化的权威预期输出
~~~

依赖方向是单向的：适配器和 CLI 可以调用公开评测 API；指标模块可以依赖契约和隔离 backend；契约、registry 校验和 bundle 校验不得导入模型适配器或指标重依赖。

### 4.1 公开评测请求

评测请求至少必须标识：

- 解码后的合成表或不可变样本 artifact；
- Dataset Profile 标识符及校验和；
- 协议配置及版本；
- 请求的指标标识符和版本；
- 适用的 evaluator 与硬件配置；
- 可用时的模型/run 来源；
- 主体类型：适配器运行或外部合成表；
- 种子集合；
- 输出 bundle 位置；以及
- 资源与失败政策。

引擎必须在计算前解析全部身份。未知字段、缺少必需身份、不兼容的指标/profile 组合以及未经审阅的正式声明，必须在昂贵计算开始前校验失败。

### 4.2 执行图

指标依赖必须显式。例如：结构校验先于所有正式指标；可复用编码视图可以供多个指标使用；原始 TRTR/TSTR/Dummy 结果先于 Local Utility retention；每目标 ratio 先于 Global Utility；Atomic Result 先于聚合；只有通过校验的 incomplete bundle 才能 finalization。

每个节点记录内容寻址的输入、输出、实现版本、种子、状态、耗时和资源观测。只有全部身份输入一致时，resume 才能复用节点。

## 5. 交付顺序与依赖门

| 阶段 | 交付物 | 依赖 | 当前状态 | 退出门槛 |
|---|---|---|---|---|
| P0 | 可信开发基线 | 无 | 已通过 | 核心测试可在最小环境收集；仓库测试与参考测试已隔离 |
| P1 | 契约、registry、profile 与 incomplete bundle writer | P0 | 已通过；[Linux 证据已留存](../evidence/evaluation/p1-foundation-run-31018595264.json) | 无效契约可确定性失败；round-trip 与 schema 测试通过 |
| P2 | 首个垂直切片：外部表 -> 结构门 -> Shape/Trend -> finalized bundle | P1 | 已通过；[Linux 证据已留存](../evidence/evaluation/p2-shape-trend-run-31025796906.json) | 在 Linux/Python 3.11 上通过直接锁定来源等价和 bundle 校验 |
| P3 | 完整 Validity 子系统和显式预处理边界 | P2 | 已通过；[Linux 证据已留存](../evidence/evaluation/p3-validity-run-31036844043.json) | 无隐藏修复或缺失值修改；规则和失败测试通过 |
| P4 | Local 与 Global Utility | P1、P3 | 诊断门与有限范围真实来源运行时等价已通过；[工程](../evidence/evaluation/p4-utility-run-31053624769.json)和[运行时](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json)证据已留存；数据集规模准入待完成 | 原始 arms、状态语义、profile 身份、来源/公式验证和精确 pilot 聚合均通过 |
| P5 | 高阶 Fidelity 与经验 Privacy 工作包 | P2、P3 | 未开始 | 只有已解决并批准的指标推进；被阻止的指标保持排除 |
| P6 | 资源感知 orchestration、Efficiency、cache 与 resume | P2 | 未开始 | 阶段核算和复用完整性在声明的硬件配置下通过 |
| P7 | 数据集聚合、不确定性、兼容组和 leaderboard snapshot | 视情况依赖 P2-P6 | 未开始 | 不兼容结果无法合并；覆盖率和发布门通过 |
| P8 | Legacy 迁移、文档、打包、CI 和发布证据 | P0-P7 | 未开始 | 所声明发布类别的 public-preview 或 official-release 门通过 |

依赖稳定后，P3、P4、P5 和 P6 的部分工作可以并行。每个参与指标和数据集分别通过准入门之前，P7 不能用于发布排名。

## 6. 阶段工作包

### 6.1 P0 — 可信基线

任务：

- 为 Python 3.11 增加根打包元数据、显式 core/evaluation/model/development extras 和控制台入口。
- 配置 pytest 默认只收集本仓库拥有的测试。上游和 `research_inputs/` 测试只能通过显式 provenance/parity job 运行。
- 让顶层导入保持轻量；把可选模型和指标导入移到工厂之后，并在缺少 extra 时给出可操作提示。
- 建立格式化、lint、静态类型、schema 校验、单元测试和文档链接命令。
- 把当前 51 个仓库测试记录为迁移基线；将每个测试分类为 unit、integration、smoke 或 legacy-regression。
- 增加 Linux/Python 3.11 CI，仅安装 core 依赖并运行元数据、schema、CLI help 和 core 测试。
- 把 `research_inputs/` 视为不可变审阅输入，排除在打包、普通测试发现和运行时导入路径之外。

退出证据：

- clean checkout 可以安装 core 包；
- 导入契约、数据集元数据和 CLI help 不需要 PyTorch、AutoGluon、SDMetrics 或模型运行时；
- 默认 pytest 发现只包含已声明的仓库测试；以及
- 先前的收集失败已通过回归测试或 CI 配置检查得到防护。

### 6.2 P1 — 契约与身份基础

任务：

- 为 Dataset Profile、Metric Registry entry、协议配置、Evaluation Request、Atomic Result、阶段记录、manifest、metadata、summary 和 artifact index 实现有版本的 schema。
- 实现六种指标结果状态及稳定原因代码校验。
- 实现有限数值、原始/派生分离、方向、支持计数和聚合影响不变量。
- 创建数据驱动的 Metric Registry 加载器；把旧静态描述迁移为明确的 legacy 记录。
- 创建 Dataset Profile 加载器，以及把现有上游 `info.json` 导入非正式 profile 的工具。
- 定义协议解析、不可变身份哈希、canonical JSON、安全 YAML 加载和 bundle 相对路径校验。
- 实现带原子文件替换、事件日志和确定性内容 fingerprint 的 incomplete Run Result bundle writer。
- 增加列出和校验 metric record、Dataset Profile、协议配置和 result bundle 的 CLI 命令。

退出证据：

- schema 正例、反例、未知字段、版本、路径穿越、非有限值和 round-trip 测试通过；
- 等价请求具有相同 fingerprint，科学上不同的请求具有不同 fingerprint；
- 中断的 writer 留下可审计 incomplete bundle，而不会留下假的 finalized bundle；以及
- 缺少必需证据字段时不能推进 registry 生命周期状态。

完成证据（2026-08-05）：以上 P1 表面均已实现。专用只读 workflow 已在 Linux x86-64 与 Python 3.11.15 上通过，见 [GitHub Actions run 31018595264](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31018595264)；其[机器可读证据](../evidence/evaluation/p1-foundation-run-31018595264.json)已留存，SHA-256 为 `3013e913c58adf0c03c6ec30118879c522a87f4682d1cceb99f8778115c7da5a`。该证据只验证工程契约，没有执行科学指标，也不作来源等价声明。

### 6.3 P2 — 首个端到端垂直切片

范围刻意限制为：一个外部或适配器生成的单表、结构门、来源等价 Shape、来源等价 Trend、Atomic Result、摘要视图和 bundle finalization。

任务：

- 把 CSV、Parquet 和 DataFrame 输入解析为同一个规范语义表，不允许有损强制转换。
- 实现行数、列集合、唯一性、序列化、顺序和安全转换校验。
- 在隔离的评测依赖配置中包装锁定的 SDMetrics Column Shapes 与 Column Pair Trends report 行为。
- 为每个可评测列和列对保留一个 Atomic Result，包括输入计数和来源原始输出。
- 只计算来源定义的等列/等列对摘要以及单独命名的基准库报告视图。
- 实现 synthetic-vs-test、synthetic-vs-train 和版本化 real-vs-real 参考接口；尚未解决的参考构造保持 diagnostic。
- 生成全部必需 Run Result 文件，校验后最后写入校验和，并不可变 finalization。
- 增加不依赖模型训练的 `evaluate-table` 和 `validate-result` CLI 路径。

等价测试套件：

- 数值、分类、Boolean、datetime 和混合列对正常情形；
- 适用时覆盖常量、空、单行、缺失、未见类别、零范围和不同样本行数；
- report 级预处理与聚合对比直接锁定来源调用；
- 精确来源 revision、解析参数、warning、容差和状态；以及
- 与底层替代行为进行有意对比，防止在来源等价标识符下误用 Spearman 或 common-bin 混合列对行为。

退出证据：

- 直接权威调用和 benchmark 路径结果在批准的等价协议下吻合；
- 两个相同请求生成语义等价的 bundle 和 fingerprint；
- 结构失败阻止正式下游指标，但仍生成有效 incomplete/failed bundle；
- 每个请求的列和列对都有分母完整的记录；以及
- 高阶组件达到 protocol-frozen 前不输出总 Fidelity 分数。

### 6.4 P3 — Validity 与预处理边界

任务：

- 为 nullability、finiteness、整数语义、已审边界、类别、字符串格式和 datetime 范围实现每列硬规则。
- 实现带稳定标识符和适用性规则的已审跨列约束。
- 保留原始解码输出；如获允许，另行记录 evaluator 规范化或修复视图。
- 把数据 onboarding 拆分为原始注册、权利/来源审阅、schema 画像、显式预处理、split 生成和物化。
- 用必需的预处理请求和完整 cleaning report 替换静默删除与隐式填补。
- 所有学习型预处理器只在 train 上拟合；对学习状态和转换后 schema 计算校验和；政策变化产生新的 dataset-view 身份。
- 在讨论并批准相关数据集和策略之前，不选择插补策略。

退出证据：

- 手算规则、畸形 schema、有损转换、无约束和宽度敏感性测试通过；
- 测试证明 test 数据不能影响拟合的预处理状态；
- 原始 validity 不会因 evaluation-only 视图被修复而提高；以及
- 除显式声明的序列化转换外，单纯注册保持字节不变。

### 6.5 P4 — Utility

Local Utility 任务：

- 在同一个 held-out real test set 上实现完全一致的 Dummy、TRTR 和 TSTR arms。
- 经 pilot 审阅后冻结独立的分类和回归 evaluator profile。
- 为每个 evaluator、目标和种子保留 Macro-F1/RMSE 主原始值和适用的次要值。
- 把基线调整的 retention 实现为单独标识的 benchmark-derived 结果。
- 编码缺失类别、恒定目标、TRTR 相对 Dummy 过弱、predictor 失败和资源失败，不得删除任务。

Global Utility 任务：

- 在聚合层实现经审阅、由 TabStruct Equation 4 描述的 Global Utility ratio 公式：分类目标使用 Balanced Accuracy ratio，数值目标使用逆 RMSE ratio，最后进行等目标均值。该论文只作为公式来源，不是运行时或代码依赖。
- 把 Full-tuned、Tiny-default 和任何锁定 TabEval predictor profile 保持为不同指标身份和兼容组。
- 默认排除 identifier；其他每个目标排除都必须在 Dataset Profile 中给出理由。
- 绝不裁剪大于一的 ratio，也绝不静默省略零或非有限分母。
- 优先包装批准的权威 predictor 实现。若必须修改源码，实现暂停并进入审阅。

退出证据：

- 手算、原始 arm 不变量、类别支持失败、恒定目标、多分类、回归边界和全部目标分母测试通过；
- 随机算法的可复现性和容差政策已预声明；
- 声称来源等价的所选 profile 通过来源等价验证；以及
- Local Utility 与 Global Utility 保持为不同输出和子榜单。

### 6.6 P5 — 高阶 Fidelity 与经验 Privacy

高阶任务：

- 在 C2ST 影响 Fidelity 前，通过 pilot 冻结预处理、discriminator、split、平衡、种子、不确定性和 AUROC-complement 变换。
- 在来源专用 diagnostic 标识符下保留 GReaT RF discriminator accuracy。
- 在 mixed-table embedding 和积分行为解决前，Integrated Alpha-Precision/Beta-Recall 保持 experimental。

Privacy 任务：

- 分别实现精确 train collision 和 synthetic internal duplication diagnostic。
- 包装锁定的 SDMetrics DCR 距离；添加单独标识的 held-out 校准和分布摘要。
- 在冻结 privacy suite 前定义并验证至少一个 membership-inference threat model。
- 只有 Dataset Profile 具备已审 sensitive/quasi-identifier role 和已批 threat model 时，才加入 attribute inference。
- 在 paper/code 差异裁决前保持 Authenticity excluded。
- 在 Delta Presence 的语义和失败行为得到科学解决前，将其排除在正式评分之外。

退出证据：

- 每个 privacy 输出都标识 attacker knowledge、member/non-member 构造、表示、模型和方向；
- collision 与 DCR 边界/等价套件通过，包括 null 和 zero-range 行为；
- 任何 privacy diagnostic 都不被描述为形式化 privacy guarantee；以及
- 未解决的指标只以明确 experimental 或 excluded 记录存在。

### 6.7 P6 — Orchestration、Efficiency、cache 与 resume

任务：

- 把执行扩展为 prepare、train、sample、validate、evaluate、aggregate 和 report 阶段记录。
- 记录 wall time、可靠时的 CPU time、峰值 RAM、峰值 accelerator memory、行吞吐、请求/实际样本数、warm-up 政策和被排除的 setup time。
- 定义 hardware profile，并禁止跨 profile 的 Efficiency 排名。
- 在隔离进程或环境中运行可选指标 backend，并设置显式时间、内存和失败边界。
- 实现内容寻址的阶段 cache 和 resume，不得改变结果身份或隐藏此前失败。
- 日志结构化并删除 secrets 和不安全路径；把确定性科学输出与时间戳、host-specific 诊断分开。

退出证据：

- 强制 timeout、模拟 out-of-memory、中断、retry、stale-cache 和 partial-success 测试通过；
- cache 复用证明所有身份输入吻合，并在阶段 metadata 中可见；
- Efficiency 测量在命名硬件配置下、声明容差内可复现；以及
- 一个可选指标失败不会删除已完成的 Atomic Result。

### 6.8 P7 — 聚合与榜单发布

任务：

- 聚合前校验每个输入 bundle 并构造兼容组。
- 按政策规定顺序把 Atomic Result 聚合到 run、seed、dataset、suite 和 snapshot。
- 实现不确定性区间、两两完整性、覆盖率、失败/未定义分母核算、并列和纠正/supersession 记录。
- 分离 Native 与 Standardized Tuning comparison track。
- 强制执行 Official、Partial/Diagnostic 和 Community publication class 及其证据要求。
- 从同一批已校验记录生成不可变 Leaderboard Snapshot bundle、人类可读表和机器可读导出。
- 禁止展示代码重新计算科学值或改变排序规则。

退出证据：

- 人为构造的不兼容协议、split、预处理、指标、种子、硬件和 tuning 记录无法合并；
- 缺失或失败贡献不能提高覆盖率或从分母中消失；
- 可以从声明的 bundle 确定性重建 snapshot；以及
- 没有独立准入记录的模型、数据集或指标不能进入 Official Results。

### 6.9 P8 — 迁移与发布

任务：

- 把 `standardized_summary.json` 和 `tabstruct-aligned-v1` 标记为 legacy，冻结其 schema，并在声明的迁移窗口内保留只读 importer。
- 当缺少必需 Atomic Result 证据时，绝不能把 legacy 摘要转换为 Official Result。
- 让适配器评测通过新引擎，同时在安全情况下保留 train/sample 行为和现有 artifact 位置。
- 用已批准的模型状态维度和证据记录替换旧的 `implemented` inventory 语言。
- 更新英文 README、教程、示例、架构、metric card、dataset card、故障排除和贡献者指南；按计划提供中文审阅翻译。
- 在完成各自审计后，增加 license、third-party notice、citation、contributor acknowledgement、security policy、code of conduct 和发布 checklist。
- 在 Linux/Python 3.11 上测试 clean installation、table-only evaluation、一个适配器 smoke run、result validation 和 diagnostic comparison。

退出证据：

- legacy 与新输出不会因文件名、schema、CLI 标签或文档而混淆；
- clean-checkout quickstart 不依赖开发者本地路径或未声明数据并通过；
- 发布声明与实际生命周期、资格和支持记录一致；以及
- Repository Quality Standard 中每个适用发布门都有基于证据的决定。

## 7. 验证策略

### 7.1 必需测试层

| 层 | 目的 | 必需示例 |
|---|---|---|
| Contract | 强制 schema 和不变量 | version、enum、未知字段、有限值、path、hash |
| Formula unit | 校验 benchmark 数学 | 可手算正常与边界情况 |
| Source parity | 校验权威行为 | 在共享 fixture 上直接锁定调用对比 wrapper |
| State and negative | 防止有利的静默失败 | empty、constant、missing class、timeout、dependency failure |
| Integration | 校验子系统边界 | profile -> table -> metric -> bundle -> validator |
| End-to-end | 校验用户工作流 | Linux/Python 3.11 上的外部表和一个真实适配器 |
| Determinism | 校验科学身份 | 重复 seed、进程隔离、cache reuse、canonical serialization |
| Migration | 保留有意兼容性 | legacy reader、deprecation warning、不得正式升级 |
| Security and publication | 保护发布 artifact | path traversal、unsafe YAML、secret/path redaction、manifest allowlist |

Mock 测试适合验证控制流，但不能满足 source parity、真实 smoke、科学验证或 release-support 门槛。

### 7.2 Golden fixture 政策

Golden fixture 必须小型、合成、可再分发、可人工检查且有版本。每个 fixture 记录权威来源 revision、依赖 lock、调用参数、原始预期输出、允许容差以及任何平台容差的理由。更新 golden value 必须有审阅记录；依赖升级不会自动授权重新生成。

### 7.3 CI 分区

- `core`：不含模型或指标重依赖；运行契约、profile、CLI 元数据和 bundle 校验。
- `evaluation-unit`：确定性指标公式和状态测试。
- `source-parity`：每个权威 backend 使用隔离锁定环境。
- `adapter-smoke`：选定真实适配器；必要时定期或按硬件标签运行。
- `release`：clean installation、文档链接、artifact allowlist、安全/许可证证据和 quickstart。

普通测试禁止网络访问。需要下载的测试使用预批准缓存输入并校验 checksum。

## 8. 证据与审阅控制

每个工作包必须产出：

- 实现 commit 和已变更公开契约清单；
- 测试与不可变 fixture；
- 解析后的依赖 lock 和许可证记录；
- 指标或模型生命周期更新；
- 当公式或语义变化时的科学 reviewer 决定；
- 兼容性影响与迁移说明；
- 文档与局限更新；以及
- 针对退出门槛的机器可读评估。

源码修改、新插补政策、数据集硬约束、指标变换、predictor profile、threat model、聚合权重和正式阈值都是审阅检查点。做出或改变其中任何选择前，实施暂停并进入讨论。

## 9. 阻止发布的待定事项

路线图可以绕开下列事项继续推进，但在解决前，受影响输出不能成为正式结果：

- 数据集专用插补策略和 missing-indicator 政策；
- 正式 real-vs-real 参考构造；
- Trend 来源忠实版本与单独命名 common-bin 变体的角色；
- C2ST evaluator 与不确定性配置；
- Local Utility evaluator profile 以及是否提供 clipped retention 视图；
- Global Utility predictor profile；
- Integrated Alpha-Precision/Beta-Recall embedding；
- membership- 和 attribute-inference threat model；
- Authenticity paper/code 差异；
- Delta Presence 科学角色；
- support、容差、宽表列对采样和资源限制的 pilot 阈值；以及
- Core Dataset Suite、Core Model Set 和 hardware profile。

这些事项不能成为编造临时正式默认值的理由。在获得批准前，对应 registry 记录停留在适当的早期生命周期阶段，其数值只能是 diagnostic 或 excluded。

## 10. 里程碑与完成定义

### M1 — 评测基础

P0 和 P1 已通过，M1 的 Linux/Python 3.11 证据已留存。P2 现在也为两个非正式的 source-parity-validated 指标记录留存了权威证据。

### M2 — 首份可信报告

P2 已在 [GitHub Actions run 31025796906](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31025796906) 通过。用户可以评测兼容的合成表，并获得包含结构校验、Shape 和 Trend 证据的 finalized、validated bundle。在仓库级发布门也通过的前提下，这是第一个可以支持 public preview 的评测切片。

### M3 — 基准维度

批准子集的 P3 至 P6 通过。Validity、Utility、选定 Privacy diagnostic、已冻结时的高阶 Fidelity 和 Efficiency 具备显式生命周期记录与失败语义。

### M4 — 可发布基准库

声明发布类别的 P7 和 P8 通过。只有全部参与模型、数据集、指标、协议记录和 bundle 也分别通过独立门槛后，才能发布 Official ranking。

代码存在、mock 测试通过或一台机器生成表格都不代表实施完成。只有适用阶段退出门、生命周期证据、兼容性检查、文档和发布评估全部完成，才算完成。

## 11. 紧接着的实现增量

P4 现在使用经过 P3 审阅的不可变模型视图，以及单独由校验和绑定的 held-out real test。其有限范围 Linux/Python 3.11 来源运行时 pilot 已通过，来源/适配器聚合值严格一致，并真实训练了 XGB/KNN/TabPFN。下一道门是对 Adult/Sick 已审阅目标与多个种子进行数据集规模验证，留存高基数省略、稳定性、墙钟时间、内存和失败证据；之后才能决定是否冻结 profile。P5 不得把仍为诊断状态的 P4 profile 当成 Official Results 组件。

## 12. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [P3 有效性与预处理指南](P3_VALIDITY_AND_PREPROCESSING.zh-CN.md)
- [P4 Local 与 Global Utility 指南](P4_UTILITY.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [指标来源审阅](METRIC_SOURCE_REVIEW.zh-CN.md)
- [数据集配置规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [结果规范](RESULT_SPECIFICATION.zh-CN.md)
- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [仓库质量标准（英文规范原文）](../QUALITY_STANDARD.md)
