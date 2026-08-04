# 结果规范

英文规范：[RESULT_SPECIFICATION.md](RESULT_SPECIFICATION.md)

- 状态：设计基线
- 结果 schema 版本：0.1.0
- 最后更新：2026-08-03

本文件是英文规范的对应中文译文。若两者存在歧义，以英文规范为准。

## 1. 目的

本规范定义基准执行、原子指标、汇总、来源、失败和发布产物的持久化机器可读表示。

其目标是：

- 完整科学追溯；
- 明确失败语义；
- 安全恢复和缓存复用；
- 独立验证；
- 静态榜单生成；
- 在不静默改写历史的情况下保持长期兼容。

## 2. 结果层级

结果分为四层。

### 2.1 Atomic Result

某个声明范围内的一次指标观测，例如列、列对、目标、评测器、攻击、计时阶段、群体或种子。

### 2.2 Run Result

一个模型、一个数据集及 view、一个划分、一个 Comparison Track、一份最终配置和一个生成随机种子。

Run Result 包含 Atomic Results 和阶段证据。当训练、采样、验证或评测失败时，它仍然作为失败 bundle 有效。

### 2.3 Dataset Summary

针对一个模型—数据集组合，对全部必需 Run Results 做带版本聚合，通常包括五个生成随机种子。它引用不可变 Run Results，而不是复制或替换它们。

### 2.4 Leaderboard Summary

针对声明的数据集套件做带版本聚合。它引用 Dataset Summaries，并包含榜单快照所需的兼容性、覆盖率、不确定性和排名证据。

任何高层聚合都必须能够从其引用的低层结果复现。

## 3. 身份模型

### 3.1 运行身份

每个 Run Result 必须包括：

- run_id：不可变不透明标识；
- run_fingerprint：兼容性定义输入的确定性哈希；
- repository_commit；
- protocol_version；
- result_schema_version；
- metric_registry_version；
- 数据集身份；
- 划分身份；
- 模型与适配器身份；
- Comparison Track；
- 最终配置哈希；
- 生成随机种子；
- 评测器 profile；
- 环境 profile；
- 硬件 profile。

run_id 区分执行尝试。run_fingerprint 标识声称使用相同科学输入的执行。多个 run_id 可以共享一个 fingerprint，但除非协议显式选择，否则重复尝试不得被静默平均。

### 3.2 确定性 fingerprint

Fingerprint 输入必须使用有文档的规范序列化，并包含所有可能改变科学含义的字段。它必须排除 wall-clock 开始时间、主机名和日志路径等易变字段。

具体 fingerprint 算法和规范化方式随 schema 版本管理。

### 3.3 标识符

稳定标识符应使用小写 ASCII，并可安全用于可移植路径。显示名称和本地化标签单独存储。

Bundle 内路径必须是 POSIX 风格相对路径。禁止绝对开发者路径、父目录穿越和环境专属 home 路径。

## 4. Bundle 类型

bundle_type 为下列之一：

- run；
- dataset_summary；
- leaderboard_snapshot。

每个 bundle 在 finalization 后不可变。纠正会创建具有新标识和 supersedes 引用的新 bundle。

## 5. Run bundle 布局

已 finalization 的 Run Result 使用：

~~~text
result_bundle/
├── manifest.json
├── metadata.json
├── config.yaml
├── environment.json
├── metrics.parquet
├── summary.json
├── checksums.sha256
├── logs/
│   ├── events.jsonl
│   ├── stdout.log
│   └── stderr.log
├── stages/
│   ├── prepare.json
│   ├── train.json
│   ├── sample.json
│   ├── validate.json
│   ├── evaluate.json
│   ├── aggregate.json
│   └── report.json
└── artifacts/
    └── index.json
~~~

只有 manifest 明确说明不适用时才可以省略可选文件。失败运行仍然写入 metadata、config、environment、阶段记录、日志、校验和，以及失败前产生的全部 Atomic Results。

## 6. 序列化规则

- JSON 和 JSON Lines 使用 UTF-8。
- YAML 通过安全 loader 加载，且不得包含可执行 tag。
- 时间戳使用 UTC RFC 3339 格式。
- 数值指标列使用 64 位浮点数。
- 序列化 JSON 不得包含 NaN、正无穷或负无穷。
- 未计算的数值为 null。
- 枚举值使用 schema 定义的精确小写标识。
- 文件路径相对于 bundle 并使用正斜杠。
- 人类可读消息不能替代稳定原因代码。
- 结构化文件声明自身 schema 版本。

用于哈希的规范 JSON 采用排序对象键、规定的 Unicode 规范化形式，并且没有无意义空白。

## 7. manifest.json

Manifest 是 bundle 索引。它必须包含：

- bundle_id；
- bundle_type；
- bundle_schema_version；
- created_at；
- finalized_at；
- finalization_status；
- 运行或汇总身份；
- 必需和可选文件清单；
- 文件媒体类型；
- 引用的外部产物；
- supersedes 与 invalidates 关系；
- 生产者仓库 commit；
- 校验和算法。

finalization_status 为下列之一：

- incomplete；
- finalized；
- invalidated；
- withdrawn。

Incomplete bundle 不得进入榜单。

## 8. metadata.json

metadata.json 记录科学身份和执行结果。

必需部分为：

- identity；
- protocol；
- dataset；
- model；
- implementation；
- comparison_track；
- seeds；
- evaluator；
- execution；
- coverage；
- provenance；
- review；
- status。

### 8.1 数据集身份

数据集元数据包括：

- dataset_id；
- dataset_version；
- dataset_view；
- dataset_profile_version；
- split_id；
- raw、canonical、split 与 preprocessing 校验和；
- 每个划分的行列数；
- feature、target、identifier、sensitive 和 quasi-identifier 角色。

### 8.2 模型身份

模型元数据包括：

- model_id；
- model_family；
- 上游仓库或包；
- 不可变版本；
- reproduction target；
- modification status；
- patch 标识；
- adapter 标识与版本；
- validation level；
- eligibility track；
- support level。

### 8.3 执行结果

执行元数据包括：

- 请求动作；
- 开始和结束时间戳；
- 终止阶段；
- 运行状态；
- 请求与实际合成行数；
- 超时和资源限制；
- 中断与恢复祖先关系；
- 警告；
- 失败类别和原因代码；
- 产物引用。

运行状态为下列之一：

- success；
- partial；
- failed；
- cancelled；
- invalidated。

## 9. config.yaml

config.yaml 是执行使用的完整最终配置，不只是用户输入。

它必须包括：

- 协议和 Comparison Track；
- 模型与数据集选择；
- 应用默认值和覆盖值后的全部模型参数；
- 预处理与后处理配置；
- 请求样本数；
- 生成与评测器随机种子；
- 指标选择与版本；
- 适用时的调参 profile；
- 资源限制；
- 硬件请求；
- 输出与保留政策；
- 配置来源与优先级证据。

除非 schema 明确允许扩展，未知配置字段必须导致验证失败。

不得嵌入密钥。经过脱敏的 secret reference 可以记录需要外部凭证，但不得记录其值。

## 10. environment.json

environment.json 记录执行环境：

- 操作系统、发行版、kernel 与架构；
- Python 实现与版本；
- 已安装依赖锁标识和包清单；
- 加速框架；
- CPU 型号、逻辑与允许线程数、RAM；
- GPU 型号、设备数、VRAM、驱动、CUDA 与相关库；
- locale 与时区；
- 确定性设置；
- 容器或环境镜像身份；
- 对计算有重要影响且已移除密钥的环境变量；
- 仓库 dirty-state 指示；
- 官方硬件 profile 标识。

官方结果应从干净仓库状态产生。如果受控例外允许 dirty state，则必须记录精确 patch 哈希和审阅证据。

## 11. metrics.parquet

metrics.parquet 是 Atomic Results 的规范表。每一行代表一个 scope 的一个标量观测。大型分布或矩阵存储为单独的内容寻址产物，并在表中保存汇总标量和引用。

### 11.1 必需列

| 列 | 类型 | 说明 |
|---|---|---|
| result_schema_version | string | 原子结果 schema 版本 |
| run_id | string | 父 Run Result |
| protocol_version | string | 评测协议 |
| dataset_id | string | 稳定数据集标识 |
| dataset_version | string | 数据集版本 |
| dataset_view | string | 带版本 view 标识 |
| split_id | string | 官方划分 |
| model_id | string | 稳定模型标识 |
| comparison_track | string | native 或 standardized-tuning |
| generation_seed | int64 | 生成器随机种子 |
| metric_id | string | Metric Registry 标识 |
| metric_version | string | 精确指标版本 |
| dimension | string | 评测维度 |
| scope_type | string | column、pair、target、evaluator、attack、group、phase 或 dataset |
| scope_id | string | scope 类型内的稳定标识 |
| evaluator_id | string | 可空评测器标识 |
| evaluator_version | string | 可空评测器版本 |
| task_type | string | 可空分类或回归任务 |
| state | string | 结构化指标状态 |
| raw_value | float64 | 可空原始值 |
| normalized_value | float64 | 可空归一化值 |
| aggregate_contribution | float64 | 可空预声明贡献 |
| reference_value | float64 | 可空基线或目标 |
| unit | string | 可空单位 |
| raw_direction | string | maximize、minimize、target、distributional 或 descriptive |
| weight | float64 | 预声明聚合权重 |
| n_reference | int64 | 参考样本数 |
| n_synthetic | int64 | 合成样本数 |
| n_valid | int64 | 有效观测数 |
| n_excluded | int64 | 排除观测数 |
| reason_code | string | 可空稳定原因 |
| reason_detail | string | 可空诊断详情 |
| warning_codes | list[string] | 结构化警告 |
| artifact_ref | string | 可空 bundle 相对产物 |
| computed_at | timestamp | UTC 完成时间 |

Schema 演化可以在兼容的 minor schema 更新中增加可空列。删除字段或改变字段含义需要新的不兼容 schema 版本。

### 11.2 数值不变量

当状态为 computed：

- 必需原始或结构化输出必须存在；
- 每个标量必须有限；
- 计数必须非负；
- weight 必须匹配指标契约；
- direction 和 unit 必须匹配 Metric Registry。

当状态不是 computed：

- raw_value、normalized_value 和 aggregate_contribution 必须为 null；
- reason_code 必须存在；
- 可知时仍必须报告分母计数。

## 12. summary.json

summary.json 是可复现视图，不是事实源。

它包含：

- 运行身份和终止状态；
- 结构与内容有效性汇总；
- 维度组成分数；
- Local 与 Global Utility 汇总；
- 隐私风险诊断；
- 效率测量；
- 指标状态与分母计数；
- 警告与失败汇总；
- 原子行或产物链接；
- 聚合实现与版本；
- 该次运行是否具有数据集聚合资格的声明。

每个汇总值都必须能从 metrics.parquet、阶段证据和带版本聚合契约复现。禁止人工编辑。

## 13. 阶段记录

每个阶段记录包括：

- 阶段名称与版本；
- 状态；
- 依赖阶段标识；
- 输入 fingerprint；
- 最终动作；
- 开始和结束时间戳；
- 耗时；
- 适用时的进程退出码；
- 日志引用；
- 输出清单与校验和；
- 警告；
- 失败类别与原因；
- 缓存决策；
- 重试次数；
- 恢复祖先关系。

阶段状态为下列之一：

- pending；
- running；
- succeeded；
- failed；
- skipped；
- cancelled；
- invalidated。

Skipped 阶段需要稳定原因。没有兼容缓存证据时不得将其视为成功。

## 14. 日志

events.jsonl 是规范结构化事件日志。适用时，stdout.log 和 stderr.log 保留外部进程流。

日志必须：

- 使用时间戳和严重级别；
- 标识阶段与组件；
- 保留上游退出状态与工作目录上下文，但不暴露不安全绝对路径；
- 脱敏凭证与 token；
- 避免嵌入完整受限数据集或合成行；
- 记录截断；
- 在 bundle 移动后仍然有用。

人类可读日志不能替代结构化失败元数据。

## 15. 产物

artifacts/index.json 记录每个重要产物：

- artifact_id；
- 角色；
- 媒体类型；
- 相对路径或批准的外部 URI；
- 字节大小；
- 校验和；
- 生产阶段；
- 保留类别；
- 发布类别；
- 许可证或数据权利分类；
- 适用时的加密或访问要求。

默认不包含大型 checkpoint、真实数据、逐行合成数据和攻击轨迹。外部产物引用必须不可变或内容寻址，并说明访问预期。

不得仅为了检查结果身份、指标或汇总而要求使用不安全对象反序列化格式。

## 16. 校验和与 finalization

checksums.sha256 以有文档的确定性顺序列出除自身外每个已 finalization 普通文件，并使用 bundle 相对路径。

Finalization 要求：

1. 所有必需文件存在；
2. schema 验证通过；
3. 跨文件身份一致；
4. 每个本地产物引用存在；
5. 将不可变 bundle 标识、finalized 时间戳和 finalized 状态写入 manifest；
6. 为除校验和文件自身外的每个已 finalization 普通文件生成校验和；
7. 校验和通过；
8. 哈希后没有已 finalization 文件发生变化。

如果 finalization 失败，bundle 保持 incomplete，且没有发布资格。

## 17. 缓存与恢复

只有声明的全部输入 fingerprint、代码与依赖身份、配置、数据集、随机种子和上游身份均匹配时，阶段输出才可以复用。

缓存条目必须记录：

- 生产者运行与阶段；
- 完整兼容键；
- 产物校验和；
- 创建时间；
- 验证时间；
- 保留政策；
- 被撤销时的失效原因。

Resume 创建链接到祖先的新执行尝试。它不得改写祖先的日志或元数据。

## 18. Dataset Summary bundle

Dataset Summary 引用全部必需 Run Result bundle ID 与 fingerprint。它包含：

- 模型—数据集身份；
- 预期与观测随机种子；
- 接受与拒绝的运行及原因；
- 逐种子数值；
- 数据集级汇总；
- 不确定性；
- 状态与覆盖计数；
- 兼容性验证；
- 聚合版本；
- 官方准入状态。

Fingerprint 相同的重复尝试需要显式选择规则。即使选择了有效重试，失败尝试也保持被引用。

## 19. Leaderboard Snapshot bundle

Leaderboard Snapshot 引用 Dataset Summaries，并记录：

- 数据集套件版本；
- 合格模型集合；
- 协议与 Comparison Track；
- 指标和结果 schema 版本；
- 效率视图适用的硬件 profile；
- 聚合与 bootstrap 配置；
- 覆盖门槛；
- 排名与并列组；
- 失效或排除结果引用；
- 发布资产；
- 审阅者批准。

结构化快照是静态 HTML、Markdown、CSV、JSON 和 Parquet 发布视图的数据源。

## 20. 验证与兼容性

项目必须提供 result-bundle 验证器，检查：

- 文件与 schema 完整性；
- 枚举与类型正确性；
- 有限数值规则；
- 校验和；
- 跨文件身份；
- Metric Registry 兼容性；
- Dataset Profile 兼容性；
- 协议兼容性；
- 聚合重算；
- 禁止的绝对路径；
- 密钥模式；
- 不安全产物类型；
- 覆盖门槛；
- 发布分类。

验证输出本身是带版本证据，并链接到准入审查。

## 21. 保留、发布与隐私

保留类别包括：

- 永久证据；
- 发布证据；
- 可复现缓存；
- 临时诊断；
- 受限。

默认发布元数据、汇总指标、配置、环境、校验和与经批准日志。真实数据、逐行合成数据、敏感攻击输出和大型模型产物需要独立的权利与隐私审查。

外部产物的删除或到期不得使已发布数值来源产生误导。Bundle 记录复现是否依赖已不再保留的产物。

## 22. Schema 演化

兼容新增使用新的 minor schema 版本。不兼容字段含义、必需字段删除、身份变化或序列化变化使用新的 major schema 版本。

迁移工具必须：

- 保留原始 bundle；
- 记录源与目标 schema 版本；
- 保持确定性；
- 报告有损字段；
- 写入新 bundle；
- 重新计算校验和；
- 绝不编造缺失科学证据。

## 23. 最小元数据示例

~~~json
{
  "identity": {
    "run_id": "run-example",
    "run_fingerprint": "sha256:pending",
    "result_schema_version": "0.1.0"
  },
  "protocol": {
    "protocol_version": "benchmark-v1",
    "metric_registry_version": "pending"
  },
  "dataset": {
    "dataset_id": "example",
    "dataset_version": "1",
    "dataset_view": "canonical",
    "split_id": "official-v1"
  },
  "model": {
    "model_id": "example-model",
    "adapter_version": "pending"
  },
  "comparison_track": "native",
  "seeds": {
    "generation_seed": 0
  },
  "status": {
    "run_status": "partial",
    "terminal_phase": "evaluate"
  }
}
~~~

本示例仅用于说明，不是完整有效的 bundle。

## 24. 相关规范

- [评测协议](EVALUATION_PROTOCOL.zh-CN.md)
- [榜单政策](LEADERBOARD_POLICY.zh-CN.md)
- [指标治理](METRIC_GOVERNANCE.zh-CN.md)
- [数据集 Profile 规范](DATASET_PROFILE_SPEC.zh-CN.md)
- [实施路线图](IMPLEMENTATION_ROADMAP.zh-CN.md)
- [仓库质量标准](../QUALITY_STANDARD.md)
