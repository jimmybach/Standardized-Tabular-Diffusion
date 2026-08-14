# P7 聚合与榜单发布

状态：工程基线已实现。本文规定 P7 代码契约；更上位的科学政策仍以 [LEADERBOARD_POLICY.zh-CN.md](LEADERBOARD_POLICY.zh-CN.md) 为准。

## 1. 范围与声明边界

P7 把经过校验并 finalized 的 Run Result bundle 转换为不可变 Leaderboard Snapshot bundle。它实现：

- fail-closed 输入与兼容性校验；
- 已声明的 Atomic Result → run → generation seed → dataset → suite 聚合层级；
- 保留 dataset/seed 结构的确定性百分位不确定性区间；
- 覆盖率、失败 seed、缺失 seed 与两两完整性分母；
- Native 与 Standardized Tuning track 分离；
- Official、Partial/Diagnostic 与 Community publication class；
- 独立准入记录与经审阅的纠正记录；以及
- 从同一个结构化 snapshot 生成 JSON、CSV、HTML 与 Markdown。

P7 本身不会让任何结果自动成为 Official。它不会把 Fidelity、Utility、Validity、Privacy 和 Efficiency 合成一个总分，不会发布两两优越性结论，不会跨硬件/软件 profile 归一化，也不会猜测缺失证据。

## 2. 冻结的聚合契约

一份 Snapshot Request 只选择一个精确的 metric、protocol、dataset suite、comparison track、publication class、seed 分母和聚合程序。

P7 v1 层级为：

1. 校验每个 finalized Run Result bundle，并读取请求指标对应的 Atomic Result。
2. 在一个 run 内，只对指标明确声明的 `aggregate_contribution` 求和。正权重必须精确合计为一；必需项失败、贡献缺失、非有限数值或权重不完整都会使 run 成为未计算状态。
3. 在一个 dataset 内，对已计算的 generation-seed 分数等权平均。失败和缺失 seed 始终保留在固定预期分母中，绝不会提高覆盖率。
4. 在 suite 内，对可用 dataset summary 进行等 dataset 权重宏平均。行数、predictor 数、target 数和 seed 数都不能让某个 dataset 获得额外权重。
5. 发布结构化 snapshot，并只使用其中已保存的 leaderboard row 派生所有展示文件。

Partial/Diagnostic 可以展示基于可用贡献的聚合值，但必须同时明确不完整覆盖率并保持 Official rank 为空。Official 必须完整覆盖。

## 3. 兼容性边界

P7 绝不静默混合异质结果。所有 active 输入必须在适用身份上精确一致：

- protocol ID、version 与 checksum；
- metric ID、version、dimension 与 direction；
- comparison track；
- evaluator profile；
- hardware profile；
- software/environment checksum；
- Atomic Result schema version；以及
- 每个 dataset 内的 Dataset Profile、dataset view、split 与 dataset version。

因此，Native 与 Standardized Tuning 必须生成不同 snapshot。即使展示指标名称相同，不同 evaluator、hardware、software、preprocessing view 或 split 的结果也不能合并。

经审阅的 invalidation 可以把一个不兼容 bundle 保留在 snapshot 输入历史中，同时把它从 active 聚合集合中排除。被作废 bundle、原因、审阅者、证据和替代身份仍完整可审计。

## 4. 发布类别

### 4.1 Official Results

Official Snapshot Request 必须声明恰好五个 generation seed，并启用已冻结的并列规则。每个 model/dataset 单元必须具备五个全部已计算的 run。此外，还需要以下对象的 active approved admission record：

- repository release；
- dataset suite；
- model；
- dataset；
- metric；
- protocol；
- comparison track；以及
- 每个被接受的 Run Result bundle。

metric registry entry 还必须是 `release-supported`、允许 Official Results，并具有 approved release decision。任何缺失、rejected、withdrawn、含糊或已被 supersede 的准入都会 fail closed。P7 机制的实现不会自动提升任何现有结果。

Official rank 使用未四舍五入的完整精度分数。满足已冻结“实用等价边界 + 区间重叠”规则的条目属于同一个 tie group，并共享该组占据的首个名次。组内点估计顺序仅以 `diagnostic_order` 保留，作用只是导航。

### 4.2 Partial/Diagnostic Results

该类别接收有科学用途但不完整的证据。它发布分数、不确定性、覆盖率、失败计数与 diagnostic ordering，但所有 `rank` 均为 null。工程检查、pilot、不完整 suite 和等待准入的结果默认使用此类别。

### 4.3 Community Results

Community 同样不得产生 Official rank，并且 Snapshot Request 必须额外提供提交者、源码仓库和不可变 revision 来源。Community 结果只有经过独立 Official 流程后才可能升级。

## 5. 不确定性、并列与两两记录

v1 Snapshot Request 记录 bootstrap 方法、重复次数、置信水平与随机种子。P7 从科学身份派生确定性 salt，因此相同声明输入的重复重建会产生逐字节一致的结构化和展示文件。

- Dataset 区间对 generation seed 进行 bootstrap。
- Suite 区间先重采样 dataset，再在抽中的 dataset 内重采样 seed。
- 没有已计算值时仍保留完整区间配置，并把上下界记为 null。

两两记录公开固定的预期 dataset-seed 单元数、实际配对单元数、完整性比例和配对原始差值均值。`superiority_claim` 固定为 `not-issued`；视觉顺序不是总体优越性的证据，也不暗示已执行 multiple-comparison procedure。

## 6. 准入与纠正记录

Admission record 是独立审阅产物，不是 model run 自行写入的标志。其 subject identity 必须严格使用对应 subject type 的精确字段集合。每条记录包含 decision、适用 publication class、reviewer、UTC 审阅时间、证据引用和被 supersede 的旧 decision ID。

证据引用只能是安全的仓库相对路径或不含凭据的 HTTPS URL。绝对路径、父目录穿越、Windows 路径、非 HTTPS URL 与嵌入凭据都会被拒绝。

Correction record 支持：

- `invalidate`：从 active 聚合中删除受影响 bundle；或
- `supersede`：使用同一 model/dataset/seed slot 且处于同一科学兼容组的一份经审阅 bundle 替换它。

禁止出现重复 active attempt。P7 绝不会自动选择最新、最高分或其他更有利的 attempt。

## 7. Snapshot Request

Request 是由 `snapshot-request.schema.json` 校验的版本化 JSON 对象。关键字段包括：

- 精确 repository release、protocol、dataset suite 与 metric 身份；
- 一个 publication class 与 comparison track；
- 已排序、互不重复的预期 generation seed；
- 冻结的 v1 run/seed/dataset 聚合规则；
- 确定性的 hierarchical-percentile bootstrap 配置；
- tie method 与 equivalence margin；
- exact evaluator/hardware compatibility policy；以及
- Community submission provenance，非 Community 类别则为 null。

未知字段会被拒绝。Official request 如果不是恰好五个 seed，或禁用了 tie method，会在读取 bundle 前直接失败。

## 8. 命令行流程

安装 P7 依赖：

```powershell
python -m pip install ".[leaderboard]"
```

构建新的不可变 snapshot；按需重复 `--bundle`、`--admission` 和 `--correction`：

```powershell
std-tabular-diffusion build-leaderboard `
  --request path/to/snapshot-request.json `
  --bundle path/to/run-seed-1 `
  --bundle path/to/run-seed-2 `
  --admission path/to/admission.json `
  --correction path/to/correction.json `
  --output artifacts/leaderboard-snapshot
```

独立校验已发布 snapshot：

```powershell
std-tabular-diffusion validate-leaderboard `
  --snapshot artifacts/leaderboard-snapshot
```

builder 会拒绝覆盖非空输出目录。

## 9. 不可变 snapshot 布局

```text
leaderboard-snapshot/
├── manifest.json
├── checksums.sha256
├── leaderboard.json
├── leaderboard.csv
├── leaderboard.html
└── leaderboard.md
```

`leaderboard.json` 是规范的结构化发布记录。CSV、HTML 与 Markdown 都是其中已存储 row 的确定性渲染；它们不会重新计算分数、不确定性、覆盖率、rank 或顺序。

Snapshot 同时记录：

- `input_fingerprint`：绑定 Snapshot Request、输入 bundle 引用、admission 与 correction；以及
- `snapshot_fingerprint`：绑定完整科学 snapshot 内容。

finalized manifest 为每个发布文件记录 SHA-256 与 byte size。校验器会强制执行精确文件 allowlist、manifest/schema 有效性、两个 fingerprint、文件 checksum、request/snapshot 身份一致性、覆盖率分母、model/dataset 单元完整性、pairwise record 完整性，并逐字节重新生成和核对所有展示文件。

## 10. 验证与 CI

运行 P7 聚焦测试：

```powershell
python -m pytest `
  tests/evaluation/test_p7_leaderboard.py `
  tests/evaluation/test_p7_run_bundle_integration.py

python -m standardized_tabular_diffusion.validation.p7_leaderboard `
  --output artifacts/p7-local.json `
  --require-primary-family-environment
```

`.github/workflows/p7-leaderboard-validation.yml` 在作为主平台族的 hosted Windows/Python 3.11 和作为次级兼容平台族的 Linux/Python 3.11 上执行 contract、negative、finalized-bundle integration、lint、type、package 与 exit-gate 检查。

## 11. 当前限制与下一阶段边界

- P7 每次发布一个固定 metric/protocol/suite/track/class snapshot。多 snapshot 网站和交互式跨 snapshot 过滤属于 P8 发布工作。
- pairwise completeness 与 raw difference 仅为描述性信息；尚未启用 superiority test 和经审阅的 multiple-comparison policy。
- P7 能强制执行已批准 admission，但不能自行产生科学批准。准入决定仍是独立审阅产物。
- 缺少 Atomic Result 的历史 legacy summary 不能经 P7 升级；其只读迁移边界属于 P8。
