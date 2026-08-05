# P2 Shape 与 Trend 评测

- 状态：已通过 Linux/Python 3.11 权威验证；仍仅为诊断协议
- 协议：`p2-shape-trend@0.2.0`（draft、诊断用途）
- 指标身份：`sdmetrics-column-shapes@1.0.0` 与 `sdmetrics-column-pair-trends@1.0.0`
- Official Results 准入：否
- 最后更新：2026-08-05

## 范围与信任边界

P2 是本仓库第一条完整评测路径。它接收解码后的参考表和合成表，执行 Dataset Profile 结构门禁，运行两个精确的上游 SDMetrics property，为每个规范列和无序列对保留一条 Atomic Result，并以原子方式发布校验和完整的 Run Result bundle。

该运行时路径绝不导入 `evaluation/tabstruct.py`。TabStruct 仍然只是研究参考材料，P1 之前的评测器仍然只是旧版诊断兼容路径。

选定来源为官方 [SDMetrics 仓库](https://github.com/sdv-dev/SDMetrics/tree/ba8842f2ba04ce914f698cc1cf746ca12338ab0e) commit `ba8842f2ba04ce914f698cc1cf746ca12338ab0e`，发行版本 `0.28.3.dev0`，许可证为 MIT。后端会对已安装包全部 121 个 Python 源文件经 LF 换行规范化后的字节计算哈希，并要求源码树摘要为 `784beda5c7a63d5ebb5fe74f98d00db3a2e018a29b2f32f643bf857750a6c2a9`；同时要求经 LF 规范化的已安装 MIT 许可证摘要为 `1310119ad2a00b68f05d86309aea0bf1d3853e747d4f5646b1bf60a5a07f09a8`。该规范化消除不同 checkout 或 wheel 构建换行策略的影响，但不会忽略任何可执行内容或法律文本差异。版本、源码树或许可证不一致时，评测会在报告任何指标前停止。

## 结构门禁

文件输入可以是 UTF-8 CSV 或 Parquet；Python 门禁也接受 pandas DataFrame。执行指标前，两个输入必须同时满足：

1. Dataset Profile 有效，并且规范列顺序与其列记录完全一致；
2. 列名是唯一的非空字符串，列集合与 profile 完全相同；
3. 列按 profile 的规范顺序重排；
4. 数值列和整数列的转换不产生无效值或无穷值；整数在规范 signed-int64 范围内保持精确，不经由 float64 转换；
5. 布尔值和日期时间值采用显式、失败即停止的转换规则；
6. 参考表和合成表均不允许缺失值；
7. 合成行数等于请求值；CLI 默认使用参考表行数。

该门禁不会静默填补、修复、裁剪、去重，也不执行属于 P3 的取值域或跨列有效性规则。失败时会写入带稳定原因码的结构化 `validate` 失败阶段，留下可审计的 incomplete bundle，并且不会创建 `metrics.parquet`。

## 精确来源语义

### Column Shapes

上游 property 按 SDMetrics 语义类型选择指标：

- 数值和日期时间：`KSComplement`，即一减两样本 Kolmogorov-Smirnov 统计量；
- 类别和布尔：`TVComplement`，即一减经验类别频率之间的全变差距离；
- 不支持的类型：上游不计算，P2 保留为 `not_applicable` Atomic Result。

来源 property 分数是所有有限列分数的算术平均。P2 将来源分数原样写为 `raw_value`，单独记录恒等归一化，对有限来源结果赋予等权重，并以绝对和相对误差 `1e-12` 验证 Atomic Result contribution 能重建来源 property 分数。

### Column Pair Trends

上游 property 评测每个受支持的无序列对：

- 连续—连续列对使用 Pearson `CorrelationSimilarity`，分数为 `1 - |r_real - r_synthetic| / 2`；
- 离散—离散列对使用联合频率 `ContingencySimilarity`；
- 混合列对按照锁定来源行为，分别在真实表与合成表内对连续列独立离散化，再使用 `ContingencySimilarity`；
- 不支持的语义类型列对保留为显式 `not_applicable` Atomic Result。

锁定的 Quality Report 默认值属于指标身份的一部分：`num_rows_subsample=50000`、真实数据绝对相关阈值 `0.5`、真实数据 Cramér's V 关联阈值 `0.3`。真实关系未超过相应阈值时，来源会返回非有限分数。P2 将该决定记录为 `not_applicable`，并使用原因码和警告码 `below_source_threshold`；绝不会静默删除列对。property 分数是所有有限贡献列对分数的来源算术平均，P2 会再从 Atomic Result 独立重建该值。

当输入超过 50,000 行时，锁定的来源实现会使用未显式传入 random state 的 pandas 抽样。为此，P2 要求恰好一个 evaluator seed；执行来源代码时使用该种子控制旧版 NumPy 随机状态，对来源调用进行串行化，并在结束后恢复调用方原有状态。这样既保留官方计算，又使重复评测可复现；即使输入没有触发抽样，该种子仍属于请求身份。

## Atomic Result 与摘要

每个规范列和无序列对都使用 Dataset Profile `column_id` 构造稳定 scope。每行记录精确的指标/版本、数据集/view/split/模型身份、状态、原始值与归一化值、聚合 contribution、来源 evaluator 身份、参考/合成计数、有效/排除计数、警告、原因详情，以及指向保留的来源明细 artifact 的引用。

支持的状态包括 `computed`、`mathematically_undefined`、`insufficient_support`、`not_applicable`、`implementation_failure` 和 `resource_failure`。只有来源判定可贡献且为有限值的结果参与聚合。`summary.json` 会公开覆盖率和状态计数；失败与未定义结果不能从分母中消失。

P2 分别报告两个来源 property 分数。它有意不生成合并或总体 Fidelity 分数，不执行数据集聚合，也不产生可进入榜单的结果。

## Finalized bundle

成功运行包含请求、环境、metadata、Atomic Results Parquet、summary、全部七个阶段记录、事件日志、结构证据、规范化保存的 SDMetrics 原始明细表、artifact index、manifest 和 `checksums.sha256`。外部参考表和合成表采用内容寻址，但不会复制进 bundle。

每个阶段输出的校验和必须与 manifest 一致。`checksums.sha256` 覆盖除自身之外的所有 finalized 普通文件。finalized 状态作为最后一个原子提交标记写入，因此在此替换前发生中断时，manifest 仍保持 `incomplete`；finalized bundle 会拒绝 writer 继续修改。

`validate-result` 还会把 Parquet 中每一行重新解析为 Atomic Result 契约，检查科学身份和 scope 唯一性，重算状态/警告覆盖率与两个 P2 property 分数，核对本地产物的大小/checksum/media type/producer stage，并将外部产物 provenance 与 Evaluation Request 对齐。

## CLI

安装隔离的评测依赖并运行：

~~~bash
python -m pip install ".[evaluation]"

std-tabular-diffusion evaluate-table \
  --reference path/to/reference.csv \
  --synthetic path/to/synthetic.csv \
  --dataset-profile configs/datasets/adult-uci-2-v1.json \
  --output artifacts/evaluation/run-001 \
  --model-id my-model \
  --comparison-track native

std-tabular-diffusion validate-result --bundle artifacts/evaluation/run-001
~~~

`--expected-rows` 可以覆盖默认的参考表行数。`--generation-seed` 记录所提供合成产物的生成身份；`--evaluator-seed` 控制锁定来源在超过 50,000 行时的抽样，并在较小输入下仍被记录。

## 验证边界

专用 workflow 会在 Linux/Python 3.11 上安装精确官方来源、证明完整源码树身份、比较 wrapper 与直接权威调用的分数和明细 DataFrame、覆盖数值/类别/布尔/日期时间/混合与边界输入、精确重复一个带种子的 50,001 行来源抽样案例、检查分母完整的 Atomic Result、对同一请求构建两个语义等价的 finalized bundle、测试失败短路与中断安全、执行 lint 和类型检查，并生成机器可读证据。

P2 通过后，这两个记录只推进到 `source-parity-validated`。协议冻结、Dataset Suite 准入、Official Results 资格、总体 Fidelity 定义和仓库 release support 仍然是后续相互独立的门槛。

P2 退出门槛已在 [GitHub Actions run 31025796906](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31025796906) 通过；仓库已留存其精确的[机器可读证据](../evidence/evaluation/p2-shape-trend-run-31025796906.json)。
