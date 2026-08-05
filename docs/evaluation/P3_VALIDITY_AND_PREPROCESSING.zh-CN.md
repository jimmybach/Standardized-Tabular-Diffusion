# P3 有效性与显式预处理边界

英文原文：[P3_VALIDITY_AND_PREPROCESSING.md](P3_VALIDITY_AND_PREPROCESSING.md)

- 状态：通过 Linux/Python 3.11 验证的诊断性实现
- 协议：`p3-validity@0.3.0`
- 指标版本：`1.0.0`
- 主要环境：Linux 与 Python 3.11
- 是否允许进入 Official Results：否

## 1. 范围

P3 在不可变的解码后合成表上实现两个本仓库原生定义的指标：

- `std-tabular-column-validity@1.0.0`；
- `std-tabular-constraint-validity@1.0.0`。

P3 同时固定仓库的显式缺失值预处理边界。所有需要学习的填补状态只能在真实训练集上拟合；验证集和测试集只能使用冻结状态转换，不能重新拟合。该路径绝不填补或静默修复生成数据。

P3 不发布榜单，不定义总体 benchmark 分数，不审批仍待讨论的数据集约束，也不修改任何上游生成模型实现。其公式由本仓库定义，不得归因于 PAFT、TabStruct 或 SDMetrics。

## 2. 结构校验与内容有效性

P3 结构门检查解释表格所必需的条件：

- CSV 或 Parquet 可读取；
- 列名唯一；
- 模型视图列集合和行数严格一致；
- 审计字段和忽略字段不进入模型输出分母；
- 基础类型可以安全转换。

随后，P3 会保留内容违规并进行评分。缺失值、非有限数、整数列中的小数、未知类别、越界值或经审阅约束的违规都不会被静默修正。这与 P2 为保证上游来源兼容而采用的更严格结构门不同。选择 `p2-shape-trend` 时仍保持原有 P2 行为。

## 3. Dataset Profile 有效性契约

支持 P3 的 Dataset Profile 必须声明 `validity.contract_schema_version: 1.0.0`、经审阅的硬列规则、经审阅的硬跨列约束、软诊断项和未解决审阅项。

每条硬规则或约束都必须包含：

- 稳定标识符和版本；
- 获准的证据类别；
- 证据引用；
- `hard` 严重级别；
- 封闭、声明式的规则类型和参数。

允许的证据类别包括权威数据字典、数据集文档、法律或政策要求，以及有记录的人工审阅。禁止任意 Python 表达式和动态代码加载。

### 3.1 列规则类型

实现支持：

| 规则 | 含义 |
|---|---|
| `not_null` | 缺失单元格违反模型输入非空规则。 |
| `finite` | 非缺失数值必须是有限数。 |
| `integer` | 数值必须是可由有符号 int64 表示的有限数学整数。 |
| `allowed_values` | 值必须属于显式或 Dataset Profile 声明的词表。 |
| `bounds` | 值必须满足经审阅的闭区间上下界。 |
| `regex` | 字符串必须完整匹配经审阅的正则表达式。 |
| `length` | 字符串长度必须满足经审阅的闭区间限制。 |
| `unique` | 列中每个非缺失值必须恰好出现一次。 |
| `datetime_range` | 解析后的日期时间必须位于经审阅的闭区间内。 |

选择器可以依据列标识符、语义类型、模型输入可空性、角色和必需的 domain 字段选列。每个规范模型视图列必须至少匹配一条硬规则，否则 P3 在评分前关闭失败。

### 3.2 跨列约束类型

封闭约束语言支持比较、条件域、互斥、带显式容差的数值求和等式、允许组合和函数依赖。每条约束还必须声明适用条件和缺失值处理方式。

适用条件可以是全部行、所有相关列非缺失的行、某列等于指定值的行，或某列属于指定集合的行。如果一条经审阅约束没有任何适用行，系统保留 `not_applicable` 状态，绝不会伪造满分 1。

## 4. 指标定义

对规范列 (j)：

```text
valid_cell_rate_j =
    同时满足该列所有适用硬规则的单元格数
    / 合成数据行数

column_validity_score = 所有规范列 valid_cell_rate_j 的等权平均
```

对经审阅约束 (k)：

```text
constraint_satisfaction_rate_k =
    满足约束 k 的适用合成行数
    / 适用合成行数

constraint_validity_score =
    对至少有一条适用行的经审阅约束做等权平均
```

维度总分为：

```text
如果 constraint_validity_score 可计算：
    validity_score = 0.5 * column_validity_score
                   + 0.5 * constraint_validity_score
否则：
    validity_score = column_validity_score
```

`fully_valid_row_rate` 单独报告，绝不进入 `validity_score`。因为随着表格变宽，一行至少违反一条规则的概率会机械性上升。测试会明确验证：相同的列级有效性可以产生不同的完全有效行比例。

## 5. Atomic Result 与证据

P3 输出：

- 每个规范模型视图列一个已计算 Atomic Result；
- 每条经审阅跨列约束一个 Atomic Result；
- 如果没有经审阅约束，则输出一个显式的 `no-reviewed-constraints` 不适用记录。

`artifacts/validity-details.json` 保存规则标识符、证据引用、逐规则违规数量、逐列有效和无效数量、逐约束适用/满足/违规数量、组件得分和完全有效行诊断。文件会记录输入未被修改且未进行合成数据修复，但不会把真实或合成数据行复制进结果包。

最终结果包验证会解析 Parquet 中的每个 Atomic Result，重新计算两个组件分数和 `validity_score`，并交叉核对规则详情、计数、scope、不可变输入声明、summary、metadata、artifact inventory、manifest 和 checksums。

## 6. 当前已审阅数据集行为

Adult 和 Sick 诊断 Dataset Profile 目前只启用其已审阅模型输入契约和校验和锁定来源文档支持的规则：

- 模型输入非空；
- 数值有限；
- 已声明字段的整数语义；
- 已声明的类别或布尔词表。

训练集观察到的最小值和最大值只是软诊断，不是硬生成 domain。Adult 的 education 与 education-number 一致性、权威数值边界、Sick 医学范围以及医学跨列约束仍待解决。P3 不会自动提升它们。新增或修改硬约束必须经过证据审阅并更新 Dataset Profile 版本。

## 7. 显式缺失值预处理

集中式预处理实现采用已批准的 v1 策略：

- 数值特征：真实训练集中已观察值的算术平均数；
- 类别特征：真实训练集中已观察值的众数；
- 众数并列：按 Unicode 规范化后的词典序确定，保证确定性；
- 目标列：缺失即报错，绝不填补；
- 训练特征整列缺失：关闭失败；
- 验证集和测试集：只使用冻结训练状态进行转换；
- 生成样本：拒绝或按无效值评分，绝不填补；
- 缺失指示列：默认关闭，显式启用时必须版本化。

文件工作流输出转换后的数据划分、`imputation-state.json` 和 `preprocessing-manifest.json`。manifest 记录输入输出哈希、清理报告、学习状态哈希与指纹、转换后 schema 指纹、策略/配置指纹和派生 Dataset View token。因此策略或转换后 schema 变化会产生不同的数据视图身份。非空输出目录不会被覆盖。

数据集注册会保留字节完全一致的源文件副本并记录两份哈希。单独物化的规范 CSV 会被明确标注为序列化转换。注册过程不删除行，也不填补值。

## 8. CLI 用法

为了保持向后兼容，P2 仍是默认协议。使用 P3 时必须显式选择：

```powershell
std-tabular-diffusion evaluate-table `
  --protocol p3-validity `
  --reference real.csv `
  --synthetic synthetic.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --expected-rows 32561 `
  --output results/adult-validity
```

重新验证最终证据：

```powershell
std-tabular-diffusion validate-result --bundle results/adult-validity
```

如果真实原始数据划分含缺失值，模型训练前运行独立预处理命令：

```powershell
std-tabular-diffusion preprocess-missing-values `
  --train-csv raw/train.csv `
  --test-csv raw/test.csv `
  --numerical-column age --numerical-column hours `
  --categorical-column workclass --categorical-column occupation `
  --target-column income `
  --output-dir processed/adult-imputed-v1
```

## 9. 准入状态与限制

两个 P3 指标均为 `unit-validated`、本仓库原生定义的诊断指标。它们不是来源等价性指标，因为不存在定义这些仓库公式的上游实现。它们尚未达到 `protocol-frozen` 或 `release-supported`，不能进入 Official Results。

P3 专用 CI 覆盖手工可计算规则、畸形 Profile、无约束行为、适用性、宽度敏感性、无隐藏修复、只在训练集拟合、schema 与策略身份、注册字节保留、结果包重算、CLI、Linux/Python 3.11 打包、lint 和类型检查。它已在 [GitHub Actions 运行 31036844043](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31036844043) 中通过；精确的[机器可读证据](../evidence/evaluation/p3-validity-run-31036844043.json)已留存，SHA-256 为 `bc63a2df553036ee7e161ce81c6f264dace950f3fe414ba2f8195d8e557e401d`。
