# REaLTabFormer 验证协议

状态：已通过并永久保留证据

协议：`realtabformer-official-package-parity-v1`

目标：方法作者官方 `realtabformer==0.2.4` 表格模型包

支持的验证环境：Linux、Python 3.11

## 声明边界

本协议检查标准化 `realtabformer` 适配器是否保留了选定的官方表格模型执行语义。它让“直接调用经校验和锁定的官方包”和“通过适配器调用”使用相同的带类型训练表、GPT-2 配置、训练控制、检查点重载方式、采样控制和随机种子，并比较二者结果。

强制验证通过后，适配器可以在已测试的表格路径上提升为 `native-parity-validated`。这不代表 REaLTabFormer 已经 `benchmark-eligible`，不代表它可以进入 Official Results，不证明完整论文规模下的统计质量，也不等于 `release-supported`。官方 sensitivity-based stopping 路径、关系表模型、数据集准入、中心评测、资源预算和发布责任仍是独立门槛。

## 已审计的权威来源与发行制品

权威来源是[世界银行的方法作者仓库](https://github.com/worldbank/REaLTabFormer)：标签 `v0.2.4`、提交 `73f239643f9ea5abc877f685ce927e986302ac2d`、树 `aa4431468f040fc485f82e7e15238c57eef05753`。选定制品是官方 [PyPI 0.2.4 发行版](https://pypi.org/project/realtabformer/0.2.4/)：

- 文件名：`realtabformer-0.2.4-py3-none-any.whl`；
- 大小：49,890 字节；
- SHA-256：`852436c5c82a0bf470ca7e9063e5a4f3e250b3ff5b9c8f6c50113c1e9ba76486`；
- 许可证：MIT，Copyright 2022 Aivin V. Solatorio；
- 发行形式：可选包依赖；本仓库不内置 REaLTabFormer 源码。

当前标签和仓库默认分支指向同一个提交。wheel 与标签源代码归档中共有的 11 个源码文件逐字节相同；wheel 只额外包含一个空的 `rtf_tokenizer.py`。源代码归档、wheel、源许可证、包元数据以及 16 个带哈希的已安装文件都分别进行了校验和锁定。

协议会拒绝改名、大小不符、符号链接、内容被修改或含路径穿越条目的 wheel。它检查包身份、Python 要求、声明依赖、纯 Python 标签、MIT 许可证、所有 `RECORD` 路径/大小/哈希，以及每个锁定文件安装后的内容。直接运行依赖也必须与冻结验证版本完全一致。

## 适配器语义

适配器：

- 接受一个无缺失值的单表，且必须恰好有一个分类或回归目标；
- 严格按照 `DatasetSpec` 列顺序读取数据，并要求声明角色恰好覆盖每一列；
- 检查数值特征和回归目标均为有限数值；
- 在不修改官方 tokenizer 的前提下，将声明为类别型的特征和分类目标转换为字符串；
- 将 `RunSpec.seed` 传给官方构造器和 Hugging Face 训练参数；
- 将官方检查点、sensitivity 样本、周期保存和最终模型目录全部放在 `output_dir` 下；
- 保存官方 `rtf_config.json`、`rtf_model.pt` 及适用的官方制品；
- 记录包身份、变换后训练表、有效控制参数和逐文件检查点完整性元数据；
- 加载前核对这些元数据，并要求只能解析出一个明确的模型目录；
- 使用 PyTorch `weights_only=True` 约束官方加载器；
- 在调用未修改的官方 `sample()` 前，重置 Python、NumPy 和 PyTorch 随机生成器。

缺失值默认报错。有缺失值的数据集必须先使用只在训练划分上拟合的中心插补模块；适配器不会从验证集或测试集学习填充值。

## 已记录的兼容边界

官方源码文件没有被修改。共记录六项显式边界：

1. 输出隔离：把官方目录参数设置到 `output_dir` 下；
2. 声明类型：原生路径和适配器路径都在调用前将类别角色转换为字符串；
3. 采样种子：两条路径都使用请求的种子重置随机生成器；
4. 仅在 Transformers 导入期间禁用未使用的 torchvision 探测；
5. 将官方状态字典加载限制为 `weights_only=True`；
6. 在调用未修改的官方 `save()` 前，把 `full_save_dir` 表示为含义完全相同的路径字符串。

第 6 项用于规避 v0.2.4 的序列化缺陷：官方 `save()` 会把两个同类 `Path` 属性转成字符串，却遗漏 `full_save_dir`，因此新构造模型会在 `json.dumps` 中失败。该处理不改变模型、张量、优化器、预处理或采样状态。由于这一处理和受限加载器改变了调用边界的运行语句，本集成保守标记为 `compatibility-patched`，而不是 `adapter-only`。

## 支持的控制项

适配器恢复官方默认值：1,000 个 epoch、batch size 为 8。它支持有界的训练比例、早停、mask 和数值 token 化构造参数；经过筛选的 Hugging Face `TrainingArguments`；所有官方表格 fit sensitivity 控制项；自定义 `GPT2Config` 映射（数据推导出的词表和特殊 token 字段仍由 REaLTabFormer 管理）；用于 smoke test 的确定性训练行数限制；以及官方采样生成参数。

未知控制项默认报错。外部训练上报默认关闭。验证中的微型配置只有一层 GPT-2，仅用于限制 CI 成本，不是推荐的 benchmark 配置。

## 冻结等价性用例

强制协议使用三个确定、混合类型、无缺失值的固定表，每个 24 行：

1. 二分类；
2. 多分类；
3. 回归。

每个表包含两个数值特征、一个类别特征和一个目标。每种任务分别使用种子 0、19、73，共 9 个独立用例。每个用例用真实官方 GPT-2 实现进行一个受限 epoch 的训练，关闭 sensitivity stopping（`n_critic=0`），随后保存、重新加载模型并请求 7 行样本。

两条独立路径为：

- 原生：直接官方构造器 → `fit` → `save` → `load_from_dir` → `sample`；
- 适配器：标准化 `train` → 完整性校验后的官方检查点 → 标准化 `sample`。

## 强制通过条件

9 个用例必须全部通过。门槛要求：

1. Linux/Python 3.11 环境、wheel、安装文件、许可证和依赖身份完全一致；
2. 检查点键顺序和张量值完全一致；
3. 将输出根目录和基于时间的实验 ID 替换为占位符后，官方保存配置在语义上完全一致；
4. 标准化序列化之前的原始 DataFrame 完全一致；
5. 原生与适配器样本 CSV 逐字节一致；
6. 输出行数和规范列顺序准确；
7. 不含缺失值和非有限数值，类别输出不超出训练域；
8. 适配器的包、训练表、有效参数和检查点完整性元数据有效；
9. 所有用例结束后，锁定的官方包文件仍保持不变。

任何不一致、依赖漂移、平台错误、不安全制品、模型目录歧义或完整性失败都会闭合失败，并保留诊断 JSON 制品。

## 已知边界

- 固定用例证明的是包装等价性，不是论文规模训练预算下的生成质量。
- 官方 sensitivity-based stopping 路径虽然已暴露，但不由本协议提升状态；它需要单独的、受资源约束的验证研究。
- 官方关系模型接收相互关联的父表和子表。当前仓库的规范契约是单表，因此关系模式不在本适配器已验证范围内。
- Transformer 训练和自回归采样明显重于传统 baseline；各数据集的序列长度、时间和显存预算仍待研究。
- 条件 seed input 和高级生成参数是官方专家接口；进入 Official Results 前必须在 benchmark profile 中冻结。
- 官方包支持早于 3.11 的 Python，但本仓库正式支持的发布环境仅为 Linux/Python 3.11。

## 证据

[GitHub Actions 运行 `30950369908`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30950369908) 在 Linux、Python 3.11.15 上通过全部 9 个任务/种子用例。该运行核验了官方 wheel 和 16 个带哈希的已安装文件；每个用例中的原生/适配器检查点张量与文件完全一致，保存配置语义一致，原始样本相同，最终 CSV 逐字节一致。所有输出均为请求的 7 行规范列，数值有限、类别域有效且无缺失值。

永久证据记录为 `docs/evidence/realtabformer/native-parity-run-30950369908.json`，SHA-256 为 `0c6047efc3463aa21fa4b2e6aeed66858cbc29bfd5a9e836f330d975ec0cfa07`。该文件从制品 `8908863813` 逐字节保留；制品归档摘要为 `sha256:03ae72ed21ea357c466a9c7f9ee3b29a1c2e5e29ec8fcc2305c9dc7a7f2f8147`。PR head 为 `7db46e00452ce5cc25d28d8b484c9d6ee14de5b3`，证据中记录的 PR merge checkout 提交为 `fb2f03dd579bb4d1847fa18395696ed698c8ce58`。

因此，REaLTabFormer 在官方表格 `n_critic=0` 路径上现为 `native-parity-validated`。它仍然是 `experimental` 和 `unsupported`；本证据不会提升 sensitivity stopping、关系模式、中心 benchmark 评测、数据集准入、资源预算或发布支持状态。
