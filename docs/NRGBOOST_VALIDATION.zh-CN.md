# NRGBoost 验证协议

状态：已通过并永久保留

协议：`nrgboost-native-parity-v1`

目标：方法作者官方 `nrgboost==0.0.3` 包

支持的验证环境：Linux、Python 3.11

## 声明边界

本协议检验标准化 `nrgboost` 适配器是否构造与官方直接调用相同的带类型输入表和官方 `Dataset`，是否原样传递数据集、训练、采样与随机种子参数，保存并重新加载相同模型，以及是否生成相同输出。

强制运行通过后，适配器可以提升为 `native-parity-validated`。这不等于 NRGBoost 已经 `benchmark-eligible`，不代表可进入正式结果，不证明统计评测质量，也不等于 `release-supported`。数据集准入、中心评测冻结、运行时限制和发布责任仍是独立门槛。

## 已审计权威来源与发行包

已审计实现为[方法作者仓库](https://github.com/Ajoo/nrgboost)的 `v0.0.3` 标签，对应提交 `feef73a3edb20b911c2f7214b13f810909ef20ad`、树 `e3e84bacc7236a36af93c3d214de14bd308d2767`。支持的制品是官方 [PyPI 0.0.3 发行](https://pypi.org/project/nrgboost/0.0.3/)中面向 CPython 3.11 和 manylinux 2.28 x86-64 的 wheel：

- 文件名：`nrgboost-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl`；
- SHA-256：`dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb`；
- 许可证：MIT；
- 发行形式：可选包依赖；本仓库不复制、不修改 NRGBoost 源码。

协议检查 wheel 文件名与摘要、安全归档路径、包元数据、Python 和 ABI 标签、声明依赖、源码与 wheel 许可证哈希、编译扩展、随包 OpenMP 运行库、wheel `RECORD` 中每个带哈希条目、安装分发根目录以及公开类导出。PyPI Trusted Publishing 来源将发行制品绑定到锁定标签提交。

NRGBoost 0.0.3 提供 Linux 和 macOS wheel，当前不支持 Windows；源码构建需要 C 编译器和 OpenMP。因此本仓库以 Linux/Python 3.11 为权威环境，不会把 Windows 上的自行源码构建解释为等价证据。

## 适配器语义

适配器保持官方包的薄封装边界：

- 按声明列顺序读取规范训练 CSV；
- 按官方要求，将声明的分类特征和分类任务目标转为 pandas `category`；
- 拒绝缺失值，要求先显式调用基准预处理模块；
- 构造官方 `nrgboost.Dataset` 和 `nrgboost.NRGBooster`；
- 每次传入新的训练参数映射副本，因为官方 `fit` 会从该映射中取出部分条目；
- 向 `NRGBooster.fit` 和 `NRGBooster.sample` 同时传递 `RunSpec.seed`；
- 保存和加载官方基于 joblib 的检查点格式；
- 仅按规范列顺序写出请求的最终链样本。

检查点格式在加载时可以执行 Python。因此适配器默认只加载运行输出目录内非符号链接的普通文件。外部检查点只能在审查来源与完整性后显式启用不安全覆盖。

## 支持的控制项

适配器在不改变算法的前提下暴露官方标量数据集和训练参数。数据集参数包括 `num_bins`、定点数推断、可选显式离散化类型，以及两个有序分类推断开关。训练参数包括树数、收缩与线搜索、树规模与分裂、数据/模型叶约束、初始混合、特征比例、模型采样链、刷新率、预热、温度、最小增益、JIT 选择和线程数。在官方调用前会对取值范围进行检查。

采样暴露输出行数、Gibbs 步数、可选 boosting 轮次、温度、线程数和随机种子。`output_full_chain` 固定为 `false`：返回 `num_samples × num_steps` 个链状态会违反标准化 `num_samples` 行数契约。

## 冻结等价性用例

强制协议使用两个确定、无缺失、混合类型表，每个包含 36 行：

1. 分类：两个数值特征、一个分类特征和一个分类目标；
2. 回归：两个数值特征、一个分类特征和一个数值目标。

每个表分别使用种子 0、19 和 73，共形成 6 个独立原生/适配器用例。有界 CI 配置使用单线程拟合 3 棵树、256 个模型样本、4 条链和小型树限制；采样请求 16 行、12 个 Gibbs 步和单线程。这些设置会运行真实编译实现，同时限制 CI 成本；它们是验证固件，不是推荐的基准超参数。

每个用例运行两条独立路径：

- 原生：直接官方 `Dataset` → `NRGBooster.fit` → `save` → `load` → `sample`；
- 适配器：标准化 `train` → 官方检查点 → 标准化 `sample`。

## 强制通过条件

6 个用例必须全部通过。门槛要求：

1. 冻结 Linux/Python 3.11 环境和官方 wheel 身份完全一致；
2. 适配器制品清单和结构化元数据完全一致；
3. 原生与适配器检查点字节完全一致；
4. 两个检查点都声明序列化版本 `0.0`，包含恰好 3 棵树，并保留规范变换列；
5. 原生和适配器样本 CSV 字节完全一致，DataFrame 也完全一致；
6. 输出行数和列顺序正确，数值有限，分类值位于学习域内，且无缺失值；
7. 训练和采样都不改变传统 NumPy 全局随机状态。

任何不一致、制品缺失、平台错误、依赖漂移、不安全 wheel 路径、未验证安装文件或比较失败都会闭合失败，并保留诊断 JSON 制品。

## 已知边界

- 精确适配器等价性只证明封装保留了选定的官方执行；不证明 3 棵树的烟雾固件具有完整论文质量。
- 官方采样是近似的，成本随 Gibbs 步数线性增长。基准超参数和运行时预算需要后续数据集级研究。
- 缺失数据不属于模型适配器契约。插补必须仅在训练划分上拟合，由预处理层记录，并在 NRGBoost 之前应用。
- 高级显式 `discretization_types` 是官方专家接口。任何非默认映射在进入正式结果前都必须写入数据集档案。
- 本协议验证分类和回归表的生成。`NRGBooster.predict` 的预测用法不属于标准化生成适配器范围。

## 证据

[GitHub Actions 运行 `30922326384`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30922326384) 在 Linux、Python 3.11.15 上通过全部 6 个任务/种子用例。它将 22 个带哈希的已安装文件与锁定 wheel 逐一核对，使用种子 0、19 和 73 执行分类与回归固件，并在每个用例中生成字节级一致的原生/适配器检查点和样本 CSV。

永久证据记录为 `docs/evidence/nrgboost/native-parity-run-30922326384.json`，SHA-256 为 `5958c67261e8c25e60d58891efd5d27f8e8bb6439852862064e831f630cbe56c`。运行绑定到仓库提交 `4cd32c8beedd116c6385463d41cf9cba8b1d5438`；下载的 GitHub 制品 ID 和摘要也已写入来源锁。因此 NRGBoost 现为 `native-parity-validated`，但基准准入与发布支持仍待完成。
