# BN 验证协议

状态：已通过；已保留 Linux/Python 3.11 官方包方案等价性证据

协议 ID：`pgmpy-bn-recipe-parity-v1`

支持的验证平台：Linux、Python 3.11、CPU

## 范围与声明边界

本协议验证仓库声明的离散贝叶斯网络方案，是否与直接调用未经修改的官方 `pgmpy==1.1.2` 包完全一致。该方案先对数值列进行确定性的分位数离散化，再使用 pgmpy 的 BIC 评分爬山搜索学习 DAG，使用 BDeu 先验估计条件概率分布，最后使用 pgmpy 前向采样器生成数据。

pgmpy 是权威的贝叶斯网络通用库，并不是某一篇 BN 合成论文的原始实现。因此，权威验证通过后，`bn` 只能针对“官方包 + 本仓库明确声明的方案”晋级为 `native-parity-validated`。这不表示复现了某篇论文的表格，也不验证其他离散化器、评分函数、先验、结构搜索或采样方法；更不表示 BN 已经 `benchmark-eligible`、可进入 Official Results 或已经 `release-supported`。

## 来源权威性、包身份与许可证

- 官方仓库：`pgmpy/pgmpy`。
- PyPI 包：`pgmpy==1.1.2`。
- 官方 wheel：`pgmpy-1.1.2-py3-none-any.whl`。
- Wheel SHA-256：`e55c78763a4a45dd644a13b250cea86af0c7e08590cf35de489624f34a4d9a0b`。
- Wheel 大小：2,446,383 字节。
- Git 标签：`v1.1.2`；注释标签对象 `ff663f9203c5075b2367707917016efafed03593`。
- 锁定提交：`617cb48af678a7a471aad81d523ca95d2095430f`。
- 锁定 tree：`6c7adc00a479f540b2215889b1fac99a7b0b8a9c`。
- 许可证：MIT；锁定许可证 SHA-256 为 `89171dcc8977530b0c101fbbb1c1d34caee998fc7def9eded629753cd2616a15`。

PyPI Trusted Publishing 来源证明将 1.1.2 的两个发行文件绑定到锁定的 GitHub 标签提交。Wheel 中全部 636 个 `pgmpy/` 文件，都与源码发行包和锁定标签中的对应文件逐字节一致；Git 标签另有 12 个仅存在于仓库的文件。模型运行前，协议会验证 wheel 摘要、全部 649 个成员、`RECORD` 中全部 648 个带哈希条目、包元数据和依赖声明、许可证、九个关键运行文件及其官方 Git blob 身份、已安装发行包和公开类位置。全部模型案例结束后会再次执行来源检查。

该注释标签没有密码学签名。本项目如实记录 Trusted Publishing 来源、不可变包摘要、标签提交/tree、逐字节对比和 Git blob 锁，而不会声称它是签名发布。

## 声明方案与适配边界

验证方案使用：

- `KBinsDiscretizer`：六个分位数箱、序数编码、`quantile_method="averaged_inverted_cdf"`、`subsample=None`；
- `HillClimbSearch`：`scoring_method="bic-d"`、输出 DAG、最大入度 2、100 次迭代、tabu 长度 20、epsilon 为 `1e-4`；
- `DiscreteBayesianEstimator`：`prior_type="BDeu"`、等效样本量 5、单任务；
- `BayesianModelSampling.forward_sample`：单任务并使用标准化随机种子。

仓库适配器不修改 pgmpy 源码，只提供标准 benchmark 所需的边界：

- 使用明确声明的数值/类别角色，并要求恰好一个分类或回归目标；
- 在显式训练集拟合的预处理模块执行前，严格拒绝缺失值和非有限数值；
- 将常量数值列确定性地处理为单一状态；
- 将所有规范列加入图节点，包括孤立变量；
- 对已验证的确定性路径强制使用 CPU 和单任务；
- 完成后恢复调用方的进程级 NumPy 随机状态；
- 严格检查请求行数、规范列顺序、状态域、缺失值和数值有限性。

适配器只接受已验证的评分函数、先验、采样器、分位数方法和 DAG 返回类型。改变这些选择时，必须定义新的独立方案并重新建立等价性协议。

## 安全持久化边界

官方对象不会使用 pickle 保存。适配器写入有类型的 `pgmpy-discrete-bn-state` JSON checkpoint，其中包含包身份、方案、离散化边界、常量值、类别/状态名、图边和 CPD。加载时会验证完整 schema，使用官方 `TabularCPD` 重建官方 `DiscreteBayesianNetwork`，执行 `check_model()`，然后调用未经修改的官方采样器。旁路文件记录 checkpoint SHA-256 和数据处理声明。因此，经审阅的外部 BN checkpoint 无需开启不安全 pickle 选项即可加载。

Checkpoint 有意不保留逐行训练数据，也不包含可执行对象。这属于数据最小化，不是隐私保证：箱边界、类别水平、图结构、CPD 和训练帧指纹仍可能暴露拟合数据的信息。训练后的 checkpoint 仍须实施访问控制和保留期限审查。

## 冻结环境

工作流使用 CPython 3.11 和 `requirements-bn-validation.txt` 中的精确依赖集合。它通过 HTTPS 下载官方 wheel，安装前验证 SHA-256，以不解析依赖的方式安装该 wheel 和本仓库，并要求 `pip check` 通过。协议会拒绝非 Linux 主机、非 3.11 解释器或任何冻结发行版本不一致的环境。

等价 Linux 命令见英文协议中的“Frozen environment”部分。

## 冻结对照

协议覆盖二分类、多分类和回归。每个确定性夹具包含 60 行无缺失数据、相关数值列、常量数值列、一个类别列和一个目标列。每种任务使用随机种子 `0`、`19`、`73`，每次请求 13 行，共形成九个案例。

每个案例中，直接路径独立预处理已经持久化的规范 CSV，并直接调用官方类；适配器路径通过 `BNAdapter` 完成训练、保存、重新加载和采样。所有案例都必须在不使用数值容差的前提下同时证明：

1. 预处理状态和离散训练帧哈希完全一致。
2. 学到的图边，以及每个 CPD 的变量、证据顺序、基数、状态名和概率完全一致。
3. JSON 恢复后的官方模型完全一致，并通过官方模型校验。
4. 原始离散样本、最终 CSV 字节和重新读取的 DataFrame 完全一致。
5. 两条路径均恢复调用方的 NumPy 状态。
6. 产物清单和 checkpoint/样本哈希旁路文件有效。
7. 输出具有请求行数和规范列、合法的取值域与范围、有限数值且无缺失值。
8. Checkpoint 是不可执行 JSON，不包含逐行训练数据，不夸大隐私保证，并明确要求继续控制训练产物访问权限。
9. 验证后已安装的官方包保持不变。

## 执行与晋级规则

权威命令为：

```bash
python -m standardized_tabular_diffusion.validation.bn \
  --repo-root . \
  --output-dir /tmp/bn-validation \
  --evidence-path /tmp/bn-evidence.json \
  --wheel-path /tmp/pgmpy-wheel/pgmpy-1.1.2-py3-none-any.whl
```

`.github/workflows/bn-validation.yml` 执行该命令，并将 JSON 证据保留 90 天。包、依赖、适配器、checkpoint schema 或协议发生任何变化后都必须重新运行。只有 Linux/Python 3.11 产物通过、经检查并原样保留到 `docs/evidence/bn/` 后，状态才允许晋级。

## 已保留结果

GitHub Actions 运行 [`30967779298`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30967779298) 已在 Linux 和 Python 3.11.15 环境通过。二分类、多分类、回归及随机种子组合形成的全部九个案例都通过了所有精确比较，包括预处理和离散训练帧、学习到的图边、全部 CPD、JSON 恢复后的官方模型、原始离散样本、最终 DataFrame 和 CSV 字节、产物元数据、安全状态声明以及调用方 NumPy 状态恢复。

经审查的证据已逐字节保留在 `docs/evidence/bn/native-parity-run-30967779298.json`，SHA-256 为 `6463f178fb4d30a4dc0925db207a814cf1d7d0ab85ed75b26e619ec4b26d9ad8`。GitHub artifact ID 为 `8915417956`，归档摘要为 `sha256:6dbedd1970b51ab5243e8da35b052d8d8df780cd2c2e81ac2adf756b4cae5654`，source lock 已同时交叉引用这两项记录。

因此，BN 针对“精确官方 pgmpy 包 + 本仓库声明方案”晋级为 `native-parity-validated`。在中央评测、数据集准入、运行资源、治理和发布门槛分别通过前，它仍是 `experimental`、`unsupported`，并排除在 Official Results 之外。本结论不声称论文原生等价性，也不覆盖其他 BN 方案。
