# ARF 验证协议

状态：已通过；已保留 Linux/Python 3.11 官方包等价性证据

协议 ID：`arfpy-official-package-parity-v1`

支持的验证平台：Linux、Python 3.11、CPU

## 范围与声明边界

本协议验证标准化 ARF 适配器是否使用未经修改、由方法作者维护的官方 Python 包 `arfpy==0.1.1` 完成混合类型 FORDE/FORGE 生成，并与直接原生调用得到完全相同的保留密度状态和生成结果。该包与原始 R 包同属 `bips-hb` 组织，并将 ARF 作者 Kristin Blesch 和 Marvin N. Wright 列为包作者。因此，它是方法作者提供的官方 Python 实现，而不是无关的本地重实现。

权威运行通过后，适配器可针对这一精确 Python 包晋级为 `native-parity-validated`。该结论不证明它与独立 R 包在数值或行为上跨语言等价，不复现论文表格，也不意味着 ARF 已经 `benchmark-eligible`、可以进入 Official Results 或已经 `release-supported`。这些结论分别受评测、数据集、运行资源、治理和发布门槛约束。

## 来源权威性、包身份与许可证

- 官方 Python 仓库：`bips-hb/arfpy`。
- 相关原始 R 仓库：`bips-hb/arf`。
- PyPI 包：`arfpy==0.1.1`。
- PyPI 源码发行包：`arfpy-0.1.1.tar.gz`。
- 源码发行包 SHA-256：`88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707`。
- 源码发行包大小：11,841 字节。
- 锁定 Git 提交：`6f737baaaa589f7ac3ff59f0d739ce04b0f1381c`。
- 锁定 Git tree：`68b6fc5d28578a5c21bef560bd28f4c0d2d6401c`。
- 许可证：MIT，Copyright 2023 Kristin Blesch and Marvin Wright。

该仓库没有为 0.1.1 建立 Git 标签或 GitHub Release，因此这里在文件层建立来源关系：发行包与锁定提交共有的六个文件——`LICENSE`、`README.md`、`setup.py` 和三个 `arfpy` Python 文件——均与该提交的 Git blob 逐字节一致。PyPI 在该提交约十四分钟后发布源码发行包。模型执行前，协议会验证归档名称、大小、SHA-256、全部 20 个成员、全部 16 个普通文件的哈希、元数据、依赖声明、MIT 许可证、记录的 Git blob 身份、安装包元数据、已安装运行源码哈希、已安装 `RECORD` 哈希的自洽性、导入位置和导出的 `arfpy.arf.arf` 类。

## 适配器契约

仓库自有适配器不修改上游源码。它会：

- 要求精确的 `arfpy==0.1.1`，并在每次训练和采样时验证三个已安装运行源码文件；
- 支持包含数值列和类别列的扁平单表分类与回归数据；
- 将声明的类别特征和分类目标转换为 pandas categorical 类型，并检查所有声明的数值均为有限值；
- 在显式 benchmark 预处理模块仅使用训练数据拟合数值均值和类别众数之前，拒绝缺失值；
- 支持官方 ARF 参数 `num_trees`、`delta`、`max_iters`、`early_stop`、`min_node_size`，以及 scikit-learn 执行参数 `n_jobs`；
- 支持官方 FORDE 参数 `dist="truncnorm"`、`oob=false` 和非负 `alpha`；
- 要求 CPU，因为该官方实现基于 scikit-learn 随机森林；
- 将标准化随机种子应用于 NumPy/pandas/SciPy 操作和 `RandomForestClassifier.random_state`，完成后恢复调用进程原有的全局 NumPy 状态；
- 写出样本前验证请求行数、规范列顺序、无缺失值和有限数值输出。

`arfpy==0.1.1` 虽然暴露了 `forde(oob=True)`，但该路径引用了构造函数从未创建的属性。适配器对 `oob=true` 默认报错，不会暗中修改官方源码；当前验证和上游默认路径均为 `oob=false`。

## 安全持久化边界

0.1.1 没有提供保存/加载 API。对完整官方对象使用 pickle，不仅加载时可能执行代码，还会保留拟合后的森林和经过编码的逐行训练数据；而 `forde()` 完成后，`forge()` 并不使用这两部分对象。

因此，适配器使用有类型的 `arfpy-forge-state` JSON checkpoint，只保存未经修改的官方 `forge()` 实际读取的属性：列与类型元数据、类别水平、叶节点边界与覆盖率、连续密度参数和类别概率。分布边界中的非有限浮点数使用显式 JSON 标签表示，并能精确恢复。加载时先创建一个未初始化的官方 `arfpy.arf.arf` 实例，恢复上述属性，再调用官方方法。旁路 SHA-256 记录用于发现意外修改或不完整写入。由于解析该 checkpoint 不会执行 Python 代码，经审阅的外部 checkpoint 无需启用不安全 pickle 覆盖选项即可加载。

这一持久化转换属于适配器边界，不是新的 ARF 算法。正式协议会将每个恢复后的 FORGE 属性与仍在内存中的原生对象逐项比较，并要求生成 CSV 完全一致。

不保留逐行数据是一项数据最小化措施，不代表差分隐私，也不构成信息不泄露保证。叶节点边界、密度参数、类别水平和规范训练帧指纹仍可能暴露拟合数据集的信息。因此，checkpoint 必须与其他已训练模型产物采用相同的访问控制和保留期限审查。

## 冻结环境

工作流安装 CPython 3.11 和 `requirements-arf-validation.txt` 中的全部精确依赖。它只下载官方 PyPI 源码发行包，在构建前验证 SHA-256，使用非隔离且不浮动的构建环境完成构建，以不解析依赖的方式安装本仓库，并要求 `pip check` 通过。协议会拒绝非 Linux 主机、非 3.11 解释器和任何依赖版本不一致。

等价的 Linux 命令为：

```bash
python -m pip install --upgrade "pip==25.1.1"
python -m pip install -r requirements-arf-validation.txt
curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/arf-sdist/arfpy-0.1.1.tar.gz "https://files.pythonhosted.org/packages/95/6f/a61794959d3860e23f5f2de5886b61154d40c246b38eedebf19d22e4cc35/arfpy-0.1.1.tar.gz"
echo "88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707  /tmp/arf-sdist/arfpy-0.1.1.tar.gz" | sha256sum --check
python -m pip install --no-deps --no-build-isolation --use-pep517 /tmp/arf-sdist/arfpy-0.1.1.tar.gz
python -m pip install --no-deps .
python -m pip check
```

## 冻结对照

协议覆盖二分类、多分类和回归。每个确定性夹具包含 60 行无缺失数据、两个强相关数值列、一个类别列和一个目标列。列之间的强依赖可保证每个案例都真实执行一次对抗细化迭代，而不是在初始判别器后直接结束。

对于每个夹具和随机种子 `0`、`19`、`73`，两条路径均使用 20 棵树、`delta=0`、最多一次对抗迭代、关闭 early stopping、最小叶节点大小 2、单线程、截断正态 FORDE、`oob=false`、`alpha=0`，并请求 13 行。原生路径直接构造 `arfpy.arf.arf`、调用 `forde()` 和 `forge()`；适配器路径通过 `ARFAdapter` 完成相同操作，并包含安全的训练/采样持久化边界。

九个案例必须同时满足：

1. PyPI 归档、安装包、运行源码、类身份、依赖版本和 MIT 许可证均与锁定记录一致。
2. 原生对抗循环确实执行，且原生与适配器调用均恢复调用方的 NumPy 状态。
3. JSON 恢复后的原始列、factor/object 掩码、类别水平、树数量、分布类型、叶节点边界、连续参数、类别概率和 OOB accuracy 序列完全一致。
4. Checkpoint 是不可执行 JSON，并明确声明不包含逐行训练数据和随机森林。
5. Checkpoint 与样本元数据哈希有效，标准化产物清单正确标识模型和夹具。
6. 原生与适配器 CSV 字节和重新读入的 DataFrame 完全一致，恰有 13 行规范列，数值有限，类别值位于训练集已观察域内，且无缺失值。
7. 验证完成后，已安装官方包仍保持逐字节不变。

协议不使用数值容差；确定性等价必须精确成立。

## 执行与晋级规则

权威命令为：

```bash
python -m standardized_tabular_diffusion.validation.arf \
  --repo-root . \
  --output-dir /tmp/arf-validation \
  --evidence-path /tmp/arf-evidence.json \
  --sdist-path /tmp/arf-sdist/arfpy-0.1.1.tar.gz
```

`.github/workflows/arf-validation.yml` 执行该命令并保留 JSON 证据 90 天。包、依赖、适配器、checkpoint schema 或协议发生变化后均须重新运行。只有 Linux/Python 3.11 结果通过、产物完成人工审阅，并且未经修改地保留到 `docs/evidence/arf/` 后，状态才允许晋级。

## 已保留结果

GitHub Actions 运行 [`30964711614`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30964711614) 已在 Linux 和 Python 3.11.15 环境通过。二分类、多分类、回归与随机种子组合形成的全部九个案例均通过所有精确比较，其中包括恢复后的 FORGE 状态和生成 CSV 字节。经审阅的证据已逐字节保留在 `docs/evidence/arf/native-parity-run-30964711614.json`，其 SHA-256 为 `959753701a3a615afe841c32a37bb2f2610be3a6ad421ac6476ab6f50573783f`，并已从 source lock 交叉引用。

因此，ARF 针对这一精确官方 Python 包的状态已晋级为 `native-parity-validated`。在独立的 benchmark、数据集、运行资源、治理和发布门槛通过前，它仍为 `experimental`、`unsupported`，并排除在 Official Results 之外。本结论不声称 R/Python 跨语言等价性。
