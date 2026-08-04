# CTAB-GAN+ 原生等价性验证

状态：已在 Linux/Python 3.11 上通过；适配器为 `native-parity-validated`

## 声明边界

本协议只回答一个有限问题：在冻结的验证环境中，标准化适配器是否保持了锁定且未经修改的算法作者版 CTAB-GAN+ 源码行为？

验证通过后，适配器可以晋级为 `native-parity-validated`，但不能因此成为 `benchmark-eligible`、进入 Official Results 或成为 `release-supported`。算法作者仓库没有声明许可证文件或许可证表达式。因此，本仓库不再分发其源码；即使技术验证通过，公开发布仍需要先明确源码使用权，并分别完成统一评测、数据集、运行资源和维护责任等门槛。

## 权威源码

- 仓库：`https://github.com/Team-TUD/CTAB-GAN-Plus`
- 提交：`6a6f90188cca3dac2c533fd5e8e7f20de074365b`
- 根目录 tree：`f5a08d81b0309d6635bf1c7a646965a34913fa93`
- `model/` tree：`645e6a9d5129346f5d4e29085f1bafd5de4531fd`
- 上游提交目的：修正 WGAN-GP 框架中的 critic 迭代

codeload 压缩包的 URL、字节数和 SHA-256，以及五个必需运行时文件各自的字节数和 SHA-256，均冻结在 `standardized_tabular_diffusion/resources/upstream/ctabgan-plus-source-manifest.json` 中。

官方仓库没有 Python 打包元数据。用户通过以下命令直接从算法作者仓库获取锁定源码：

```bash
python -m standardized_tabular_diffusion.cli materialize-model-source --model ctab-gan-plus
python -m standardized_tabular_diffusion.cli model-source-status --model ctab-gan-plus
```

下载器会先验证整个压缩包，再将五个运行时文件提取到 Git 忽略的 `.cache/upstream-sources/` 目录。上游 Adult/King 数据、Notebook、生成数据、`.DS_Store` 和字节码都不会复制进本仓库。

## 已退役的内嵌快照

仓库最初在 `TabDDPM-main/CTAB-GAN-Plus/` 下包含 18 个 CTAB-GAN+ 文件，共 146,977 字节。与当前官方提交对比后发现了实质差异，包括：

- 用 DataFrame 构造器替代官方 CSV 构造器；
- 删除官方的监督式训练/测试划分；
- critic 迭代次数与训练循环结构不同；
- 向上游类中加入可配置的优化器和设备行为；
- 采样批大小、随机种子和超时行为不同；
- 另有一份重复的 `model copy/` 源码。

这些并非单纯的导入路径适配。因此，该快照已从当前工作树删除，不能再标记为官方实现或等价性已验证。它仍可通过 Git 历史恢复；本次工作没有重写 Git 历史。

## 适配边界

适配器不修改任何官方源码文件，只执行以下操作：

1. 导入前验证五个官方运行时文件的哈希；
2. 隔离上游通用的 `model` 命名空间，避免与其他内嵌 baseline 冲突；
3. 生成临时 CSV，以满足官方构造器只接收 CSV 路径的接口；
4. 从标准 Dataset Specification 推导类别列和任务角色；
5. 在官方 `fit()` 构建网络之前，将公开的模型控制参数设置到官方 synthesizer 对象；
6. 在官方调用前后设置并恢复 Python、NumPy 和 PyTorch 随机状态；
7. 使用官方 synthesizer 和逆预处理器实现标准化的指定采样条数；
8. 加载 pickle 前校验源码清单和 checkpoint 哈希。

官方内部训练划分默认保留为 `0.2`。将它改成零会重新引入旧快照中的语义补丁，而不是复现官方方法。数据集特定的 `mixed_columns`、`log_columns`、`general_columns`、`non_categorical_columns` 和 `integer_columns` 可以在 Dataset Specification 元数据或动作参数中声明，但适配器会校验每一个列名。

缺失值不属于本适配器的输入契约。调用 CTAB-GAN+ 前，必须通过显式预处理模块，仅在训练集上拟合均值/众数填补器。

## 冻结环境

权威环境为 Linux + Python 3.11，主要依赖为：

- PyTorch 2.3.0 CPU；
- NumPy 1.26.4；
- pandas 2.2.3；
- scikit-learn 1.5.2；
- SciPy 1.13.1；
- six 1.17.0；
- tqdm 4.66.5。

上游 README 列出的 PyTorch 1.9.1、scikit-learn 0.24.1 等旧依赖无法组成受支持的 Python 3.11 环境。在实现本协议前，锁定的官方提交已在上述现代依赖上完成未经源码修改的真实运行。最终兼容性声明以冻结 CI 为准。

`dython` 被有意排除：它只由上游独立评测模块导入，生成过程不需要它，中央 benchmark 评测器也不会调用该模块。

## 冻结用例

强制协议同时覆盖分类和回归。每个无缺失值夹具包含 40 行、两个数值特征、一个类别特征和一个目标列。随机种子 0、19、73 共形成六个独立用例。

每个用例都执行两条完整的一轮 CPU 训练路径：

- 原生路径：直接使用官方类完成构造、参数设置、`fit`、pickle、synthesizer 采样和逆预处理；
- 适配器路径：对同一份已验证源码执行标准化 `train` 和 `sample`。

有界模型使用批大小 8、潜变量维度 8、生成器/判别器通道数 4、两个 8 单元分类器层、单线程和 13 行生成数据。它真实覆盖 WGAN-GP、预处理、下游损失、序列化和采样代码，但不是推荐的 benchmark 质量超参数。

## 强制通过标准

六个用例必须全部通过。协议要求：

1. Linux/Python 3.11 和所有冻结依赖版本完全一致；
2. 五个官方运行时文件全部匹配源码清单；
3. 原生路径与适配器路径的生成器张量、预处理状态、条件生成器、源训练帧和配置签名完全一致（不直接比较 pickle 原始字节，因为 PyTorch 序列化会写入对象特定的存储标识）；
4. 两条路径的 CSV 样本字节和 DataFrame 完全一致；
5. 产物清单以及 checkpoint/样本元数据完全正确；
6. 请求行数、列顺序、数值有限性、类别取值域和缺失值约束完全满足；
7. 原生路径和适配器路径都恢复全局 NumPy 随机状态。

任何源码、环境、元数据、checkpoint、样本或状态差异都会失败关闭，并保留诊断 JSON 产物。

## 已知边界

- 一轮训练等价性夹具不能证明论文级模型质量。
- CPU 等价性不声明 GPU 训练达到字节级一致。
- 官方预处理会再次执行监督式划分；benchmark 数据集配置必须记录实际使用的训练子集。
- 任意 mixed/general/log 列角色需要逐数据集审查。
- pickle 可执行代码。除非使用现有显式不安全覆盖并完成来源审查，适配器会拒绝外部或哈希不一致的 checkpoint。
- 即使技术等价性通过，上游没有许可证仍是独立的发布阻断项。

## 证据流程

`.github/workflows/ctabgan-plus-validation.yml` 会把锁定源码下载到临时 runner 目录，安装冻结的 Linux/Python 3.11 CPU 环境，运行真实六用例协议，并将 JSON 产物保留 90 天。源码、适配器、协议或冻结环境一旦变化，都必须重新执行强制工作流并审阅、固化新的证据记录。

## 已保留证据

GitHub Actions [运行 `30926267432`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30926267432)中的[作业 `92049288002`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30926267432/job/92049288002)，已在 Linux、Python 3.11.15 和 PyTorch 2.3.0 CPU 环境中通过本协议。该运行通过 GitHub 测试合并提交 `48837271b693b8af396f4f35cb68707b5c52e5bc` 验证了拉取请求分支提交 `473af6334d6f367b75b35736370c4dfa6adf85bf`。

分类/回归与三个随机种子组成的六个用例全部通过。每个用例的原生路径与适配器路径均具有完全一致的 checkpoint 状态签名和逐字节一致的样本 CSV，同时满足清单、元数据、随机状态恢复、13 行列顺序、无缺失值、数值有限性和类别取值域要求。该运行还依据源码清单 SHA-256 `a76cb5e64fec6d99aae2df2d66a51598bd72ae26bc7f2e0e3104bf5a1dc1652a` 验证了全部五个官方运行时文件。

经审阅的 JSON 已永久保存在 `docs/evidence/ctabgan-plus/native-parity-run-30926267432.json`，其 SHA-256 为 `df3bbf0dd46d34e8d57551048c7b7abe60340eddb3738e31d400e44344c5e5f2`。对应的 GitHub artifact ID 为 `8899232990`，压缩包摘要为 `sha256:f3abfc1e2bbd69d7858e2ce1e5b1ab0099e9b8fa06b95711a762dd16a06a2729`，到期日为 2026-11-02。该证据只提升适配器的验证级别；在上游许可证和其他独立准入门槛解决前，CTAB-GAN+ 仍为 `experimental`、`unsupported`，不得进入 Official Results，也不得作为发布支持模型。
