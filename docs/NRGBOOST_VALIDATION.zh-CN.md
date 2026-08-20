# NRGBoost 验证协议

状态：已通过；已保留 Linux 原生等价证据与原生 Windows 最小真实运行证据

协议：`nrgboost-native-parity-v1`

目标：方法作者官方 `nrgboost==0.0.3` 包

权威等价性验证环境：Linux、Python 3.11

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

## Windows 诊断性源码构建

Windows 是本仓库的主要用户平台，因此项目另设一个证据强度明确较低的路径，用于验证官方实现能否在 Windows 上实际运行。`tools/build_nrgboost_windows.ps1` 下载 PyPI 官方源码包 `nrgboost-0.0.3.tar.gz`，强制校验 SHA-256 `7b9e6a2a951755a75f34f1ec1185e82c4038938de6d126b046d46ce0624bbda0`，并且不会修改解压后的源码。随后脚本会：

1. 根据 `tools/nrgboost-windows-toolchain.explicit.txt` 创建精确锁定的 conda-forge MinGW-w64 5.3.0 环境；锁文件包含每个包归档的 MD5；
2. 从已安装的官方 CPython DLL 生成 GNU 格式的 CPython 3.11 导入库；
3. 使用官方声明的 OpenMP 参数构建原生 CFFI 扩展；
4. 使用 `delvewheel==1.13.0` 打包所需的 MinGW/OpenMP 运行库；
5. 在第二个全新环境中安装构建结果，执行 `pip check`，导入编译扩展，并实际调用其 64 位采样器。

在 PowerShell 中使用 64 位 CPython 3.11 执行：

```powershell
.\tools\build_nrgboost_windows.ps1 -PythonExe C:\path\to\python.exe
```

旧版 MinGW 后端要求 `WorkRoot` 不存在或为空、只包含 ASCII 字符且总长度不超过 60 个字符；默认路径位于 `%LOCALAPPDATA%`，满足这些要求。命令会在 `WorkRoot\output` 中生成本地 wheel 和来源记录 JSON。该 wheel 只是诊断性本地制品，本仓库不会提交或再分发它，也不会将其称为作者发布的 Windows 发行包。wheel 的 ZIP 元数据与构建时间有关，因此每次构建记录实际摘要，而不把某次本地构建摘要冻结成官方发行摘要。

该流程已在 Windows x86-64、CPython 3.11.15 上从头到尾实跑：锁定源码未经补丁即可编译，修复后的 wheel 能在全新环境安装，`pip check` 通过，编译扩展采样器也通过。该证据只支持 Windows 功能探针；永久保留的 Linux 运行仍是唯一的 `native-parity-validated` 权威证据。

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

训练阶段写入不可变的 `nrgboost_metadata.json`，其中包含检查点 SHA-256。采样阶段在反序列化前校验该摘要，并把采样记录单独写入 `nrgboost_sample_metadata.json`；它不会改写复制过来的训练记录。这样，流水线可以证明采样使用了完全相同的训练制品，同时没有修改它们。

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

## 已保留的原生 Windows 最小真实运行结果

Windows 审计构建流程已从 SHA-256 为 `7b9e6a2a951755a75f34f1ec1185e82c4038938de6d126b046d46ce0624bbda0` 的官方源码发行包重新执行。流程使用校验和锁定的 MinGW-w64/OpenMP 工具链，没有修改任何源码，生成 SHA-256 为 `24d852ebad1687bb4598ac0c739922f1f4cc4a1496ce37b06630c4439622f739` 的诊断 wheel，并通过全新环境安装、`pip check`、导入和编译后 64 位采样器检查。该 wheel 不提交、不再分发；经审阅的来源记录保留在 `docs/evidence/nrgboost/windows-source-build-provenance-20260820.json`。

提交 `3271298` 上的第一次 Pipeline V2 尝试在模型执行前停止，因为干净模型环境暴露出一项验证器依赖漏项：Pipeline V2 使用了 `packaging`，但 NRGBoost 锁文件未声明它。问题 `RF-CORE-008` 已在安装 extra 和独立 Windows V2 锁中精确加入 `packaging==26.3`，Linux 权威等价环境锁保持不变。失败尝试保留在 `docs/evidence/nrgboost/windows-v2-probe-dependency-failure-3271298.json`。

仓库提交 `8fb0afe1857487fd1bdc7cbe25781974ef66af5d` 上的干净重跑在原生 Windows 11、Python 3.11.15 和 CPU 上通过 `pipeline-v2-native-windows-v1`。确定性的 256 行 Adult 派生夹具完成一次有界五树拟合，并复用于种子 `17` 和 `29` 的生成；每份结果包含 16 行规范、无缺失数据，两份结构均有效、结果彼此不同，训练产物保持不变。随后种子 17 的结果完成中央 `p3-validity` Result Bundle 最终化与校验，待定文件数为零。通过证据保留在 `docs/evidence/nrgboost/windows-v2-real-function-8fb0afe.json`，SHA-256 为 `e1ee8c473a19950cc66bb933911bf4e5aa507f295519ad936ffa6d21dd518420`。

该结论只证明所记录诊断构建和有界配置下的 Windows 最小真实功能。它不会把本地 wheel 变成官方发行版，也不会替代 Linux 官方 wheel 的原生等价权威；同时不证明代表性规模质量、榜单资格、Official Results 准入或发布支持。
