# CoDi 验证协议

状态：已针对固定的 TabSyn 快照完成原生一致性验证

## 声明边界

本集成的验证目标是 `amazon-science/tabsyn` 在提交 `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` 中发布的 CoDi benchmark 快照，而不是另一份论文作者仓库的未修改副本。

论文作者仓库 `ChaejeongLee/CoDi` 被固定在提交 `8da2af242e7c43cba86b9ff5a86d05d3411b4ed5`，仅用于来源追踪。该仓库没有检测到许可证文件。两者共有十个同名源码路径，其中五个逐字节一致、五个不同；TabSyn 还单独提供了采样入口。因此，即使验证成功，也只证明与具有许可证的 TabSyn benchmark 快照一致，不证明与论文作者原版一致，不复现论文表格，也不代表获准进入 Official Results 或已经获得发布支持。

## 源码与依赖完整性

本地 CoDi 子树包含 11 个文件，与 TabSyn tree `85c16ccfb76fbf00db6b30450ca47e9928efa8d3` 逐字节一致。失败即关闭的执行 manifest 共覆盖 24 个文件：

- 11 个 CoDi 文件；
- TabSyn dispatcher 和共享数据运行时；
- 根目录依赖文件、README、Apache-2.0 许可证及 NOTICE。

训练前以及训练或采样后，所有文件都会按 LF 规范化规则校验。任何锁定文件发生变化都会直接拒绝执行，不会因为存在其他运行假设而放宽完整性要求。

共享运行时导入名为 `zero` 的模块，但对应的正确发行包是 `libzero==0.0.8`，不是 PyPI 上另一个同名为 `zero` 的项目。官方 wheel 的 SHA-256 固定为 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`。验证工作流使用 `--no-deps` 安装它，因为其历史性的 `torch<2` 元数据不能描述我们验证的 PyTorch 2.3 环境。其导入时回退路径依赖 `tqdm`，因此协议另行将其固定为 `4.66.5` 并校验版本。

在支持的 Python 3.11 环境中，应先安装 CoDi extra，再显式安装锁定的研究工具 wheel：

```bash
python -m pip install ".[codi]"
python -m pip install --no-deps "libzero==0.0.8"
```

## 支持的数据契约

CoDi 接受具有一个目标列的已处理单表数据，任务类型可以是 `binclass`、`multiclass` 或 `regression`。运行前，适配器要求：

- 数据集标识符只能是安全的单个名称，不能是路径；
- train/test CSV 非空、列名唯一，而且列结构完全一致；
- 数值、类别和目标索引必须精确划分全部列；
- `idx_name_mapping` 必须与规范 CSV 列顺序一致；
- NumPy 数组不能使用 pickle，并且形状和值必须与 CSV 表示一致；
- 不允许缺失值或非有限数值；
- 二分类目标在训练集中必须恰好有两个类别，多分类至少有三个；
- 按任务类型放置目标后，必须至少有一个连续扩散分量和一个离散扩散分量；
- 连续训练列不能是常数，因为固定快照中的 min-max transformer 没有常数列特判。

分类任务的目标属于离散分量，回归任务的目标属于连续分量。因此，支持能力取决于最终扩散分量，而不只是特征列表是否被描述为“混合类型”。存在缺失值时，必须先调用中央的、仅在训练集上拟合的填补模块。

## 仅适配层的运行边界

仓库内追踪的上游源码不做补丁。`standardized_tabular_diffusion/compat/codi_launcher.py` 明确实现三个调用桥接：

1. `codi-cpu-device-count-v1`：只向固定 loader 的 batch 整除检查报告一个逻辑执行设备，避免纯 CPU 环境除以零；数据、batch 大小、模型、损失和优化器均不改变。
2. `codi-output-checkpoint-root-v1`：只改变源码模块用于推导 checkpoint 目录的局部文件锚点，使官方 `model_con.pt` 和 `model_dis.pt` state dict 写入 `output_dir`，而不是污染追踪源码。
3. `codi-exact-sample-count-v1`：先在完整训练数据上拟合官方 transformer，然后以确定性的重复或截断方式仅改变采样占位张量的行数；学习到的变换、类别、权重和逆扩散方程不改变。

适配器还提供确定性随机种子、CPU 或经过验证的 CUDA 选择、线程数以及官方架构和训练控制。未知参数会直接报错。

## Checkpoint 与输出安全

CoDi 会产生两个 PyTorch state-dict 文件。由于 PyTorch 序列化内容属于需要信任的可执行内容，适配器只接受当前 `output_dir` 内由本次训练产生的 checkpoint 对，拒绝外部路径和符号链接。训练 metadata 会记录两个 SHA-256、源码身份、数据 schema、随机种子、设备及完整生效配置。源码、schema 或 checkpoint 发生漂移时，采样会拒绝继续。

生成 CSV 必须具有精确请求行数、规范列顺序、零缺失值、有限的连续列，而且离散值必须属于训练阶段拟合的取值域。单独的 sample metadata 会把输出哈希与两个 checkpoint 哈希绑定。

## 原生一致性协议

Linux/Python 3.11 协议运行九个真实案例：

- 二分类、多分类和回归；
- 每种任务分别使用随机种子 0、19 和 73；
- 混合数值与类别特征，以及一个按任务类型处理的目标列；
- 对连续和离散两个扩散模型进行受限但真实的训练；
- 从十二行训练数据中请求七行样本，以实际覆盖精确行数桥接。

每个案例都会建立两个隔离源码根目录。native 根目录使用通过 checksum 校验的源码，并仅为验证临时透明地覆盖 CPU 运行、小网络、两个扩散步、一个 epoch multiplier、确定性种子和七行采样。adapter 根目录保持 checksum 完全一致，并通过兼容 launcher 接收等价的公开控制。

协议要求：连续 checkpoint 状态精确一致、离散 checkpoint 状态精确一致、CSV 在字节和 DataFrame 层面完全一致、行数和 schema 精确、数值有限、零缺失值、manifest 与 metadata 有效、执行后源码仍然完整，并且 adapter 源码树下不存在 checkpoint。

验证环境由 `requirements-codi-validation.txt` 固定，并使用 CPU 版 PyTorch `2.3.0`。即使验证失败也会上传证据。

九个案例均在 GitHub Actions [运行 `30941940893`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30941940893) 的 Linux、Python 3.11.15 和 PyTorch 2.3.0 CPU 环境中通过。每个案例的两个 checkpoint 状态均精确一致，生成 CSV 在字节和 DataFrame 层面均精确一致，七行请求样本全部存在，而且所有安全性及源码完整性断言均通过。经审阅的证据已永久保留在 [`docs/evidence/codi/native-parity-run-30941940893.json`](evidence/codi/native-parity-run-30941940893.json)。这只支持把固定的 TabSyn 快照晋级为 `native-parity-validated`。

## 尚未完成的门槛

快照一致性通过不会自动使 CoDi 成为 `benchmark-eligible` 或 `release-supported`。剩余门槛包括中央指标执行、数据集 Profile 准入、全规模运行特征评估、正式 benchmark 配置审批、许可证与治理审查，以及发布审查。若要声明论文原始实现一致性，还必须获得可合法使用的论文作者源码，并单独完成等价性判断。
