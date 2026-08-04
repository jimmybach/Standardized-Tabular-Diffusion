# TabSyn 验证协议

状态：已在 Linux/Python 3.11 通过；已完成原生一致性验证

协议 ID：`tabsyn-native-parity-v1`

支持的验证平台：Linux、Python 3.11、PyTorch 2.3 CPU

## 范围与声明边界

本协议验证标准化 TabSyn 适配器是否在不修改仓库中已跟踪的 VAE、潜空间扩散、解码和 EDM 采样器源码的情况下，调用固定版本的作者官方实现。覆盖范围包括失败即终止的源码完整性检查、真实混合类型 VAE 与扩散训练、采样、三个确定性随机种子、产物完整性，以及原生路径与适配器路径的精确对照。

验证通过后，适配器可以从 `smoke-validated` 提升到 `native-parity-validated`。但这不等于 TabSyn 已达到 `benchmark-eligible`、可以进入 Official Results track，或已经 `release-supported`。完整数据集上的质量、中央评测协议、隐私与公平性审查、运行资源阈值、更广泛任务覆盖、依赖维护和发布责任仍是相互独立的门槛。

## 源码权威与补丁处理

- 方法源码：`amazon-science/tabsyn`，提交 `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`。
- Git tree：`cb10c6da6e4b5c6f27261dfa0e4c593df9cc19ca`。
- 完整性清单：`standardized_tabular_diffusion/resources/upstream/tabsyn-source-manifest.json`。

仓库此前修改过六个官方文件，并包含一个本地 `zero` 替代模块。六个文件均已恢复为官方版本，本地替代模块也已删除。冻结清单现在覆盖 20 个文件，包括 TabSyn 主入口、VAE、扩散、解码与采样实现、共享数据工具、依赖声明，以及 Apache 许可证、NOTICE 和 README。任一哈希不匹配都会在模型运行前失败。随仓库附带的其他 baseline、上游评测脚本、数据、图片、检查点和生成产物不属于本次 TabSyn 主实现验证范围，需要分别审计。

上游 requirements 写的是 `zero`，但 TabSyn 实际导入的研究工具 API 来自 `libzero`；名为 `zero` 的发行包是无关的电路分析项目。官方 `libzero==0.0.8` wheel 的校验值已记录。由于其旧元数据要求 `torch<2`，验证环境以 `--no-deps` 安装该官方 wheel，并单独固定 NumPy、pynvml、PyTorch 和 tqdm。本协议已证明这种依赖解析在 PyTorch 2.3 上保持一致性。

## 适配器契约

仓库自有启动器只做调用层控制，并直接导入官方实现：

- `device="cpu"` 映射到 CPU，`cuda` 和 `cuda:<index>` 映射到指定 CUDA 设备；设备不可用时明确报错；
- 在官方模块执行前统一设置 Python、NumPy 和 PyTorch 随机种子；
- 把请求的设备、采样步数和样本行数传给官方采样器；
- VAE 与扩散训练仍使用官方硬编码的正式训练周期和模型结构；
- 由于官方源码没有暴露 VAE/扩散 epoch 参数，适配器会拒绝以前的本地 epoch 控制；
- 官方 TabSyn 使用耦合的固定 VAE/扩散检查点目录，因此拒绝显式外部 `checkpoint_path`；
- 内部潜变量与 PyTorch 检查点必须是 TabSyn 工作树内的普通文件，不能是符号链接。

采样会加载 PyTorch 序列化文件，而这类文件在加载时可能执行代码。只应使用在已审计 TabSyn 工作树中生成或经明确放置的检查点，并在执行前核验来源。

## 冻结环境

工作流安装 CPython 3.11、PyTorch 2.3.0 CPU，以及 `requirements-tabsyn-validation.txt` 中的精确依赖。若实际核心版本与冻结值不同，协议直接失败。

等价的本地安装命令为：

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabsyn-validation.txt
python -m pip install --no-deps "libzero==0.0.8"
python -m pip install --no-deps .
```

## 冻结对照

协议针对随机种子 `0`、`19` 和 `73` 分别从已验证清单创建两份隔离源码副本。两份副本使用同一份无缺失值混合类型二分类夹具：24 行训练数据、12 行测试数据、两个数值特征、一个类别特征和一个类别目标。

官方 TabSyn 硬编码了 4,000 个 VAE epoch、10,001 个扩散 epoch、4 个数据加载 worker、宽度 1,024 的扩散 MLP，以及默认指向 CUDA 的采样调用。源码完整性检查通过后，协议只在两份一次性副本中应用完全相同且预先声明的运行控制：2 个 VAE epoch、2 个扩散 epoch、0 个 worker、扩散宽度 64、4 个采样步、12 行输出，并显式传递 CPU 设备。这些控制不会写入仓库跟踪的上游源码；它们使真实 CI 可执行，同时让两条路径使用完全相同的官方函数和数学过程。该夹具只用于运行和一致性验证，不是模型质量 benchmark。

原生路径直接调用官方根目录 `main.py` 完成 VAE 训练、扩散训练和采样；仅验证环境中的 `sitecustomize.py` 会在官方入口执行前设置选定种子。适配器路径使用同一随机种子和运行控制调用仓库启动器。每个随机种子均必须满足：

1. 冻结范围内 20 个源码哈希全部匹配；
2. 原生路径与适配器路径的运行覆盖完全一致；
3. VAE 模型、编码器、解码器、最佳扩散模型和第 0 个 epoch 扩散模型的所有键与张量完全一致；
4. 完整潜变量数组逐元素一致且全部为有限值；
5. 最终生成 CSV 逐字节一致；
6. 恰好生成 12 行且四列结构符合预期；
7. 所有生成数值都是有限值；
8. 两个标准化产物清单都正确标识 TabSyn。

本协议不使用数值容差；确定性一致性必须完全相等。

## 执行与状态提升规则

权威执行命令为：

```bash
python -m standardized_tabular_diffusion.validation.tabsyn \
  --repo-root . \
  --output-dir /tmp/tabsyn-validation \
  --evidence-path /tmp/tabsyn-evidence.json
```

`.github/workflows/tabsyn-validation.yml` 执行该命令，并保留 JSON 证据 90 天。源码、依赖、适配器命令或协议若发生变化，已有证据立即失效，必须重新运行。

本协议已在 [GitHub Actions run 30871758645](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30871758645) 中通过，对应仓库提交 `54d419642842d7146d6afa4aa1b3d5167301c51c`。保留产物 ID 为 `8878140935`，摘要为 `sha256:72e6488aa48357f03b685e101e0c218ef73fbefc40229963ca4eee80b9dca57c`。证据的精确永久副本位于 `docs/evidence/tabsyn/native-parity-run-30871758645.json`，文件 SHA-256 为 `3b74600a9c6d5e4e841cf56bd128ac7d17b70a6d186b48a3de78d8ca476d8089`。

因此，TabSyn 当前状态为 `native-parity-validated`，但仍保持 `experimental` 与 `unsupported`；在数据集、中央评测、治理、运行资源和发布门槛分别通过之前，不得进入 Official Results。
