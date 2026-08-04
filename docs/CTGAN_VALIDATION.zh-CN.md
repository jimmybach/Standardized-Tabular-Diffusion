# CTGAN 验证协议

状态：首次强制性 Linux/Python 3.11 运行尚未完成；适配器仍为 `adapter-complete`

协议 ID：`ctgan-native-parity-v1`

支持的验证平台：Linux、Python 3.11、PyTorch 2.3 CPU

## 范围与声明边界

本协议验证标准化 CTGAN 适配器是否使用与原生直接调用完全相同的数据、构造参数、随机种子、持久化 API 和采样请求，调用经校验和锁定的官方 `ctgan==0.12.1` 包。验证范围包括包身份、真实混合类型训练、保存与加载、三个确定性随机种子、产物清单，以及原生路径与适配器路径的精确比较。

强制运行通过后，CTGAN 可以提升为 `native-parity-validated`。这不代表 CTGAN 已经 `benchmark-eligible`、可以进入 Official Results，或已经 `release-supported`。数据集准入、模型质量评测、隐私与公平性审查、运行资源阈值、依赖维护和发布责任仍是独立门槛。

## 权威来源、包身份与许可证

- 方法作者仓库：`sdv-dev/CTGAN`。
- 发布标签：`v0.12.1`。
- Git 提交：`826da23f8f9385ad15fd206ecad691e04cb0ccdc`。
- Git tree：`164a4e877a6db2ca51b3cd7dbb22cbc18af536cb`。
- PyPI wheel：`ctgan-0.12.1-py3-none-any.whl`。
- Wheel SHA-256：`38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18`。
- 许可证表达式：`BUSL-1.1`。

PyPI 记录表明，该 wheel 通过可信发布机制从同一发布提交生成。模型执行前，协议会检查下载 wheel 的哈希、wheel 元数据、已安装发行包版本、许可证元数据、所有带哈希的已安装 `RECORD` 条目，以及实际导入模块的位置。若 CTGAN 从仓库内的本地源码解析，协议将直接失败。

BUSL-1.1 不是 OSI 开源许可证。0.12.1 允许非生产使用，并在上游许可证所定义的受限“合成数据创建服务”之外授予额外生产使用权。本仓库不分发 0.12.1 源码，而是将其作为可选包安装。本协议只进行研究验证，不构成法律意见，也不代表可以提供商业服务。Official Results 和正式发布仍需单独完成许可证审查。

## 旧源码副本的处理

仓库在 `TabDDPM-main/CTGAN/CTGAN/` 下包含旧的 CTGAN `0.5.2.dev0` 副本，其声明 Python `<3.10`、PyTorch `<2`，许可证为 MIT。CTGAN 适配器已不再导入该副本，验证协议也会拒绝从仓库本地解析包。由于尚未验证的 TVAE 适配器仍与该副本耦合，因此本次 CTGAN 专项修改暂时保留旧目录；将在 TVAE 验证阶段统一处理，而不是在本次任务中静默删除。

## 适配器契约

仓库自有适配器执行以下操作：

- 从精确版本的官方包导入 `CTGAN`，并拒绝其他版本；
- 通过官方 `enable_gpu` 和 `set_device` 接口映射基准的 CPU/GPU 请求；
- 转发官方公开的构造参数，不修改上游源码；
- 在训练前调用官方 `set_random_state`；
- 将类别特征和分类目标标记为离散列；
- 使用官方 `save` 与 `load` API；
- 拒绝加载符号链接或输出目录外可执行代码的检查点，除非用户显式启用现有的不安全外部检查点选项；
- 按规范数据集列顺序返回指定行数。

官方 CTGAN 要求输入不含缺失值。有缺失值的数据集必须先调用仓库预处理模块；数值均值和类别众数只能在训练集上拟合。

## 冻结环境

工作流安装 CPython 3.11、官方 PyTorch 2.3.0 CPU wheel、`requirements-ctgan-validation.txt` 中的精确依赖，并且只有在 SHA-256 校验通过后才安装 CTGAN wheel。仓库包以不解析可选依赖的方式安装。验证开始前，`pip check` 必须通过。协议会拒绝非 Linux 平台、非 Python 3.11 解释器或任何发行包版本不匹配。

等价的 Linux 安装命令为：

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ctgan-validation.txt
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest /tmp/ctgan-wheel "ctgan==0.12.1"
echo "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18  /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## 冻结对照

对于随机种子 `0`、`19` 和 `73`，两条路径使用相同的 40 行无缺失值混合类型二分类夹具：两个数值特征、一个类别特征和一个类别目标。两者均使用一个真实训练 epoch、批大小 20、PacGAN 分组大小 10、16 维嵌入、宽度为 16 的生成器/判别器层、CPU 执行，并请求 12 行样本。该有界夹具只用于执行与一致性验证，不是模型质量 benchmark。

原生路径直接构造、设种子、训练、保存、加载并采样官方 `CTGAN` 类；适配器路径通过 `CTGANAdapter` 完成相同操作。每个随机种子必须同时满足：

1. wheel、已安装包、许可证元数据和已安装文件哈希均与锁定记录一致；
2. 构造参数完全一致；
3. 所有保留的生成器键与张量值完全一致且均为有限值；
4. 夹具转换数组与数据采样器状态完全一致；
5. 采样后的 NumPy 和 PyTorch 模型随机状态完全一致；
6. 记录的生成器/判别器损失完全一致；
7. 生成 CSV 字节和重新读取的 DataFrame 均完全一致，恰好 12 行且列顺序规范；
8. 数值均为有限值，类别值属于训练集已观察域，且不存在缺失值；
9. 标准化产物清单正确标识 CTGAN 和验证夹具。

本协议不使用数值容差；确定性一致性必须完全相等。

## 执行与状态提升规则

权威命令为：

```bash
python -m standardized_tabular_diffusion.validation.ctgan \
  --repo-root . \
  --output-dir /tmp/ctgan-validation \
  --evidence-path /tmp/ctgan-evidence.json \
  --wheel-path /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl
```

`.github/workflows/ctgan-validation.yml` 执行该命令并保留证据 90 天。包、依赖、适配器或协议一旦变化，都必须重新运行。在通过证据经过审阅并永久固化之前，CTGAN 仍为 `adapter-complete`、`experimental`、`unsupported`，并排除在 Official Results 之外。
