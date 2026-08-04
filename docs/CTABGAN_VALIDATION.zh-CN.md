# CTAB-GAN 验证说明

状态：native-parity-validated；Official Results 与发布支持仍待完成

## 声明边界

本文档验证统一 `ctab-gan` 适配器是否与方法作者的 CTAB-GAN 实现一致。它不代表 CTAB-GAN 已进入 Official Results，也不代表该模型已达到发布支持状态。数据集准入、中央评测协议、完整规模运行资格、治理审阅和发布测试仍是相互独立的门槛。

当前支持路径仅限分类。官方 `DataPrep` 始终执行带分层抽样的监督式训练/测试划分，因此把源码中名义上的 regression 参数描述为已验证回归接口会夸大官方实现的实际能力。

## 官方源码身份

- 仓库：`https://github.com/Team-TUD/CTAB-GAN`
- 提交：`73d4e315a2a51cf16c97ed8a00d2dad456cfce8a`
- 提交树：`3ef0223477193400d88344ff66b7ac6ffeefa173`
- `model/` 树：`89ad16bce9f0f6c23f393d9b6b2959ce8ef64bf9`
- 许可证：Apache-2.0

上游没有 tag 或 release，因此使用不可变提交和 Git 树身份锁定方法作者的最新提交。codeload 压缩包的长度与 SHA-256、每个选定文件的 Git blob、规范化字节数和规范化 SHA-256，均冻结在 `standardized_tabular_diffusion/resources/upstream/ctabgan-source-manifest.json` 中。

`TabDDPM-main/CTAB-GAN/` 只分发七个选定文件：两份上游许可证/署名文件、上游 README，以及生成所需的四个源码文件。上游数据集、生成 CSV、notebook 和上游评测模块均被有意排除；正式 benchmark 指标只能由中央评测器产生。

在仓库声明的文本规范化之后，选定源码与官方文件完全一致：哈希前将 CRLF 和 LF 统一为 LF，并规范为一个结尾换行。没有任何可执行语句被修改。

## 旧快照处置

初始导入包含 15 个 CTAB-GAN 文件，共 78,185 个 Git blob 字节。与锁定的官方树比较，在统一换行后，九个共有路径中只有两个完全一致，七个不同。该分叉修改了公开构造函数，绕过了官方分层划分，暴露了不同的优化器/设备控制，并采用不同的采样数量和随机种子行为。

这套语义分叉已从当前工作树删除，但未重写 Git 历史。数据集专用 `columns.json`、本地 train/tune/pipeline 包装脚本、修改后的评测器和本地 `model/__init__.py` 均不属于支持实现。保留的七个文件来自锁定的方法作者提交，并完整保留原许可证和署名。

## Python 3.11 兼容边界

官方依赖列表面向 scikit-learn 0.24.1，该环境不适合作为受支持的 Python 3.11 环境。官方 transformer 使用：

```python
BayesianGaussianMixture(self.n_clusters, ...)
```

在 scikit-learn 1.5.2 中，`n_components` 仍是同一参数，但改为只能通过关键字传入。因此适配器安装运行时兼容桥 `ctabgan-sklearn-keyword-only-v1`，把原整数原值转发为 `n_components=self.n_clusters`。该桥不改变估计器、参数值、源码文件或训练操作。协议的原生路径和适配器路径会彼此独立地安装同一兼容桥。未来若出现会改变算法语义的兼容问题，必须另行审阅并升级协议版本。

## 适配器契约

适配器会：

1. 在导入前验证全部七个选定源码文件；
2. 隔离上游通用的 `model` 模块命名空间；
3. 保留官方默认 `test_ratio=0.2`，除非显式提供合法比例；
4. 从 Dataset Profile 推导类别列和整数列，同时允许经过审阅的显式覆盖；
5. 要求恰好一个分类目标，并强制把目标纳入类别列；
6. 在训练集拟合的显式预处理完成前拒绝缺失值；
7. 设置 Python、NumPy、PyTorch 及可用 CUDA 的随机种子，并在结束后恢复调用方的随机状态和线程状态；
8. 序列化官方模型类，并通过源码、环境、配置、兼容桥和校验和元数据绑定 checkpoint；
9. 通过官方 synthesizer 和官方逆预处理实现显式采样行数；
10. 仅通过仓库的可信可执行产物边界接受 pickle checkpoint。

## 冻结的等价协议

协议 `ctabgan-native-parity-v1` 包含六个真实训练与采样用例：

- 变体：平衡二分类和四分类；
- 种子：`0`、`19`、`73`；
- 源数据行数：`40`；
- 列类型：连续、整数、类别和目标；
- 生成行数：`13`；
- 训练配置：1 epoch、batch size 8、随机维度 8、4 个通道、分类器维度 `[8, 8]`、单 CPU 线程。

每个用例中，原生侧直接构造并训练官方类；适配器侧独立进入统一 train/sample 契约。只有以下条件全部成立时，用例才通过：

- 选定源码身份和七个文件校验和有效；
- 适配器 manifest 以及 checkpoint/sample 元数据完整；
- 生成器张量的名称、形状、dtype 和字节完全一致；
- 预处理器、transformer 和条件生成器的序列化签名完全一致；
- 原始数据帧和有效配置签名完全一致；
- 原生侧与适配器侧的样本 CSV 字节和解析后 DataFrame 完全一致；
- 行数和列顺序完全正确；
- 数值有限、类别值属于源数据域、没有缺失值；
- 两条路径结束后调用方 NumPy 状态均保持不变。

协议会记录 pickle 文件哈希，但不比较 pickle 文件字节，因为 Python/PyTorch 序列化可能包含与语义无关的对象身份细节。模型状态则在张量和拟合组件层面进行语义且字节精确的比较。

## 冻结环境

权威工作流使用 Linux、Python 3.11、CPU PyTorch 2.3.0 和 `requirements-ctabgan-validation.txt`，任何列出的版本漂移都会失败。本地 Windows 运行只能作为诊断证据，不能用于提升验证状态。

完整性与协议命令：

```bash
python -m standardized_tabular_diffusion.cli model-source-status \
  --model ctab-gan \
  --source-dir TabDDPM-main/CTAB-GAN

python -m standardized_tabular_diffusion.validation.ctabgan \
  --source-dir TabDDPM-main/CTAB-GAN \
  --output-dir /tmp/ctabgan-validation \
  --evidence-path /tmp/ctabgan-evidence.json
```

`.github/workflows/ctabgan-validation.yml` 执行权威协议，并将 JSON 证据保留 90 天。源码、适配器、兼容行为、协议或环境任何变化都会使旧证据失效，必须重新运行并审阅新证据。

## 已保留的 Linux 证据

Pull Request 工作流运行 [`30930939961`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30930939961) 中的作业 [`92065163118`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30930939961/job/92065163118) 已在 Linux/Python 3.11 上通过。经审阅的 artifact 证明：

- 二分类/多分类 × 种子的 6/6 用例全部通过；
- 生成器/checkpoint 状态签名完全一致；
- 样本 CSV 字节和解析后的 DataFrame 完全一致；
- 七个选定源码文件、适配器 manifest 和元数据全部有效；
- 数值、类别域和缺失值检查全部通过；
- 两条路径均恢复了 NumPy 随机状态和全局 warning 状态。

经审阅的 JSON 已永久保存在 `docs/evidence/ctabgan/native-parity-run-30930939961.json`，SHA-256 为 `41788d11578c55530b55fbf392412de361ec2769c63329a3174fa15c6905d0c6`，大小为 96,534 字节。对应 GitHub artifact ID 为 `8901113892`，压缩包摘要为 `sha256:0ce878b402b284ae34e5f13d96e52480f110a7873d511670b7707b6e4cee04ae`，到期日为 2026-11-02。

## 当前决定

经审阅的 Linux 证据将 CTAB-GAN 提升为 `native-parity-validated`。它仍是 `experimental`、`unsupported`，且不进入 Official Results。Apache-2.0 已解决源码再分发问题，但数据集准入、中央评测、完整规模运行资格、治理和发布测试仍待完成。
