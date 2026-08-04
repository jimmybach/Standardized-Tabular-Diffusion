# SMOTE 验证协议

状态：已在 Linux/Python 3.11 上通过；适配器为 `native-parity-validated`

协议 ID：`smote-native-parity-v1`

支持的验证平台：Linux、Python 3.11

## 范围与声明边界

本协议验证标准化 `smote` 适配器是否使用与原生直接调用完全相同的输入表、目标列、采样器选择、构造参数、随机种子、重采样请求和输出选择，调用经过校验和锁定的官方 `imbalanced-learn==0.14.2` 包。验证覆盖官方 `SMOTE`、`SMOTENC` 和 `SMOTEN`，并检查包身份、确定性输出、拟合后采样器状态、产物元数据及原生路径与适配器路径的精确一致性。

SMOTE 是仅用于分类任务的经典过采样参考方法，不是联合表格分布生成模型，不能作为生成模型的同类参与排名。其输出由原始训练行和插值得到的少数类行共同组成；若请求不同的输出行数，适配器会再从该合并表中抽样。因此，它适合下游分类效用实验，但不能把输出当作独立合成数据集参与生成保真度、隐私或记忆化排名。

强制运行通过后，适配器可以提升为 `native-parity-validated`。这不等于已经 `benchmark-eligible`、可以进入 Official Results，或已经 `release-supported`。经典参考方法的独立赛道、数据集准入、冻结评测规则、泄漏控制、运行阈值和发布责任仍是彼此独立的门槛。

## 权威来源、包身份与许可证

- 权威仓库：`scikit-learn-contrib/imbalanced-learn`。
- 发布标签：`0.14.2`。
- Git 提交：`8504e95f0160f61d1b617ca66f779646d2ee609e`。
- Git tree：`af452de62e0f5c3d7e65fdc44a32dc97078152f2`。
- PyPI wheel：`imbalanced_learn-0.14.2-py3-none-any.whl`。
- Wheel SHA-256：`f9b81c47231aa1e3a71a1e4b3cc85b42e3b14f85e3a36922f3323c4da23605ef`。
- 许可证：MIT。

模型执行前，协议会检查 wheel 文件名与哈希、压缩包路径安全、包元数据、Python 版本要求、源码与 wheel 中的许可证哈希、压缩包成员数量、已安装发行包版本、公开类身份、wheel `RECORD` 中全部 123 个带哈希文件，以及实际导入位置。本仓库不分发也不修改 imbalanced-learn 源码。

主要一手资料为 [0.14.2 release](https://github.com/scikit-learn-contrib/imbalanced-learn/releases/tag/0.14.2)、[PyPI 发行包](https://pypi.org/project/imbalanced-learn/0.14.2/)，以及 [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)、[SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html) 和 [SMOTEN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTEN.html) 的官方 API 文档。

## 适配器契约

本仓库拥有的适配器：

- 只接受恰好一个目标列且至少一个特征列的分类数据集；
- 遇到缺失值时默认报错，并要求先执行只在训练集上拟合的显式预处理；
- 全数值特征调用官方 `SMOTE`；
- 混合类型特征调用官方 `SMOTENC`，并直接传入分类列名；
- 全分类特征调用官方 `SMOTEN`；
- 原样传递 `random_state`、`k_neighbors` 和可 JSON 序列化的 `sampling_strategy`，不修改上游源码；
- 对未安装或非 0.14.2 的包、无效邻居数、单类别数据和类别样本过少等情况直接报错；
- 按数据集规范恢复列顺序，并记录源数据、平衡后数据和最终输出的行数；
- 不创建持久化模型检查点，因为每次重采样都从规范训练划分重新计算。

适配器已移除旧有的本地序数编码、取整、裁剪和逆变换。官方 `SMOTENC` 可以直接接收 DataFrame 和分类列名；再叠加一层本地编码会产生额外语义，不能继续称为严格原生包装。

## 输入与评测政策

输入只能来自规范训练划分。验证集和测试集不得参与拟合、近邻搜索、重采样、预处理统计量计算或输出行数选择。需要缺失值预处理时，数值均值和分类众数只能在训练集上拟合，再应用到其他划分。

默认使用 `sampling_strategy="auto"`。公开适配器默认 `k_neighbors=5`；冻结一致性用例使用 `k_neighbors=3`，以便在小规模少数类上执行真实算法。最小类别必须至少包含 `k_neighbors + 1` 行。

SMOTE 结果只能放入明确标注的“经典过采样参考”部分。下游分类器应在重采样后的训练表上训练，并在完全未改动的真实测试划分上评估。不得把合并后的 SMOTE 表当作生成表与训练数据计算生成质量，因为它按设计保留原始记录。

## 冻结环境

工作流安装 CPython 3.11、`requirements-smote-validation.txt` 中的精确依赖，并仅在 SHA-256 校验通过后安装官方 wheel。仓库包安装时不解析可选依赖；验证开始前 `pip check` 必须通过。平台不是 Linux、解释器不是 Python 3.11，或任一发行包版本不符时，协议都会失败。

等价的 Linux 安装命令为：

```bash
python -m pip install -r requirements-smote-validation.txt
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest /tmp/smote-wheel "imbalanced-learn==0.14.2"
echo "f9b81c47231aa1e3a71a1e4b3cc85b42e3b14f85e3a36922f3323c4da23605ef  /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## 冻结对照

协议包含三个确定性、无缺失值的二分类用例：

1. 两个数值特征，调用 `SMOTE`；
2. 两个数值特征加一个分类特征，调用 `SMOTENC`；
3. 两个分类特征，调用 `SMOTEN`。

每个用例包含 18 行源数据，类别数量为 12 和 6。`sampling_strategy="auto"` 与 `k_neighbors=3` 生成 24 行平衡表，再确定性选择 20 行。对种子 `0`、`19` 和 `73`，原生直接调用和适配器路径都必须满足：

1. wheel、发布版本、已安装包、许可证和已安装文件身份与锁定记录一致；
2. 官方采样器类、模块和构造参数完全一致；
3. 适用时，拟合后的采样策略、特征元数据、分类特征选择、编码器类别、近邻矩阵和 SMOTENC 中位数状态完全一致；
4. 不修改全局 NumPy 随机状态；
5. 平衡后行数和目标类别数量完全一致；
6. 输出 CSV 字节和重新读取的 DataFrame 完全一致；
7. 数值输出均为有限值，分类值保持在观测域内，且不存在缺失值；
8. 产物清单和 `smote_metadata.json` 精确描述本次执行。

验证不设置数值容差：全部 9 个“采样器变体 × 随机种子”用例都必须确定性精确一致。

## 留存结果

GitHub Actions 运行 [`30918785254`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30918785254) 在 Linux、Python 3.11.15 环境中通过全部 9 个“采样器变体 × 随机种子”用例。运行将 123 个带哈希的已安装文件与锁定 wheel 逐一核对，分别以种子 0、19 和 73 执行官方 SMOTE、SMOTENC 和 SMOTEN，并在每个用例中得到字节级一致的原生/适配器 CSV 输出。

经审阅的 JSON 已永久保存在 `docs/evidence/smote/native-parity-run-30918785254.json`，SHA-256 为 `1b375b93c332327dd2118c2aad9420497008be1390078e1e48e79f8270f74863`。对应 GitHub 产物 ID 为 `8896180932`，产物摘要为 `sha256:ecc8167acd739762bdaca258d5dcb6a5f23648b45e6adba9c6900c14063d1aa6`。临时 Actions 产物过期后，仓库中的永久副本仍为权威记录。

## 执行与提升规则

权威命令为：

```bash
python -m standardized_tabular_diffusion.validation.smote \
  --repo-root . \
  --output-dir /tmp/smote-validation \
  --evidence-path /tmp/smote-evidence.json \
  --wheel-path /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl
```

`.github/workflows/smote-validation.yml` 执行该命令，并保留证据产物 90 天。包、依赖、适配器、用例或协议发生任何变化，都必须重新运行。经审阅的通过证据将适配器提升为 `native-parity-validated`；在上述独立门槛全部满足前，它仍为 `experimental`、`unsupported`，不能进入 Official Results，也不能进入联合生成模型排名。
