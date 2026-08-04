# TabDiff 验证协议

状态：候选协议；等待 Linux/Python 3.11 执行

协议 ID：`tabdiff-native-parity-v1`

支持的验证平台：Linux、Python 3.11、CPU

## 范围与声明边界

本协议验证标准化 TabDiff 适配器是否在不改变训练、采样或确定性输出的情况下调用已固定版本的作者官方实现。覆盖范围包括源码完整性、真实混合类型训练与采样冒烟测试、原生命令与适配器的一致性、产物完整性，以及对上游随机种子限制的明确处理。

验证通过后，适配器可以从 `smoke-validated` 提升到 `native-parity-validated`。但这不等于 TabDiff 已达到 `benchmark-eligible`、可以进入 Official Results track，或已经 `release-supported`。数据集准入、中央指标验证、隐私与公平性审查、依赖维护和发布责任仍是彼此独立的门槛。

## 源码权威与评测器处理

- 方法源码：`MinkaiXu/TabDiff`，提交 `5ecdb3356261aea72716cc9a779f31d7ad083bf4`。
- Git tree：`052a505cb1fbee5cbc705eeb0717d90d706ffb91`。
- 完整性清单：`standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json`。

仓库此前对 `eval/mle/mle.py` 做过语义修改，涉及估计器设置、运行设备、目标函数、随机性、错误处理和边界指标语义。该补丁已经移除，并已按规范化换行后的结果恢复为固定版本的官方文件。冻结范围内的 27 个文件现在均与作者官方源码一致；任一哈希不匹配都会在模型运行前直接失败。

恢复的上游评测器用于保持源码完整性和验证原生运行路径，但它不是正式榜单指标的权威实现。正式 benchmark 结果必须使用本仓库中单独版本化并经过审阅的中央评测引擎。

## 适配器契约

适配器仅进行调用层映射：

- `device="cpu"` 映射为官方参数 `--gpu -1`；
- `cuda` 和 `cuda:<index>` 映射为相应的官方 GPU 编号；
- 受控运行默认关闭 Weights & Biases；
- 默认启用官方 `--deterministic` 选项；
- 对产物目录之外的显式 PyTorch 检查点，必须明确确认信任，因为加载它可能执行代码。

固定版本的官方 CLI 不支持配置随机种子；其确定性模式会把 Python、NumPy 和 PyTorch 的种子固定为 0。因此适配器会拒绝非零 `RunSpec.seed`，而不是静默忽略。本协议只验证 seed 0，不声明支持多随机种子或可配置种子。若要增加可配置种子，必须对上游源码修改进行单独审阅。

## 冻结环境

工作流安装 CPython 3.11、PyTorch 2.3.0 CPU，以及 `requirements-tabdiff-validation.txt` 中的精确依赖。这是 benchmark 支持的验证环境，并不表示完全复刻了上游原始的 Python 3.10/CUDA 11.7 环境。

等价的本地安装命令为：

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabdiff-validation.txt
python -m pip install --no-deps .
```

## 冻结对照

协议根据已验证的源码清单创建两个相互隔离的源码副本，因此不会向工作源码树写入数据、检查点或运行结果。两个副本使用完全相同的确定性混合类型二分类夹具：30 行训练数据、14 行测试数据、两个数值特征、一个类别特征、一个类别目标，且不包含缺失值。

上游 `--debug` 并不适合作为冒烟测试：它仍保留 8,000 个训练 epoch，并把采样 batch 设为 10,000 行。因此，在源码完整性验证通过后，协议会对两个隔离副本应用完全相同且预先声明的 TOML 超参数：1 层 Transformer、时间维度 64、4 个优化 epoch、4 个扩散时间步、batch size 32、每 2 个 epoch 验证一次、采样 batch size 32。这只改变实验配置，不修改 Python 算法源码。两条路径均使用 CPU、关闭在线日志，并固定 seed 0。数字形式的列名用于覆盖上游绘图路径。该夹具只用于运行与一致性验证，不是模型质量 benchmark。

原生路径直接调用 `main.py` 完成训练和测试，适配器路径执行同样的 train 与 sample 操作。以下条件必须全部满足：

1. 冻结范围内 27 个源码哈希均与清单一致；
2. 两份预声明 TOML 运行配置及缓存运行配置在语义上完全一致；
3. 第 4 个 epoch 的检查点中每个张量完全一致；
4. 训练阶段生成样本与 density 指标完全一致；
5. 最终生成 CSV 逐字节一致；
6. 最终上游 DCR 指标完全一致；
7. 恰好生成 12 行且四列结构符合预期；
8. 所有生成数值均为有限值；
9. 两个标准化产物清单均正确标识 TabDiff。

本协议不使用数值容差；确定性对照要求完全一致。

## 执行与状态提升规则

权威执行命令为：

```bash
python -m standardized_tabular_diffusion.validation.tabdiff \
  --repo-root . \
  --output-dir /tmp/tabdiff-validation \
  --evidence-path /tmp/tabdiff-evidence.json
```

`.github/workflows/tabdiff-validation.yml` 在 Linux/Python 3.11 上执行该命令，并保留 JSON 证据 90 天。在成功运行的证据被复制到 `docs/evidence/tabdiff/` 且由 source lock 引用之前，registry 继续保持 `adapter-complete`。源码、依赖、适配器命令或协议若发生变化，已有证据立即失效，必须重新运行。
