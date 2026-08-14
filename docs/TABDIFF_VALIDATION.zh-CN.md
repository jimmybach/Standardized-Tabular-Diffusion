# TabDiff 验证

状态：已完成原生一致性验证；可配置随机种子与 Windows/Adult 真实功能验证均已通过

协议：`tabdiff-native-parity-v1`、`tabdiff-adult-real-function-windows-v1`

## 声明边界

现有证据证明：经过校验和锁定的作者官方实现可以由标准适配器正确调用；seed 0 保持官方确定性路径；一个真实 TabDiff 模型能够在主要 Windows + Python 3.11 + GPU 环境中完成训练，并生成三份完整 Adult 表格。

这不代表 TabDiff 已经 `benchmark-eligible` 或 `release-supported`，也不代表 Adult、诊断分数或任何具体运行已进入 Official Results。数据集、模型、结果、完整中央评测、治理与发布准入仍是彼此独立的门槛。

## 源码权威

- 仓库：`MinkaiXu/TabDiff`。
- 提交：`5ecdb3356261aea72716cc9a779f31d7ad083bf4`。
- Git tree：`052a505cb1fbee5cbc705eeb0717d90d706ffb91`。
- 清单：`standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json`。
- 许可证：MIT。

冻结范围内的 27 个文件在规范化换行后均与作者官方源码一致。此前对 `eval/mle/mle.py` 的本地语义修改仍保持删除状态。上游评测代码仅用于源码保真和运行诊断；本仓库的正式评测使用独立版本化的中央评测引擎。

## 已审计的适配边界

受版本控制的 `TabDiff-main` 源码保持不变。适配器只应用校验和锁定、失败即关闭的运行时边界：

1. `tabdiff-configurable-seed-overlay-v1` 在内存中把官方写死的 6 处 seed 0 替换为用户请求的非负种子，不改变模型、损失、优化器、日程、预处理或采样方程。
2. `tabdiff-config-path-overlay-v1` 接入公共 `RunSpec.upstream_config_path`。完整 TOML 原样传入；未提供时仍使用官方默认配置。
3. `tabdiff-pytorch-reduce-lr-verbose-bridge-v1` 仅接收并丢弃 PyTorch 2.8 已移除、只影响日志的 `verbose` 参数。
4. `tabdiff-diagnostic-plot-bypass-v1` 仅关闭某些 Windows 非 ASCII 路径下会失败的可选 `density_plots.png` 渲染；合成表和全部序列化指标仍会生成。

适配器还负责 CPU/CUDA 映射、确定性执行、默认关闭在线日志、把不同种子的 `samples.csv` 隔离到各自目录、记录实际运行身份，以及在加载外部 PyTorch checkpoint 前要求明确的信任授权。

## 随机种子与原生一致性

原 Linux + Python 3.11 + PyTorch 2.3 CPU 协议会在两个隔离副本中，用完全相同的混合类型夹具和缩短 TOML 对比原生命令与适配器。缓存配置、checkpoint 张量、生成 CSV 字节和上游指标都必须严格一致。该协议已在 [GitHub Actions run 30866879879](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30866879879) 通过；不可变证据保存在 `docs/evidence/tabdiff/native-parity-run-30866879879.json`。

扩展协议在保持 seed 0 严格一致的同时，增加了可配置种子检查：相同非零种子必须得到字节完全一致的样本，不同种子必须产生不同样本，每份运行记录必须保存实际种子。由于官方默认 `dequant_dist="none"` 明确不会还原整数列，原生一致性诊断可以显式绕过标准整数输出门；这个绕过不能用于标准化结果。

## 整数还原问题及修复

第一次真实 Adult 运行发现了一个实际输出契约问题：TabDiff 成功生成了 32,561 行，但官方默认 `dequant_dist="none"` 使 Adult 的 6 个整数列出现小数。验证器拒绝了该表，没有修补，也没有降低标准。

TabDiff 官方预处理本身已支持 `dequant_dist="round"`：其训练正向变换不做任何改动，官方逆变换则对声明的整数列执行 `numpy.rint`。因此，保留的真实功能配置选择了这一原生模式。重新训练得到的 checkpoint SHA-256 与失败的默认配置运行完全相同（`4319e6938a1ae4619cdd17a995d71f5de0d50c450ff096754e6ef6ab2e0a26f0`），证明该变化只影响输出逆变换。适配器现在会拒绝含小数的整数列，并提示标准运行使用上游 `round` 模式；它不会静默修复合成数据。

## Windows/Adult 真实功能协议

保留的缩短配置继续使用官方 10,622,977 参数网络、优化器、训练 batch size、可学习日程和 50 步扩散过程，只调整：

- `data.dequant_dist`：从 `none` 改为官方 `round`，以满足 schema 的整数逆变换；
- 训练轮数：8,000 改为 20；
- 周期验证：每 2,000 轮改为最后一轮；
- 采样 batch size：10,000 改为 4,096，以适配 16 GB 显存。

这是真实完整 Adult 数据上的端到端功能验收，不是最终质量训练。运行环境为 Windows、Python 3.11.15、PyTorch 2.8.0+cu128 和 NVIDIA GeForce RTX 5080，结果包括：

- seed 0 训练一次；
- 同一个 checkpoint 复用于生成种子 3、4、5；
- 每张表 32,561 行、15 列；
- 无缺失值和非有限数值；
- 所有整数列为整数且位于审阅范围内；
- 所有类别均属于审阅域；
- 三份样本哈希彼此不同。

模型运行证据位于 `docs/evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json`。

## 中央评测接入

随后，seed 3 的输出在不修补合成数据的情况下进入公共 `evaluate-table` 路径：

- P3 成功 finalization，结构门通过，完整有效行比例为 1.0；
- P2 成功 finalization，生成 27 个 computed Atomic Result，并由冻结的中央实现计算 Column Shapes 与 Column Pair Trends 诊断值。

精确 request fingerprint 与 bundle checksum 记录在 `docs/evidence/tabdiff/adult-central-route-windows-rtx5080-20260814.json`。这些数值只是诊断结果，不属于 Official Results。

## 复现

在锁定的验证环境中运行源码/种子一致性协议：

```bash
python -m standardized_tabular_diffusion.validation.tabdiff \
  --repo-root . \
  --output-dir /tmp/tabdiff-validation \
  --evidence-path /tmp/tabdiff-evidence.json
```

在 TabDiff GPU 环境中运行 Windows 真实功能协议：

```powershell
python -m standardized_tabular_diffusion.validation.tabdiff_adult_real_function `
  --repo-root . `
  --output-root artifacts/tabdiff-real-function/adult-windows-rtx5080-real-function-v2 `
  --evidence-path docs/evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json
```

输出根目录和上游实验目录必须不存在。数据、样本和 checkpoint 都保留为 Git 忽略的本地产物。
