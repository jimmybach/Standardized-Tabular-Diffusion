# Goggle 源码与验证记录

状态：原生等价性已验证；榜单与发布门槛仍未完成<br>
协议：`goggle-method-author-native-parity-v1`<br>
正式环境：Linux + Python 3.11

## 范围

本文明确本仓库中的 `goggle` 指向哪份权威源码、适配层允许改变什么，以及状态升级前必须取得什么证据。本文不代表该模型已具备正式榜单资格或发布支持。

复现目标是 ICLR 2023 论文 *GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure* 的方法作者实现：

- 仓库：`https://github.com/vanderschaarlab/GOGGLE`；
- 提交：`1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0`；
- 仓库树：`2d6a54f6d6f4d156890bf4e035119dbb483a46d0`；
- `src/goggle` 树：`6dcaae801859f63e173537445548a50cd1f8625b`；
- 许可证：MIT，Copyright 2023 Tennison Liu；
- 固定压缩包 SHA-256：`62dc6c98a2067d950513b4fe6343715f03a6a096990241fc6143b18fb56aaf65`。

源码清单固定了 18 个文件，包括许可证、作者信息、README、官方环境和构建声明，以及所有运行时 Python 模块。物化器先校验压缩包大小与摘要，只解压清单内路径，再按已声明规则规范化文本并逐文件复验。源码保存在被 Git 忽略的缓存中，不作为可被随意修改的本地副本提交。

## 已退役的旧副本

原 `TabSyn-main/baselines/goggle` 并不是方法作者原版。它包含 11 个文件；与官方包共享的 9 个路径在文本规范化后全部不同。实质差异包括：

- 不同的 `fit` 接口和外部数据加载器；
- 不同的编码器/解码器宽度及批大小默认值；
- 移除了官方验证集划分和早停逻辑；
- 改变了检查点位置和采样约束；
- 改变了图解码器和 RGCN 的导入。

当前代码树已经删除该副本，但没有重写 Git 历史。它不再作为原生证据，也不会被描述为官方实现。

## 适配边界

适配器直接调用未修改的官方 `GoggleModel.fit`。官方模型、图结构学习器、编码器、解码器、损失函数、交替优化器、带随机种子的训练/验证划分、验证选择、早停和 state-dict 序列化均保留在上游源码中。

只有以下五类操作位于上游源码之外：

1. **源码与产物安全。** 执行前后都校验源码。训练时把 `output_dir` 设为工作目录，使官方的 `tmp/<dataset>.pt` 不会写入源码树；随后只移动未改变张量内容的 state dict 到 `output_dir/model.pt`。采样使用 `torch.load(..., weights_only=True)`，并拒绝未经明确授权的外部检查点。
2. **数值表接口。** 官方实验接收有限数值 `DataFrame`。数值特征只用真实训练集拟合总体标准化，与 `StandardScaler` 等价；类别特征也只用训练集确定类别并进行确定性 one-hot 编码。唯一目标列仍在联合建模向量内。
3. **输出接口。** 请求的正整数行数直接传给未修改的 `Goggle.model.sample` 核心。数值列做逆标准化，one-hot 块用训练类别上的 argmax 恢复，分类目标映射到最近的已记录类别编码。这样把官方接口中“参考 DataFrame 的行数”改成明确的行数参数。
4. **旧 Synthcity 导入边界。** 官方模块会提前导入 Synthcity 0.2.2 的指标和 `Schema`，但训练与核心采样都不会执行它们。Python 3.11 适配层只提供这些导入名称；如果真的实例化 `Schema`，立即报错。正式评测始终使用仓库的中央版本化评测器。
5. **未使用的 RGCN 导入边界。** 官方 `GraphDecoder.py` 在默认同构 GCN/SAGE 路径下也会提前导入 `RGCNConv`。当 `torch-sparse` 不存在时，未使用的符号被替换为“一旦实例化就报错”的占位符。已验证的 GCN 路径从不构造它；`decoder_arch="het"` 仍必须安装官方编译扩展栈，并且不在当前验证结论内。

没有上游补丁文件，也没有修改任何官方可执行语句。

## 数据接口

适配器支持分类和回归表格，要求：

- 至少包含数值或类别特征中的一种；
- 恰好一个目标列；
- 非空的真实训练 CSV，且列及顺序与规范数据集完全一致；
- 数值必须有限；
- 不允许缺失值。

遇到缺失值默认报错。用户必须先调用中央填补模块；该模块仅在真实训练集上拟合数值均值和类别众数。适配器不会静默填补。

模型联合生成特征和目标。预处理元数据、训练配置、源码身份、检查点摘要和变换拟合范围记录在 `goggle-model-metadata.json`。采样必须同时拥有该元数据和带哈希的运行配置；检查点或配置被篡改时会拒绝运行。

## 支持的参数

公开参数覆盖官方构造与训练参数：编码器/解码器宽度和层数、异构节点编码、GCN/SAGE/异构解码器选择、图阈值、图先验与掩码、KL 与图损失权重、交替优化开关、学习率、权重衰减、轮数、批大小、耐心值、日志间隔、随机种子、设备和线程数。

默认值与方法作者源码一致：编码器和解码器宽度 64、两层、GCN、阈值 0.1、`alpha=beta=0.1`、交替优化、学习率 0.005、权重衰减 0.001、1,000 轮、批大小 32、耐心值 50、日志间隔 100。未知参数、非法范围、非方阵先验和非二值掩码都会在执行前报错。

## 正式等价性协议

强制工作流安装固定的 CPU 环境：PyTorch 2.3.0、DGL 1.1.3、torch-geometric 2.5.3、NumPy 1.26.4、pandas 2.2.3 和 scikit-learn 1.5.2。随后每个案例都会物化并校验两份相互隔离的官方源码。

九个案例覆盖：

- 二分类、多分类和回归；
- 随机种子 0、19 和 73。

每个案例使用 12 行混合类型训练数据并请求 7 行样本。独立原生路径直接调用官方 `GoggleModel.fit` 与 `Goggle.model.sample`；标准路径使用相同的变换后输入和有效配置调用公开适配器。只有同时满足以下条件，案例才通过：

- 检查点的所有键、形状、类型和张量值完全一致；
- 核心原始采样数组完全一致；
- 最终样本 DataFrame 与 CSV 字节完全一致；
- 行数和规范列顺序完全一致；
- 数值有限且无缺失；
- 适配器元数据精确描述固定源码与有效配置；
- 检查点位于源码树之外；
- 执行后 18 个源码文件仍全部匹配清单。

九个案例已在 GitHub Actions 运行 [`30945676747`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30945676747) 中全部精确通过。经检查的 JSON 以原始字节提交为 `docs/evidence/goggle/native-parity-run-30945676747.json`，SHA-256 为 `1dbcf50194505820cac0650ba72d519f4f331008bbcaac635f8eb846bec7da59`。工作流产物保留 90 天；仓库中的永久副本是长期证据记录。

## 使用方式

首次使用先物化固定源码：

```bash
python -m standardized_tabular_diffusion.cli materialize-model-source --model goggle
python -m standardized_tabular_diffusion.cli model-source-status --model goggle
```

在 Linux/Python 3.11 安装运行环境并执行 smoke preset：

```bash
python -m pip install "standardized-tabular-diffusion[goggle]"
python -m standardized_tabular_diffusion.cli run --config configs/smoke/goggle-adult-smoke.json
```

正式协议：

```bash
python -m standardized_tabular_diffusion.validation.goggle \
  --repo-root . \
  --output-dir /tmp/goggle-validation \
  --evidence-path /tmp/goggle-evidence.json
```

## 尚未完成的门槛

即使精确原生等价性通过，Goggle 仍保持 `experimental` 和 `unsupported`。正式榜单资格还需要冻结的中央评测协议、获批的数据集档案、资源策略验证、代表性规模运行，以及对尚未验证的 SAGE/异构解码器路径作出明确决定。正式发布支持还需要完成打包、安装、安全、文档和长期维护审查。
