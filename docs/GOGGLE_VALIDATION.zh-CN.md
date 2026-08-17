# Goggle 源码、PyTorch 图后端与验证记录

状态：已保留方法作者 GCN 核心的原生等价证据；纯 PyTorch 后端仍在完成正式验证<br>
当前协议：`goggle-pytorch-graph-backend-parity-v2`<br>
主要运行环境：Windows 11、Python 3.11、PyTorch 2.8<br>
独立图算子判定环境：Linux、Python 3.11、DGL 1.1.3

## 范围

本文严格区分两个不能混为一谈的结论：

1. 固定校验的方法作者版 Goggle GCN 核心，已经在历史协议 `goggle-method-author-native-parity-v1` 中使用 DGL 完成原生等价验证。
2. 当前普通运行环境不修改 Goggle 源码，但会用一个范围很小、可独立验证的 PyTorch 图后端，替代 Windows 上不可用的 DGL 依赖。

第二项属于“依赖兼容性重实现”，不能表述成“未修改的官方 DGL 运行路径”。它本身不会让 Goggle 自动获得 Official Results、正式榜单或发布支持资格。

## 权威源码

复现目标是 ICLR 2023 论文 *GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure* 的方法作者实现：

- 仓库：`https://github.com/vanderschaarlab/GOGGLE`；
- 提交：`1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0`；
- 仓库树：`2d6a54f6d6f4d156890bf4e035119dbb483a46d0`；
- `src/goggle` 树：`6dcaae801859f63e173537445548a50cd1f8625b`；
- 许可证：MIT，Copyright 2023 Tennison Liu；
- 固定压缩包 SHA-256：`62dc6c98a2067d950513b4fe6343715f03a6a096990241fc6143b18fb56aaf65`。

源码清单固定了 18 个文件。物化时会校验压缩包及每个选定文件，训练和采样后再次复验。源码保存在 Git 忽略的缓存中；仓库没有上游补丁文件，也没有改写任何官方可执行语句。

旧的 `TabSyn-main/baselines/goggle` 副本与方法作者源码存在实质差异，现已退役，不作为官方证据。

## 为什么替换 DGL

本项目主要支持原生 Windows 与 Python 3.11。官方 DGL 没有覆盖当前 Windows/Python/PyTorch 组合的 wheel；如果自行编译并发布私有 wheel，就会把编译器、二进制兼容和长期维护风险转移给每位用户。

经源码审计，已验证的 Goggle GCN 路径只执行三类 DGL 接口：

- 根据有序源节点和目标节点张量构造有向同构图；
- 对重复图进行不相交批处理；
- 执行带边权的同构 `GraphConv`。

`standardized_tabular_diffusion.compat.goggle_torch_graph` 只实现这段已审计接口，明确不是通用 DGL 替代品。普通安装和运行 Goggle 不再需要 DGL 或 PyTorch Geometric；DGL 只保留在独立验证环境中。

## 图算子语义

后端 `standardized-goggle-torch-graph@1.0.0` 对齐 Goggle 使用的 DGL `GraphConv` 1.1.3 语义：

- 保留边顺序和批图节点偏移；
- 按 `none`、`left`、`right`、`both` 模式使用结构出度和入度归一化；
- 边权只缩放消息，不会重新定义结构度数；
- 输入维度大于输出维度时先乘权重再聚合，否则先聚合再乘权重；
- 权重使用 Xavier uniform 初始化、偏置初始化为零，state dict 键仍为 `weight` 和 `bias`；
- 偏置与激活顺序和 DGL 一致；
- 除非显式允许，否则零入度节点直接报错。

Goggle 学到的邻接矩阵包含对角边，因此受支持路径通常不会出现零入度节点。任何超出上述小范围契约的输入都会明确报兼容性错误，而不是静默近似 DGL。

## 支持的模型路径

公开适配器只支持 `decoder_arch="gcn"`。`sage` 和 `het` 使用不同算子，目前没有经过验证的兼容实现，因此会在训练前被拒绝。为保证未修改的上游包能够导入，适配层只为这些未执行导入提供“一旦使用就报错”的占位符，不需要安装无关的编译扩展。

官方模型、图结构学习器、编码器、GCN 解码器结构、损失函数、交替优化器、带种子的训练/验证划分、早停和 state-dict 序列化仍全部位于固定校验的上游源码中。

## 其余适配边界

以下操作明确位于上游源码之外：

1. 校验源码和产物身份，并把官方相对检查点写入限制在 `output_dir`；
2. 只在真实训练集上拟合数值标准化和确定性类别 one-hot 编码；
3. 把请求行数传给未修改的 `Goggle.model.sample` 核心，再执行已记录的逆变换；
4. 为未使用的旧 Synthcity 与异构导入提供 fail-on-use 占位符；
5. 导入未修改的 Goggle 源码时，注入已记录的纯 PyTorch 图后端。

模型元数据 schema 2 会记录后端 ID、版本、语义目标、源码身份、变换、运行配置和产物摘要。采样会拒绝旧 schema 1 元数据，以及任何后端、源码、配置或检查点不一致。后端迁移前训练的模型需要重新训练。

## 数据接口

适配器支持分类和回归表格：数值与类别特征至少有一类非空，恰好一个目标列，列顺序必须规范，数值必须有限，并且不允许缺失值。缺失值默认报错，用户必须先调用只在真实训练集上拟合的均值/众数中央填补模块。目标列仍与特征联合生成。

## 验证

历史保留运行 [`30945676747`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30945676747) 已在 DGL 1.1.3 下证明方法作者 GCN 核心的精确原生等价性。不可变证据保存在 `docs/evidence/goggle/native-parity-run-30945676747.json`；它不证明这次新增的依赖替换。

协议 v2 新增两层独立检查：

1. 使用 DGL 1.1.3 对照图构造、批处理、前向输出、特征/边权/参数梯度、state dict、两种矩阵乘法顺序、激活函数及全部归一化模式；
2. 执行二分类、多分类、回归与随机种子 `0`、`19`、`73` 组合成的九个端到端案例。

每个端到端案例中，参考路径使用“未修改 Goggle + DGL”，候选路径使用“同一份 Goggle + PyTorch 后端”。通过条件包括：检查点张量完全一致、原始样本完全一致、最终 DataFrame 和 CSV 字节完全一致、行列接口准确、元数据有效、源码执行前后不变。正式 v2 结论仍需一次干净的 Linux/DGL 1.1.3 工作流；本地 Windows 诊断不能替代这一判定环境。

Windows GPU 真实功能是另一道门：必须在没有 DGL 和 PyTorch Geometric 的情况下，使用 PyTorch 2.8.0+cu128 与声明的 RTX 5080 完成训练和采样，再通过中央结构校验。这只能证明真实功能可运行，不代表生成质量或榜单资格。

## 使用方式

首次使用先物化固定源码：

```powershell
python -m standardized_tabular_diffusion.cli materialize-model-source --model goggle
python -m standardized_tabular_diffusion.cli model-source-status --model goggle
```

验证过的 Windows GPU profile 需要先从 PyTorch 官方 CUDA 源安装 PyTorch，再安装模型 extra：

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[goggle]"
python -m standardized_tabular_diffusion.cli run --config configs/smoke/goggle-adult-smoke.json
```

smoke 配置刻意保持很小，默认使用 CPU；Windows V2 验证计划会把设备覆盖为 CUDA。

只有在冻结的 Linux 验证环境中才运行 DGL 判定协议：

```bash
python -m standardized_tabular_diffusion.validation.goggle \
  --repo-root . \
  --output-dir /tmp/goggle-validation \
  --evidence-path /tmp/goggle-evidence.json
```

## 尚未完成的门槛

GCN 核心仍保留 `native-parity-validated` 来源结论，但当前 PyTorch 后端还需要正式 v2 保留证据。Goggle 仍是 `experimental` 和 `unsupported`。正式榜单资格还要求获批数据集、冻结的中央评测协议、代表性规模资源验证和单独准入。除非未来分别实现并验证，否则 SAGE 和异构解码继续保持不支持。
