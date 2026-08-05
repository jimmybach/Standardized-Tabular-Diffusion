# NFlow 验证协议

状态：协议已实现；尚待 Linux/Python 3.11 权威证据

协议 ID：`nflows-maf-tabular-recipe-parity-v1`

支持的验证平台：Linux、Python 3.11、CPU

## 范围与声明边界

`nflows` 是通用 normalizing-flow 工具库，并不是某个表格数据合成论文的原生实现。因此，本仓库中的 `nflow` 是由未经修改的官方 `nflows==0.14` 类和本仓库明确定义的 benchmark 配方组成的模型。它对数值变量进行标准化，对类别变量进行序数编码，训练 masked affine autoregressive flow（MAF），生成连续输出，再将类别坐标舍入、裁剪并映射回仅由训练集拟合得到的类别域。

权威运行通过后，`nflow` 只能针对“官方包 + 明确声明的配方”晋级为 `native-parity-validated`。这不代表复现了任何论文表格，也不代表与 Neural Spline Flows 或其他表格 flow 方法等价；它不验证其他 flow 架构或类别表示方式，不会自动使模型成为 `benchmark-eligible`、进入 Official Results 或成为 `release-supported`。

## 来源权威性、包身份与许可证

- 权威工具库仓库：`bayesiains/nflows`。
- PyPI 包：`nflows==0.14`。
- 官方源码发行包：`nflows-0.14.tar.gz`。
- 源码发行包 SHA-256：`6299844a62f9999fcdf2d95cb2d01c091a50136bd17826e303aba646b2d11b55`。
- 源码发行包大小：45,784 字节。
- Git 标签：轻量标签 `v0.14`。
- 锁定提交：`64b856c081e5f07521b32be99da262e8338fbfe8`。
- 锁定 Git tree：`83057958f8773e35044e3aa5c13ac9c06c4a3994`。
- 许可证：MIT；源码标签中 `LICENSE.md` 的 SHA-256 为 `74a24abd8e13ac55286f5a8396a88c20da9f67a64cbc5daa8999f31843a8b948`。

0.14 版于 2020-12-02 上传至 PyPI，没有使用 Trusted Publishing，也没有独立发行签名；轻量标签及其提交同样未签名。因此，本仓库记录不可变的 PyPI 摘要、提交/tree 锁和字节比较结果，而不宣称存在加密验证的发布者来源证明。

协议在执行前验证全部 96 个归档成员、80 个普通文件、包元数据、依赖声明、9 个关键运行文件，以及全部 42 个 `nflows/` 包文件的确定性聚合摘要。这 42 个包文件与锁定 Git tree 字节完全一致；安装完成后及所有模型案例结束后还会再次验证。

官方 Git 标签包含 MIT `LICENSE.md`，但 PyPI 源码发行包遗漏了该文件，仅在元数据中声明 `License: MIT`。本仓库既不内嵌也不修改该包。该打包缺陷会被明确记录，不能误写为“发行包内许可证文件已验证”。

## 明确声明的预处理与模型配方

适配器为每个规范表格列分配一个 flow 坐标：

- 声明的数值特征以及回归目标使用仅在训练集拟合的 `StandardScaler` 均值和尺度进行标准化；
- 声明的类别特征以及分类目标先转换为字符串，再按照训练集拟合、按字典序排列的类别做序数编码；
- 生成后的类别坐标被舍入到最近整数、裁剪到已观察类别范围内，再解码为类别值。

默认正式配方包含四次以下结构重复：

1. 官方 `RandomPermutation`；然后
2. 官方 `MaskedAffineAutoregressiveTransform`：64 个隐藏单元、两个残差块、ReLU、确定性 mask、无 context、无 dropout、无 batch normalization。

基础分布为官方 `StandardNormal`。训练使用 float32、Adam（`lr=1e-3`、betas 0.9/0.999、epsilon `1e-8`、无 weight decay、无 AMSGrad）、大小为 512 的确定性乱序 batch、10 个 epoch、零 worker、单 CPU 线程和标准化 seed。为了让九组权威比较的成本可控，验证 fixture 使用同一配方族中的两个 layer、16 个隐藏单元、一个残差块、batch size 16 和三个 epoch。不支持的架构替换会直接报错，并必须建立独立协议。

这种连续序数表示是 benchmark 配方选择，并不是原生离散 flow 机制。在进入榜单之前，其统计质量必须经过中心评测器验证。

## 适配器边界与输入契约

适配器不修改任何官方包源码，只提供本仓库需要的标准化边界：

- 恰好一个分类或回归目标；
- 规范列名唯一，数值/类别角色覆盖完整且互不重叠；
- 在显式调用仅由训练集拟合的缺失值预处理模块之前，严格拒绝缺失值和非有限数值；
- 验证路径仅支持 CPU 单线程确定性执行；
- 严格保证请求行数和规范列顺序；
- 检查数值输出有限性与类别域；
- 恢复调用者原有的进程级 PyTorch 随机状态和线程数。

## 安全持久化边界

原有的 `model.pkl` 已被移除。适配器现在写出：

- `model.nflow.json`：保存包身份、配方、训练帧指纹、预处理统计量、类别集合、训练 loss、tensor 清单和保守的隐私声明；
- `model.nflow.weights.npz`：只保存无 object dtype 的 NumPy 数组，并重新装载到由声明配方构造的官方架构中。

加载时会拒绝符号链接、超大文件、路径穿越、加密或重复归档成员、异常 tensor 名、object dtype、错误的形状/类型、非有限 tensor，以及任何文件或 tensor 哈希不一致。NumPy 始终使用 `allow_pickle=False`；先根据锁定配方重建官方 `Flow`，再严格加载 state。经审阅的外部 checkpoint 因此不再需要开启不安全可执行 checkpoint 开关。

这些文件不保存逐行训练数据，也不包含可执行 Python 对象。这属于数据最小化，并不构成隐私保证：缩放统计量、类别域、loss 和训练权重仍可能泄露信息或记忆训练记录。训练产物仍需访问控制与保留期限审查。

## 冻结环境

工作流使用 CPython 3.11、CPU-only PyTorch 2.3.0，以及 `requirements-nflow-validation.txt` 中的精确依赖集合。它通过 HTTPS 下载官方源码发行包，在构建前验证 SHA-256，在不进行依赖解析和构建隔离的情况下安装该包，然后以不解析依赖的方式安装本仓库，并要求 `pip check` 通过。

等价的 Linux 命令见英文协议中的 Frozen environment 一节。

## 冻结比较

协议覆盖二分类、多分类和回归。每个无缺失 fixture 含 48 行、一个相关数值列、一个常量数值列、一个类别列和一个目标列。每个变体使用 seed `0`、`19`、`73` 并请求 13 行，因此共形成九个案例。

每个案例中，直接路径会独立预处理已落盘的规范 CSV，并直接构建、训练和采样官方类；适配器路径则通过 `NFlowAdapter` 完成训练、序列化、重新加载和采样。每个案例必须精确满足：

1. 训练集拟合的预处理状态与每个 epoch 的 loss 一致；
2. 官方模型全部 state tensor 的名称和字节值一致；
3. 重新加载后的原始连续样本、最终 DataFrame 和 CSV 字节一致；
4. artifact manifest 和 checkpoint/sample 哈希有效；
5. JSON/NumPy 持久化不可执行、不包含逐行训练数据、不夸大隐私，并声明训练产物需要访问控制；
6. 两条路径均恢复调用者的全局 PyTorch 状态和线程数；
7. 输出行数、规范列、数值有限性、类别域和无缺失值契约全部满足；
8. 验证后安装的官方包保持不变。

## 执行与晋级规则

权威命令为：

```bash
python -m standardized_tabular_diffusion.validation.nflow \
  --repo-root . \
  --output-dir /tmp/nflow-validation \
  --evidence-path /tmp/nflow-evidence.json \
  --sdist-path /tmp/nflows-0.14.tar.gz
```

`.github/workflows/nflow-validation.yml` 执行该命令并将 JSON artifact 保留 90 天。包、依赖、适配器、checkpoint schema、预处理、架构、优化器或协议发生任何变化都必须重新运行。只有 Linux/Python 3.11 结果通过、人工检查完毕，并将原始证据不变地保存在 `docs/evidence/nflow/` 后，才允许晋级。

## 当前结果

协议与工作流已经实现，但尚未保留权威 artifact。因此，`nflow` 当前仍为 `adapter-complete`、`experimental`、`unsupported`，并且在所需运行通过、证据完成审阅并提交之前继续排除在 Official Results 之外。
