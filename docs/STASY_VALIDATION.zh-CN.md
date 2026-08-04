# STaSy 验证协议

状态：实现已完成；等待强制性的 Linux/Python 3.11 证据

## 声明边界

本集成以 Amazon 官方 TabSyn 仓库提交 `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` 中发布的 STaSy baseline 快照为验证对象。它**不**声称与论文作者单独仓库在字节或行为上完全一致。

论文作者仓库为 `JayoungKim408/STaSy`，审计提交为 `3dcc660db26e31cc1bbb00cfc14bf7687fde448d`。该仓库未检测到许可证文件或 GitHub 许可证声明。TabSyn 快照具有不同的执行结构；经行尾规范化后，14 个共享路径中有 12 个不同。因此：

- 本项目只重新分发 Apache-2.0 的 TabSyn 快照源码；
- 验证名称是 **TabSyn 快照一致性**，而不是论文原始实现一致性；
- 模型继续位于 Experimental track；
- 即使技术一致性通过，Official Results 和发布支持仍然分别受阻。

## 源码完整性

运行时清单锁定 30 个文件：

- `TabSyn-main/baselines/stasy/` 下选定的全部 17 个 Python 文件；
- TabSyn 调度入口和共享预处理模块；
- Apache-2.0 许可证、NOTICE、README 和依赖声明。

本地 17 个 STaSy Python 文件在声明的 `lf-one-final-newline` 规范化后全部匹配锁定的 TabSyn 源码树。被跟踪的 Python 字节码、数据集、检查点、生成数据、其他 baseline 和 TabSyn 主模型不属于本次 STaSy 执行范围。每次训练和采样都会在导入上游代码之前校验全部 30 个文件。

## 适配器边界

被跟踪的上游源码保持不变。`standardized_tabular_diffusion/compat/stasy_launcher.py` 补充快照中缺失或无效的控制能力：

- 明确选择 CPU 或经过验证的 CUDA 设备；
- 固定 Python、NumPy 和 PyTorch 随机种子；
- 有效控制 epoch 数、batch size、score network 宽度、SDE scale 数、worker 数和线程数；
- 选择快照自带的 ODE 或 predictor-corrector 采样器；
- 精确控制采样行数；
- 将检查点从被跟踪的源码目录重定向到 `output_dir`；
- 默认保留 self-paced learning，除非用户明确关闭。

Python 3.11 环境使用 scikit-learn 1.5.2，其中 `OneHotEncoder` 已将快照使用的 `sparse` 参数更名为 `sparse_output`。适配器外的 `stasy-sklearn-onehot-keyword-v1` 桥接会把未改变的 `False` 值转发给新参数；编码器和稠密输出均不会改变。

这些控制项配置快照自身 `get_config` 返回的对象，并调用其原生预处理、score model、SDE loss、EMA、检查点、采样和逆变换函数。本项目没有使用替代性重实现模型。

原快照根 CLI 虽然接受 epoch 和采样行数参数，但 STaSy 实际并不读取它们；STaSy 模块内部还会无条件选择 CUDA，并把检查点写进源码目录。专用兼容边界使这些行为变为明确且可测试的契约，避免把无效参数误报为已生效。

## 运行环境

在 Python 3.11 中安装 STaSy extra，然后以不解析依赖的方式安装官方 `libzero==0.0.8` wheel：

```bash
python -m pip install ".[stasy]"
python -m pip install --no-deps "libzero==0.0.8"
```

TabSyn 共享运行时代码以 `zero` 为导入名使用研究工具包，但该 API 实际由 `libzero` 发行；PyPI 上名称确为 `zero` 的包与本项目无关。官方 wheel 被锁定为 `libzero-0.0.8-py3-none-any.whl`，SHA-256 为 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`。它的旧元数据要求 `torch<2`，因此已验证环境忽略该旧依赖元数据，并分别锁定包括 PyTorch 2.3.0 在内的实际依赖。

## 数据契约

适配器通过 TabSyn 处理后数据布局支持二分类、多分类和回归，同时支持数值与类别特征。目标列遵循快照针对不同任务的拼接方式。

执行前会拒绝缺失值和非有限数值。含缺失值的数据集必须先使用中央预处理模块；数值均值和类别众数只在真实训练集上拟合。这样既不会暗中采用不同的模型内缺失值策略，也能防止测试集信息泄漏。

数据集标识符不得包含路径分隔符或路径穿越片段。训练集和测试集数组必须是无需 pickle 的 NumPy 文件，行数必须一致且非零，至少包含一组特征，只能有一个目标列，并且元数据中的行数必须与数组一致。

## 检查点安全与可复现性

STaSy 检查点采用可执行 pickle 的 PyTorch 格式。因此适配器会：

- 只在配置的 `output_dir` 下写入检查点；
- 拒绝符号链接或外部提供的检查点；
- 记录检查点 SHA-256、源码清单身份、实际训练配置、随机种子和设备；
- 采样前复核检查点哈希和源码身份；
- 记录样本 SHA-256、行数、列和采样配置。

smoke preset 故意使用很小的网络和 predictor-corrector 步数，只用于证明集成行为，不能作为生成质量配置。

## 强制一致性协议

Linux/Python 3.11 工作流会执行 9 个真实案例：

- 二分类、多分类和回归；
- 随机种子 `0`、`19` 和 `73`；
- 数值与类别混合特征；
- 使用快照原生 loss、optimizer、EMA 和检查点函数训练一个真实 epoch；
- 使用快照 predictor-corrector 实现精确生成 12 行数据。

每个案例会创建两个隔离源码根目录。原生参考路径通过透明的临时源码覆盖获得受限 CPU 配置；适配器路径保持校验和完全一致，并通过公开兼容边界获得等价参数。协议要求：

- 模型、optimizer、EMA、step 和 epoch 检查点状态精确一致；
- 生成 CSV 的字节和 DataFrame 均精确一致；
- 请求的行数与列数精确一致；
- 输出不存在缺失值或非有限数值；
- 适配器 manifest 和 metadata 有效；
- 适配器执行后 30 个运行文件仍然完全匹配；
- 检查点和生成产物均未写入上游源码树。

验证环境由 `requirements-stasy-validation.txt` 冻结，并使用 CPU 版 PyTorch `2.3.0`。即使验证失败也会上传证据。在成功产物经过检查并保留到仓库之前，registry 不得从 `adapter-complete` 晋级。

## 尚未完成的门槛

快照一致性通过也不会自动使 STaSy 成为 `benchmark-eligible` 或 `release-supported`。剩余门槛包括中央指标执行、数据集 Profile 准入、全规模运行特征验证、配置审批和发布审查。若要声明论文原始实现一致性，还必须获得具有明确许可证的论文作者源码，并单独作出等价性判断。
