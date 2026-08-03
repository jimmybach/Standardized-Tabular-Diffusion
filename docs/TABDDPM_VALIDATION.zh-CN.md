# TabDDPM 验证协议

状态：可执行协议，等待首个留存的 Linux 验证结果

协议 ID：`tabddpm-native-parity-v1`

支持的验证平台：Linux、Python 3.11、CPU

## 范围与结论边界

本协议用于验证标准化 TabDDPM 训练与采样适配器是否在不改变配置和确定性输出的前提下调用固定版本的作者官方实现。验证范围包括源码完整性、真实端到端 smoke 运行、适配器与原生路径的一致性、生成产物完整性，以及三个预先声明种子组合下的可复现性。

协议通过后，适配器验证等级可以依次提升到 `smoke-validated` 和 `native-parity-validated`。但这不足以让 TabDDPM 自动成为 `benchmark-eligible`，也不代表它可以进入正式榜单或达到 `release-supported`。数据集准入、统一评测协议、隐私与公平性审阅、依赖维护和发布负责人仍是彼此独立的门槛。

## 官方来源与源码完整性

- 模型源码：`yandex-research/tab-ddpm`，提交 `b476257dd460b778ba09eb97f7a51d6490fa17f8`。
- 源码树：`b0b380892ae2fdcedadaac52a6334ad36a5d60ce`。
- 运行依赖：PyPI 官方包 `libzero==0.0.8`，SHA-256 为 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`。
- 完整性清单：`standardized_tabular_diffusion/resources/upstream/tabddpm-source-manifest.json`。

仓库最初导入了 58 个与上游一致的范围内文件，但漏掉了官方 `lib/` 包的全部六个文件，因此真实原生流水线无法运行。现在这些文件已从固定的官方提交恢复。原来的三个 `zero` 兼容文件也被否决：它把同一个种子同时赋给 Python、NumPy 和 PyTorch，而官方 libzero 会使用错开的种子，因此两者随机性语义并不相同。现在已替换为官方 wheel 中七个逐字节一致的运行模块，并保留其 MIT 许可证。

协议对 64 个范围内的 TabDDPM 文件进行统一换行后的 SHA-256 校验，对七个 libzero 模块及其许可证进行逐字节校验。任何不一致都会在模型运行前直接失败。

## 固定环境

工作流安装以下环境：

- CPython 3.11；
- PyTorch 2.3.0 CPU；
- `requirements-tabddpm-validation.txt` 中固定版本的依赖；
- 使用 `--no-deps` 安装 `rtdl==0.0.9`。

最后一点是有意为之。`rtdl==0.0.9` 的包元数据声明 `torch<2`，但本协议会实际验证固定 TabDDPM 源码所使用的有限接口在受支持的 PyTorch 2.3 环境中是否正常。这是环境兼容决策，不是模型源码补丁。官方 libzero 源码被精确 vendoring 也是因为其旧版依赖元数据与当前环境冲突。

官方入口位于 `scripts/` 子目录，而 `lib` 和 `zero` 是上游仓库根目录下的同级包。因此适配器会把上游根目录添加到 `PYTHONPATH` 最前面，原生对照命令使用完全相同的环境。这只是调用层适配，不修改上游源码或运行语义。

等价的本地安装命令为：

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps "rtdl==0.0.9"
python -m pip install -r requirements-tabddpm-validation.txt
python -m pip install --no-deps .
```

## 预先声明的对照方法

协议会创建一个确定性的纯数值二分类小型数据：训练集 24 行、验证集 8 行、测试集 8 行。配置包含三个数值特征、`[16, 16]` 的 MLP、三个优化步骤、四个扩散时间步和十二行生成结果。它只用于运行与一致性验证，不用于衡量模型质量。

三个“训练种子、采样种子”组合分别为 `(0, 23)`、`(17, 47)` 和 `(101, 89)`。对每个组合：

1. 原生路径直接执行 `scripts/pipeline.py --train` 和 `scripts/pipeline.py --sample`；
2. 标准化适配器使用其训练和采样接口执行同一配置；
3. 除输出专用的 `parent_dir` 外，两个配置必须完全一致；
4. 原始模型和 EMA 模型必须具有完全一致的参数键与张量值；
5. 所有 NumPy 生成文件的清单、数据类型、形状和元素必须完全一致；
6. 所有数值型生成结果必须为有限值；
7. loss CSV 必须完全一致；
8. 必须生成十二行标签；
9. 两个适配器产物清单必须有效，并正确标识模型和测试数据。

本协议不设置数值容差：所有确定性对照都要求精确相等。

## 执行与证据

权威执行命令为：

```bash
python -m standardized_tabular_diffusion.validation.tabddpm \
  --repo-root . \
  --output-dir /tmp/tabddpm-validation \
  --evidence-path /tmp/tabddpm-evidence.json
```

`.github/workflows/tabddpm-validation.yml` 会在 Linux/Python 3.11 上执行该命令，并将 JSON 证据保留 90 天。在成功运行链接写入来源锁定记录之前，注册表必须保持 `adapter-complete`。以后只要源码、依赖、适配器命令或协议发生变化，原证据就失效，必须重新运行验证。
