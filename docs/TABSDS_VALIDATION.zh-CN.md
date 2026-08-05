# TabSDS 验证协议

状态：已由保留证据的 Linux 权威运行提升为 `native-parity-validated`

协议：`tabsds-official-source-parity-v1`

目标：方法作者 Python `simple` shuffle 路径

支持的验证环境：Linux、Python 3.11

## 声明边界

本协议把适配器与校验和锁定的方法作者 Python 源码直接比较。保留证据的权威运行通过后，可对选定的 simple-shuffle 路径及声明的精确行数边界标记 `native-parity-validated`。这不等于获得再分发或正式发布授权：上游仓库未声明许可证。

## 已审计源码

权威来源为 [echaibub/TabSDS](https://github.com/echaibub/TabSDS) 的唯一锁定提交 `866501495069c7e1300bdea91c411f1947d19f2f`，树为 `0f237c7b4fa02e06c525f29d1d83ff5c460816ee`。源码压缩包为 119,407 字节，SHA-256 为 `292011aab0153ca8f7cc90c21dd4acbcbd2a22da557ab080883555b7ab0cf82a`。系统只按需获取两个必需的 notebook 辅助文件，并在使用前校验路径、规范化字节数和 SHA-256。

## 适配边界

官方 Python 实现以 notebook 辅助文件形式提供，其中一个文件假设 notebook 全局命名空间已有 `numpy` 和 `pandas`。适配器在执行逐字节一致的源码前重建该命名空间，不修改或翻译任何上游语句。

输入必须是一张无缺失、数值有限、角色声明完整且只有一个分类或回归目标的混合类型表。训练阶段只保存类型化 JSON 配方状态、模式、源码身份、行数和训练文件摘要。采样调用官方 `tab_sjppds(..., shuffle_type="simple")`。因为官方一次调用固定返回与输入相同的行数，大于训练表的请求通过重复原始调用完成，并且只截断最后一个块。

## 强制等价用例

二分类、多分类和回归各使用 0、19、73 三个种子，共 9 个用例。每个用例从 37 行训练表请求 53 行，同时覆盖官方首块和明确的重复/截断边界。官方直接路径与适配器的 CSV 必须逐字节一致；源码树和调用者 NumPy 状态必须保持不变。

## 已知边界

- 仅验证 Python simple-shuffle 路径，不对 R 代码或其他 shuffle 模式作声明。
- 精确等价不代表质量或隐私结论。
- 上游未声明许可证；源码仅获取到忽略的缓存中，本仓库不再分发。
- Official Results、发布支持、中央评测和数据集准入仍是独立且未完成的门槛。

## 证据

GitHub Actions 运行 [`30974574593`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30974574593) 已在 Linux、Python 3.11.15 环境通过。二分类、多分类、回归与三个种子组成的 9 个用例，在官方直接路径和适配器路径间获得完全一致的 DataFrame 与 CSV 字节，并实际覆盖从 37 行训练表请求 53 行的重复/截断边界。经审阅的 JSON 已逐字节保留在 `docs/evidence/tabsds/native-parity-run-30974574593.json`，SHA-256 为 `11cfa96a3221944ebb6d423fdddf8660f278e7f6b108dff500fe39a1f9b07b66`，并已从 source lock 交叉引用。上游许可证缺失仍然阻止再分发与发布。
