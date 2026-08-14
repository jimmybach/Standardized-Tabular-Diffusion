# 故障排查

## Python 或平台不匹配

发布目标是 64 位 CPython 3.11。若锁定依赖没有 wheel、转而尝试本地编译，请先检查 `python --version` 和系统位数，再用 Python 3.11 重建环境。

## 可选依赖缺失或版本错误

安装错误信息指向的最小 extra，例如 `.[quickstart]`、`.[evaluation]`、`.[validity]` 或模型专属 extra。适配器会主动拒绝其他上游版本，因为“能导入”不等于“已验证”。

## 中央评测找不到 Profile 或表格

适配器评测需要解码后的合成表、已审阅的 `dataset_profile_path` 和真实训练参考表；P4/P5 还需要独立真实测试表。路径可在 `evaluation` 配置中显式指定，也可由已物化 DatasetSpec 解析。

## 结构校验失败

不要在评测器中修复生成结果。核对列名与顺序、CSV/Parquet 格式、行数、缺失值、类型和数据集视图。真实输入有缺失值时，先使用只在训练集拟合的均值/众数预处理，再重新训练与生成。

## 输出目录已存在

Result Bundle、旧格式导入、快照和快速开始目录都不可变。请选择新目录，不要删除或修改保留证据来制造通过结果。

## 旧 summary 被拒绝

只接受 `schema_version: 1.0` 且 `protocol_name: tabstruct-aligned-v1` 的冻结格式。使用 `import-legacy-summary`，不要交给 `validate-result` 或 `build-leaderboard`。任何未知字段（包括 Official 声明）都会失败。

## checkpoint 被阻止

Pickle 和许多 PyTorch 格式能执行代码。优先使用同一运行在 `output_dir` 下生成的 checkpoint。只有在独立核验来源与完整性后，才可显式启用不安全外部 checkpoint。

## CUDA 不可用或显存不足

核对适配器的 Windows/CUDA 支持、PyTorch 是否识别 GPU，以及依赖环境是否匹配验证记录。只能调整明确可调的参数；这可能把结果转入 `standardized-tuning`。声明 GPU 的证据不得静默退回 CPU。

## 诊断快照没有 rank

这是正常行为。`partial-diagnostic` 与 `community` 只有诊断顺序，没有 Official rank；正式排名需要所有独立准入和覆盖门通过。

提交 issue 时请提供命令、Python/平台版本、稳定 reason code 和脱敏日志。不要附带私人数据行、凭证或未经审阅的 checkpoint。
