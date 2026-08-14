# 快速开始

状态：软件发布候选示例，仅用于诊断。

## 支持环境

主环境为 Windows 11 x86-64 + CPython 3.11；Linux x86-64 + CPython 3.11 是次要可移植环境。

## 干净安装

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[quickstart]"
std-tabular-diffusion quickstart --output artifacts/quickstart
~~~

Linux 使用 `source .venv/bin/activate` 激活环境，其余命令不变。

该命令使用包内自带的、小型人工 Apache-2.0 数据，不下载数据集，也不使用任何私人输入。它会依次运行：

1. 未修改的官方 `imbalanced-learn==0.14.2` SMOTE；
2. 中央 `p3-validity@0.3.0` 评测器；
3. Result Bundle 校验；
4. 不含 Official 排名的不可变 `partial-diagnostic` 快照。

可再次独立校验：

~~~powershell
std-tabular-diffusion validate-result --bundle artifacts/quickstart/evaluation-result
std-tabular-diffusion validate-leaderboard --snapshot artifacts/quickstart/diagnostic-snapshot
~~~

`valid: true` 只表示 schema、校验和、跨文件不变量及该诊断流程通过；不表示 SMOTE 是联合生成模型，也不表示它已获得发布支持、榜单资格、隐私保证或科学优越性。

## 评测自己的解码表格

准备经过审阅的 Dataset Profile，并使用全新的输出目录：

~~~powershell
std-tabular-diffusion evaluate-table --protocol p3-validity --reference real_train.csv --synthetic synthetic.csv --dataset-profile dataset-profile.json --output artifacts/my-result
std-tabular-diffusion validate-result --bundle artifacts/my-result
~~~

P4、P5 还必须提供 `--real-test`。最终证据不可覆盖，不要复用输出目录。缺失值必须先通过训练集拟合的预处理流程；常见问题见[故障排查](TROUBLESHOOTING.zh-CN.md)。
