# 快速开始

状态：软件发布候选示例，仅用于诊断。

## 支持环境

主环境为 Windows 11 x86-64 + CPython 3.11；Linux x86-64 + CPython 3.11 是次要可移植环境。

## 干净安装

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install ".[quickstart]"
std-tabular-diffusion quickstart --output artifacts/quickstart
~~~

如果 Windows 没有 `py` launcher，请先激活一个 Python 3.11 环境后运行 `python -m venv .venv`，或将 `py -3.11` 换成可信 CPython 3.11 可执行文件的绝对路径。如果 `python --version` 不是 3.11，请不要继续。Linux 使用 `source .venv/bin/activate` 激活，其余命令不变。

`.[quickstart]` 适用于源码 checkout。安装已下载的发行包时，使用 `path/to/standardized_tabular_diffusion-0.1.0rc1-py3-none-any.whl[quickstart]` 或对应的 `.tar.gz[quickstart]`。未来发布到包注册表后，再使用 `standardized-tabular-diffusion[quickstart]`。

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

## Adult 从下载到评测的完整流程

以下命令会验证用户工作区、Adult 官方下载、标准预处理、真实 SMOTE 训练/生成、P3 评测与 Result Bundle 校验。请在希望创建 `user-workspace` 和 `artifacts` 的父目录中运行。

下面第一条命令只适用于源码 checkout。如果已经按上文使用 `[quickstart]` 安装了 wheel 或源码归档，请直接复用该环境；`quickstart` extra 已包含此流程需要的依赖。不要在源码 checkout 之外使用 `.` 安装形式。

~~~powershell
python -m pip install ".[quickstart,data]"
New-Item -ItemType Directory -Force user-workspace | Out-Null

std-tabular-diffusion materialize-dataset --dataset adult --workspace user-workspace
std-tabular-diffusion materialization-status --dataset adult --workspace user-workspace

std-tabular-diffusion create-diagnostic-dataset-profile `
  --dataset adult `
  --workspace user-workspace `
  --output user-workspace/adult-diagnostic-profile.json

std-tabular-diffusion example-config `
  --model smote `
  --dataset adult `
  --device cpu `
  --training-seed 17 `
  --generation-seed 17 `
  --num-samples 128 `
  --dataset-profile user-workspace/adult-diagnostic-profile.json `
  --protocol p3-validity `
  --output-dir artifacts/adult-smote-001 `
  --save-config user-workspace/adult-smote-001.json

std-tabular-diffusion run --config user-workspace/adult-smote-001.json --workspace user-workspace
std-tabular-diffusion validate-result --bundle artifacts/adult-smote-001/evaluation-result
~~~

自动生成的 profile 会明确保持 `official_eligible: false`。它仅让仓库的“模型输入不得含缺失值”规则能在诊断性 P3 运行中执行，不能替代对取值域、跨列约束、隐私角色、权利和榜单准入的专业审阅。每次都应使用新的 `--output-dir`，不要覆盖已完成的运行产物。

## 评测自己的解码表格

准备经过审阅的 Dataset Profile，并使用全新的输出目录：

~~~powershell
std-tabular-diffusion evaluate-table --protocol p3-validity --reference real_train.csv --synthetic synthetic.csv --dataset-profile dataset-profile.json --output artifacts/my-result
std-tabular-diffusion validate-result --bundle artifacts/my-result
~~~

P4、P5 还必须提供 `--real-test`。最终证据不可覆盖，不要复用输出目录。缺失值必须先通过训练集拟合的预处理流程；常见问题见[故障排查](TROUBLESHOOTING.zh-CN.md)。
