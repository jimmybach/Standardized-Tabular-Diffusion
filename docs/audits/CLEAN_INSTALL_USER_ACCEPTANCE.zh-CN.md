# 干净安装与用户流程验收

状态：2026-08-20 已在本地通过。验收对象是以仓库 commit `6b1669e415e7b6589e5fb04ba880405efbd5a788` 为基础、包含未提交修改的候选版本。

项目总路线定位：这是[项目总路线阶段 9](../PROJECT_ROADMAP.zh-CN.md)的本地验收证据。[PR #34](https://github.com/jimmybach/Standardized-Tabular-Diffusion/pull/34) 合并后，精确 `main` merge commit 通过包括 Windows/Linux P8 发布门在内的所有托管工作流，阶段 9 因此完成。

## 声明边界

本次验收证明，该候选版本能在原生 Windows/Python 3.11 主要目标上完成构建、安装，并且能在源码 checkout 之外使用。范围包括发行包清洁性、用户工作区、官方数据集获取与预处理、代表性 CPU/GPU 流程、中央评测和 Result Bundle 校验。

它不会将任何数据集、模型、运行、指标或结果准入 Official Results，不会把适配器提升为 `release-supported`，不评价生成质量，不会把记录的 GPU 结果推广到其他硬件，也不替代每个模型的上游等价证据。本地验收候选最初包含未提交修改；后续已合并 commit 通过了托管 Windows/Linux 集成门。这会关闭项目总阶段 9，但不会关闭独立的正式 tag 发布 checklist。

## 已验收环境

- 原生 Windows 11 x86-64
- CPython 3.11.15
- CPU 路径：官方 `imbalanced-learn==0.14.2` SMOTE
- GPU 路径：官方 `ctgan==0.12.1`、PyTorch 2.8.0+cu128、CUDA 12.8
- GPU：NVIDIA GeForce RTX 5080
- 数据集：校验和锁定的 UCI Adult 官方训练/测试文件

## 验收矩阵

| 范围 | 验收动作 | 结果 |
|---|---|---|
| 构建 | 使用 Python 3.11 构建一个 wheel 和一个源码包 | 通过 |
| 归档安全 | 检查必需成员、不安全路径、链接/设备、重复成员、运行产物、凭证、保留证据和开发机绝对路径 | 通过 |
| wheel 安装 | 在新虚拟环境中安装 wheel 与 `quickstart` 依赖，并运行 `pip check` | 通过 |
| 源码包安装 | 在另一个新虚拟环境中安装源码包与 `quickstart` 依赖，并运行 `pip check` | 通过 |
| 安装包执行 | 在 checkout 之外执行包导入、元数据命令、quickstart、Result Bundle 校验和快照校验 | 通过 |
| 路径可移植性 | 在同时含空格与中文字符的目录下重复安装包流程 | 通过 |
| 数据获取 | 通过 HTTPS 下载 Adult 官方归档，并在解压前校验注册的 SHA-256 | 通过 |
| 工作区隔离 | 将 Adult 物化到显式用户工作区，在后续 CLI 进程中重新发现，并校验 manifest | 通过 |
| 缺失值 | 仅在训练数据上拟合数值均值和类别众数，再不重新拟合地转换独立测试集 | 通过 |
| CPU 流程 | 执行 Adult SMOTE 训练、生成、P3 评测、bundle finalization 与独立校验 | 通过 |
| GPU 流程 | 执行有界 Adult CTGAN CUDA 训练、生成、P3 评测、bundle finalization 与独立校验 | 通过 |
| CLI 与文档 | 校验文档中的工作区、profile、配置、运行、错误边界和输出目录行为 | 通过 |

## 代表性结果

安装包的 CPU 流程物化了精确的 32,561 行 Adult 训练集和 16,281 行测试集。训练集限定的预处理验证学得数值均值 `20.0` 和类别众数 `NY`，并使用这些训练统计量填补独立测试 fixture。最终 wheel 中的 SMOTE 运行生成 128 行，完成 Result Bundle `run-c2a5d562b1b542e4ba73a3897d23f50a`：18 个指标范围存在、0 个 pending、2 个明确不适用。

安装包的 GPU 流程先确认了预期的失败边界：仅从默认包索引安装 CTGAN extra 会解析到只支持 CPU 的 PyTorch。显式安装 PyTorch 官方 CUDA 12.8 build 后，环境正确报告 PyTorch 2.8.0+cu128、CUDA 可用与预期 RTX 5080。最终 wheel 完成了一次真实的单 epoch CTGAN 运行、生成 32 行，并最终完成 Result Bundle `run-c72ffaa7091c40d4b891a747e59942a1`：18 个指标范围存在、0 个 pending、2 个明确不适用。

这些数量只证明流程完整，不表示指标值良好或科学质量已被证明。

## 发现并修正的缺陷

1. 安装包的数据集命令原先默认写入包目录，而不是用户工作区。
2. 缺少上游源码 checkout 时，物化 manifest 无法生成可被后续命令发现的 DatasetSpec。
3. 显式工作区会在顶层运行和 P6 子进程 worker 中丢失。
4. `example-config` 生成的评测设置不能直接运行，而且未暴露关键的 seed、device、样本数和 profile 控制。
5. 旧 profile importer 正确地保持为非 Official，但已安装包没有一条安全的 P3 功能 profile 路径；现已增加故意保持诊断性和不准入状态的生成命令。
6. 源码包可能携带包含开发机路径的保留证据；现在会排除证据目录，并对两种归档执行敏感或不可移植内容检查。
7. 发布 workflow 原先在一个环境中先后安装 wheel 和源码包；现在会为每种发行包创建独立新环境。
8. 文档可能让用户误以为安装模型 extra 就保证 CUDA 可用；现在 GPU 声明要求显式安装 PyTorch 官方 CUDA build 并校验设备。
9. 归档路径校验可能受到 CI 主机路径规则影响。现在 Windows 与 Linux 会一致拒绝 POSIX/Windows 绝对路径、Windows 保留名或非法名称，以及大小写不敏感路径碰撞。

随后进行的 2026-08-21 集成审阅在 Python 3.11.15 上通过当前环境可执行的完整测试（`638 passed`，`11` 项因可选依赖跳过），在 Python 3.13.5 上通过 `640` 项测试（`9` 项因可选依赖跳过）；全仓库 Ruff、36 个源文件的 Mypy，以及使用 Python 3.11 重新构建 wheel/源码包并检查归档也全部通过。精确 commit 的干净安装由托管 Windows/Linux 发布门执行，确保它发生在提交之后，不会被追溯归因给较早的本地产物。

## 保留证据与剩余发布门

可移植的机器可读验收记录为 [clean-install-user-acceptance-windows-py311-6b1669e.json](../evidence/audits/clean-install-user-acceptance-windows-py311-6b1669e.json)。发行包会故意排除 `docs/evidence/`；证据保留在仓库历史中，不作为包 payload 分发。

随后，精确已合并 commit 已通过托管 Windows 平台家族与 Linux/Python 3.11 Core、P8 发布 CI，其中包括重新构建 wheel/源码包、检查归档和分别进行干净安装。在创建正式 tag 前，仍须针对所选 tag 候选重复适用的安全/历史与法务检查，发布产物校验和，并独立安装已发布产物。这些属于阶段 10 发布门，不会否定当前工程结果。
