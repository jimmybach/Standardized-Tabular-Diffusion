# 平台支持政策

## 1. 发布目标

主要发布环境是**原生 Windows 11 x86-64 与 CPython 3.11**。只有核心安装、命令行接口、仓库自有测试、软件包构建，以及所有声明为 `release-supported` 的工作流均在干净的主要环境中通过后，版本才能声明仓库级发布支持。

Linux x86-64 与 CPython 3.11 是次要兼容环境。部分权威上游包和历史等价协议只适用于 Linux，因此仍必须保留 Linux CI；但 Linux 通过不能替代 Windows 发布门。

macOS 目前不在支持范围内。这是对已测试发布支持范围的声明，并不表示每个命令必然无法在 macOS 上运行。

## 2. CPU 与 GPU profile

最小化库、元数据命令、数据集工具、表结构校验和可在 CPU 上运行的诊断功能，必须在 Windows 主要环境中不依赖 GPU 也能工作。

依赖 GPU 的运行使用独立的硬件 profile。没有兼容性证据时，不得将某一 GPU profile 的结果推广到其他 profile。首个 P4 Windows GPU pilot 只针对：

- 原生 Windows 11 x86-64；
- CPython 3.11；
- 具有 16 GiB 级显存的 NVIDIA GeForce RTX 5080；
- 完整记录的 NVIDIA 驱动和 CUDA 运行时；
- 校验和锁定的 TabPFN 分类与回归检查点；以及
- 完全冻结的 Python 依赖集。

该 pilot 不会使 RTX 5080 成为整个仓库的强制依赖；它只用于准入一个硬件特定的 Global Utility 执行 profile。

## 3. 证据解释

历史 Linux/Python 3.11 证据保持不可变，并继续支持其原本建立的声明。当声明的上游运行时只适用于 Linux 时，这些证据可以支持来源或适配器等价，但不能证明当前的 Windows 发布支持。

以下声明相互独立：

- `native-parity-validated`：在记录的环境中与声明的上游目标等价；
- `benchmark-eligible`：通过相应科学协议的准入；
- `release-supported`：在 Windows 主要环境中提供安装、兼容性、文档和维护支持。

更换主要发布平台不会删除历史证据，也不会自动提升或降低适配器的等价状态；它会新增一道 Windows 发布兼容门。

## 4. CI 与发布门

每个候选发布版本必须通过：

1. Windows 11/Python 3.11 核心 CI、打包和干净安装 smoke test；
2. 每个声明为 `release-supported` 组件的 Windows 原生验证；
3. 每个依赖 GPU 的正式结果所声明的 Windows GPU profile；
4. Linux/Python 3.11 次要兼容 CI；
5. 明确记录仅有 Linux 上游等价证据、但尚无 Windows 发布支持的组件。

不得编辑历史证据文件来反映本次政策变更。新的 Windows 证据必须使用新的协议或环境身份。
