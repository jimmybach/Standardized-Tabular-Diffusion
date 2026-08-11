# 平台支持政策

## 1. 发布平台家族与精确目标

主要发布平台家族是 **Windows x86-64 与 CPython 3.11**，面向用户的精确发布目标是**原生 Windows 11 x86-64 与 CPython 3.11**。

这两个声明相关，但并不相同。GitHub 的 `windows-latest` 托管 runner 当前提供 Windows Server 镜像。它通过后可以证明主要 Windows 平台家族的兼容性，但**不得**被表述为精确的 Windows 11 准入。精确准入必须由原生或自托管 Windows 11 运行证明，并记录 OS 版本与 build、架构、Python 版本、仓库 commit 和已安装依赖集合。

仓库中的可移植平台分类器有意只识别 Windows/Python 平台家族。Python 标准平台 API 无法可靠区分所有共享相同内核 build 的 Windows 11 消费者版本与 Windows Server 版本，因此该分类器不得被当作精确 OS 版本的证明。

Linux x86-64 与 CPython 3.11 是必须保留的次要兼容环境。部分权威上游包和历史等价协议只适用于 Linux，但 Linux 通过不能替代 Windows 平台家族 CI，也不能替代精确的 Windows 11 发布门。

macOS 目前不在支持范围内。这是对已测试发布支持范围的声明，并不表示每个命令一定无法在 macOS 上运行。

## 2. CPU 与 GPU profile

最小化库、元数据命令、数据集工具、表结构校验和可在 CPU 上运行的诊断功能，必须先在主要 Windows 平台家族中工作，并在发布前通过精确 Windows 11 目标的验证。

依赖 GPU 的运行使用独立的硬件 profile。没有兼容性证据时，不得将某一 GPU profile 的结果推广到其他 profile。首个 P4 Windows GPU pilot 只针对：

- 原生 Windows 11 x86-64；
- CPython 3.11；
- 具有 16 GiB 级显存的 NVIDIA GeForce RTX 5080；
- 完整记录的 NVIDIA 驱动和 CUDA 运行时；
- 校验和锁定的 TabPFN 分类与回归检查点；以及
- 完全冻结的 Python 依赖集。

该 pilot 不会使 RTX 5080 成为整个仓库的强制依赖；它只用于准入一个硬件特定的 Global Utility 执行 profile。

## 3. 证据解释

历史 Linux/Python 3.11 证据保持不可变，并继续支持其原本建立的声明。当声明的上游运行时只适用于 Linux 时，这些证据可以支持来源或适配器等价，但不能证明 Windows 发布支持。

以下证据范围必须相互分开：

- **托管 Windows 平台家族 CI：**在被记录的托管 Windows 镜像上证明可移植安装、测试、CLI 与打包兼容性；
- **精确 Windows 11 准入：**在声明的原生 Windows 11 目标上证明发布行为；
- **硬件 profile 准入：**在一个完整标识的 CPU/GPU 与依赖 profile 下证明行为。

以下生命周期声明也相互独立：

- `native-parity-validated`：在记录的环境中与声明的上游目标等价；
- `benchmark-eligible`：通过相应科学协议的准入；
- `release-supported`：在精确发布目标上提供安装、兼容性、文档和维护支持。

更换主要发布平台不会删除历史证据，也不会自动提升或降低适配器的等价状态；它会新增 Windows 平台家族门和精确目标门。

## 4. CI 与发布门

每个候选发布版本必须通过：

1. 托管 Windows/Python 3.11 主要平台家族的核心 CI、打包和干净安装 smoke test；
2. 针对同一候选 commit 的原生或自托管 Windows 11/Python 3.11 精确目标运行；
3. 每个声明为 `release-supported` 组件的精确目标验证；
4. 每个依赖 GPU 的正式结果所声明的 Windows GPU profile；
5. Linux/Python 3.11 次要兼容 CI；
6. 明确记录仅有 Linux 上游等价证据、但尚无 Windows 发布支持的组件。

不得编辑历史证据文件来反映本次政策变更。新证据必须使用新的协议或环境身份。托管 Windows Server 证据与精确 Windows 11 证据必须分别命名和报告。
