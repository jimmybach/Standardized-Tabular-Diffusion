# 核心 Baseline 联合集成验证

状态：已在累计 Linux/Python 3.11 候选提交上通过

候选提交：`9da6e556a091f2501af4cd80ed938feaccb34055`

## 目的

TabDDPM、TabDiff 和 TabSyn 首先在分别从仓库审计基线派生的独立分支上通过了原生一致性协议。本次集成验证用于证明：三套已审计实现、适配器、依赖解析、源码清单、证据记录和保守状态声明可以同时存在于同一个仓库状态中。它不会取代各模型自己的验证协议，也不会扩大其声明范围。

累计分支完整保留了三个独立验证分支的提交历史，根基线仍为 `codex/repository-audit`。本次工作没有合并或关闭任何验证 PR 或审计 PR。

## 集成审阅

本次集成统一处理了注册表、模型清单、来源锁、上游审计、运行状态、第三方声明、验证包入口和测试中的共享修改，同时发现并修正了两个跨分支问题：

- TabDDPM 已在适配器注册表中晋级，但独立模型清单仍保留旧状态；现在两者均为 `native-parity-validated`。
- 恢复 TabSyn 主实现源码不等于其单独附带的 CoDi baseline 已完成审计；CoDi 因此继续保守标记为 `compatibility-patched`。

Windows checkout 还暴露出 TabDDPM 的 libzero 许可证使用原始字节哈希，而 Git 可能规范化其行尾。清单现在明确记录 `license_sha256_lf`，验证器也证明 LF 与 CRLF checkout 会得到相同的规范化哈希。官方 Python 模块仍按字节精确校验，主源码仍依照预先声明的规范化行尾规则进行哈希检查。

## 联合源码与证据检查

三套源码检查已在同一个工作树中同时通过：

| 模型 | 固定源码版本 | 范围内文件 | 验证用例 |
|---|---|---:|---|
| TabDDPM | `b476257dd460b778ba09eb97f7a51d6490fa17f8` | 64 个官方文件及 7 个精确 libzero 模块 | `(训练, 采样)` 种子 `(0, 23)`、`(17, 47)`、`(101, 89)` |
| TabDiff | `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | 27 | 官方确定性种子 `0` |
| TabSyn | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | 20 | 种子 `0`、`19`、`73` |

三份此前永久保留的证据仍与各自声明的 SHA-256 完全一致。随后又在同一个累计候选提交上生成了新证据：

| 检查 | GitHub Actions 运行 | Artifact ID | Artifact 摘要 |
|---|---:|---:|---|
| Core CI | [30873942339](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942339) | — | — |
| TabDDPM 原生一致性 | [30873942377](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942377) | `8878855123` | `sha256:72da27768abb9c92a1e4b04932f80e2d18691d304baf65853674d8ce00e90f5d` |
| TabDiff 原生一致性 | [30873942340](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942340) | `8878859972` | `sha256:dc4e3b9d5a2426a1354451d334af6fb82a65a1be8b3b9c909b619950912244c1` |
| TabSyn 原生一致性 | [30873942394](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942394) | `8878857161` | `sha256:5ee1339c00528bdc3a514d868ae1198e5e579be95766a443437ca83a3e187473` |

每份下载证据都指向该累计候选提交，并通过了与永久模型证据相同的精确比较断言。机器可读的联合证据索引位于 `docs/evidence/core-baselines/native-parity-integration-9da6e55.json`，其 SHA-256 为 `08aebe5409ffb88c980f24e0d36b12b589930717f6c7dd6b473749e79a36c860`。

## 本地质量门禁

- Ruff：通过。
- mypy：配置范围内的 14 个源码文件全部通过。
- pytest：161 项通过、3 项跳过；跳过项均属于已记录的可选运行时或平台条件。
- 源码包与 wheel 构建：通过。
- 三套源码完整性验证器：已同时通过。

全仓库 Git diff 报告的尾随空格仅存在于经过字节哈希或规范化哈希校验的官方上游文件中。为了保持与权威源码的对应关系，这些内容被有意保留，不能为了格式整洁而改写。

## 声明边界与失效规则

三个模型当前均为 `native-parity-validated`、`experimental` 和 `unsupported`。它们尚未达到 `benchmark-eligible`，不得进入 Official Results，也不是 `release-supported`。数据集准入、中央评测、完整模型质量 benchmark、运行资源阈值、隐私与公平性审查、依赖维护和发布责任仍是相互独立的门槛。

如果主源码范围、源码清单、锁定的验证依赖、适配器命令映射或模型专属协议发生变化，对应的一致性证据立即失效，必须重新运行。仅更新本次集成说明文档不会改变已验证候选提交的运行行为。
