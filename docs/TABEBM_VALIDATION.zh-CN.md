# TabEBM 验证协议

状态：已提升为 `smoke-validated`；未执行受门控的完整 TabPFN 生成

协议：`tabebm-official-package-core-validation-v1`

目标：方法作者 `tabebm==2025.8.19` 包及 `tabpfn==2.1.2`

支持的验证环境：Linux、Python 3.11

## 声明边界

这项门槛刻意只标记 `smoke-validated`，不是原生等价验证。官方完整 `generate()` 会构造 TabPFN-v2 估计器，并可能下载需要接受外部条款和凭证的权重。公共 CI 不能绕过或假装接受这些条款。因此协议只验证包/源码身份、可确定执行的官方核心辅助函数、安全预处理与检查点，以及使用测试替身的精确官方调用边界，并明确记录 `full_tabpfn_generation_executed=false`。

## 已审计分发物

权威来源为 [andreimargeloiu/TabEBM](https://github.com/andreimargeloiu/TabEBM) 的提交及发布标签 `72eb78dab896c7a8f39c4dcc288c834fd72eff2b`，树为 `627af984e9447bf1a88f1d13e4c766704738ec28`。PyPI 源码分发物 `tabebm-2025.8.19.tar.gz` 为 19,178 字节，SHA-256 为 `6111611326747a680f93dfadcbac1d602ce20cb722b9b6cbff1f556b9f48d503`，许可证为 Apache-2.0。安装包中的两个文件与锁定标签逐字节一致。

## 适配边界

TabEBM 只支持分类。适配器要求一个目标、至少两个目标类别、至少一个特征、特征角色完整、数值有限且无缺失。数值特征使用训练集均值和总体标准差进行标准化；类别特征和目标使用确定性的排序映射。类型化 JSON 保存这些状态与训练文件摘要，不保留训练行。

除非显式设置 `allow_gated_model=true` 或等价环境开关，否则采样失败关闭。明确授权后，适配器把编码后的 `X`、`y`、SGLD 参数和标准化种子传给官方 `TabEBM.generate()`。官方按类别返回等长块，适配器为每类请求 `ceil(N/类别数)` 行，再按排序类别确定性轮询截断为恰好 `N` 行。即使上游依据 `torch.cuda.is_available()` 选择设备，适配器也会落实显式 CPU 请求。

## 强制冒烟检查

工作流验证源码分发物的全部普通文件、关键源码摘要、元数据、安装运行文件摘要和安装 `RECORD`。它直接执行官方能量计算、带种子的代理负样本构造和全训练集划分辅助函数。二分类和多分类适配器用例通过门控 `generate()` 主体的确定性替身，验证安全 JSON、参数精确传递、类别块检查、逆变换和精确行数。

## 后续提升要求

若要超过 `smoke-validated`，必须另行获得授权，在已接受 TabPFN 条款且凭证可用的 Linux 环境中执行真实官方生成路径。该验证需覆盖多个种子和分类数据，比较官方直接路径与适配器的输入、逐类输出、最终行和 CSV，并保留依赖及模型身份；中央评测和资源门槛仍需独立通过。

## 已知边界

- 官方方法和适配器均不声明支持回归。
- 冒烟验证不证明真实 TabPFN 生成样本相等。
- TabPFN 模型条款、凭证、缓存和产物访问控制由运行者负责。
- 不包含差分隐私保证。

## 证据

GitHub Actions 运行 [`30974574544`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30974574544) 已在 Linux、Python 3.11.15 环境通过。该运行验证了锁定源码分发物与安装包，执行了可确定运行的官方核心辅助函数，并通过二分类和多分类的安全状态/调用委托边界用例。经审阅的 JSON 已逐字节保留在 `docs/evidence/tabebm/smoke-validation-run-30974574544.json`，SHA-256 为 `8d461e440440d73213f31efe1b8086e9c78fed299822da2fe203ea62af3c21dc`，且明确记录 `full_tabpfn_generation_executed=false`。该证据只支持 `smoke-validated`，不支持原生等价声明。
