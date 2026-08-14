# P8 迁移与发布

状态：实现与精确原生 Windows 11/Python 3.11 退出门已在提交 `aae531b` 通过；托管发布证据待完成。

## 1. 目的

P8 建立从适配器输出到版本化评测证据的唯一公共路径，隔离 P2 以前的 summary，并定义软件发布门。它不修改指标公式、聚合权重、数据划分、模型算法或科学准入。

## 2. 旧格式边界

带有 `protocol_name: tabstruct-aligned-v1` 的 `standardized_summary.json` 冻结在 schema `1.0`，不再支持生成。在 `0.1.x` 迁移线内，`import-legacy-summary` 可把已有文件逐字节保存为：

~~~text
legacy-import/
  checksums.sha256
  legacy_import_record.json
  source/standardized_summary.json
~~~

导入器拒绝未知字段、重复键、非有限数、符号链接、覆盖、篡改和非旧协议身份。记录始终声明 `legacy-diagnostic`、`not-converted`、无 Atomic Result 证据且禁止 Official。该兼容面不会早于 `0.2.0` 删除，延期必须有书面发布决定。

## 3. 中央评测路径

公开适配器评测会解析原始解码样本、真实参考表、已审阅 Dataset Profile、精确协议、轨道、生成种子和评测种子，并写入新的 `evaluation-result/` Run Result bundle。适配器本地指标引擎不再被调用；训练、采样和安全产物位置保持不变。

P6 把最终 Result Bundle 目录作为评测阶段所需输出。其 operational aggregate/report 仍只索引执行证据；科学聚合与发布只由 P7 完成。

## 4. 发布快速开始

包内快速开始使用已声明的人工 Apache-2.0 输入和精确官方 `imbalanced-learn==0.14.2`，生成 SMOTE 表、最终 P3 bundle 和 `partial-diagnostic` 快照。它不联网，也不产生 Official rank。

## 5. 发布资产与验证

根目录包含 Apache-2.0 许可与范围说明、第三方清单、引用信息、贡献者致谢、DCO、Security Policy、行为准则、变更记录和发布清单。

最终 `0.1.0` 标签前，托管 Windows/Python 3.11 必须通过干净构建安装、核心门、旧格式迁移、纯表评测、快速开始及双重校验；Linux/Python 3.11 必须通过可移植子集；原生 Windows 11/Python 3.11 必须重复快速开始和发布清单；并在发布 commit 重新审计包内容、链接、法律清单、秘密和受限数据。

这些只证明软件发布就绪。模型支持、榜单资格、数据集/指标/运行准入及 Official 发布仍互相独立。

留存的[原生 Windows 11/Python 3.11 证据](../evidence/evaluation/p8-native-windows11-py311-aae531b.json)通过了全部八项 P8 软件退出门，并用校验和锁定实现表面。Windows 11 的注册表仍会暴露 NT `10.0` 兼容版本，因此该门同时绑定工作站产品类型 `WinNT` 与 build `26200`，而不是直接相信 `platform.release()`。
