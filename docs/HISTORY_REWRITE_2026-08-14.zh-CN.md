# 仓库 Git 历史净化说明

状态：所有公开分支的历史已于 2026-08-14 完成重写；GitHub 托管的隐藏 Pull Request 引用与缓存页面尚待平台侧清除。

## 原因

首个带标签的软件版本发布前，维护者审计了仓库的全部历史，发现其中仍有已废弃的 Adult/Sick 行级数据物化副本、一份来源未能完全确认的 Sick 派生数据、运行产物、synthetic 目录镜像和模型/VAE 检查点。这些对象都不应属于长期维护的源码历史，任何活跃分支都不再需要它们，发行包也已经排除这些内容。

维护者在保留私有可恢复归档后，批准了一次性的破坏性历史重写。重写从每个公开分支中移除了以下路径族：

- `artifacts/`
- `data/uploads/`
- `materialized_datasets/`
- `TabDiff-main/data/adult/`
- `TabDiff-main/data/sick/`
- `TabDiff-main/synthetic/`
- `TabSyn-main/data/adult/`
- `TabSyn-main/data/sick/`
- `TabSyn-main/synthetic/`
- `TabSyn-main/tabsyn/ckpt/`
- `TabSyn-main/tabsyn/vae/ckpt/`

包内的人工 quickstart 表格被特意保留。它是本仓库原创、遵循 Apache-2.0 的测试数据，不是真实个人数据集。

## 保留与验证

历史重写使用 `git-filter-repo` 2.47.0。在修改任何远端引用之前，已创建覆盖全部 22 个分支和 31 个 Pull Request head 的私有镜像与 bundle，并通过 `git fsck`、`git bundle verify` 和 SHA-256 校验。完整的新旧提交映射随私有归档保留，不予公开，因为公开它会直接暴露已移除对象图的识别符。

验证结果如下：

- 重写前后 `main` 的 Git tree ID 完全相同，均为 `967e78c6d3232ebdc23b557edf62a34195948809`；
- 22 个公开分支名、从这些分支可达的全部 147 个提交和全部作者身份均已保留；
- 覆盖分支与 Pull Request head 的完整重写映射包含 152 个提交，没有提交被映射为全零或删除身份；
- 重写分支中唯一仍可达的数据/产物类路径是 `standardized_tabular_diffusion/resources/quickstart/train.csv`；
- 代表性行级数据与检查点 blob ID 已不存在于重写后的对象库；
- 重写后仅公开分支的 pack 约为 8 MiB，清理前约为 129 MiB；
- Gitleaks 8.30.1 扫描了重写后的全部分支历史。四个命中均是不可变 P6 证据中经审阅的 SHA-256 硬件比较身份，不是密钥；它们在重写历史中的精确指纹已记录于 `.gitleaksignore`。

由于当前 `main` 的文件树没有变化，此次重写不会改变源码、文档、包内容、科学输入、结果或结论，仅改变提交身份与祖先链。保留证据中记录的净化前提交 ID 仍是对原始验证运行的如实记录；私有映射和相同 tree 校验构成迁移审计轨迹。

## GitHub 平台边界

强制更新分支引用不会更新 GitHub 为已关闭 Pull Request 保留的只读 `refs/pull/*/head` 引用。GitHub 会拒绝维护者更新这些隐藏引用。因此，在 GitHub Support 清除受影响的 Pull Request 引用和缓存页面之前，已知的净化前对象 ID 仍可能从 GitHub 取回。普通克隆只获取重写后的维护分支历史；但显式获取 GitHub Pull Request 引用的镜像克隆，在平台侧清除完成前仍可能获取净化前对象图。

该限制被视为发布阻塞项。在维护者完成 GitHub Support 步骤，或形成一份明确且经审阅的风险决定前，仓库不会声称 GitHub 已物理清除全部旧对象，也不会创建首个发布标签。GitHub 在[从仓库中移除敏感数据](https://docs.github.com/zh/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)指南中说明了这一限制。

## 已有克隆

任何在 2026-08-14 之前克隆过本仓库的人，都应将自己未合并的独立工作单独保存，然后重新克隆仓库。在旧克隆中直接拉取重写历史，可能会使旧对象图和分叉分支引用继续留在该克隆中。

贡献者的作者身份、姓名、邮箱与时间戳均已保留。提交 ID 发生变化，是因为 Git 提交 ID 包含其父历史。
