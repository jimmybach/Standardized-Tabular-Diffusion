# Repository History Sanitization Notice

Status: public branch history rewritten on 2026-08-14; GitHub-hosted hidden pull-request references and cached views are pending platform-side purge.

## Reason

Before the first tagged software release, the maintainers audited all repository history and found obsolete row-level Adult/Sick materializations, one unverified Sick derivative, generated run artifacts, synthetic-directory mirrors, and model/VAE checkpoints. None belonged in the maintained source history. No active branch required these objects, and the release distributions already excluded them.

The maintainers approved a one-time destructive rewrite after retaining a private recovery archive. The rewrite removed these path families from every public branch:

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

The packaged artificial quickstart table is intentionally retained. It is repository-authored Apache-2.0 test data, not a real-person dataset.

## Preservation and verification

The rewrite used `git-filter-repo` 2.47.0. Before any remote mutation, a private mirror and bundle covering all 22 branches and 31 pull-request heads were created and verified with `git fsck`, `git bundle verify`, and SHA-256. The full old-to-new commit map is retained with that private archive and is deliberately not published because it would provide direct identifiers for the removed object graph.

Verification established that:

- the pre- and post-rewrite `main` trees have the identical Git tree ID `967e78c6d3232ebdc23b557edf62a34195948809`;
- all 22 public branch names, all 147 commits reachable from those branches, and all author identities were preserved;
- the complete branches-and-pull-head rewrite map covered 152 commits, and no commit mapped to an all-zero/deleted identity;
- the only data/artifact-like path reachable from rewritten branches is `standardized_tabular_diffusion/resources/quickstart/train.csv`;
- representative removed row-level data and checkpoint blob IDs are absent from the rewritten object database;
- the rewritten branch-only pack is approximately 8 MiB, compared with approximately 129 MiB before cleanup; and
- Gitleaks 8.30.1 scanned the full rewritten branch history. Its four findings were reviewed SHA-256 hardware-comparison identities in immutable P6 evidence, not secrets; their exact rewritten-history fingerprints are recorded in `.gitleaksignore`.

Because the current `main` tree did not change, the rewrite does not change source code, documentation, package contents, scientific inputs, results, or conclusions. It changes commit identities and ancestry only. Retained evidence that records a pre-sanitization commit ID remains an honest record of the original validation run; the private mapping and identical-tree check provide the migration audit trail.

## GitHub platform boundary

Force-updating branch references does not update GitHub's read-only `refs/pull/*/head` references for closed pull requests. GitHub rejects maintainer attempts to update those hidden references. Consequently, a known pre-sanitization object ID can remain retrievable from GitHub until GitHub Support removes the affected pull-request references and cached views. A normal clone of the maintained branches receives only rewritten history, but a mirror that explicitly fetches GitHub pull-request references can still receive the pre-sanitization graph until that platform-side purge completes.

This limitation is tracked as a release blocker. The repository will not claim that GitHub has physically purged every old object, and it will not create the first release tag, until the maintainer has completed the GitHub Support step or recorded an explicit reviewed risk decision. GitHub documents this limitation in its guidance for [removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Existing clones

Anyone who cloned the repository before 2026-08-14 should preserve any independent work separately and clone the repository again. Pulling the rewritten history into an old clone can leave the old object graph and divergent branch references in that clone.

Contributor authorship, names, email identities, and timestamps were preserved. Commit IDs changed because a Git commit ID includes its parent history.
