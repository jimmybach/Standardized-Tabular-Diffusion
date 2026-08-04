# Third-Party Source Inventory

This file is a release-preparation inventory, not a substitute for the license files distributed with each component. The three primary source snapshots are pinned in the [upstream source audit](docs/UPSTREAM_SOURCE_AUDIT.md) and the machine-readable [source lock](standardized_tabular_diffusion/resources/upstream/source-lock.json). Nested projects and transitive dependencies still require separate review before public release.

| Component | Location | Declared upstream | Included license | Current treatment |
|---|---|---|---|---|
| TabDDPM | `TabDDPM-main/` | `yandex-research/tab-ddpm` at `b476257dd460b778ba09eb97f7a51d6490fa17f8` | MIT, `TabDDPM-main/LICENSE.md` | The imported core matched all 58 compared upstream blobs. A local `zero` compatibility substitute remains unapproved and parity-unvalidated. |
| TabDiff | `TabDiff-main/` | `MinkaiXu/TabDiff` at `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | MIT, `TabDiff-main/LICENSE` | The imported non-data snapshot matched all 30 compared blobs. `eval/mle/mle.py` was later changed semantically and is excluded from official metrics. The license file matches upstream byte-for-byte; malformed quote characters originate upstream and must not be silently rewritten. |
| TabSyn and bundled baselines | `TabSyn-main/` | `amazon-science/tabsyn` at `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | Apache-2.0, `TabSyn-main/LICENSE`, with `TabSyn-main/NOTICE` | The 20-file primary TabSyn execution scope is restored to and checksum-frozen against the official tree. Compatibility behavior is outside upstream source, and retained native parity passed. Bundled baseline snapshots still require independent license/source review. |
| CTAB-GAN snapshot | `TabDDPM-main/CTAB-GAN/` | `Team-TUD/CTAB-GAN` | Apache-2.0, `TabDDPM-main/CTAB-GAN/LICENSE` and `License.txt` | Nested source snapshot; exact revision and relationship to the root snapshot still need review. |
| CTGAN snapshot | `TabDDPM-main/CTGAN/CTGAN/` | `sdv-dev/CTGAN` | MIT, `TabDDPM-main/CTGAN/CTGAN/LICENSE` | Nested source snapshot; exact revision and package parity still need review. |

Public datasets are downloaded locally from their authoritative publishers and are not bundled in the Python distribution:

| Dataset | Authority and citation | Declared license | Repository treatment |
|---|---|---|---|
| Adult | Becker, B. and Kohavi, R. (1996), UCI Machine Learning Repository, DOI `10.24432/C5XW20` | CC BY 4.0 as declared by UCI | Archive and selected-member hashes, strict parsing, fixed split, preprocessing, and attribution are recorded in `sources.json`, `adult-uci-2-v1.json`, and the reviewed Dataset Profile. |
| Thyroid Disease (`sick` view) | Quinlan, R. (1986), UCI Machine Learning Repository, DOI `10.24432/C5D010` | CC BY 4.0 as declared by UCI | Archive and selected-member hashes, strict parsing, fixed split, preprocessing, and attribution are recorded in `sources.json`, `sick-uci-102-v1.json`, and the reviewed Dataset Profile. |

Package-backed adapters also depend on separately installed third-party projects. Their packages, versions, licenses, source authority, and validation evidence must be captured in the release lock and model evidence record. A package being importable does not make its adapter official-track or release-supported.

Do not delete or replace an upstream license or notice while its source is distributed. If upstream code is patched, retain a reviewable patch record in addition to its original attribution.
