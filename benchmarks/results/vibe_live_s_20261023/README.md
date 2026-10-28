<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retained VIBE observations

The campaign was interrupted by the provider on 2026-10-24; the full 24-dataset protocol did not finish. `manifest.json` indexes 20 complete HMS grids and only completed comparator arms, totaling 5,488 exported groups. The final snapshot is a historical observation: its running states do not indicate active work.

`interrupted-complete-arm-exports/` preserves byte-identical official Parquet exports, threshold standings, native-result digests and truth extraction receipts. Same-container comparisons exist for Celeba, Coco, ImageNet-CLIP, Llama, Yi and Yandex; Llama/Yi also retain failed MLANN-RF attempts in the historical records. Cross-container or mixed-container results remain diagnostic. Peak process RSS was not measured, and resource cost estimates are not invoices.

Threshold standings can be regenerated offline from each Parquet with the existing `STANDING` program in `../../vibe/modal_vibe.py`; use the versions of Python and Polars recorded in `interrupted-complete-arm-exports/local-export-environment.json`. Receipts bind each Parquet to its SHA-256 and identify the original raw inputs. The five prior completed collections are retained for the recorded numerical reproduction check.

Raw predictions, timings and truth arrays are external historical artifacts; their digests do not independently validate recall. Absolute paths and Modal object IDs inside byte-preserved evidence identify original observations and are not required local files. This directory does not authorize a new remote run.

The manifest records the immutable archive commit for omitted operational scripts, repeated snapshots, and large inventories. Retrieve an archived file with `git show <archive-commit>:<repository-path>`. Original evidence bytes, including historical metadata and formatting, are preserved.
