# Source and implementation provenance

propaxis was extracted and generalized from Aleksandar Tomašević's Populism Axis research project. The source snapshot was `bada4a800b1260d65c9271bb67d7a641cf54a52d` on its `tidy-main` branch.

The main source concepts and files were:

| Source | Contribution |
| --- | --- |
| `src/populism_axis/geometry.py` | Unit-normalized centroid differences and passage centering. |
| `src/populism_axis/contextual_axis.py` | Clause-aware sentence caps and overlapping passage construction. |
| `src/populism_axis/encoders.py` | E5 input rendering and attention-mask mean pooling. |
| `scripts/frozen_reference.py` | Estimating reference scaling parameters once for subsequent scoring. |
| `scripts/score_texts.py` | Passage selection, NLI checking, result records, and evidence output. |
| `anchors/proposition_anchors.json` | The forty P/E propositions supplied by `presets.populism()`. |
| `anchors/claim_hypotheses.json` | The ten hypotheses supplied by the populism preset. |

The package refactors these methods into a standalone library. It adds validation of construct specifications, generic component names, persistent instruments, binary thresholds, and a notebook editor. No research corpus, cached embedding matrix, private correspondence, run log, or fitted benchmark result is included.

The [method guide](method.md) records differences that affect scores. Published or archived research performance does not automatically transfer to this package's defaults or to a user's new reference dataset.

The package's MIT license covers its code and included construct specification. Downloaded models retain their upstream terms. Saved instruments record resolved model commits rather than relying on a moving `main` revision when reloaded.
