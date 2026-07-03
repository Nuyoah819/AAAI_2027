# V64-V68 ELSS Anchor and Distrust Verdict

## Scope

This report records the V64-V68 optimization sequence for the unified APTC
pipeline. The goal remains the governance target: solve the attribute-structure
mismatch heterophily-noise association problem and reach SOTA on at least 6 of
9 datasets without label leakage, dataset-specific branches, legacy final heads,
or post-hoc metric selection.

## Implemented Mechanisms

- V64A: spectral-subspace Gram alignment against a fixed graph-smoothed SVD
  embedding.
- V64B: V64A with post-80 release to avoid late over-constraint.
- V65A: V40A core training configuration with `final_label_mode=aptc`, used to
  test whether V40A's training path alone explained historical high scores.
- V66A: ELSS/anchor-subspace embedding used only as a Gram training anchor.
- V66B: ELSS/anchor-subspace embedding converted to a soft q anchor, then
  absorbed through the existing V62 release/reliability path.
- V67A/B: post-release anchor distrust gates based on unsupervised
  posterior-anchor agreement.
- V68A: low-agreement teacher-guard boost to preserve the epoch-80 solution on
  anchor-unsafe graphs.

All variants kept final readout in the unified APTC/KMeans path. No legacy
subspace-refine final label head was enabled.

## Full-Run Results

Best current single full-run variant from this sequence is V67A/V67B depending
on metric tradeoff. V67A is the most balanced full-run result:

| Dataset | ACC | NMI | ARI | ACC Gap to Target |
| --- | ---: | ---: | ---: | ---: |
| ACM | 0.9336 | 0.7534 | 0.8121 | -0.0026 |
| DBLP | 0.9145 | 0.7322 | 0.7983 | -0.0224 |
| PubMed | 0.6344 | 0.2615 | 0.2329 | -0.1273 |
| Wiki | 0.4915 | 0.4538 | 0.2806 | -0.1525 |
| Flickr | 0.4540 | 0.3478 | 0.2356 | -0.3619 |
| BlogCatalog | 0.9007 | 0.7624 | 0.7821 | -0.0165 |
| Squirrel | 0.2605 | 0.0231 | 0.0211 | -0.0838 |
| Texas | 0.7158 | 0.4460 | 0.5331 | -0.0350 |
| Chameleon | 0.3276 | 0.1504 | 0.0643 | -0.0926 |

The best per-dataset result among V63B-V68A remains 0/9 ACC SOTA, but V66B/V67A
substantially improved the high-anchor-reliability datasets:

| Dataset | Best Variant | ACC | Previous V63B ACC |
| --- | --- | ---: | ---: |
| ACM | V67A/V67B | 0.9336 | 0.9081 |
| DBLP | V67A | 0.9145 | 0.7266 |
| PubMed | V68A | 0.6405 | 0.5163 |
| Wiki | V67A | 0.4915 | 0.4216 |
| Flickr | V67B | 0.4659 | 0.2866 |
| BlogCatalog | V67A/V67B | 0.9007 | 0.8507 |

## Verdict

The useful mechanism is ELSS q-anchor absorption, not Gram-only subspace
alignment. V66B proved that a stronger low-rank anchor can be absorbed by the
unified model and move ACM/DBLP/BlogCatalog close to SOTA.

The remaining bottleneck is anchor safety on heterophily-heavy or anchor-weak
graphs. Post-hoc final selection is not acceptable, and teacher boosting did
not recover Texas/Squirrel. The next route should make the anchor itself
mixture-aware: retain ELSS q supervision only where local structure-attribute
evidence supports it, and use compactness-only or self-distillation elsewhere.
