# CSTC Tuning Record

Date: 2026-06-29
Protocol: fixed seed 42, no test-metric-based model selection.

## Fixed Default Configuration

The main reported run uses `CODE/configs/cstc_default.json` on all datasets with the same CSTC forward path, loss family, and final assignment rule.

## Stability Probe

Initial smoke and main runs showed very low posterior entropy and unstable view-gate dominance. A label-free autoencoder pretraining stage was added to `CODE/cstc/model.py` before prototype initialization. This change is a training-stability correction and is not dataset-specific.

Probe command:

```powershell
conda run -n aaai-e2e-subspace python CODE\run_cstc_experiments.py --datasets pubmed wiki squirrel chameleon --device cuda --output CODE\results\cstc_pretrain_probe_results.csv --diagnostics-dir CODE\results\diagnostics_pretrain_probe
```

Probe observation:

| Dataset | Before ACC/NMI/ARI | After ACC/NMI/ARI | Interpretation |
| --- | --- | --- | --- |
| PubMed | 57.31 / 13.83 / 14.49 | 60.43 / 19.21 / 17.45 | Pretraining improves representation/prototypes |
| Wiki | 36.38 / 33.20 / 16.50 | 43.33 / 42.49 / 25.42 | Pretraining helps mixed graph assignment |
| Squirrel | 22.26 / 0.31 / 0.18 | 22.63 / 0.28 / 0.16 | Heterophily failure remains |
| Chameleon | 30.00 / 4.87 / 3.63 | 29.69 / 5.20 / 3.87 | Essentially unchanged |

The final main result uses the post-pretraining code because it is a unified training stability improvement, not a dataset-specific selector.

## Planned Search Space

`CODE/tune_cstc.py --dry-run` generated `CODE/results/tuning/cstc_tuning_plan.csv` with this fixed space:

| Hyperparameter | Values |
| --- | --- |
| `temperature` | 0.20, 0.24 |
| `refine_repel` | 0.15, 0.25 |
| `target_hetero_ratio` | 0.24, 0.32 |

Selection rule is not yet activated. The current plan explicitly states: diagnostics-only; no test metric used for final selection.

## 2026-06-29 Optimization Continuation

Protocol: fixed seed 42, same CSTC forward path for all datasets, no dataset-specific module/loss/head/post-processing, and no multi-restart selection. All commands used `conda run -n aaai-e2e-subspace ...`.

### Diagnosis Before Modification

The previous main diagnostics showed low posterior entropy, high ambiguous DTC mass, brittle view-gate dominance, and a weak degree-only structural signal in edge concordance. A separate implementation bug was also found in candidate-edge construction: after KNN/graph edges were deduplicated, `graph_source` was assigned by position rather than by actual edge membership, so edge priors could be misaligned.

### Iteration Log

| Round | Unified change | Probe/main outcome | Decision |
| --- | --- | --- | --- |
| Round 1 | Added common-neighbor/two-hop structural evidence, removed over-strong confidence logit standardization, added entropy curriculum, view balance, ambiguity penalty, and prototype refresh. Also fixed edge-source membership in `data.py`. | Probe improved ACM/Texas/Squirrel but hurt PubMed. | Kept core evidence/edge-source fixes; continued diagnosis. |
| Round 2 | Added a fixed spectral feature bank from original features, normalized low-pass features, and residual features. | PubMed improved, but ACM/Texas/Squirrel regressed. | Kept code as optional but disabled by default; fixed feature bank is not robust enough. |
| Round 3 | Added differentiable assignment-graph consistency: high-confidence contracted edges attract transport posteriors, low-confidence edges softly repel. | Main results improved 7/9 datasets, especially DBLP, BlogCatalog, Texas. PubMed/Wiki regressed. | Kept as core CSTC upgrade. |
| Round 4 | Added a label-free initial posterior anchor with decaying weight. | Probe weakened PubMed/Squirrel/Texas. | Disabled by default and recorded as failed stabilizer. |
| Round 5 | Added static feature cosine evidence to edge concordance, combining learned/raw/static attribute agreement with structural evidence. | Best current unified implementation. ACM and DBLP became close to SOTA; BlogCatalog/Texas/Flickr improved substantially; PubMed remains a failure point. | Adopted as current default. |
| Round 6 | Relaxed view consistency/balance and raised late entropy floor. | Probe regressed Texas and did not help PubMed. | Reverted to Round 5 defaults. |

### Current Default Configuration Notes

- `use_spectral_feature_bank` is present but set to `false` because fixed pre-input spectral concatenation helped PubMed but damaged heterophilic graphs in a unified setting.
- `anchor_weight_start` and `anchor_weight_end` are set to `0.0` because the label-free anchor reduced useful topology adaptation.
- `view_balance_weight` remains `0.05` and `view_consistency_weight` remains `0.03`; although the view gate mean appears close to uniform, relaxing these terms degraded stability.

### Current Best Main Results

The current reported results are from `CODE/results/cstc_main_results.csv`, copied from `CODE/results/cstc_round5_main_results.csv`.

| Dataset | ACC | NMI | ARI | ACC gap to SOTA | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| ACM | 89.39 | 65.38 | 71.15 | -4.23 | Close but still below strongest baseline. |
| DBLP | 91.92 | 75.36 | 80.61 | -1.77 | Closest current success case. |
| PubMed | 56.36 | 14.48 | 12.73 | -19.81 | Main regression/failure case. |
| Wiki | 43.41 | 42.24 | 24.36 | -21.41 | Similar to previous CSTC; assignment remains weak. |
| Flickr | 30.39 | 14.12 | 8.74 | -53.50 | Improved but still far from SOTA. |
| BlogCatalog | 64.05 | 48.15 | 45.89 | -27.67 | Large improvement, still far below SOTA. |
| Squirrel | 25.82 | 2.14 | 1.65 | -8.61 | Gap narrowed but absolute clustering remains weak. |
| Texas | 47.54 | 38.53 | 23.69 | -27.54 | Improved, but ACC/ARI still far below SOTA. |
| Chameleon | 32.32 | 6.67 | 4.98 | -9.70 | Modest improvement. |

### Bottleneck After Multiple Rounds

The current line improves edge-level concordance and lets DTC influence transport assignment, which explains the strong jumps on ACM/DBLP/BlogCatalog/Texas. However, the model still uses a relatively shallow MLP encoder and a fragile self-training transport objective. The remaining gap is unlikely to close through small scalar tuning alone. The most promising next step is a unified stronger encoder/pretraining strategy that remains label-free and end-to-end, such as graph-filtered masked reconstruction, contrastive edge calibration from confidence quantiles, or a lightweight graph propagation encoder that preserves CSTC's DTC and transport path.

## 2026-06-30 Clustering-Head Optimization Continuation

Protocol: fixed seed 42, same CSTC forward path for all datasets, no dataset-specific module/loss/head/post-processing, and no multi-restart selection. All experiments used `conda run -n aaai-e2e-subspace ...`.

### Round 7: Unified MoE Transport Head

Change: added a unified ensemble/MoE transport head in `CODE/cstc/model.py`. The head keeps the original transport posterior and adds three shared expert prototype banks. Each expert uses the same Sinkhorn refinement path, and a learned soft gate mixes expert posteriors. The expert count and path are identical for all datasets.

Probe datasets: DBLP, PubMed, Texas, Flickr.

| Dataset | Round 5 ACC/NMI/ARI | Round 7 ACC/NMI/ARI | Outcome |
| --- | --- | --- | --- |
| DBLP | 91.92 / 75.36 / 80.61 | 90.61 / 72.83 / 78.09 | Regressed |
| PubMed | 56.36 / 14.48 / 12.73 | 56.15 / 14.22 / 12.50 | Similar/slightly worse |
| Texas | 47.54 / 38.53 / 23.69 | 46.45 / 36.26 / 22.07 | Regressed |
| Flickr | 30.39 / 14.12 / 8.74 | 26.27 / 9.75 / 5.60 | Regressed |

Diagnosis: expert gates were near-uniform and posterior entropy became even lower on several graphs. The MoE experts behaved like averaged noisy prototype banks rather than useful specialized clusterers.

Decision: keep MoE code as an optional unified head (`num_transport_experts > 0`) but do not enable it by default.

### Round 8: Lightweight Expert Mixture

Change: reduced expert influence so the base transport head keeps 75% weight and expert mixture contributes 25%; lowered expert consistency and balance losses.

| Dataset | Round 8 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 90.61 / 72.42 / 78.00 | Still below Round 5 |
| PubMed | 56.33 / 14.08 / 12.63 | Similar to Round 5 |
| Texas | 47.54 / 38.60 / 23.80 | Roughly matches Round 5 |
| Flickr | 25.24 / 10.51 / 5.81 | Worse |

Decision: not adopted as default.

### Round 9: Fused-Space Prototype Initialization

Change: initialized KMeans prototypes in the transport fusion space instead of encoder space.

| Dataset | Round 9 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 91.15 / 73.76 / 79.03 | Below Round 5 |
| PubMed | 46.04 / 8.56 / 8.15 | Severe regression |
| Texas | 49.18 / 36.80 / 23.25 | ACC improved, NMI/ARI not enough |
| Flickr | 24.99 / 10.59 / 5.81 | Worse |

Diagnosis: fused-space initialization helps some small/heterophilic cases but destabilizes PubMed and Flickr. It is not robust enough as a unified default.

Decision: reverted.

### Round 10: Prototype Separation Regularizer

Change: added a uniform prototype separation loss to discourage collapsed transport prototypes.

| Dataset | Round 10 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 92.04 / 75.61 / 80.91 | Slightly improves DBLP |
| PubMed | 56.38 / 14.39 / 12.70 | Similar to Round 5 |
| Texas | 42.08 / 29.25 / 18.15 | Severe regression |
| Flickr | 29.45 / 13.24 / 8.62 | Below Round 5 |

Decision: code retained but `prototype_separation_weight` set to `0.0` by default.

### Round 11: Temperature Curriculum

Change: early training uses softer transport temperature and anneals back to the default temperature.

| Dataset | Round 11 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 91.20 / 74.00 / 79.11 | Below Round 5 |
| PubMed | 56.32 / 14.76 / 12.89 | Slight ARI/NMI improvement only |
| Texas | 46.45 / 36.98 / 23.00 | Below Round 5 |
| Flickr | 24.70 / 10.84 / 6.26 | Worse |

Decision: code retained but `temperature_start_scale` set to `1.0` by default, disabling the curriculum.

### Round 12: Default Reproduction After Failed Head Trials

After reverting failed default settings, a nine-dataset reproduction was run and saved to `CODE/results/cstc_round12_main_results.csv` with diagnostics in `CODE/results/diagnostics_round12_main`.

| Dataset | ACC | NMI | ARI | ACC gap to SOTA |
| --- | ---: | ---: | ---: | ---: |
| ACM | 89.36 | 65.29 | 71.06 | -4.26 |
| DBLP | 91.92 | 75.36 | 80.61 | -1.77 |
| PubMed | 56.36 | 14.49 | 12.73 | -19.81 |
| Wiki | 43.41 | 42.24 | 24.36 | -21.41 |
| Flickr | 30.32 | 14.07 | 8.70 | -53.57 |
| BlogCatalog | 64.05 | 48.09 | 45.82 | -27.67 |
| Squirrel | 25.84 | 2.13 | 1.65 | -8.59 |
| Texas | 47.54 | 38.53 | 23.69 | -27.54 |
| Chameleon | 32.28 | 6.49 | 4.86 | -9.74 |

### Conclusion of 2026-06-30 Continuation

The requested unified MoE/ensemble clustering head was implemented and tested, but it did not improve CSTC. The current bottleneck is not merely the clustering head form; the weak cases need stronger label-free representation learning and graph-aware pretraining. Continuing to stack transport experts or scalar clustering-head regularizers is unlikely to close the remaining SOTA gap without a more principled encoder/pretraining upgrade.

## 2026-07-01 Graph-Aware Pretraining and Encoder Continuation

Protocol: fixed seed 42, same CSTC forward path for all datasets, no dataset-specific module/loss/head/post-processing, and no multi-restart selection. All experiments used `conda run -n aaai-e2e-subspace ...`.

### Motivation

After the failed clustering-head rounds, the next hypothesis was that CSTC's remaining gap comes from representation quality rather than final assignment mechanics. The continuation therefore tested unified label-free pretraining and encoder upgrades while preserving DTC, spectral decoupling, and the same final transport posterior.

### Round 13: Masked Graph-Aware Pretraining with Edge Calibration

Change: upgraded pretraining from plain autoencoding to masked feature reconstruction, graph-neighborhood reconstruction, and edge calibration from label-free structural/attribute evidence.

| Dataset | Round 12 ACC/NMI/ARI | Round 13 ACC/NMI/ARI | Outcome |
| --- | --- | --- | --- |
| DBLP | 91.92 / 75.36 / 80.61 | 89.11 / 69.70 / 74.62 | Regressed |
| PubMed | 56.36 / 14.49 / 12.73 | 39.81 / 1.44 / 1.77 | Severe regression |
| Texas | 47.54 / 38.53 / 23.69 | 36.61 / 17.26 / 9.45 | Severe regression |
| Flickr | 30.32 / 14.07 / 8.70 | 35.41 / 19.63 / 14.36 | Large improvement |

Diagnosis: edge calibration helps noisy Flickr but is too aggressive for PubMed, Texas, and DBLP.

Decision: not adopted as default.

### Round 14: Weaker Edge Calibration

Change: reduced `pretrain_edge_weight` from `0.08` to `0.02`.

| Dataset | Round 14 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 90.02 / 72.72 / 76.79 | Still below Round 12 |
| PubMed | 42.71 / 6.99 / 6.36 | Still severe regression |
| Texas | 46.45 / 36.95 / 24.21 | Roughly recovers Texas |
| Flickr | 27.29 / 14.17 / 9.28 | Loses Flickr gain |

Decision: not adopted.

### Round 15: Masked + Graph Reconstruction Only

Change: disabled edge calibration and kept masked feature reconstruction plus graph-neighborhood reconstruction.

| Dataset | Round 15 ACC/NMI/ARI | Outcome |
| --- | --- | --- |
| DBLP | 91.23 / 73.95 / 79.14 | Below Round 12 |
| PubMed | 56.85 / 14.42 / 13.63 | Slight ACC/ARI improvement |
| Texas | 49.73 / 35.90 / 22.98 | ACC improves, NMI/ARI worse |
| Flickr | 22.36 / 9.13 / 5.03 | Severe regression |

Decision: not adopted as default because the improvement is not robust across heterophilic graphs.

### Round 16-17: Residual Encoder Capacity

Change: replaced the shallow MLP encoder with a residual MLP block. Round 16 combined it with graph reconstruction; Round 17 kept only masked autoencoding.

| Dataset | Round 16 ACC/NMI/ARI | Round 17 ACC/NMI/ARI | Outcome |
| --- | --- | --- | --- |
| DBLP | 91.10 / 73.71 / 78.90 | 91.10 / 73.73 / 78.90 | Below Round 12 |
| PubMed | 58.57 / 16.62 / 16.35 | 58.35 / 16.44 / 16.10 | PubMed improves |
| Texas | 47.54 / 27.55 / 19.96 | 46.45 / 36.12 / 22.05 | Mixed/regressed |
| Flickr | 23.89 / 11.00 / 6.65 | 24.59 / 12.61 / 7.78 | Regressed |

Diagnosis: higher encoder capacity helps PubMed but hurts Flickr/DBLP stability under the current CSTC loss.

Decision: reverted to the original shallow encoder.

### Round 18-20: DTC-Gated Edge Contrast During Main Training

Change: added an optional DTC-gated edge contrast loss. Homophilic contracted edges pull embeddings together; heterophilic contracted edges are softly pushed apart. Round 20 used a graph-level DTC hetero-mass adaptive weight instead of dataset-specific logic.

| Dataset | Round 18 ACC/NMI/ARI | Round 19 ACC/NMI/ARI | Round 20 ACC/NMI/ARI | Outcome |
| --- | --- | --- | --- | --- |
| DBLP | 92.01 / 75.94 / 80.76 | 91.96 / 75.60 / 80.66 | 91.92 / 75.30 / 80.62 | Mostly preserves DBLP |
| PubMed | 55.76 / 13.76 / 12.14 | 55.95 / 13.94 / 12.31 | 56.37 / 14.49 / 12.74 | No real gain |
| Texas | 41.53 / 30.42 / 17.83 | 41.53 / 29.52 / 17.36 | 47.54 / 38.50 / 23.62 | Adaptive recovers Texas |
| Flickr | 38.10 / 20.64 / 14.72 | 30.86 / 14.58 / 9.24 | 27.82 / 12.82 / 8.21 | Strong fixed weight helps Flickr but not unified |

Diagnosis: DTC-gated edge contrast can substantially improve Flickr, but the same fixed strength harms Texas/PubMed. The hetero-mass adaptive version protects Texas/PubMed but loses the Flickr gain.

Decision: code retained as optional (`edge_contrast_weight > 0`) but disabled by default.

### Conclusion of 2026-07-01 Continuation

The graph-aware pretraining and encoder upgrades provide useful evidence but are not yet robust enough to replace the Round 12 default. The strongest local signal is DTC-gated edge contrast on Flickr; the strongest PubMed signal is residual encoder capacity. These improvements conflict under one shared configuration, so the next credible research step is not another scalar sweep. It should be a unified adaptive mechanism that learns when to apply graph reconstruction or edge contrast from label-free diagnostics without becoming dataset-specific.
