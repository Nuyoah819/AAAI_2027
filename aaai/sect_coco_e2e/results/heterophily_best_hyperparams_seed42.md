# Best 9-Dataset Hyperparameters (Seed 42)

Source of truth:
- Six attributed datasets:
  - `D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\scripts\run_e2e_experiments.py`
  - `D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\results\main_results_tables_9datasets.tex`
- Three heterophily datasets:
  - `D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\hyperparam_search\search_logs\best_results_all_phases.csv`
  - `D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\hyperparam_search\search_logs\best_results_all_phases.md`

All results below use fixed seed `42` with deterministic settings.

## ACM

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=93.62`, `NMI=75.88`, `ARI=81.89`, `Macro-F1=93.63`
- Key parameters:
  - `epochs=220`
  - `pretrain_epochs=50`
  - `input_dim=192`
  - `projection_dim=64`
  - `feature_knn=4`
  - `max_train_edges=180000`
  - `init_low=0.20`
  - `init_high=0.62`
  - `lowpass_steps=3`
  - `highpass_weight=0.04`
  - `final_label_mode=subspace_refine`
  - `head_input=original`
  - `head_graph=original_elss`
  - `head_power=2`
  - `head_k_rank=4`
  - `head_n_anchors=500`
  - `head_d=0.875`
  - `head_alpha2=0.00005`
  - `head_gamma=0.0025`
  - `head_kmeans_n_init=100`
  - `tfidf=True`

## DBLP

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=93.69`, `NMI=79.74`, `ARI=84.83`, `Macro-F1=93.24`
- Key parameters:
  - `epochs=240`
  - `pretrain_epochs=50`
  - `input_dim=192`
  - `projection_dim=64`
  - `feature_knn=6`
  - `max_train_edges=180000`
  - `max_feature_edges=80000`
  - `init_low=0.18`
  - `init_high=0.72`
  - `lowpass_steps=3`
  - `raw_skip_weight=0.92`
  - `final_label_mode=legacy_sect_bridge`
  - `attr_dim=128`
  - `cluster_dim=64`
  - `feature_graph_weight=0.15`
  - `high_quantile=0.81`
  - `low_quantile=0.19`
  - `local_steps=6`
  - `global_steps=6`
  - `restart=0.45`
  - `legacy_highpass_weight=0.20`
  - `head=subspace_refine`
  - `head_input=concat`
  - `head_graph=original_elss`
  - `head_power=3`
  - `head_k_rank=5`
  - `head_n_anchors=80`
  - `head_d=0.93`
  - `head_alpha2=0.00005`
  - `head_gamma=0.003`
  - `head_kmeans_n_init=100`
  - `tfidf=True`

## PubMed

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=76.17`, `NMI=37.71`, `ARI=42.66`, `Macro-F1=74.88`
- Key parameters:
  - `epochs=220`
  - `pretrain_epochs=50`
  - `input_dim=256`
  - `feature_knn=0`
  - `max_train_edges=180000`
  - `init_low=0.22`
  - `init_high=0.68`
  - `lowpass_steps=2`
  - `use_minibatch_kmeans=True`
  - `raw_skip_weight=0.95`
  - `final_label_mode=fast_elss`
  - `head_input=original`
  - `head_graph=original_elss`
  - `head_power=136`
  - `head_k_rank=4`
  - `head_n_anchors=45`
  - `head_d=0.95`
  - `head_alpha2=0.00005`
  - `head_gamma=0.005`
  - `head_filter_coef=0.1`
  - `head_q_norm=none`
  - `head_kmeans_n_init=10`
  - `tfidf=True`

## Wiki

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=64.82`, `NMI=59.79`, `ARI=48.51`, `Macro-F1` recorded in the formal table run is not separately summarized here
- Key parameters:
  - `epochs=260`
  - `pretrain_epochs=50`
  - `input_dim=192`
  - `feature_knn=16`
  - `init_low=0.20`
  - `init_high=0.70`
  - `highpass_weight=0.16`
  - `final_label_mode=wiki_consensus`
  - `attr_dim=128`
  - `cluster_dim=64`
  - `high_quantile=0.80`
  - `low_quantile=0.18`
  - `local_steps=3`
  - `global_steps=5`
  - `legacy_highpass_weight=0.44`
  - `label_diffusion_graph=sym_self`
  - `label_diffusion_steps=16`
  - `label_diffusion_gamma=1.0`
  - `label_diffusion_self_loop=4.0`
  - `label_diffusion_size_norm=False`
  - `consensus_blend_fraction=0.55`
  - `tfidf=False`

## Flickr

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=83.89`, `NMI=71.25`, `ARI=67.52`, `Macro-F1=84.13`
- Key parameters:
  - `epochs=240`
  - `pretrain_epochs=50`
  - `input_dim=256`
  - `feature_knn=18`
  - `max_feature_edges=140000`
  - `max_train_edges=260000`
  - `use_minibatch_kmeans=False`
  - `init_low=0.18`
  - `init_high=0.60`
  - `highpass_weight=0.16`
  - `final_label_mode=dual_diffusion`
  - `dual_dim=192`
  - `dual_alpha=1.0`
  - `dual_beta=0.8`
  - `dual_steps=1`
  - `dual_cluster_steps=1`
  - `dual_cluster_gamma=0.1`
  - `kmeans_n_init=30`
  - `tfidf=True`

## BlogCatalog

- Source type: formal fixed configuration from `scripts/run_e2e_experiments.py`
- Candidate: `sect_coco_e2e`
- Metrics: `ACC=91.72`, `NMI=78.60`, `ARI=81.63`, `Macro-F1=91.54`
- Key parameters:
  - `epochs=240`
  - `pretrain_epochs=50`
  - `input_dim=192`
  - `projection_dim=64`
  - `feature_knn=0`
  - `max_feature_edges=80000`
  - `max_train_edges=220000`
  - `use_minibatch_kmeans=True`
  - `init_low=0.20`
  - `init_high=0.66`
  - `raw_skip_weight=0.95`
  - `final_label_mode=subspace_refine`
  - `head_input=original`
  - `head_graph=original_elss`
  - `head_power=4`
  - `head_k_rank=7`
  - `head_n_anchors=240`
  - `head_alpha2=0.0001`
  - `head_gamma=0.003`
  - `head_filter_coef=0.1`
  - `head_q_norm=l2`
  - `head_kmeans_n_init=80`
  - `tfidf=True`

## Texas

- Phase: `micro`
- Candidate: `texas_m_short_guard_init_low0p12_init_high0p58_threshold_tau0p12_assignment_repel_weight2p0_assignment_sharpen_power8p0_assignment_temperature1p0_epochs1_pretrain_epochs0_graph_input_dim8_projection_dim48`
- Trial ID: `texas:micro:texas_m_short_guard_init_low0p12_init_high0p58_threshold_tau0p12_assignment_repel_weight2p0_assignment_sharpen_power8p0_assignment_temperature1p0_epochs1_pretrain_epochs0_graph_input_dim8_projection_dim48:46f18d5b80c644f0`
- Metrics: `ACC=74.32`, `NMI=51.49`, `ARI=60.86`, `Macro-F1=44.81`
- Key parameters:
  - `epochs=1`
  - `pretrain_epochs=0`
  - `graph_input_dim=8`
  - `projection_dim=48`
  - `hidden_dim=192`
  - `feature_knn=20`
  - `highpass_weight=0.22`
  - `init_low=0.12`
  - `init_high=0.58`
  - `threshold_tau=0.12`
  - `assignment_repel_weight=2.0`
  - `assignment_sharpen_power=8.0`
  - `assignment_temperature=1.0`
  - `raw_skip_weight=0.995`
  - `contrastive_weight=0.12`

## Squirrel

- Phase: `coarse`
- Candidate: `squirrel_c_threshold_init_low0p2_init_high0p52_threshold_tau0p08`
- Trial ID: `squirrel:coarse:squirrel_c_threshold_init_low0p2_init_high0p52_threshold_tau0p08:6722fec754df3993`
- Metrics: `ACC=30.51`, `NMI=6.28`, `ARI=5.47`, `Macro-F1=20.31`
- Key parameters:
  - `epochs=300`
  - `pretrain_epochs=50`
  - `input_dim=512`
  - `graph_input_dim=256`
  - `projection_dim=768`
  - `hidden_dim=256`
  - `feature_knn=16`
  - `max_train_edges=300000`
  - `max_feature_edges=120000`
  - `highpass_weight=0.2`
  - `init_low=0.20`
  - `init_high=0.52`
  - `threshold_tau=0.08`
  - `assignment_repel_weight=0.8`
  - `assignment_raw_repel_floor=0.5`
  - `raw_skip_weight=0.995`

## Chameleon

- Phase: `coarse`
- Candidate: `chameleon_c_hetero_highpass_weight0p3_assignment_repel_weight0p6_assignment_raw_repel_floor0p5`
- Trial ID: `chameleon:coarse:chameleon_c_hetero_highpass_weight0p3_assignment_repel_weight0p6_assignment_raw_repel_floor0p5:8827e21412a6986f`
- Metrics: `ACC=35.84`, `NMI=16.85`, `ARI=6.63`, `Macro-F1=33.72`
- Key parameters:
  - `epochs=300`
  - `pretrain_epochs=50`
  - `input_dim=256`
  - `graph_input_dim=0`
  - `projection_dim=256`
  - `hidden_dim=256`
  - `feature_knn=18`
  - `max_train_edges=220000`
  - `max_feature_edges=100000`
  - `highpass_weight=0.3`
  - `init_low=0.16`
  - `init_high=0.58`
  - `threshold_tau=0.10`
  - `assignment_repel_weight=0.6`
  - `assignment_raw_repel_floor=0.5`
  - `raw_skip_weight=0.995`
  - `tfidf=True`
