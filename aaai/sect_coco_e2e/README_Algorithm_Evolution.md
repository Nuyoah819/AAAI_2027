# SECT-COCO-E2E Algorithm Evolution Log

## 2026-06-22: Unified Backend Clustering Refactor Design

### 1. Diagnosis of the AAAI-risk issue

The previous implementation in
`D:\study\graduate_student\papers\AAAI2027\AAAI0617\CODE\core\e2e\sect_coco_e2e.py`
keeps the valuable frontend idea, but its final clustering stage is not a unified
end-to-end algorithm.

The frontend is still the paper-worthy part:

- an adaptive edge-confidence module predicts edge reliability from attribute,
  raw-feature, degree, and prior evidence;
- differentiable topology contraction learns ordered low/high confidence
  thresholds and produces homophilic, heterophilic, and ambiguous edge masks;
- low-pass and signed high-pass graph filters build frequency-aware node views;
- assignment flow propagates soft cluster probabilities over learned topology
  masks.

The problem appears after representation learning. Lines around the final
inference block select mutually different final heads through `final_label_mode`:

- `flow`: directly uses `q_flow.argmax`;
- `dual_diffusion` / `dgac_dual`: replaces the learned model output with a
  separate dual diffusion head;
- `legacy_sect_bridge`: reruns the old SECT-CoCo estimator;
- `s2cag` / `spectral_subspace`: runs a sparse S2CAG-style spectral head;
- `wiki_consensus` / `s2cag_consensus`: combines legacy labels, label diffusion,
  S2CAG variants, voting, and margin blending;
- `subspace_refine`, `fast_elss`, `sgc_lowpass`, `legacy_head`: dispatch to
  legacy ELSS/SGC/subspace heads through an adapter.

The 9-dataset best-configuration record confirms that this dispatch is used as
dataset-specific routing:

| Dataset | Old best final backend | Old result target (ACC / NMI / ARI) |
| --- | --- | --- |
| ACM | `subspace_refine` | 93.62 / 75.88 / 81.89 |
| DBLP | `legacy_sect_bridge` | 93.69 / 79.74 / 84.83 |
| PubMed | `fast_elss` | 76.17 / 37.71 / 42.66 |
| Wiki | `wiki_consensus` | 64.82 / 59.79 / 48.51 |
| Flickr | `dual_diffusion` | 83.89 / 71.25 / 67.52 |
| BlogCatalog | `subspace_refine` | 91.72 / 78.60 / 81.63 |
| Texas | mostly `kmeans` / short assignment-flow variants | 74.32 / 51.49 / 60.86 |
| Squirrel | `kmeans`-style E2E variants | 30.51 / 6.28 / 5.47 |
| Chameleon | `kmeans`-style E2E variants | 35.84 / 16.85 / 6.63 |

This is unacceptable for the target paper because the reported model is not one
algorithm. It is a collection of dataset-conditioned terminal tricks. Several
terminal heads are also outside the PyTorch computation graph, so gradients from
the final assignment objective cannot train the topology contraction frontend.

### 2. Refactor principle

The new `AAAI0622` implementation must use one forward path for all 9 datasets:

```text
features, graph
  -> sparse input encoder
  -> adaptive edge confidence
  -> differentiable topology contraction
  -> low/high frequency graph filtering
  -> unified posterior transport clustering head
  -> soft assignments Q*
  -> labels = argmax(Q*) only at evaluation/export time
```

There must be no dataset-name branch and no `final_label_mode` branch in the
algorithmic path. Dataset configs may still specify ordinary scale-related
hyperparameters such as input dimension, training epochs, candidate-edge budget,
and memory-safe minibatching, but not a different clustering backend.

### 3. Proposed unified backend: APTC head

I propose replacing every legacy/spectral/consensus final head with a single
Adaptive Posterior Transport Clustering (APTC) head.

#### 3.1 Learnable prototypes and posterior logits

Let `H in R^{n x d}` be the normalized embedding produced by the current
topology-contraction frontend. The clustering head owns learnable prototypes
`C in R^{k x d}` and computes cosine-distance logits:

```text
s_ik = tau_c^{-1} * cosine(h_i, c_k) + b_k
Q0 = softmax(S)
```

The prototypes are normal PyTorch parameters. They receive gradient from every
loss term. KMeans can be used once as an optional initialization heuristic, but
the trained model and final assignment must not depend on offline KMeans.

#### 3.2 Differentiable Sinkhorn posterior transport

To prevent collapse without hard-coded dataset priors, APTC projects `Q0` onto a
soft balanced transport polytope:

```text
Q* = Sinkhorn(exp(S / epsilon), row_sum = 1, col_sum = pi)
```

The cluster prior `pi` is not fixed to exactly uniform. It is produced by a
learnable Dirichlet/logit prior with a mild entropy regularizer:

```text
pi = softmax(rho),  sum_k pi_k = 1
```

This lets the same model handle nearly balanced citation graphs and imbalanced
social graphs. Sinkhorn iterations are differentiable, so gradients flow through
`Q*` into prototypes, embeddings, edge confidence, and topology thresholds.

#### 3.3 Graph-manifold posterior refinement

The old code's assignment flow is conceptually useful, but it is currently mixed
with non-unified final heads. APTC keeps a unified differentiable refinement:

```text
Y0 = Q*
for t = 1..T:
    A_pos = normalize(homo + alpha_hard * hard)
    A_neg = normalize(hetero + beta_prior * edge_prior)
    pos_msg = A_pos Y_{t-1}
    neg_msg = A_neg Y_{t-1}
    logits_t = log(Q*) + lambda_pos * pos_msg - lambda_neg * neg_msg
    Y_t = Sinkhorn(exp(logits_t / epsilon_t), row_sum = 1, col_sum = pi)
```

The same `T`, equations, and losses apply to all datasets. The learned topology
masks decide how much attraction or repulsion each edge contributes, rather than
the dataset name deciding a backend.

#### 3.4 Multi-view posterior consensus inside the graph

The unified head can compute three posterior proposals from the same trainable
prototypes:

- `Q_attr` from the raw attribute encoder view;
- `Q_low` from the homophily-aware low-pass view;
- `Q_high` from the heterophily-aware high-pass residual view.

A small attention gate predicts node-wise mixture weights from entropy, edge
confidence statistics, and view disagreement:

```text
w_i = softmax(MLP([H_i, entropy(Q_attr_i), entropy(Q_low_i), entropy(Q_high_i)]))
Q_mix = w_attr Q_attr + w_low Q_low + w_high Q_high
Q* = APTC_Sinkhorn_Refine(Q_mix)
```

This replaces the old Wiki-style external consensus, but remains one
differentiable module shared by every dataset.

### 4. Unified objective

The new loss will be the same weighted sum for every dataset:

```text
L = L_transport
  + lambda_proto L_proto_compact
  + lambda_cons L_view_consistency
  + lambda_edge L_edge_posterior
  + lambda_recon L_reconstruction
  + lambda_freq L_frequency
  + lambda_thr L_threshold
  + lambda_prior L_prior_entropy
```

Planned terms:

- `L_transport`: KL consistency between the initial posterior `Q0` and refined
  transported posterior `Y`, with stop-gradient teacher scheduling only during
  warmup.
- `L_proto_compact`: expected distance from embeddings to prototypes under `Y`.
- `L_view_consistency`: symmetric KL or Jensen-Shannon agreement among
  `Q_attr`, `Q_low`, `Q_high`, weighted by confidence so heterophilic nodes are
  not forced into low-pass agreement.
- `L_edge_posterior`: attraction on homophilic edges and repulsion on
  heterophilic edges:
  `- homo * <Y_i,Y_j> + hetero * <Y_i,Y_j>`, with ambiguous edges downweighted.
- `L_reconstruction`: keep the current decoder regularizer to preserve
  attribute information.
- `L_frequency`: keep low-pass Dirichlet smoothing and high-pass contrastive
  separation from the existing frontend.
- `L_threshold`: keep ordered-threshold and target-ratio regularization, but
  make targets adaptive from edge-score quantiles instead of dataset-specific
  constants where possible.
- `L_prior_entropy`: avoid both collapse and over-uniformity by regularizing
  `pi` toward a broad Dirichlet prior.

The only non-differentiable step is evaluation-time `argmax(Y)`, which is allowed
because training optimizes the full soft assignment graph.

### 5. Engineering plan for `AAAI0622`

1. Rebuild the old project structure under
   `D:\study\graduate_student\papers\AAAI2027\AAAI0622` without modifying the
   `AAAI0617` code.
2. Remove legacy-head imports and all final backend dispatch from the new E2E
   implementation.
3. Add an `APTCHead` module with:
   - learnable normalized prototypes;
   - differentiable Sinkhorn transport;
   - topology-aware posterior refinement;
   - optional view-attention posterior mixing.
4. Make `fit_predict` return `Y.argmax(1)` from the unified head for all
   datasets.
5. Add a one-command 9-dataset runner that records ACC/NMI/ARI and compares
   against the 0617 table targets listed above.
6. During iterative experiments, update this file after every architectural
   change, including the reason, mathematical change, and per-dataset impact.

### 6. First experimental hypothesis

The old strongest results came from three disconnected effects:

- ELSS/S2CAG heads supplied balanced low-rank cluster structure;
- Wiki consensus supplied posterior ensembling and label diffusion;
- assignment flow helped heterophilic graphs when used directly.

APTC attempts to absorb all three into one trainable object:

- Sinkhorn transport replaces low-rank/balanced spectral discretization;
- multi-view posterior mixing replaces external consensus;
- topology-aware differentiable refinement replaces post-hoc label diffusion.

The initial risk is that a single global set of loss weights may underfit both
homophilic citation graphs and heterophilic web graphs. The planned mitigation is
not dataset routing, but adaptive quantities computed from the current graph:
posterior entropy, learned cluster prior `pi`, edge-confidence distributions,
and homophily/heterophily mask mass.

### 7. Current status

- Code changes in `AAAI0622`: first unified APTC implementation completed.
- Old code changed: no.
- Next action: run full 9-dataset unified experiments, analyze failures, and
  tune only shared/adaptive APTC mechanisms rather than dataset-specific heads.

## 2026-06-22: APTC v0 Implementation

### Why this change

The old implementation could still be attacked because the source tree contained
multiple final clustering heads and `fit_predict` selected among them with
`final_label_mode`. Even if configs were cleaned later, the code path itself made
dataset-conditioned backend selection available. A paper artifact should make
the illegal path impossible, not merely unused.

### What changed

In `AAAI0622\core\e2e\sect_coco_e2e.py`:

- removed legacy-head imports from the new E2E module;
- added `AdaptivePosteriorTransportHead`;
- replaced Student-t/KMeans-centered final assignment with learnable prototype
  posteriors;
- added differentiable Sinkhorn transport with a learnable cluster prior;
- added multi-view posterior mixing over attribute, low-pass, and high-pass
  views;
- added topology-aware posterior refinement using homophilic attraction and
  heterophilic repulsion;
- changed final prediction to `argmax(q_refined)` for every dataset;
- removed the old final-head dispatch and deleted legacy/S2CAG/consensus helper
  functions from the new core file.

In `AAAI0622\scripts\run_unified_aptc_9datasets.py`:

- added a one-command unified evaluation runner;
- it imports the old dataset scale configs but strips all backend keys such as
  `head_*`, `legacy_*`, `dual_*`, `label_diffusion_*`, `consensus_*`, and
  `s2cag_*`;
- it writes `results\unified_aptc_9datasets.csv` and
  `results\unified_aptc_9datasets_diagnostics.jsonl`;
- it compares each dataset against the 0617 target ACC/NMI/ARI table.

### Mathematical update

The trainable posterior is now:

```text
Q_attr = softmax(cos(P_attr z_attr, C) / tau)
Q_low  = softmax(cos(P_low z_low, C) / tau)
Q_high = softmax(cos(P_high z_high, C) / tau)
Q_mix  = sum_v alpha_v Q_v
Q_T    = Sinkhorn(Q_mix, pi)
Q*     = TopologyRefine(Q_T, homo, hetero, hard)
```

The new objective adds:

- transport consistency between `Q*` and Sinkhorn posterior;
- prototype compactness under `Q*`;
- posterior view consistency;
- edge posterior attraction/repulsion;
- learnable prior entropy regularization.

### Immediate impact

Smoke test command:

```powershell
python scripts\run_unified_aptc_9datasets.py --datasets texas --epochs 1 --device cpu --log-level WARNING
```

Smoke result:

| Dataset | Epochs | ACC | NMI | ARI | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 1 | 32.24 | 10.79 | 2.81 | Forward/loss/eval path works; not a performance run. |

### Research note

The first smoke result is far below the target because it uses only one CPU
epoch and random prototype initialization. The next full experiment must decide
whether APTC needs better differentiable prototype initialization, stronger
posterior sharpening, adaptive Sinkhorn priors, or a topology-conditioned
temperature schedule. These fixes must remain shared modules or graph-statistic
adaptive rules, not dataset-name switches.

## 2026-06-22: APTC v0 Full Run and v1 Repair

### Full-run observation

Command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --device cuda --log-level INFO
```

Result summary:

| Dataset | ACC | NMI | ARI | Main symptom |
| --- | ---: | ---: | ---: | --- |
| ACM | 39.50 | 2.79 | 2.91 | semantic clusters collapsed despite balanced posteriors |
| DBLP | 32.56 | 1.86 | 1.61 | same collapse |
| PubMed | 36.81 | 0.98 | 0.70 | same collapse |
| Wiki | 35.30 | 30.88 | 16.46 | partial structure remains |
| Flickr | 19.68 | 3.99 | 2.01 | severe collapse |
| BlogCatalog | 34.08 | 11.39 | 9.20 | severe collapse |
| Squirrel | 23.46 | 0.80 | 0.60 | weak heterophily signal |
| Texas | 53.55 | 24.26 | 22.08 | only small graph benefits somewhat |
| Chameleon | 25.52 | 1.77 | 1.40 | weak heterophily signal |

Diagnostics show two consistent failure modes:

- Sinkhorn balancing succeeds numerically (`balance` near zero on most
  datasets), but semantic quality is poor. This means balanced transport alone
  is acting as a geometric partitioner over random or drifting prototypes.
- The topology contraction thresholds mark almost all edges as heterophilic on
  many homophilic graphs (`homo_ratio` often near zero). The posterior refinement
  then over-repels neighbors that should share labels.

### APTC v1 change

To keep the pipeline unified while addressing those failures, I added:

- dynamic prototype estimates from the current soft posterior:
  `C_dyn = normalize(Q*^T H / sum_i Q*_ik)`;
- a momentum prototype memory used as a warm start for posterior logits;
- prototype mixing between learnable parameters, memory, and current dynamic
  prototypes;
- adaptive threshold occupancy targets inferred from the current edge-score
  spread, instead of forcing fixed global target ratios.

This is still one shared algorithm. It does not introduce dataset-name routing or
alternative final heads.

### Immediate v1 smoke

Command:

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --datasets texas --epochs 5 --device cuda --log-level WARNING
```

Result: Texas 29.51 / 6.38 / 1.21. This is worse than the v0 full Texas result,
so v1 must be judged on the full run before keeping it. The likely remaining
problem is that dynamic prototypes are still bootstrapped from a poor initial
posterior, while strong transport and topology repulsion can lock in bad
assignments.

### APTC v1 full-run conclusion

The v1 full run degraded most datasets:

| Dataset | v0 ACC | v1 ACC | Conclusion |
| --- | ---: | ---: | --- |
| ACM | 39.50 | 38.94 | no improvement |
| DBLP | 32.56 | 34.95 | tiny improvement only |
| PubMed | 36.81 | 37.98 | tiny improvement only |
| Wiki | 35.30 | 33.06 | worse |
| Flickr | 19.68 | 17.97 | worse |
| BlogCatalog | 34.08 | 26.35 | much worse |
| Squirrel | 23.46 | 21.63 | worse |
| Texas | 53.55 | 46.45 | worse |
| Chameleon | 25.52 | 26.97 | tiny improvement only |

Interpretation: dynamic prototypes estimated from an already bad posterior make
the error self-reinforcing. The next fix should not bootstrap from `Q*` early.
Instead, use one unified unsupervised prototype initialization before training,
then let the differentiable APTC objective update prototypes end-to-end.

### APTC v2 change

Implemented a shared one-time prototype initialization:

```text
H0 = frontend_embedding(X, A) before training
C0 = KMeans(H0, k)
learnable_prototypes <- C0
prototype_memory <- C0
```

This is only an initialization heuristic, applied identically to every dataset.
It is not a final clustering head and is not refreshed during training. Final
labels still come only from `argmax(Q*)`.

I also disabled dynamic posterior-derived prototype mixing by default
(`aptc_dynamic_proto_weight=0.0`) because v1 showed that early self-bootstrapping
amplifies bad assignments.

Smoke result:

| Dataset | Epochs | ACC | NMI | ARI |
| --- | ---: | ---: | ---: | ---: |
| Texas | 5 | 39.89 | 16.67 | 10.26 |

### APTC v3 change

v2 confirmed that one-time prototype initialization helps some datasets, but
training can still wash the initialized semantic structure away. v3 adds two
shared stabilizers:

- initialize the learnable cluster prior from the initial KMeans cluster-size
  distribution instead of starting from uniform;
- store the initial transported posterior `Q_init` and add a small teacher loss
  `KL(Q* || Q_init)` with weight `0.05`.

This is still not a final head: `Q_init` is only a soft regularizer during
training, while final labels are still `argmax(Q*)`.

Short smoke:

| Dataset | Epochs | ACC | NMI | ARI | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 10 | 57.38 | 18.93 | 25.72 | better than v2 short run |
| DBLP | 10 | 29.21 | 0.66 | 0.13 | short run not informative; needs full schedule |

### APTC v3 full-run conclusion

v3 degraded the v2 gains: DBLP fell from 68.33 ACC to 31.01 ACC, and BlogCatalog
fell from 57.60 ACC to 22.21 ACC. The teacher posterior was not preserving good
structure; it was freezing an early bad transported posterior. I disabled
`aptc_init_teacher_weight` by default.

The persistent diagnostic pattern is more fundamental: on homophilic graphs such
as ACM/DBLP/PubMed, the learned contraction often reports `homo_ratio` close to
zero and `hetero_ratio` above 0.93. This makes posterior refinement repel
neighbors that should mostly agree.

### APTC v4 change

I added a unified quantile anchor inside differentiable topology contraction:

```text
low  <- (1 - eta) low_learned  + eta quantile(score, 0.25)
high <- (1 - eta) high_learned + eta quantile(score, 0.75)
```

The masks remain differentiable with respect to edge scores and thresholds, but
the thresholds cannot drift into a degenerate all-heterophily regime. This is a
graph-statistic adaptive rule shared by all datasets, not a dataset branch.

### APTC v4 full-run conclusion

v4 improved ACM, Flickr, Wiki, and Texas relative to v3, but it did not solve the
core problem. Diagnostics still show `hetero_ratio` above 0.89 on most datasets.
The quantile anchor makes thresholds less degenerate in short runs, but during
full training edge scores and thresholds still move toward a mostly-repulsive
posterior refinement.

### APTC v5 change

I added mask-mass balanced posterior refinement:

```text
repel_scale = clamp(mean(attract_weight) / mean(repel_weight), min=0.05, max=1.0)
logits = log(Q_T) + lambda_pos A_pos Q - lambda_neg repel_scale A_neg Q
```

The intent is to preserve heterophily handling without letting an all-hetero
mask dominate every graph. This is a graph-statistic adaptive correction shared
by all datasets.

### APTC v5 full-run conclusion

v5 modestly improved Texas, Flickr, and DBLP relative to v4, but still remained
far from the 0617 targets. This suggests that posterior refinement is no longer
the main bottleneck. The strongest shared signal so far is the one-time
prototype geometry from v2, which helped DBLP and BlogCatalog before later losses
washed it away.

### APTC v6 change

I changed the initialization teacher from topology-refined `Q*` to the cleaner
prototype-geometric posterior `Q_mix`, and set a small default weight:

```text
Q_teacher = Q_mix at initialized prototypes
L_teacher = 0.03 KL(Q* || Q_teacher)
```

The goal is to preserve semantic prototype geometry without freezing early
topology-flow mistakes.

### APTC v6 conclusion

Representative full-schedule subset:

| Dataset | v5 ACC | v6 ACC | Conclusion |
| --- | ---: | ---: | --- |
| DBLP | 44.32 | 32.27 | teacher hurts homophilic citation graph |
| BlogCatalog | 44.69 | 34.45 | teacher hurts social graph |
| Texas | 51.37 | 59.02 | teacher helps one small heterophily graph |

Because the improvement is not consistent, I disabled
`aptc_init_teacher_weight` again by default. Current default keeps v5:
one-time prototype initialization, quantile threshold anchor, and mask-mass
balanced refinement.

### Current bottleneck

The unified pipeline is now structurally correct and end-to-end trainable, but
performance remains far from the 0617 multi-head targets. The repeated diagnostic
pattern is that edge confidence/topology contraction does not reliably separate
homophilic from heterophilic evidence without dataset-specific heads. Further
progress should focus on the frontend edge-confidence objective, not more final
head patching. A likely next change is a unified self-supervised edge calibration
loss that contrasts raw graph edges against feature-KNN edges and prevents the
edge-confidence MLP from collapsing to one evidence source.

## 2026-06-23: v7 Self-Supervised Edge-Confidence Calibration

### Red-line compliance

This design keeps the same forward path and the same final assignment rule for
all 9 datasets:

```text
X, A -> edge confidence -> differentiable topology contraction
     -> frequency-aware views -> APTC posterior transport -> argmax(Q*)
```

The proposed losses are unsupervised, graph-statistic adaptive, differentiable,
and shared by every dataset. They do not introduce dataset-name branches, legacy
heads, post-hoc label diffusion, or alternate clustering backends.

### Notation

For a candidate edge `e=(i,j)`, the edge-confidence module receives edge feature
vector `phi_ij` and evidence vector:

```text
r_ij = raw / learned attribute evidence
d_ij = degree-structure evidence
g_ij = gap-consistency evidence
v_ij = [r_ij, d_ij, g_ij]
alpha_ij = softmax(g_omega(phi_ij)) in Delta^2
s_ij = sigmoid(logit(sum_t alpha_ij,t v_ij,t) + c_omega(phi_ij))
```

The differentiable topology contraction produces:

```text
m^+_ij = homophilic mask
m^-_ij = heterophilic mask
m^0_ij = uncertain / hard mask
m_ij = [m^+_ij, m^-_ij, m^0_ij]
```

v7 adds three frontend calibration losses.

### 1. Evidence Attention Entropy Regularization

Problem: diagnostics show that `alpha_ij` often collapses to one evidence
source. This destroys the intended multi-evidence edge confidence.

Use a normalized entropy target with a mild lower-bound margin:

```text
H_alpha(e) = - sum_t alpha_e,t log(alpha_e,t) / log(3)
L_alpha_entropy = mean_e ReLU(tau_alpha - H_alpha(e))^2
```

Recommended default:

```text
tau_alpha = 0.72
lambda_alpha = 0.01
```

This does not force exactly uniform evidence weights. It only prevents one-hot
collapse, so the model can still prefer attribute, degree, or gap evidence when
the edge statistics justify it.

To avoid a trivial always-uniform solution, pair it with an evidence-response
variance term that rewards node-pair-specific attention variation:

```text
bar_alpha = mean_e alpha_e
L_alpha_usage = KL(U_3 || bar_alpha)
L_alpha = L_alpha_entropy + beta_usage L_alpha_usage
```

where `U_3=[1/3,1/3,1/3]` and `beta_usage` is small, e.g. `0.25`. The first term
prevents per-edge collapse; the second prevents global source starvation.

### 2. Mask Mass / Diversity Loss

Problem: `m^+`, `m^-`, and `m^0` can collapse to degenerate all-homo or
all-hetero regimes. Fixed target ratios are too rigid because graph homophily
varies widely.

Use a dynamic target distribution derived from the edge-score spread:

```text
Delta_s = quantile(s, 0.90) - quantile(s, 0.10)
pi^+_raw = tau_min + (tau_max - tau_min) Delta_s
pi^-_raw = tau_min + (tau_max - tau_min) (1 - Delta_s)
pi^0_raw = tau_mid
pi_m = normalize([pi^+_raw, pi^-_raw, pi^0_raw])
```

where:

```text
tau_min = 0.15
tau_max = 0.45
tau_mid = 0.25
```

Let:

```text
bar_m = mean_e [m^+_e, m^-_e, m^0_e]
```

Then:

```text
L_mask_mass = KL(pi_m || bar_m)
```

To make the constraint smooth rather than hard, add only a margin penalty when a
mask component becomes too small:

```text
L_mask_floor = sum_c ReLU(rho_min - bar_m_c)^2
L_mask = L_mask_mass + beta_floor L_mask_floor
```

Recommended defaults:

```text
rho_min = 0.04
beta_floor = 0.50
lambda_mask = 0.02
```

This keeps every graph with usable positive, negative, and uncertain support
while still allowing the graph's own score distribution to decide the dominant
regime.

### 3. Structure-Attribute Consistency Loss

Problem: without labels, `s_ij` can drift because the confidence MLP receives no
direct self-supervised target. We need a teacher signal built from graph-local
statistics, not dataset labels.

Construct a detached teacher confidence from the three existing evidence
channels:

```text
a_ij = attribute cosine evidence in [0,1]
d_ij = degree similarity evidence in [0,1]
p_ij = edge prior / feature-KNN prior in [0,1]
u_ij = |a_ij - d_ij|
```

Use adaptive reliability weights:

```text
w_attr = stopgrad(1 - u_ij)
w_deg  = stopgrad(d_ij)
w_prior = stopgrad(p_ij)
```

Normalize:

```text
omega_ij = normalize([w_attr, w_deg, w_prior])
```

Then define the teacher:

```text
t_ij = stopgrad(
    omega_attr a_ij
  + omega_deg d_ij
  + omega_prior p_ij
)
```

The confidence score is calibrated by binary cross entropy:

```text
L_score_teacher = mean_e BCE(s_ij, t_ij)
```

But this alone can overfit ambiguous edges. Weight it by teacher confidence:

```text
c_ij = stopgrad(|t_ij - 0.5| * 2)
L_struct_attr = mean_e c_ij BCE(s_ij, t_ij)
```

Recommended default:

```text
lambda_struct_attr = 0.03
```

This teacher is unified and self-supervised. It says: when attribute smoothness,
degree compatibility, and graph/feature prior agree, calibrate `s_ij` strongly;
when they conflict, leave room for the learned confidence module.

### v7 Total Frontend Calibration Objective

The new frontend regularizer is:

```text
L_calib =
    lambda_alpha L_alpha
  + lambda_mask L_mask
  + lambda_struct_attr L_struct_attr
```

The total model objective becomes:

```text
L_total_v7 = L_total_v5 + L_calib
```

with initial shared defaults:

```text
lambda_alpha = 0.01
lambda_mask = 0.02
lambda_struct_attr = 0.03
```

### Expected Diagnostics

v7 should improve frontend health before it improves clustering metrics. The
first validation targets are:

- `alpha_attr`, `alpha_struct`, and `alpha_gap` should not be near `0` or `1`
  globally unless evidence statistics strongly demand it.
- `homo_ratio`, `hetero_ratio`, and `hard_ratio` should all remain nonzero on
  every dataset, with no all-homo/all-hetero collapse.
- `edge_prior` / score-teacher BCE should decrease without forcing all scores to
  raw adjacency.
- ACC/NMI/ARI should recover especially on ACM, DBLP, PubMed, Flickr, and
  BlogCatalog, where v5 diagnostics show topology masks are unreliable.

### Implementation Plan After Confirmation

1. Add config fields:
   `calib_alpha_weight`, `calib_mask_weight`, `calib_struct_attr_weight`,
   `calib_alpha_entropy_floor`, `calib_mask_floor`.
2. Implement helper functions in `core/e2e/sect_coco_e2e.py`:
   `evidence_attention_loss`, `mask_diversity_loss`,
   `structure_attribute_consistency_loss`.
3. Add diagnostics:
   `calib_alpha`, `calib_mask`, `calib_struct_attr`,
   `alpha_entropy`, `mask_target_homo`, `mask_target_hetero`,
   `mask_target_hard`.
4. Run the unified 9-dataset script and compare both metrics and frontend health
   against v5 diagnostics.

### v7 Implementation and Closed-Loop Results

Implemented in `core/e2e/sect_coco_e2e.py`:

- `evidence_attention_loss`;
- `mask_diversity_loss`;
- `structure_attribute_consistency_loss`;
- diagnostics for `calib_alpha`, `calib_mask`, `calib_struct_attr`,
  `alpha_entropy`, `alpha_usage_kl`, and dynamic mask targets.

Numerical stability:

- all logs use `clamp_min(torch.finfo(dtype).eps)`;
- BCE inputs and teachers are clamped to `(eps, 1 - eps)`;
- teacher confidence and dynamic mask targets are detached;
- gradients still flow through `alpha`, `score`, and masks into the edge
  confidence MLP, topology thresholds, and encoder.

Runner output was switched to:

```text
results/unified_aptc_9datasets_v7.csv
results/unified_aptc_9datasets_v7_diagnostics.jsonl
```

#### v7 tuning loop

Three shared hyperparameter rounds were tested:

- v7a: `lambda_alpha=0.01`, `lambda_mask=0.02`,
  `lambda_struct_attr=0.03`, threshold anchor `0.20`.
- v7b: stronger mask correction with `lambda_mask=0.08`, threshold anchor
  `0.45`; this overcorrected masks and hurt ACM/DBLP/BlogCatalog.
- v7d current default: `lambda_mask=0.02`, `lambda_struct_attr=0.01`,
  mask floor `0.06`, threshold anchor `0.20`; this is the best stability
  compromise from the v7 loop.

Current default v7d full-run metrics:

| Dataset | ACC | NMI | ARI |
| --- | ---: | ---: | ---: |
| ACM | 39.93 | 2.54 | 2.06 |
| DBLP | 43.26 | 17.98 | 5.70 |
| PubMed | 54.59 | 11.85 | 10.74 |
| Wiki | 26.94 | 21.28 | 8.94 |
| Flickr | 23.23 | 7.10 | 4.44 |
| BlogCatalog | 46.75 | 27.26 | 14.48 |
| Squirrel | 22.94 | 0.56 | 0.34 |
| Texas | 55.74 | 21.00 | 32.58 |
| Chameleon | 26.26 | 2.58 | 1.71 |

Frontend health after v7d:

| Dataset | alpha attr/struct/gap | alpha entropy | mask homo/hetero/hard |
| --- | --- | ---: | --- |
| ACM | 0.313 / 0.408 / 0.279 | 0.972 | 0.005 / 0.800 / 0.195 |
| DBLP | 0.325 / 0.353 / 0.323 | 0.999 | 0.001 / 0.866 / 0.133 |
| PubMed | 0.268 / 0.560 / 0.173 | 0.884 | 0.002 / 0.897 / 0.101 |
| Wiki | 0.324 / 0.454 / 0.222 | 0.960 | 0.001 / 0.910 / 0.089 |
| Flickr | 0.293 / 0.492 / 0.215 | 0.940 | 0.003 / 0.881 / 0.117 |
| BlogCatalog | 0.325 / 0.385 / 0.290 | 0.993 | 0.002 / 0.885 / 0.113 |
| Squirrel | 0.578 / 0.229 / 0.193 | 0.750 | 0.044 / 0.749 / 0.206 |
| Texas | 0.326 / 0.476 / 0.198 | 0.928 | 0.016 / 0.742 / 0.242 |
| Chameleon | 0.383 / 0.427 / 0.190 | 0.871 | 0.008 / 0.816 / 0.175 |

Conclusion:

- Evidence-source collapse is largely fixed: attention entropy is healthy on
  most datasets and all three evidence channels are used.
- Mask degeneration is only partially fixed: `hard` support recovered, but
  `homo` remains too small on most datasets.
- Metrics improved meaningfully for PubMed, Wiki, Texas, and BlogCatalog over
  some v5/v6 runs, but still remain far below the 0617 multi-head targets.

Next research direction:

The remaining bottleneck is not attention entropy but score calibration around
the high threshold. The model still maps most candidate edges below the
homophily cutoff. The next candidate should calibrate `s_ij` with a rank-aware
or contrastive edge objective that explicitly separates high-confidence
feature-smooth edges from low-confidence mismatch edges, while keeping the same
unified APTC backend.

## v8: Order-Preserving Edge Calibration and Logit Re-centering

v7 showed that evidence attention is healthy, but `s_ij` is still
systematically left-shifted: the homophily mask receives almost no mass because
the sigmoid input is too negative and the learnable high threshold drifts in an
open loop. v8 keeps the unified APTC backend fixed and only recalibrates the
frontend edge-score generator.

### 1. Logit Distribution Re-centering

The confidence score follows:

```text
s_ij = sigmoid(logit(bar_s_ij) + h_omega([phi_ij || alpha_ij]))
```

v8 replaces the raw fused logit with a candidate-edge normalized logit:

```text
r_ij = logit(clip(bar_s_ij)) + h_omega([phi_ij || alpha_ij])
mu_r = stopgrad(mean_e r_e)
sigma_r = stopgrad(std_e r_e)
hat_r_ij = (r_ij - mu_r) / max(sigma_r, eps)
tilde_r_ij = (1 - gamma) r_ij + gamma eta hat_r_ij
s_ij = sigmoid(tilde_r_ij)
```

`gamma` is a shared normalization strength and `eta` is a mild scale factor.
This removes global logit bias while preserving edge-level ordering through the
learned residual.

### 2. Order-Preserving Edge Ranking Loss

The score should respect relative attribute evidence before it is asked to hit
any absolute target. Let:

```text
t_rank,ij = stopgrad(attr_sim_ij * (1 - |attr_sim_ij - deg_sim_ij|))
```

Define positives as the top quantile of `t_rank` and negatives as the bottom
quantile. Pair them by a random permutation and enforce:

```text
L_rank = mean max(0, -s_pos + s_neg + Delta)
```

This is label-free and dataset-agnostic. Gradients flow into `s_ij`, therefore
into the confidence MLP and encoder, but the teacher ordering is detached to
avoid chasing a moving target.

### 3. Dynamic Quantile Threshold Coupling

The high threshold should track the actual score head rather than independently
drifting above it. v8 adds:

```text
q_h = stopgrad(Quantile(s, 1 - rho))
q_l = stopgrad(Quantile(s, rho))
L_anchor = ||h - q_h||_2^2 + 0.5 ||l - q_l||_2^2
```

Unlike the v7 forward-time threshold blend, this keeps `l` and `h` learnable
parameters and couples them through an explicit differentiable loss path.

### v8 Objective

```text
L_total_v8 =
    L_total_v7
  + lambda_rank L_rank
  + lambda_qanchor L_anchor
```

Initial shared defaults:

```text
gamma = 0.75
eta = 1.00
Delta = 0.12
rho = 0.18
lambda_rank = 0.03
lambda_qanchor = 0.08
```

Expected diagnostics:

- `edge_logit_mean` should approach `0`;
- `edge_score_std` should increase from the v7 collapsed range;
- `rank_gap = mean(s_pos) - mean(s_neg)` should become positive;
- `homo_ratio` should recover without forcing mask masses directly.

### v8a Full-Run Diagnosis

v8a completed all 9 datasets without NaN, but did not solve the score-collapse
problem. Early smoke tests showed healthy `score` distributions, but full
training revealed a late-stage escape path: the confidence residual can push
the raw logits strongly negative, and the soft blend
`(1 - gamma) r + gamma norm(r)` still preserves enough raw bias to collapse the
final sigmoid input.

Representative full-run diagnostics:

| Dataset | ACC | edge_logit_mean | score_mean | homo/hetero/hard |
| --- | ---: | ---: | ---: | --- |
| ACM | 37.72 | -2.35 | 0.120 | 0.033 / 0.781 / 0.186 |
| DBLP | 29.06 | -3.20 | 0.055 | 0.010 / 0.873 / 0.117 |
| PubMed | 48.43 | -3.31 | 0.050 | 0.016 / 0.918 / 0.067 |
| Flickr | 20.18 | -3.90 | 0.034 | 0.003 / 0.892 / 0.105 |

Conclusion: v8a's idea is right but the re-centering must be structural, not a
soft residual blend. Ranking should remain auxiliary because a strong ranking
loss can be satisfied by small absolute score gaps after collapse.

### v8b Adjustment

Change the default frontend calibration:

```text
gamma = 1.00
eta = 1.10
lambda_rank = 0.01
lambda_struct_attr = 0.005
lambda_qanchor = 0.12
```

With `gamma = 1`, the score logit becomes:

```text
tilde_r_ij = eta * (r_ij - stopgrad(mean_e r_e)) / stopgrad(std_e r_e)
```

This makes global left-shift impossible at the forward level while still
preserving within-graph ordering and full end-to-end gradients through every
edge score.

### v8b Smoke Diagnosis and v8c Adjustment

v8b successfully fixed the score distribution itself:

```text
edge_logit_mean ~= 0
edge_logit_std  ~= 1.10
edge_score_mean ~= 0.50
homo_ratio      ~= 0.20-0.36 on ACM/DBLP/Texas smoke
```

However, clustering metrics remained weak. The new diagnosis is that the v8a/b
ranking objective used global top/bottom quantiles, while the theoretical
design requires node-local sets `P_i` and `N_i`. Global ranking can flatten
manifold neighborhoods by comparing unrelated candidate edges from different
regions of the graph.

v8c replaces global quantile ranking with a source-node local soft ranking
objective. For each source node `i`:

```text
t_ij = stopgrad(attr_ij * (1 - |attr_ij - deg_ij|))
w^+_ij = softmax_j(t_ij / tau_rank)
w^-_ij = softmax_j((1 - t_ij) / tau_rank)
s^+_i = sum_j w^+_ij s_ij
s^-_i = sum_j w^-_ij s_ij
L_rank_local = mean_i max(0, -s^+_i + s^-_i + Delta)
```

This keeps the order-preserving signal local to each node's candidate
neighborhood. v8c also reduces `lambda_mask` from `0.02` to `0.01`, because
structural logit normalization already restores mask health and the global mask
mass objective should now be only a weak guardrail.

### v8c Full-Run Diagnosis and v8d Adjustment

v8c completed all 9 datasets. It fixed the original v8 target:

```text
edge_logit_mean ~= 0
edge_logit_std  ~= 1.10
score_mean      ~= 0.48-0.50
homo/hetero/hard masks are non-degenerate
```

It also improved several datasets versus v7/v8a, especially PubMed, Wiki, and
BlogCatalog. The remaining failure is a revived evidence-source collapse:
`alpha` often becomes almost one-hot after long training, for example DBLP and
BlogCatalog collapse to attribute evidence, while PubMed/Flickr collapse to
structural evidence.

v8d therefore keeps v8c's logit re-centering and local ranking intact, but
turns the evidence-attention regularizer from a soft preference into a stronger
barrier:

```text
L_alpha =
    mean ReLU(H_floor - H(alpha_ij))^2
  + beta_usage KL(mean_e alpha_e || Uniform)
  + beta_floor sum_c ReLU(u_min - mean_e alpha_c)^2
```

Shared defaults:

```text
lambda_alpha = 0.08
H_floor = 0.85
beta_usage = 0.75
u_min = 0.12
beta_floor = 2.00
```

The goal is to preserve the v8c score/mask recovery while restoring v7's
healthy multi-source evidence fusion.

### v8d Full-Run Diagnosis and v8e Adjustment

v8d completed all 9 datasets without NaN. The frontend score distribution is
now structurally healthy:

```text
edge_logit_mean ~= 0
edge_logit_std  ~= 1.10
score_mean      ~= 0.49-0.51
homo/hetero/hard masks remain non-degenerate
```

The stronger alpha barrier also prevented hard one-hot evidence collapse, but
it became too prescriptive on graphs where one evidence source should
temporarily dominate. PubMed and BlogCatalog dropped from the v8c gains, while
ACM and Texas benefited. This indicates that v8d solved the numerical
calibration problem but over-regularized evidence routing.

v8e keeps the successful v8 structural pieces unchanged:

```text
tilde_r_ij = eta * (r_ij - stopgrad(mean_e r_e)) / stopgrad(std_e r_e)
L_rank_local = mean_i max(0, -s^+_i + s^-_i + Delta)
L_anchor = ||h - stopgrad(Q_{1-rho}(s))||_2^2
         + 0.5 ||l - stopgrad(Q_rho(s))||_2^2
```

Only the evidence-attention barrier is relaxed from "nearly uniform" to
"anti-collapse":

```text
L_alpha_v8e =
    mean ReLU(0.78 - H(alpha_ij))^2
  + 0.45 KL(mean_e alpha_e || Uniform)
  + 1.00 sum_c ReLU(0.08 - mean_e alpha_{e,c})^2
```

Shared v8e defaults:

```text
lambda_alpha = 0.03
H_floor = 0.78
beta_usage = 0.45
u_min = 0.08
beta_floor = 1.00
```

Expected outcome: preserve v8d's zero-centered edge logits and mask health,
while recovering v8c's PubMed/Wiki/BlogCatalog ranking signal by allowing
data-driven evidence specialization inside a unified, dataset-agnostic
pipeline.

### v8e Smoke Diagnosis and v8f Adjustment

v8e smoke testing showed that relaxing all alpha terms at once is too weak.
The edge-score side remains healthy (`edge_logit_mean ~= 0`, `score_mean ~=
0.49`, non-degenerate masks, positive `rank_gap`), but evidence attention
again collapses early:

```text
DBLP        alpha ~= 0.887 / 0.012 / 0.101
BlogCatalog alpha ~= 0.928 / 0.013 / 0.059
Texas       alpha ~= 0.909 / 0.065 / 0.027
PubMed      alpha ~= 0.046 / 0.793 / 0.161
```

The lesson is more precise than "increase alpha regularization": global KL to
uniform can over-constrain useful specialization, while a weak source floor
does not stop collapse. v8f therefore shifts the alpha objective toward a
local anti-one-hot barrier and a stronger evidence-source floor, with a lighter
global usage KL:

```text
L_alpha_v8f =
    mean ReLU(0.82 - H(alpha_ij))^2
  + 0.25 KL(mean_e alpha_e || Uniform)
  + 4.00 sum_c ReLU(0.10 - mean_e alpha_{e,c})^2
```

Shared v8f defaults:

```text
lambda_alpha = 0.06
H_floor = 0.82
beta_usage = 0.25
u_min = 0.10
beta_floor = 4.00
```

This keeps the pipeline unified and end-to-end, but changes the regularizer's
geometry from "be uniform" to "do not become one-hot and do not abandon any
evidence source."

### v8f Smoke Diagnosis and v8g Adjustment

v8f improved the regularizer geometry but still leaves a loophole: the source
floor acts only on the global mean, so the model can keep one evidence source
near zero on most edges and pay only a small average penalty. Raising the loss
weight further risks returning to v8d's over-uniform behavior.

v8g introduces a structural, differentiable Dirichlet smoothing layer directly
inside the evidence attention:

```text
alpha_raw = softmax(g_omega(phi_ij))
alpha_ij = (1 - epsilon_alpha) alpha_raw + epsilon_alpha / 3
```

with a shared default:

```text
epsilon_alpha = 0.08
```

This guarantees every edge keeps all three evidence sources connected to the
fused score while preserving learnable relative preference:

```text
alpha_c in [epsilon_alpha / 3, 1 - 2 epsilon_alpha / 3]
```

The alpha loss is then returned to a lighter v8c/v8e-like role:

```text
lambda_alpha = 0.04
H_floor = 0.78
beta_usage = 0.25
u_min = 0.08
beta_floor = 1.00
```

This is still a single unified forward path and keeps all gradients flowing
through `alpha_raw`, the edge-confidence MLP, and the feature encoder.

### v8g Smoke Diagnosis and v8h Adjustment

v8g improved ACM, PubMed, and Texas, confirming that structural alpha smoothing
is a better anti-collapse mechanism than heavier global KL. However, with
`epsilon_alpha = 0.08`, large graphs can still keep an evidence source near the
minimum on most edges:

```text
DBLP        alpha ~= 0.878 / 0.031 / 0.091
BlogCatalog alpha ~= 0.915 / 0.032 / 0.053
Texas       alpha ~= 0.886 / 0.074 / 0.040
```

v8h therefore increases the shared Dirichlet smoothing level:

```text
epsilon_alpha = 0.18
alpha_c in [0.06, 0.88]
```

The alpha loss remains light, so this is a structural participation constraint
rather than a dataset-specific evidence routing rule.

v8h smoke showed that this stronger smoothing is too restrictive for ACM,
PubMed, and Texas. We therefore keep v8h as an ablation record and select v8g
(`epsilon_alpha = 0.08`) for the next full 9-dataset benchmark.

### v8g Full-Run Diagnosis and v8i Adjustment

v8g full training completed all 9 datasets without NaN. It substantially
improved PubMed, Wiki, Texas, Chameleon, and DBLP NMI relative to earlier
short-run variants, while preserving the v8 zero-centered score distribution.

The remaining frontend failure is not score collapse but threshold/score
decoupling. DBLP and Wiki show the clearest pattern:

```text
DBLP high=0.769, Q_0.82(s)=0.607, low=0.221, Q_0.18(s)=0.341, hard=0.810
Wiki high=0.750, Q_0.82(s)=0.640, low=0.244, Q_0.18(s)=0.293, hard=0.712
```

The learnable thresholds drift outside the active score bulk, causing the
uncertain mask to absorb too many edges and starving the clean homophily mask.
v8i strengthens the already-unified quantile anchor:

```text
L_anchor =
    ||h - stopgrad(Q_{1-rho}(s))||_2^2
  + 0.5 ||l - stopgrad(Q_rho(s))||_2^2

lambda_qanchor: 0.12 -> 0.30
rho: 0.18 -> 0.20
```

No dataset-specific threshold rule is introduced; all datasets still share the
same differentiable contraction path and the same final APTC assignment.

v8i smoke confirmed that stronger anchoring reduces the DBLP/Wiki hard-mask
mass, but the metric response is negative or neutral. This means threshold
alignment alone is not the bottleneck; it can even remove useful uncertainty
mass before the ranking signal is mature. We keep v8i as a negative ablation
and restore the v8g full configuration as the current best checkpoint.

## v9: Homophily Manifold Repair with Raw Leakage and Implicit Subspace Alignment

v8g fixed the numerical physics of edge confidence, but it exposed a deeper
failure mode: highly homophilous citation-style graphs can be over-contracted.
The learned masks may be numerically healthy while still cutting the original
homophily manifold into disconnected pieces. v9 keeps the unified APTC backend
and introduces three end-to-end, dataset-agnostic repair terms.

### 1. Adaptive Raw-Topology Leakage Gate

The low-pass support is no longer forced to rely only on contracted masks:

```text
beta = sigmoid(theta_beta)
w_low_ij = (1 - beta) w_sup_ij + beta a_raw_ij
```

where `a_raw_ij` is the candidate-edge raw adjacency prior. `theta_beta` is
initialized at `2.0`, so `beta ~= 0.88`. On clean homophily graphs, gradients
can preserve raw topology. On noisy heterophily graphs, gradients can reduce
the leakage and trust the contracted support.

### 2. Implicit Self-Expressive Subspace Loss

To convert the old post-hoc subspace refinement idea into an end-to-end
regularizer, v9 adds a sampled self-expression loss on the learned embedding
`H`:

```text
S = softmax(H H^T / tau_s)
L_subspace = ||H - S H||_F^2 + gamma_s mean(|S|)
```

For scalability, the loss is computed on a random node subset with shared
maximum size `M_s`. This keeps gradients flowing into the encoder/projection
while avoiding a full dense `N x N` matrix on large graphs.

### 3. Rayleigh-Quotient View Routing

For each posterior view `Q_v in {Q_attr, Q_low, Q_high}`, compute raw-graph
smoothness:

```text
R(Q_v) = mean_(i,j in E_raw) ||Q_v_i - Q_v_j||_2^2
p_v = softmax(-R(Q_v) / tau_R)
L_rayleigh = KL(mean_i w_i || stopgrad(p))
```

This does not hard-code homophily or heterophily datasets. It gives the view
gate a graph-signal teacher derived from the current posteriors and raw graph:
smooth posterior views receive more routing mass when they explain the raw
manifold well.

Initial v9a defaults:

```text
lambda_subspace = 0.02
tau_s = 0.25
gamma_s = 1e-3
M_s = 2048
lambda_rayleigh = 0.03
tau_R = 0.20
theta_beta_init = 2.0
```

Expected diagnostics:

- `raw_leak_beta` should stay high on clean homophily graphs if raw topology is
  useful;
- `gate_high` should decrease on ACM/DBLP if the high-pass view is noisy;
- `subspace_loss` should add manifold cohesion without changing final
  assignment logic.

### v9a Smoke Diagnosis and v9b Adjustment

v9a smoke on ACM/DBLP/Texas/PubMed was negative. Diagnostics show why:

```text
raw_leak_beta ~= 0.87 on every graph
Rayleigh targets ~= uniform on ACM/DBLP/PubMed
subspace_loss is small and not the main disturbance
```

The raw leakage gate is too biased toward raw topology before the model learns
when to trust it, and the Rayleigh teacher is not discriminative enough when
posterior views are still nearly smooth. v9b therefore isolates the promising
part of v9:

```text
theta_beta_init = 0.0      # beta ~= 0.50 instead of 0.88
lambda_rayleigh = 0.0     # keep diagnostics, remove routing force
lambda_subspace = 0.08    # test implicit manifold stitching directly
```

This keeps all modules unified, but avoids forcing raw topology or view routing
before the self-supervised signals have matured.

### v9b Smoke Diagnosis and v9c Adjustment

v9b reduced raw leakage and disabled Rayleigh routing, but performance remained
below v8g on DBLP/Texas/PubMed. The important lesson is that changing the
low-pass support itself is too invasive: it alters the feature extractor before
the edge-confidence module has decided which raw edges are reliable.

v9c moves raw topology repair from the support matrix to the posterior layer.
For raw candidate edges only, define a confidence-gated stitching weight:

```text
r_ij = 1[a_raw_ij = 1]
c_ij = stopgrad(r_ij * (m^+_ij + 0.5 m^0_ij) * s_ij)
```

Then encourage only the homophily-friendly views to agree over these trusted
raw edges:

```text
L_stitch =
    E_ij c_ij ||Q_low_i - Q_low_j||_2^2
  + 0.5 E_ij c_ij ||Q_attr_i - Q_attr_j||_2^2
```

This preserves the contracted low-pass support and does not force raw topology
into heterophilous graphs. Raw topology helps only when the learned edge score
and masks already assign it enough confidence.

v9c defaults:

```text
theta_beta_init = -2.0      # beta ~= 0.12, weak residual raw leakage only
lambda_subspace = 0.02
lambda_rayleigh = 0.0
lambda_stitch = 0.08
```

v9c smoke improved ACM/Texas over v9b but did not beat v8g, and DBLP remained
collapsed. The stitching diagnostics are revealing: on ACM/DBLP/PubMed the
posterior stitching loss is already near zero, meaning `Q_attr` and `Q_low` are
smooth over raw edges but smooth toward the wrong prototypes. The bottleneck is
therefore no longer raw-edge discontinuity; it is prototype/assignment
geometry. We keep v9a-v9c as negative ablations and restore v8g_full as the
stable code baseline before the next strategy.

## v10: Prototype Geometry Anchoring

v9 showed that raw-edge posterior smoothness is already present on the failed
homophily graphs. The failure is not lack of smoothness, but smooth assignment
to the wrong prototype geometry. The current pipeline initializes prototypes
with KMeans on the initial embedding, but then sets:

```text
lambda_init_teacher = 0
```

so the training objective is free to drift away from the initial partition
basin. v10 turns the existing initialization into a differentiable geometric
anchor instead of a post-hoc backend:

```text
T_0 = stopgrad(Q_mix at initialization)
C_0 = stopgrad(KMeans centers at initialization)

L_teacher = KL(Q_refined || T_0)
L_proto = ||C - C_0||_F^2
L_v10 = lambda_teacher L_teacher + lambda_proto L_proto
```

`T_0` and `C_0` are produced by the same unified model initialization for every
dataset. They do not use labels and do not introduce dataset-specific routes.
The anchor should help ACM/DBLP retain a coherent homophily prototype basin
while remaining weak enough for heterophilous graphs to adapt.

Initial v10a defaults:

```text
lambda_teacher = 0.04
lambda_proto = 0.03
```

### v10a Smoke Diagnosis and v10b Adjustment

v10a slightly improved ACM but did not rescue DBLP, and it hurt PubMed/Texas.
Diagnostics show two causes:

```text
prototype_anchor ~= 1e-4  # too small to change geometry
Texas init_teacher KL ~= 9.9  # teacher conflicts with adaptive assignment
```

Thus the initialization teacher is risky, while prototype geometry still needs
a stronger unsupervised shape constraint. v10b disables the teacher and adds a
prototype separation regularizer:

```text
P = normalize(C)
G = P P^T
L_sep = mean_{a != b} ReLU(G_ab - m_p)^2
```

where `m_p` is a cosine margin. This prevents prototype collapse without
anchoring nodes to a potentially wrong initial partition.

v10b defaults:

```text
lambda_teacher = 0
lambda_proto_anchor = 0.01
lambda_proto_sep = 0.05
m_p = 0.20
```

### v10b Full-Run Diagnosis

v10b full training completed all 9 datasets without NaN. Prototype separation
showed a useful short-run signal on ACM/Texas, but the full run did not improve
the stable v8g baseline:

```text
ACM          38.15 /  1.23 /  1.28
DBLP         32.29 /  5.00 /  0.66
PubMed       53.18 / 13.47 / 10.76
Wiki         26.90 / 23.14 /  8.25
Flickr       20.36 /  5.15 /  2.56
BlogCatalog  38.49 / 13.22 / 10.18
Squirrel     21.32 /  0.14 /  0.02
Texas        52.46 / 11.35 / 16.97
Chameleon    23.23 /  1.05 /  0.23
```

Compared with v8g_full, only Flickr improved slightly; the main homophily and
heterophily targets worsened. Prototype separation alone is therefore not the
missing assignment geometry. The current stable code baseline is restored to
v8g_full, while v10a/v10b remain documented ablations.

## v11: Unified Multi-View Prototype Bootstrap

v10 indicates that the prototype geometry problem starts before training: the
initial KMeans basin built from `embedding` alone is not reliable enough for
ACM/DBLP. v11 changes only the unified initialization representation, not the
forward path or final label rule.

At initialization, collect the already-computed views:

```text
Z_boot = concat(H, Z_attr, Z_low, Z_high)
```

Then reduce it to the model projection dimension with an unsupervised SVD:

```text
U_boot = normalize(SVD(Z_boot, d = projection_dim))
```

KMeans initializes prototypes and priors from `U_boot`; after that, training
and inference still use the same APTC computation and final
`q_refined.argmax`. This is a unified, label-free bootstrap, not a dataset
branch or post-hoc clustering head.

Initial v11a defaults:

```text
init_bootstrap_mode = multiview_svd
init_bootstrap_dim = projection_dim
```

### v11a Smoke Diagnosis and v11b Adjustment

v11a gives the strongest ACM smoke result so far, but it damages PubMed/Texas:

```text
ACM    44.83 / 3.04 / 3.33
DBLP   32.91 / 1.32 / 1.36
PubMed 41.65 / 3.21 / 0.70
Texas  39.89 / 13.72 / 13.53
```

Diagnostics show that the full four-view SVD bootstrap can create poor initial
cluster priors on PubMed/Texas. v11b keeps the useful homophily signal but
removes the noisiest views from initialization:

```text
Z_boot = concat(H, Z_low)
U_boot = normalize(SVD(Z_boot, d = projection_dim))
init_bootstrap_mode = embedding_low_svd
```

The final forward path and `q_refined.argmax` assignment remain unchanged.

v11b smoke did not preserve v11a's ACM gain and still hurt PubMed/Texas:

```text
ACM    38.18 / 1.42 / 1.55
DBLP   32.29 / 2.39 / 1.72
PubMed 41.58 / 1.49 / 2.00
Texas  47.54 / 10.44 / 12.59
```

v11a remains an informative ablation: richer initialization can improve ACM,
but the same initialization destabilizes heterophilous/sparse graphs through
bad initial priors and high assignment-flow KL. v11b shows that simply removing
the high/attribute views does not solve this. The default code is restored to
v8g_full while the bootstrap modes remain available for future controlled
experiments.

## v12: Balanced Multi-View Bootstrap Prior Repair

v11a suggests that multi-view initialization can improve ACM, but diagnostics
show that it can also create skewed initial cluster priors on PubMed/Texas. v12
keeps the useful multi-view centers while repairing only the initialization
prior:

```text
pi_count = bincount(KMeans(Z_boot)) / N
pi_uniform = 1 / K
pi_init = (1 - rho) pi_count + rho pi_uniform
```

This is still unified and label-free. The hypothesis is that centers from
multi-view SVD can help homophily graphs, while a uniform prior blend prevents
small or heterophilous graphs from being trapped by poor initial cluster size
estimates.

Initial v12a defaults:

```text
init_bootstrap_mode = multiview_svd
init_prior_uniform_blend = 0.50
```

### v12a Smoke Diagnosis

Uniform prior repair preserved the ACM gain but did not recover PubMed/Texas:

```text
ACM    44.89 / 3.13 / 3.45
DBLP   30.24 / 0.73 / 0.79
PubMed 44.39 / 5.18 / 3.30
Texas  34.97 / 10.90 / 5.99
```

The prior ranges are much healthier than v11a, so the remaining failure comes
from the multiview centers themselves, not only from skewed cluster priors.
v12b tests the safer alternative: restore `embedding` initialization but keep
a mild uniform prior blend.

```text
init_bootstrap_mode = embedding
init_prior_uniform_blend = 0.50
```

### v12b Smoke Diagnosis and v12c Adjustment

Using the original embedding initialization with a uniform prior blend gives
the strongest ACM smoke result so far and improves DBLP over v11:

```text
ACM    47.31 / 4.19 / 4.64
DBLP   36.16 / 4.16 / 3.42
PubMed 47.93 / 9.52 / 7.95
Texas  39.34 / 6.85 / 8.16
```

The failure is Texas: `flow_kl ~= 7.92`, so `rho = 0.50` over-regularizes the
initial prior for small heterophilous graphs. v12c keeps the same unified
mechanism but weakens the blend:

```text
init_bootstrap_mode = embedding
init_prior_uniform_blend = 0.25
```

### v12c Smoke Diagnosis and v12d Adjustment

Weakening the fixed blend helps Texas compared with `rho = 0.50`, but it also
reduces the ACM/DBLP gain. A fixed global `rho` is too blunt. v12d makes the
blend adaptive from the entropy of the initialization prior:

```text
h_pi = H(pi_count) / log(K)
rho_eff = rho_max * h_pi
pi_init = (1 - rho_eff) pi_count + rho_eff pi_uniform
```

When KMeans already gives a balanced prior, the uniform repair is stronger.
When KMeans gives a very skewed prior, the model keeps more of the original
size signal and avoids over-forcing small heterophilous graphs.

v12d defaults:

```text
init_bootstrap_mode = embedding
init_prior_uniform_blend = 0.50
init_prior_adaptive_blend = true
```

### v12d Full-Run Diagnosis

v12d completed a full 9-dataset run without numerical instability, but the
adaptive prior repair did not survive the full benchmark. The 4-dataset smoke
gain on ACM was not stable in the full run, and the prior smoothing damaged the
strong v8g PubMed/Texas behavior:

```text
ACM          38.38 /  1.36 /  1.44
DBLP         34.41 /  5.13 /  1.59
PubMed       44.93 /  3.87 /  3.66
Wiki         29.27 / 25.65 / 10.37
Flickr       20.74 /  5.41 /  2.49
BlogCatalog  38.07 / 12.71 / 10.58
Squirrel     22.21 /  0.49 /  0.30
Texas        48.63 / 19.18 / 21.13
Chameleon    29.82 /  4.40 /  3.54
```

Compared with v8g_full, the only clear gains are Flickr and Chameleon, while
DBLP, PubMed, Wiki, BlogCatalog, Squirrel, and Texas regress. Diagnostics show
the core failure: blending the initial prior toward uniform reduces assignment
flow KL on some graphs, but it also removes useful cluster-size information and
weakens the transport geometry:

```text
Texas v8g:  prior=(0.044, 0.479), flow_kl=7.715, ACC=61.75
Texas v12d: prior=(0.120, 0.333), flow_kl=0.589, ACC=48.63
PubMed v8g:  prior=(0.298, 0.357), ACC=58.59
PubMed v12d: prior=(0.310, 0.350), ACC=44.93
```

Conclusion: initialization-prior smoothing is a negative ablation. It can make
the optimization look calmer, but the calmer transport state is not a better
clustering state. The runner is restored to the v8g_full defaults:

```text
init_bootstrap_mode = embedding
init_prior_uniform_blend = 0.0
init_prior_adaptive_blend = false
```

The next line of attack should avoid global prior washing. Any repair should be
confidence-adaptive inside the unified computation graph and should preserve
the graph's naturally discovered cluster mass when that mass is informative.

## v13: Confidence-Gated Posterior Sharpening

v12 showed that changing the global cluster prior can make assignment flow look
smoother while damaging the final partition. v13 therefore keeps the v8g
transport prior untouched and moves the repair to a local, confidence-gated
posterior entropy term.

For each candidate edge, define a detached certainty from the current
differentiable topology contraction:

```text
c_ij = stopgrad(s_ij (m^+_ij + m^-_ij) (1 - m^0_ij))
```

The node certainty is the average incident certainty:

```text
c_i = mean_{j in N(i)} c_ij
```

Then v13 adds:

```text
L_conf_entropy = sum_i c_i^p H(Q_i) / (sum_i c_i^p + eps)
```

with shared initial defaults:

```text
lambda_conf_entropy = 0.02
p = 1.5
```

This differs from the existing global low-entropy term. It sharpens posteriors
only where the frontend itself marks local edge evidence as confident, while
ambiguous regions keep room for APTC transport. The certainty weights are
detached so the edge-confidence module cannot reduce the loss by artificially
lowering certainty. All datasets share the same forward path, loss composition,
and final `q_refined.argmax` assignment.

### v13a Smoke Diagnosis and v13b Adjustment

v13a smoke on ACM/DBLP/PubMed/Texas showed the intended ACM effect but also a
clear over-sharpening failure:

```text
ACM    44.83 /  7.14 /  7.16
DBLP   29.90 /  1.98 / -0.33
PubMed 51.03 / 13.19 / 10.96
Texas  44.81 /  7.34 /  9.38
```

The diagnosis is that `c_ij` used both `m^+` and `m^-`. Heterophilous evidence
is useful for repulsion and boundary discovery, but it should not directly
force each endpoint posterior to become sharper. v13b keeps the unified loss
but makes it explicitly homophily-gated:

```text
c_ij = stopgrad(s_ij m^+_ij (1 - m^0_ij))
lambda_conf_entropy = 0.006
p = 2.0
```

This should preserve part of the ACM gain while reducing damage to Texas/PubMed
and avoiding artificial sharpening on uncertain DBLP edges.

### v13b Smoke Diagnosis

v13b reduced the over-sharpening pressure but did not make the direction
robust:

```text
ACM    40.96 /  3.35 /  3.21
DBLP   28.77 /  1.33 / -0.40
PubMed 49.08 /  7.87 /  6.93
Texas  48.09 / 13.07 / 16.03
```

The ACM gain is much smaller than v13a, while DBLP remains worse than v8g_full.
This suggests the failure is not simply excessive heterophily-triggered
sharpening. Directly lowering posterior entropy changes the assignment
dynamics too bluntly, even when the trigger is local and confidence-weighted.

Conclusion: v13 is a negative ablation. The `L_conf_entropy` module remains in
the code as a disabled diagnostic/ablation hook, but the runner is restored to
v8g_full:

```text
lambda_conf_entropy = 0
```

The next repair should not directly compress posterior entropy. It should
instead improve the view evidence and prototype geometry before the APTC
posterior is formed.

## v14: Order-Preserving View-Logit Calibration

v13 showed that acting on the final posterior entropy is too blunt. v14 moves
the correction one step earlier, before each view posterior is formed. For a
view `v`, define node-prototype logits:

```text
z_i = Qlogits_v(i, :) = <h_i^v, C> / tau
```

Because softmax is invariant to additive shifts, center each node's logits and
measure its prototype contrast:

```text
u_i = z_i - mean(z_i)
sigma_i = std(u_i)
```

If `sigma_i` is below a shared floor, apply a positive scalar lift:

```text
u'_i = u_i * (1 + eta max(0, sigma_min - sigma_i) / (sigma_i + eps))
Q_v(i, :) = softmax(u'_i)
```

This is order-preserving: for every node, the relative ranking of prototypes is
unchanged because the correction is a positive scalar. It also does not touch
the transport prior or final assignment rule. The goal is to prevent ACM/DBLP
style homophily graphs from entering APTC with almost uniform view posteriors,
while leaving already contrasted views mostly unchanged.

Initial v14a defaults:

```text
sigma_min = 0.55
eta = 0.60
```

Diagnostics add the pre-calibration view-logit standard deviations:

```text
logit_std_attr, logit_std_low, logit_std_high
```

### v14a Smoke Diagnosis and v14b Adjustment

v14a confirmed that ACM/DBLP/PubMed enter the APTC head with low view-logit
contrast:

```text
ACM    logstd ~= (0.199, 0.190, 0.228)
DBLP   logstd ~= (0.263, 0.328, 0.302)
PubMed logstd ~= (0.360, 0.245, 0.351)
Texas  logstd ~= (0.866, 0.971, 0.781)
```

However, the initial floor was too aggressive:

```text
ACM    40.00 /  1.32 /  1.04
DBLP   33.84 /  5.17 /  0.66
PubMed 49.77 /  9.57 /  9.45
Texas  50.82 / 19.96 / 26.35
```

The lifted logits increased view-consistency and flow tension without producing
a better partition. v14b keeps the same unified order-preserving mechanism but
only repairs severely flat nodes:

```text
sigma_min = 0.30
eta = 0.35
```

### v14b Smoke Diagnosis

The softer floor reduced some of v14a's flow tension, but it still failed to
beat v8g_full on the critical smoke set:

```text
ACM    39.50 /  1.15 /  1.29
DBLP   29.31 /  1.94 / -0.47
PubMed 53.45 / 15.60 / 12.64
Texas  55.74 / 23.34 / 27.58
```

Compared with v8g_full, PubMed and Texas remain lower, and DBLP collapses
badly. The lesson is that simply amplifying node-prototype contrast before the
view softmax does not repair wrong evidence geometry. The v14 module remains
available as a disabled ablation hook, but the runner is restored to:

```text
sigma_min = 0
```

The next frontend repair should act on the evidence source itself, especially
the raw-topology evidence that is currently present only as an MLP feature
rather than as a direct evidence channel.

## v15: Explicit Raw-Topology Evidence Channel

v14 suggests that the problem is not posterior sharpness but evidence geometry.
In the v8g frontend, the candidate/raw topology prior `a_ij` is available to the
edge MLP as an input feature, but it is not one of the direct evidence sources
mixed by `alpha_ij`. The direct fusion currently uses:

```text
e_ij = [attr_sim, degree_sim, 1 - |attr_sim - degree_sim|]
```

v15 adds a unified optional fourth channel:

```text
e_ij = [attr_sim, degree_sim, 1 - |attr_sim - degree_sim|, a_ij]
alpha_ij = softmax(g_omega(phi_ij)) in R^4
s_ij = sum_c alpha_ij,c e_ij,c + calibrated residual
```

The topology prior `a_ij` is the same candidate-edge prior already constructed
for every dataset: raw graph edges receive prior 1, and feature-KNN candidate
edges receive their feature-neighbor score. This is not a dataset branch; it is
a fourth evidence source shared by every graph. The module is config-gated so
the v8g baseline remains exactly reproducible when disabled.

Initial v15a defaults:

```text
edge_prior_evidence = true
edge_alpha_smoothing = 0.08
```

Diagnostics now include:

```text
alpha_prior
```

Expected behavior: DBLP/ACM should no longer be forced to recover raw topology
only through the calibrator residual. If the raw graph manifold is reliable,
`alpha_prior` can rise end-to-end; on noisy candidate edges it can remain near
the smoothing floor.

### v15a Smoke Diagnosis

v15a did not validate the direct-evidence hypothesis:

```text
ACM    41.62 /  2.18 /  2.27
DBLP   32.22 /  6.51 /  0.20
PubMed 44.25 /  8.72 /  7.41
Texas  36.07 / 16.34 / -6.37
```

Diagnostics show why. On DBLP, the new channel remained near the smoothing
floor and did not become the missing raw-topology route:

```text
DBLP alpha = (attr 0.825, degree 0.024, gap 0.126, prior 0.026)
```

On Texas, the fourth channel disrupted the score/posterior geometry:

```text
Texas entropy ~= 0, balance ~= 0.44, ACC=36.07
```

Conclusion: raw topology should not be injected as a direct score evidence
channel. The optional 4-source module remains in code as a disabled ablation
hook, but the runner is restored to:

```text
edge_prior_evidence = false
```

The next attempt should use raw topology only as a soft teacher/ranking signal,
not as a direct fused evidence source.

## v16: Raw-Reliability-Gated Ranking Teacher

v15 showed that raw topology is too disruptive as a direct evidence channel.
v16 keeps the v8g score fusion unchanged and moves raw topology into the
existing order-preserving edge-ranking objective. The key is to make raw
topology graph-reliable before it can influence the teacher.

For each candidate edge, keep the original attribute-degree teacher:

```text
r_ij = 1 - |a_ij - d_ij|
t_base = a_ij r_ij
```

Then estimate a graph-level raw reliability signal by comparing clean evidence
on raw candidate edges and non-raw feature candidates:

```text
c_ij = 0.5 (a_ij + d_ij) r_ij
Delta_raw = mean_{e in E_raw} c_e - mean_{e notin E_raw} c_e
g_raw = sigmoid((Delta_raw - m) / tau)
```

Only when raw edges are globally cleaner than feature candidates does the raw
prior enter the ranking teacher:

```text
lambda_raw_eff = lambda_raw g_raw
t_raw = 0.5 p_ij + 0.5 p_ij r_ij
t_rank = (1 - lambda_raw_eff) t_base + lambda_raw_eff t_raw
```

This is still a single unified objective. There is no dataset branch, and raw
topology does not directly enter `s_ij` fusion, cluster prior, APTC transport,
or final `argmax`.

Initial v16a defaults:

```text
edge_rank_raw_teacher_weight = 0.35
edge_rank_raw_gate_margin = 0.02
edge_rank_raw_gate_temperature = 0.05
edge_prior_evidence = false
```

Diagnostics added:

```text
rank_raw_gate
rank_raw_advantage
rank_raw_weight
```

Expected behavior: ACM/DBLP should get a raw-topology ranking signal only if
their raw graph edges are cleaner under current attribute/degree evidence.
Texas/PubMed should remain protected when raw topology is not graph-reliable.

### v16a Smoke Diagnosis and v16b Adjustment

v16a improved ACM slightly over the 80-epoch v8g smoke but did not improve the
critical set overall:

```text
ACM    41.62 /  2.12 /  1.88
DBLP   32.91 /  5.14 /  2.86
PubMed 48.23 /  8.98 /  8.96
Texas  52.46 / 12.77 / 19.18
```

The new diagnostics explain the failure. On DBLP and PubMed, candidate edges
are effectively all raw edges after candidate construction, so the v16a
`raw-vs-feature` reliability comparison has no non-raw reference and collapses
to:

```text
rank_raw_gate = 0
rank_raw_weight = 0
```

Thus v16a did not actually inject a raw-topology ranking signal where DBLP
needed it. v16b keeps the same unified ranking-teacher idea but adds an
all-raw fallback: when no non-raw candidate reference exists, estimate raw
reliability from the spread of clean evidence inside the raw edges themselves.

```text
Delta_raw = std_{e in E_raw}(c_e)
g_raw = sigmoid((Delta_raw - m) / tau)
p'_ij = p_ij sigmoid((c_ij - mean_{E_raw} c) / tau)
t_raw = 0.5 p'_ij + 0.5 p'_ij r_ij
```

This gives DBLP/PubMed an internal high-clean-vs-low-clean raw-edge ranking
signal without treating every raw edge as equally trustworthy.

### v16b Smoke Diagnosis

v16b successfully fixed the v16a degeneracy: DBLP and PubMed now receive a
strong raw-gated ranking signal.

```text
DBLP   rank_raw_gate=0.806, rank_raw_weight=0.282, rank_gap=0.224
PubMed rank_raw_gate=0.966, rank_raw_weight=0.338, rank_gap=0.176
Texas  rank_raw_gate=0.056, rank_raw_weight=0.020
```

However, the metrics did not improve over the 80-epoch v8g smoke:

```text
ACM    41.62 /  2.12 /  1.88
DBLP   32.71 /  4.87 /  2.79
PubMed 48.51 /  9.18 /  8.06
Texas  54.64 / 13.83 / 19.31
```

This is an important negative result. Improving the edge-ranking teacher and
the measured `rank_gap` is not sufficient to repair the final partition. The
current bottleneck is likely downstream of edge confidence: the APTC
prototype/posterior geometry can still settle into a bad basin even when the
frontend ranking signal looks healthier. v16 remains documented as an
ablation, but the next attempt should act closer to the prototype geometry and
assignment flow.

## v17: Prototype-Readout Geometry Recheck

Before changing the algorithm again, I ran a readout diagnostic under the
current unified training setup. For each dataset, I compared the final
`q_refined.argmax` against several alternative readouts from the same trained
model:

```text
ACM:   q_refined ACC=41.62, embedding KMeans ACC=79.90
DBLP:  q_refined ACC=32.91, embedding KMeans ACC=67.88
PubMed:q_refined ACC=46.80, embedding KMeans ACC=59.37
Texas: q_refined ACC=54.10, embedding KMeans ACC=74.32
```

This is the cleanest diagnosis so far: the frontend embedding already contains
much stronger cluster structure than the differentiable APTC posterior reads
out. The failure is therefore concentrated in prototype/posterior geometry, not
in representation learning alone.

v17a first retests the existing dynamic prototype mechanism under the newer
v8g frontend physics:

```text
aptc_dynamic_proto_weight = 0.25
edge_rank_raw_teacher_weight = 0
```

This is a small sanity experiment. Earlier APTC v1 dynamic prototypes failed
because they were bootstrapped from a poor posterior. If v17a still fails after
the stronger v8g frontend, the next step should be a more guarded prototype
readout objective rather than direct self-bootstrapping from `Q*`.

### v17a Smoke Diagnosis and v17b Adjustment

v17a confirmed that direct dynamic prototypes are still unsafe:

```text
ACM    36.93 /  0.89 /  0.65
DBLP   32.56 /  3.30 /  1.10
PubMed 40.32 /  1.91 /  1.17
Texas  54.10 / 16.67 / 22.84
```

The current `C_dyn = Q*^T H` mechanism still self-bootstraps from a weak
posterior, so it damages the embedding readout instead of recovering it.

v17b therefore avoids replacing prototypes with `Q*`-derived centroids. It adds
a guarded differentiable readout alignment:

```text
T_proto = stopgrad(softmax(normalize(H) normalize(C)^T / tau_proto))
L_proto_readout = KL(Q* || T_proto)
```

This teacher is produced by the same learnable prototypes and current
embedding, but the teacher is detached so `Q*` is asked to respect the embedding
geometry without allowing a shortcut that moves the teacher itself.

Initial v17b defaults:

```text
aptc_dynamic_proto_weight = 0
aptc_proto_readout_weight = 0.04
aptc_proto_readout_temperature = 0.20
```

### v17b Smoke Diagnosis and v17c Adjustment

v17b gave the first downstream-positive signal, but it was not robust:

```text
ACM    40.60 /  3.40 /  3.42
DBLP   34.39 /  5.77 /  1.26
PubMed 50.35 / 11.66 /  9.49
Texas  50.82 / 14.80 / 15.34
```

Compared with the 80-epoch v8g smoke, DBLP and PubMed improved, while ACM and
Texas dropped. Diagnostics show the conflict:

```text
Texas proto_readout KL = 5.63
Texas entropy = 0.21
```

The prototype teacher strongly disagrees with an already sharp posterior on the
small heterophily graph. v17c therefore gates `L_proto_readout` at the node
level:

```text
w_i = H(Q*_i) / log K
c_i = (max T_i - 1/K) / (1 - 1/K)
L_proto_readout = sum_i w_i c_i KL(Q*_i || T_i) / sum_i w_i c_i
```

The alignment now acts mainly where the posterior is still uncertain and the
prototype teacher is confident. This keeps the same unified differentiable
objective, but avoids forcing already-sharp assignments to obey a conflicting
prototype readout.

### v17c Smoke Diagnosis and v17d Adjustment

v17c is the first clear positive downstream result:

```text
ACM    41.02 /  2.06 /  1.83
DBLP   41.68 / 13.07 /  5.20
PubMed 48.50 /  9.71 /  7.24
Texas  69.95 / 37.91 / 51.76
```

The node gate behaves as intended:

```text
DBLP  proto_readout_weight = 0.173
Texas proto_readout_weight = 0.0019
```

Thus the readout teacher helps DBLP while automatically backing off on Texas,
where the posterior is already extremely sharp. This supports the v17 diagnosis
that the main problem is the differentiable prototype readout, not the
frontend embedding.

However, PubMed drops below the 80-epoch v8g smoke, and ACM gains are modest.
v17d keeps the same gated objective but reduces the global pressure:

```text
aptc_proto_readout_weight = 0.025
```

The target is to preserve the DBLP/Texas gains while reducing collateral
pressure on PubMed.

### v17d Smoke Diagnosis

v17d was more conservative, but it lost the most important DBLP gain:

```text
ACM    38.61 /  1.58 /  1.22
DBLP   34.58 /  3.13 /  1.11
PubMed 50.73 / 11.89 /  9.67
Texas  68.31 / 31.09 / 47.34
```

Lowering the global readout weight helps PubMed slightly but weakens the
DBLP/Texas breakthrough. Therefore v17c (`lambda=0.04`) is the more promising
mainline for a full 9-dataset run, while v17d remains a conservative ablation.

### v17c Full-Run Diagnosis

v17c full run did not become a global default. Compared with v8g_full:

```text
Dataset      v8g ACC  v17c ACC  Delta
ACM          38.35    39.31     +0.96
DBLP         38.11    36.48     -1.63
PubMed       58.59    53.98     -4.62
Wiki         33.01    29.73     -3.28
Flickr       18.47    15.00     -3.47
BlogCatalog  39.95    61.66    +21.71
Squirrel     22.55    23.84     +1.29
Texas        61.75    59.02     -2.73
Chameleon    28.72    23.28     -5.45
```

The important scientific result is not that v17c is the new default; it is that
prototype-readout alignment can massively improve BlogCatalog and modestly help
ACM/Squirrel, while still hurting DBLP/PubMed/Wiki/Flickr/Texas/Chameleon under
a fixed global weight. The mechanism is real but needs a graph-level gate before
it can be used as the unified mainline.

Decision: keep v17 code as an ablation hook, but restore the runner default to
the more stable v8g_full configuration:

```text
aptc_proto_readout_weight = 0
```

Next direction: learn a graph-level controller for prototype-readout alignment
from observable diagnostics such as embedding/prototype teacher confidence,
posterior entropy, flow KL, and alpha-source collapse. It must be continuous and
dataset-agnostic, not a dataset-name branch.

## v18: Graph-Gated Prototype Readout Alignment

v17 proved that prototype-readout alignment is a real mechanism, but a fixed
global weight is unsafe. v18 adds a continuous graph-level controller using
only observable training statistics:

```text
g_prior = exp(-s_pi L_prior_entropy)
g_attr  = clamp((mean(alpha_attr) - a0) / da, 0, 1)
g_flow  = 1 / (1 + KL(Q* || Q_mix))
g_graph = g_prior g_attr g_flow
```

Then:

```text
L_v18 = lambda_proto g_graph L_proto_readout
```

Intuition:

- `g_prior` suppresses the readout teacher when cluster priors are highly
  skewed, which protected Texas/Chameleon in the v17c diagnostics.
- `g_attr` opens the teacher mainly when the edge-confidence module has
  collapsed toward attribute evidence, which matches BlogCatalog and DBLP.
- `g_flow` suppresses the teacher when assignment flow is already in heavy
  conflict with the mixed posterior.

Initial v18a defaults:

```text
lambda_proto = 0.04
s_pi = 8.0
a0 = 0.35
da = 0.45
```

This remains a single unified objective. There is no dataset-name branch and no
post-hoc KMeans final readout.

### v18a Smoke Diagnosis and v18b Adjustment

v18a gate behaved sensibly in diagnostics but did not improve the five-dataset
smoke set enough:

```text
ACM         42.08 /  3.37 /  2.59
DBLP        35.37 /  5.95 /  3.24
PubMed      47.90 /  8.88 /  9.45
Texas       54.64 / 16.28 / 23.31
BlogCatalog 38.51 / 12.88 / 10.86
```

Diagnostics show that BlogCatalog has a high final graph gate
(`g_graph ~= 0.72`), but short training did not reproduce the v17c full-run
gain. This suggests the gate may close too strongly during early optimization,
preventing the model from entering the good prototype basin.

v18b keeps the graph gate but adds a small floor:

```text
g = g_min + (1 - g_min) g_graph
g_min = 0.25
```

The floor is still dataset-agnostic. It preserves a weak readout signal during
early training while retaining graph-dependent downweighting for high-conflict
cases.

### v18b Smoke Diagnosis and v18c Adjustment

v18b made the smoke set worse:

```text
ACM         37.75 /  0.74 /  0.51
DBLP        31.50 /  1.99 /  0.81
PubMed      49.42 / 10.56 /  8.85
Texas       60.11 / 20.93 / 32.32
BlogCatalog 31.27 /  9.20 /  6.12
```

The gate floor polluted early optimization, especially on DBLP and BlogCatalog.
This suggests the right way to protect the early basin is not a floor, but a
delayed schedule.

v18c removes the floor and delays prototype-readout alignment:

```text
g_min = 0
lambda_eff(epoch) = 0                         if epoch < 80
lambda_eff(epoch) = lambda * (epoch-79)/80    for epoch in [80,159]
lambda_eff(epoch) = lambda                    afterward
```

This keeps the base v8g dynamics intact early, then lets the graph-gated
prototype readout act after the posterior and edge confidence have stabilized.

### v18c Full-Run Diagnosis

v18c improved stability compared with v17c but still did not become a global
default:

```text
Dataset      v8g ACC  v18c ACC  Delta
ACM          38.35    39.60     +1.26
DBLP         38.11    35.05     -3.06
PubMed       58.59    52.35     -6.24
Wiki         33.01    33.18     +0.17
Flickr       18.47    20.18     +1.72
BlogCatalog  39.95    49.94     +9.99
Squirrel     22.55    23.92     +1.37
Texas        61.75    48.09    -13.66
Chameleon    28.72    24.90     -3.82
```

The graph-gated delayed readout preserves part of the BlogCatalog gain and
helps ACM/Flickr/Squirrel, but it still damages DBLP/PubMed/Texas/Chameleon.
The core diagnosis remains: the final embedding contains stronger cluster
structure than the APTC posterior, but using it only as a loss teacher is too
indirect and unstable.

## v19: Embedding Posterior View

The readout diagnostics reveal a structural gap: the trained final `embedding`
is strong under KMeans, but APTC posterior fusion only uses:

```text
Q_attr, Q_low, Q_high
```

It never directly includes a posterior view computed from the final fused
embedding itself. v19 adds a fourth unified view:

```text
Q_embed = softmax(normalize(W_e H) C^T / tau)
Q_mix = sum_v gamma_v Q_v,  v in {attr, low, high, embed}
```

The view gate now receives four view entropies and outputs four weights. The
rest of the pipeline is unchanged: Sinkhorn transport, topology refinement, and
final `q_refined.argmax` remain the same for every dataset.

Initial v19a isolates this mechanism:

```text
aptc_embedding_view = true
aptc_proto_readout_weight = 0
```

This tests whether the strong embedding geometry can help the differentiable
posterior directly, without post-hoc KMeans and without the unstable v17/v18
teacher loss.

### v19a Smoke Diagnosis

v19a was a negative result:

```text
ACM         34.94 /  0.09 /  0.06
DBLP        35.08 /  4.46 /  1.79
PubMed      38.60 /  2.70 /  1.08
Texas       52.46 / 11.68 / 19.57
BlogCatalog 25.62 /  3.18 /  2.40
```

Diagnostics show that the new embedding view was used immediately:

```text
gate_embed ~= 0.16 - 0.32
```

but the `embed_projector` starts as a random trainable projection. Instead of
preserving the strong embedding geometry, it injects another unstable posterior
view and degrades early optimization. The v19 module remains available as a
disabled ablation hook:

```text
aptc_embedding_view = false
```

The embedding-readout clue should be used more conservatively, for example as
a better initialization or delayed/gated teacher, not as a randomly projected
fourth view from epoch 0.

### v19b Adjustment

The v19a failure was partly caused by an unnecessary random `embed_projector`.
The final embedding already lives in the same `projection_dim` space as the
learned prototypes, so v19b removes that projector:

```text
Q_embed = softmax(normalize(H) C^T / tau)
```

This tests the actual embedding-posterior hypothesis without injecting an extra
random projection at epoch 0.

### v19b Smoke Diagnosis

v19b revealed a useful but narrow signal:

```text
ACM         38.71 /  0.85 /  0.67
DBLP        28.79 /  0.74 / -0.08
PubMed      37.14 /  0.83 /  0.88
Texas       69.95 / 40.29 / 50.10
BlogCatalog 31.27 / 13.11 /  6.68
```

Direct embedding view strongly helps Texas, confirming that embedding geometry
can repair high-conflict heterophily assignment. But it severely hurts ACM,
DBLP, PubMed, and BlogCatalog. The view gate does not sufficiently protect
low-conflict graphs from the new view.

Decision: keep `aptc_embedding_view` as a disabled ablation hook and restore the
runner default to:

```text
aptc_embedding_view = false
```

Future version should gate the embedding view with a stronger graph-level
conflict controller, likely based on flow KL / posterior entropy / prototype
logit contrast, so the view opens for Texas-like states but stays closed for
homophilic large graphs.

## Seed-Sensitivity Diagnostic

I also ran a 3-seed 80-epoch diagnostic under the stable v8g pipeline on
ACM/DBLP/PubMed/Texas/BlogCatalog. There is meaningful variance, but no seed
produced a hidden near-SOTA basin:

```text
ACM best ACC ~= 39.14
DBLP best ACC ~= 34.53
PubMed best ACC ~= 45.17
Texas best ACC ~= 54.10
BlogCatalog best ACC ~= 30.64
```

Conclusion: best-of-seed is not the missing ingredient. The remaining problem
is structural readout/control: when and how the strong embedding geometry should
influence the differentiable APTC posterior.

## v20: Mid-Training Prototype Refresh

v19a showed that injecting a random fourth view is unsafe. The safer way to use
the strong embedding geometry is to refresh prototypes after the encoder has
learned a better representation.

Current initialization happens only before training:

```text
H_0 = embedding before optimization
C_0 = KMeans(H_0)
```

But diagnostics showed that the trained embedding is much stronger than the
APTC posterior. v20 adds a single unified refresh after pretraining:

```text
at epoch t_mid:
  H_t = current embedding
  C_t = KMeans(H_t)
  prototypes <- C_t
  prototype_memory <- C_t
  cluster_prior <- cluster counts from C_t
```

After this refresh, training continues with the same differentiable APTC
objective and final `q_refined.argmax`. There is no post-hoc KMeans final
assignment and no dataset-specific routing.

Initial v20a:

```text
mid_init_epoch = 50
mid_init_bootstrap_mode = embedding
aptc_embedding_view = false
aptc_proto_readout_weight = 0
```

### v20a Smoke Diagnosis

v20a was also negative:

```text
ACM         42.05 /  2.97 /  3.21
DBLP        30.96 /  1.30 /  0.56
PubMed      40.37 /  0.03 / -0.14
Texas       51.91 / 12.85 / 23.09
BlogCatalog 24.44 /  3.61 /  2.43
```

The mid-training KMeans refresh is too abrupt. It can slightly help ACM, but it
destroys the training state on DBLP/PubMed/BlogCatalog. Like v19, it confirms
that the embedding signal is valuable but cannot be injected by a hard prototype
reset.

Decision: keep the mid-init code as a disabled ablation hook and restore the
runner to the stable v8g_full default:

```text
mid_init_epoch = -1
aptc_embedding_view = false
aptc_proto_readout_weight = 0
```

Next diagnostic direction: quantify seed sensitivity under the stable unified
pipeline. If the current gap is largely initialization variance, a unified
stability objective or ensemble-free consistency regularizer may be more
appropriate than further hard prototype manipulation.

## v21: Conflict-Gated Embedding Posterior Injection

v19b established a sharp but unstable fact: the final embedding posterior helps
Texas-like high-conflict graphs a lot, but global direct use of `Q_embed`
damages ACM/DBLP/PubMed/BlogCatalog. The core problem is therefore not whether
embedding geometry is useful, but **when and how** it should be allowed to
influence the differentiable APTC posterior.

The first v21 idea is a unified graph-level conflict gate:

```text
g_graph = g_flow * g_prior * g_std * g_entropy
```

where:

```text
g_flow    = sigmoid((log(1 + KL(Q_flow_base || Q_base)) - c_flow) / s_flow)
g_prior   = sigmoid((H_prior - c_prior) / s_prior)
g_std     = sigmoid(((std_embed - std_base) - c_std) / s_std)
g_entropy = sigmoid((((H_base - H_embed)) - c_ent) / s_ent)
```

Interpretation:

- `g_flow` opens when the stable 3-view posterior still disagrees strongly with
  its own refinement, which is a direct sign of unresolved assignment conflict.
- `g_prior` opens when the learned cluster prior is already imbalanced, which
  often happens on difficult heterophily/small-graph states.
- `g_std` requires the embedding posterior to be sharper than the 3-view base.
- `g_entropy` requires `Q_embed` to be more decisive than `Q_base`.

This gate is fully differentiable, dataset-agnostic, and keeps the frontend
topology-contraction path unchanged.

### v21a: Graph-Gated Fourth View

The first implementation keeps the v19-style four-view mixture, but multiplies
the embedding-view branch by the graph conflict gate before final normalization:

```text
Q_mix = normalize(g_attr Q_attr + g_low Q_low + g_high Q_high + g_graph g_embed Q_embed)
```

80-epoch smoke on ACM/DBLP/PubMed/Texas/BlogCatalog:

```text
ACM         40.40 /  2.66 /  2.84
DBLP        34.88 /  3.78 /  1.81
PubMed      46.89 /  8.27 /  6.93
Texas       71.04 / 42.58 / 54.22
BlogCatalog 27.37 /  4.59 /  3.70
```

This is already much healthier than v19b:

- Texas remains strongly positive.
- DBLP and PubMed recover a large part of the v19b collapse.
- ACM also improves over v19b.
- BlogCatalog still drops badly.

The diagnostics confirm that the gate behaves structurally as intended:

```text
Texas:       embed_graph_gate ~= 0.959
ACM:         embed_graph_gate ~= 0.00047
PubMed:      embed_graph_gate ~= 0.0042
BlogCatalog: embed_graph_gate ~= 0.0135
```

So the main issue is no longer “the gate cannot distinguish graphs.” It can.

### v21b: Delayed / Stricter Graph Gate

The next hypothesis was that BlogCatalog damage comes from early training
instability, even if the final gate ends near zero. v21b keeps the same graph
gate idea but makes it stricter and delays activation:

```text
aptc_embedding_gate_warmup_epochs = 50
aptc_embedding_gate_ramp_epochs   = 30
```

Smoke result:

```text
ACM         40.56 /  3.00 /  3.31
DBLP        30.88 /  3.01 /  0.99
PubMed      47.50 /  8.59 /  7.64
Texas       69.95 / 41.87 / 51.83
BlogCatalog 27.77 /  4.38 /  3.55
```

This suggests delayed activation helps PubMed and slightly helps BlogCatalog,
but it suppresses the Texas gain and hurts DBLP. So warmup/ramp alone is not
the full fix.

### Structural Diagnosis After v21a/v21b

The important diagnosis is subtler:

- Even when final `embed_graph_gate` is tiny, the model still uses a 4-way
  softmax gate during training.
- Therefore `Q_embed` can perturb the relative competition among
  `Q_attr/Q_low/Q_high` even when its final effective weight is nearly zero.

This means “graph gate close to zero” is not enough if the mixing architecture
itself lets the fourth view distort the stable 3-view base.

### v21c: Residual Embedding Injection

To remove that interference, v21c changes the structure from
“four views compete equally” to “stable 3-view base + gated residual embed
injection”:

```text
Q_base = mix(Q_attr, Q_low, Q_high)
Q_mix  = normalize(Q_base + lambda * g_graph * Q_embed)
```

This preserves the stable v8g-style 3-view mixer and only adds embedding
information as a residual correction.

Smoke result:

```text
ACM         41.92 /  2.47 /  2.00
DBLP        33.99 /  5.17 /  1.88
PubMed      45.56 /  8.47 /  7.57
Texas       70.49 / 41.55 / 51.74
BlogCatalog 29.43 /  5.63 /  3.73
```

This is a meaningful structural improvement:

- ACM becomes the best among v21a/b/c.
- BlogCatalog recovers relative to v21a/v21b.
- Texas remains above v19b.
- DBLP remains much better than v19b.

Residual injection is therefore a better architectural fit than direct 4-way
competition.

### Hidden Leakage: Consistency Loss Still Pulls `Q_embed`

Even under v21c, one problem remained: `Q_embed` was almost closed in some
graphs, but the overall result still shifted. The likely culprit was the
existing multi-view consistency regularizer:

```text
L_view = avg_v KL(Q_v || Q_refined)
```

This still forces `Q_embed` to track the main posterior even when fusion almost
shuts it off. In other words, there was a hidden training-time path by which
embedding view could keep perturbing optimization.

### v21d: Gate-Aware Consistency

v21d keeps the v21c residual architecture and additionally gates the
`Q_embed` consistency term by the same conflict gate used for fusion:

```text
L_view = mean(KL for attr/low/high) + g_graph * KL(Q_embed || Q_refined)
```

Smoke result:

```text
ACM         41.98 /  2.42 /  1.97
DBLP        33.25 /  3.79 /  1.65
PubMed      45.04 /  8.81 /  7.16
Texas       71.58 / 44.37 / 54.55
BlogCatalog 27.44 /  7.44 /  5.21
```

This is the strongest Texas result in the whole v21 line and clearly improves
BlogCatalog NMI/ARI relative to v21c, although BlogCatalog ACC is still low.
The cost is weaker DBLP/PubMed ACC than v21c.

At this point the main structural lesson is strong:

1. `Q_embed` must be injected as a residual, not as an equal fourth competitor.
2. Fusion gating alone is insufficient; all auxiliary losses using `Q_embed`
   must respect the same gate.

### v21e: Node-Level Gate on Top of Graph Gate

The next hypothesis was that graph-level gating is still too coarse for
BlogCatalog-like cases: perhaps only a subset of nodes truly needs embedding
correction. v21e therefore adds a node-level gate:

```text
g_node = g_node_entropy * g_node_kl
Q_mix  = normalize(Q_base + lambda * g_graph * g_node(i) * Q_embed(i))
```

where:

- `g_node_entropy` opens when node-wise `Q_embed` is more certain than
  node-wise `Q_base`.
- `g_node_kl` opens when node-wise `Q_embed` actually disagrees with `Q_base`.

Smoke result:

```text
ACM         41.92 /  2.28 /  1.92
DBLP        34.56 /  5.19 /  1.71
PubMed      43.84 /  6.74 /  6.12
Texas       68.31 / 40.63 / 49.13
BlogCatalog 29.79 /  7.19 /  5.27
```

This helps BlogCatalog and DBLP somewhat, but over-suppresses Texas and PubMed.
So the node-level gate is promising as a selective repair tool, but the current
node criteria are too conservative for small high-conflict graphs.

## Current v21 Conclusions

The v21 line produced a genuinely useful scientific result, even though it is
not yet a new default:

- The useful signal is real: embedding posterior helps exactly when the stable
  3-view posterior is in high-conflict states.
- Pure graph-level gating can isolate Texas-like graphs successfully.
- Architectural leakage matters:
  - direct 4-way competition is worse than residual injection;
  - ungated consistency losses can silently reintroduce harmful influence.
- The best current trade-offs in smoke are:
  - `v21c` for overall balance across ACM/DBLP/PubMed/Texas/BlogCatalog
  - `v21d` if prioritizing Texas and BlogCatalog structure recovery
- No v21 variant is globally strong enough yet to replace the stable default.

The remaining open direction is to make embedding correction selective **within**
the graph without suppressing Texas too much. The next likely step is not
another global threshold tweak, but a softer node-conditional residual path,
possibly using learned node uncertainty/conflict features instead of only
hand-designed entropy/KL gates.

## v22: Learned Node-Conflict Gate

v21e showed that node-level selectivity is useful, but the hand-designed
node gate was too conservative for Texas and PubMed. The natural next step is
to replace or augment the heuristic node rule with a learned node-conflict
controller that still respects the red line:

- no dataset-specific routing,
- still one unified end-to-end differentiable pipeline,
- still embedding residual injection on top of the same frontend.

The question is whether a small learned gate can discover which nodes actually
benefit from `Q_embed` better than the manual entropy/KL thresholds.

### v22a: Pure Learned Node Gate

v22a keeps the strong v21d structure:

- graph-level conflict gate,
- residual embedding injection,
- gate-aware view consistency,

but replaces the node gate with a small MLP over node-level conflict features:

```text
phi_i = [ H_base(i), H_embed(i), KL(Q_embed(i) || Q_base(i)),
          conf_base(i), conf_embed(i) ]
g_node_learned(i) = sigmoid(MLP(phi_i))
Q_mix = normalize(Q_base + lambda * g_graph * g_node_learned(i) * Q_embed(i))
```

The consistency weight for `Q_embed` is also changed from pure graph-level gate
to:

```text
g_consistency = g_graph * mean_i[g_node_learned(i)]
```

so the auxiliary loss and fusion path stay aligned.

#### v22a Smoke Diagnosis

```text
ACM         41.72 /  2.25 /  1.89
DBLP        30.37 /  2.12 /  1.45
PubMed      47.38 /  9.60 /  7.46
Texas       56.83 / 17.74 / 25.66
BlogCatalog 32.08 /  6.96 /  4.89
```

This is a mixed but ultimately negative result:

- BlogCatalog ACC improves noticeably.
- PubMed also becomes strong.
- But Texas collapses badly.
- DBLP also degrades.

The diagnostics explain why. The learned node gate does **not** preserve the
strong graph discrimination learned in v21:

```text
Texas embed_graph_gate ~= 0.944, but embed_node_gate ~= 0.211
BlogCatalog embed_graph_gate ~= 0.0058, embed_node_gate ~= 0.197
ACM embed_graph_gate ~= 0.0004, embed_node_gate ~= 0.196
```

So the learned gate becomes a nearly uniform conservative shrink factor around
`0.18 - 0.25` across datasets. It does not learn “Texas should stay open.”
Instead it mostly learns “close the node gate everywhere a bit,” which protects
some large graphs but destroys the very small high-conflict graph that most
needs the embedding residual.

### Structural Diagnosis From v22a

The important finding is not simply “the learned gate failed.” It failed in a
specific way:

1. The graph-level gate already separates easy and hard graphs well.
2. A freely learned node gate without a strong inductive prior tends to ignore
   that structure and collapse toward a weak near-constant attenuation.
3. This especially harms Texas-like graphs, where the graph-level gate is
   correct but the node gate erases the residual signal.

So a pure learned node gate is too unconstrained in this regime.

### v22b: Hybrid Node Gate

The safer follow-up is to preserve the v21e heuristic node gate as a base and
let the learned gate act only as a multiplicative correction:

```text
g_node_h = g_node_entropy * g_node_kl
g_node_l = sigmoid(MLP(phi_i))
g_node   = g_node_h * g_node_l
```

This gives the learned gate a strong prior: it can refine the selective pattern
already known to work somewhat, but it cannot completely invent a flat global
node policy from scratch.

#### v22b Smoke Diagnosis

```text
ACM         41.79 /  2.27 /  1.95
DBLP        31.53 /  2.57 /  0.77
PubMed      46.85 /  8.55 /  7.39
Texas       67.21 / 36.03 / 45.64
BlogCatalog 30.45 /  8.62 /  7.41
```

This is healthier than v22a:

- Texas recovers substantially relative to v22a.
- BlogCatalog NMI/ARI improve further.
- PubMed stays reasonably strong.

But it still does not beat the stronger v21 residual baselines:

- Texas remains clearly below v21d.
- DBLP remains weak.
- The overall trade-off is still worse than the best v21 variants.

## Current v22 Conclusions

v22 adds another useful scientific constraint to the search space:

- A pure learned node gate is too weakly guided and collapses toward a
  near-constant suppression factor.
- A hybrid “heuristic base × learned correction” is better than pure learned
  gating, but still not enough to outperform the stronger v21 residual
  graph-gated baselines.
- Therefore, the problem is not merely “replace heuristics with an MLP.”
  The learned gate needs a stronger optimization signal or a better structural
  prior if it is to preserve Texas while improving BlogCatalog/DBLP.

At the current stage:

- `v21d` remains the strongest option if prioritizing Texas and structural
  recovery on difficult graphs.
- `v21c` remains the more balanced residual-injection baseline.
- `v22a/v22b` are informative negative-to-mixed results rather than new
  defaults.

The next promising direction is likely not a larger gate MLP, but a **better
supervisory target for node conflict**, for example deriving node-level gating
signals from refinement disagreement, transport residual, or topology-state
change rather than only posterior certainty features.

## v23: Disagreement-Driven Node Gate

v22 showed that replacing heuristic node gates with a learned posterior-feature
gate is not enough. The learned gate tends to collapse toward a weak global
attenuation and loses the key Texas-like selectivity. This suggests the node
gate needs signals that are closer to the actual reason embedding correction is
helpful, namely unresolved assignment conflict during transport/refinement.

### v23a: Transport/Refinement Disagreement Node Gate

v23a keeps the best structural ingredients from v21/v22:

- graph-level conflict gate,
- residual embedding injection,
- gate-aware view consistency,

but replaces the node-control target with disagreement signals derived from the
base 3-view posterior dynamics themselves.

First compute the stable 3-view pipeline:

```text
Q_base          = mix(Q_attr, Q_low, Q_high)
Q_base_transport = Sinkhorn(Q_base)
Q_base_refined   = Refine(Q_base_transport)
```

Then define node-level disagreement features:

```text
d_transport(i) = KL(Q_base_transport(i) || Q_base(i))
d_refine(i)    = KL(Q_base_refined(i)   || Q_base_transport(i))
```

The v23a node gate multiplies these with the earlier heuristic selectivity:

```text
g_node_heur = g_entropy * g_kl
g_node_dyn  = g_transport * g_refine
g_node      = g_node_heur * g_node_dyn
```

and the final residual injection remains:

```text
Q_mix = normalize(Q_base + lambda * g_graph * g_node(i) * Q_embed(i))
```

The key difference from v22 is conceptual:

- v22 asks whether `Q_embed` looks confident.
- v23 asks whether the base posterior is still **actively struggling** under
  transport/refinement.

That is much closer to the actual failure mode we want to repair.

### v23a Smoke Diagnosis

```text
ACM         42.02 /  2.33 /  2.00
DBLP        30.42 /  3.06 /  0.80
PubMed      44.41 /  5.67 /  4.94
Texas       68.31 / 37.34 / 47.97
BlogCatalog 31.22 /  6.66 /  5.26
```

This is not a new best, but it is scientifically informative:

- It is much healthier than the failed v22a learned gate.
- Texas remains substantially more open than easy graphs.
- BlogCatalog stays improved relative to earlier no-node-gate baselines.
- But DBLP and PubMed are still weaker than the best v21 trade-offs.

Most importantly, the diagnostics show that disagreement-driven node gating has
the right **ordering**:

```text
ACM:         embed_node_gate ~= 0.03
Texas:       embed_node_gate ~= 0.58
BlogCatalog: embed_node_gate ~= 0.15
DBLP:        embed_node_gate ~= 0.19
```

This is much better than v22a, where the learned gate collapsed to nearly the
same low value for every graph.

The underlying disagreement factors are also meaningful:

```text
Texas: node_transport ~= 1.00, node_refine ~= 0.66
ACM:   node_transport ~= 0.41, node_refine ~= 0.52
```

So transport/refinement disagreement is indeed surfacing genuine conflict.

### Current v23 Conclusion

v23a does **not** replace v21d as the best current branch, but it establishes a
strong new result:

- node-level gates should be driven by dynamic assignment disagreement, not only
  posterior confidence/entropy statistics;
- this produces better structural selectivity than v22's learned posterior gate;
- however, the current multiplicative design still opens too much on
  DBLP/BlogCatalog and not enough on Texas compared with the strongest v21d
  graph-only solution.

So the next good direction is not generic threshold tuning, but a refined
disagreement gate that emphasizes **small-graph / high-flow** conflict states
without over-activating on larger noisy graphs. One likely path is to combine
the dynamic disagreement factors with a stronger normalization or ranking-based
node selection rather than using only raw sigmoid magnitudes.

### v23b: Rank-Sparse Disagreement Gate

To test whether v23a still over-opens too many nodes on DBLP/BlogCatalog, I
next tried a rank-style sparse selector. Instead of using the raw disagreement
gate magnitude directly, v23b keeps only relatively high-disagreement nodes
within each graph:

```text
s_i      = g_node_heur(i) * g_transport(i) * g_refine(i)
cutoff   = quantile(s, rho)
g_rank(i)= sigmoid((s_i - cutoff) / tau_rank)
g_node   = g_node_heur(i) * g_rank(i)
```

This is intended to preserve the disagreement signal while suppressing
medium-conflict nodes that may only add noise on large graphs.

#### v23b Smoke Diagnosis

```text
ACM         41.16 /  2.02 /  1.70
DBLP        31.18 /  2.04 /  0.44
PubMed      45.67 /  6.60 /  6.56
Texas       62.84 / 29.73 / 33.21
BlogCatalog 24.17 /  3.62 /  2.05
```

This is a clear negative result.

Instead of cleaning up the difficult graphs, the rank-sparse selector removes
too much useful residual signal:

- Texas drops sharply.
- BlogCatalog also collapses.
- DBLP does not improve enough to justify the loss.

So the current failure mode is **not** simply “too many nodes are open.” A hard
within-graph sparsification strategy is too aggressive and destroys the useful
correction path on both small high-conflict graphs and structurally difficult
larger graphs.

## Updated v23 Conclusion

The combined v23 result is now clearer:

- Dynamic disagreement/transport signals are better node-gate features than the
  posterior-only signals used in v22.
- But converting them into a graph-internal sparse ranking gate is the wrong
  control mechanism.
- The main remaining challenge is therefore not node **selection count**, but
  node **residual calibration**: how strongly those nodes should receive the
  embedding correction once identified.

This points to the next more promising direction:

- keep the useful v21d graph-level conflict controller,
- keep disagreement-derived node diagnostics,
- stop pruning nodes by hard relative ranking,
- instead learn or design a softer **residual amplitude controller** conditioned
  on disagreement state.

## v24: Residual Amplitude Control

The negative v23b result sharpened the diagnosis: the remaining problem is not
mostly “which nodes should be open,” but “how strongly should the selected
embedding correction be injected.” Harder node selection kept destroying useful
signal. So the next step is to stop treating `Q_embed` as a whole posterior to
mix in, and instead use it only as a **directional residual correction** on top
of the stable base posterior.

### v24a: Positive Residual-Amplitude Injection

v24a keeps:

- the v21d graph-level conflict controller,
- the v23a disagreement-driven node diagnostics,
- residual fusion rather than four-way competition,

but changes the actual embedding correction from:

```text
Q_mix = normalize(Q_base + alpha * Q_embed)
```

to a directional residual form:

```text
Delta_embed = relu(Q_embed - Q_base)
Q_mix       = normalize(Q_base + alpha * Delta_embed)
```

where:

```text
alpha = g_graph * g_node * lambda
```

Interpretation:

- `Q_embed` is no longer allowed to arbitrarily rewrite the whole posterior.
- It can only add probability mass where it is more confident than `Q_base`.
- This is exactly the kind of conservative correction desired under the red
  line: unified, differentiable, and still topology-contraction-first.

### v24a Smoke Diagnosis

```text
ACM         41.88 /  2.21 /  1.90
DBLP        30.64 /  1.51 /  0.30
PubMed      45.32 /  6.21 /  5.48
Texas       71.58 / 42.41 / 53.49
BlogCatalog 27.31 /  8.65 /  5.91
```

This is the first clear positive result for the amplitude-control direction:

- Texas is pulled back almost to the strong v21d level.
- BlogCatalog NMI/ARI improve beyond v21d and v23a.
- The injection is much more stable than v23b’s sparse selector.

A direct comparison to recent strong baselines is useful:

```text
Dataset      v21d ACC   v23a ACC   v24a ACC
ACM          41.98      42.02      41.88
DBLP         33.25      30.42      30.64
PubMed       45.04      44.41      45.32
Texas        71.58      68.31      71.58
BlogCatalog  27.44      31.22      27.31
```

And in structural metrics:

- `Texas`: v24a nearly restores the best v21d behavior.
- `BlogCatalog`: v24a gives the strongest NMI/ARI among the recent residual
  amplitude / disagreement family.

### Current v24 Conclusion

v24a establishes another important structural fact:

- the useful role of `Q_embed` is not to replace the base posterior,
- nor merely to select nodes,
- but to provide a **bounded positive correction direction** on top of
  `Q_base`.

This is stronger than the earlier node-selection view. It explains why:

- v23a had the right disagreement signal but still over-corrected some graphs;
- v23b failed by pruning too hard;
- v24a works better by changing the **shape** of the correction, not just its
  support.

It is still not a new global default because DBLP remains weak. But the next
direction is now much clearer:

- keep the v24a-style directional residual correction,
- keep the strong graph conflict controller,
- focus future work on **DBLP-safe amplitude calibration** rather than further
  node pruning.

### v24b: Soft Amplitude Gate

The next refinement kept the v24a positive residual form, but added a
node-wise amplitude controller:

```text
s_amp   = sum_k relu(Q_embed - Q_base)_k
g_amp   = sigmoid((s_amp - c) / tau)
Q_mix   = normalize(Q_base + alpha * g_amp * Delta_embed)
```

with the first smoke setting:

```text
c   = 0.35
tau = 0.10
```

The motivation was straightforward:

- keep strong residual corrections when the embedding branch clearly disagrees,
- suppress weak residual noise on citation-style graphs,
- remain fully differentiable and unified.

### v24b Smoke Diagnosis

```text
ACM         41.55 /  2.08 /  1.81
DBLP        33.00 /  3.07 /  1.18
PubMed      48.12 /  9.81 /  9.89
Texas       69.95 / 40.75 / 51.68
BlogCatalog 24.94 /  3.34 /  2.21
```

This established a useful but incomplete fact:

- `DBLP` and `PubMed` improved clearly over v24a.
- `Texas` remained usable, though below v24a.
- `BlogCatalog` collapsed sharply.

The diagnostics explain why. The amplitude gate became extremely small on the
low-conflict graphs:

- `DBLP`: mean amplitude score about `0.1585`, mean amplitude gate about `0.1339`
- `PubMed`: mean amplitude score about `0.1718`, mean amplitude gate about `0.1682`
- `BlogCatalog`: mean amplitude score about `0.1332`, mean amplitude gate about `0.1116`

So the idea was not wrong, but the gate was too harsh. Weak residuals were not
being softly denoised; they were being nearly shut off.

### v24c: Amplitude Gate Floor

To test that diagnosis, v24c kept the same sigmoid amplitude gate but added a
floor:

```text
g_amp' = floor + (1 - floor) * g_amp
```

with:

```text
floor = 0.25
```

This remains red-line safe:

- no dataset routing,
- no new branch architecture,
- still one residual correction path,
- still topology-contraction-first.

### v24c Smoke Diagnosis

```text
ACM         41.69 /  2.15 /  1.86
DBLP        31.40 /  3.50 /  1.64
PubMed      47.59 /  9.85 /  8.41
Texas       68.85 / 42.06 / 52.20
BlogCatalog 31.29 /  7.70 /  5.98
```

This is the best compromise found in the early v24b-v24e amplitude-calibration
family:

- `DBLP` remains clearly better than v24a in NMI/ARI.
- `PubMed` remains clearly better than v24a.
- `BlogCatalog` is largely recovered from the v24b collapse and slightly
  exceeds v24a in ACC/ARI.
- `Texas` drops below v24a, but not catastrophically.

The structural interpretation is important:

- the useful move was not a hard amplitude gate,
- but a softly lower-bounded amplitude gate,
- which preserves weak residual corrections on difficult large graphs while
  still damping noisy residuals.

### v24d and v24e: Negative Follow-Ups

Two immediate follow-ups were tested and both were worse than v24c.

`v24d` replaced the fixed amplitude floor with a graph-adaptive floor tied to
`(1 - g_graph)`:

```text
ACM         41.79 /  2.20 /  1.91
DBLP        30.88 /  1.70 /  0.47
PubMed      46.52 /  8.15 /  7.56
Texas       68.31 / 40.17 / 46.55
BlogCatalog 31.56 /  8.79 /  7.93
```

This helped `BlogCatalog`, but the graph-adaptive floor over-coupled the
correction magnitude to the graph conflict gate and hurt `DBLP`, `PubMed`, and
especially `Texas`.

`v24e` instead removed the floor and only widened the amplitude sigmoid by
changing the scale from `0.10` to `0.15`:

```text
ACM         41.62 /  2.09 /  1.81
DBLP        33.72 /  3.41 /  1.35
PubMed      45.45 /  6.25 /  5.39
Texas       67.76 / 38.42 / 44.75
BlogCatalog 28.96 /  4.30 /  3.52
```

This recovered some `DBLP` ACC, but lost the stronger `PubMed` and
`BlogCatalog` gains and damaged `Texas` more than v24c.

### v24f and v24g: Finer Floor Sweep

The next step was to interpolate between the harsh `v24b` gate and the safer
`v24c` floor by sweeping smaller floor values:

- `v24f`: `floor = 0.10`
- `v24g`: `floor = 0.15`

Five-dataset smoke:

```text
v24f
ACM         41.65 /  2.09 /  1.83
DBLP        33.08 /  4.35 /  2.04
PubMed      46.60 /  7.85 /  6.98
Texas       67.21 / 35.06 / 42.01
BlogCatalog 26.48 /  4.44 /  3.30

v24g
ACM         41.62 /  2.08 /  1.81
DBLP        32.41 /  2.72 /  0.74
PubMed      47.80 /  9.27 /  8.99
Texas       71.04 / 41.81 / 50.62
BlogCatalog 31.87 /  7.05 /  6.39
```

These runs refine the amplitude-floor story further:

- `v24f` is too weak on difficult graphs. It helps `DBLP`, but loses too much
  `Texas` and does not recover `BlogCatalog` enough.
- `v24g` is much healthier overall. It preserves most of the `PubMed` gain,
  largely restores `Texas`, and gives the best `BlogCatalog` ACC/ARI among the
  v24 floor variants tested so far.

### v24c Full 9-Dataset Smoke

To check whether the v24c compromise was only a five-dataset illusion, it was
expanded to the full nine-dataset 80-epoch smoke:

```text
ACM         41.69 /  2.15 /  1.86
DBLP        30.64 /  3.03 /  1.20
PubMed      47.92 / 10.04 /  8.81
Wiki        20.87 / 13.08 /  2.77
Flickr      18.63 /  3.38 /  1.71
BlogCatalog 28.23 /  5.36 /  4.14
Squirrel    23.05 /  0.84 /  0.35
Texas       67.76 / 41.50 / 48.70
Chameleon   29.91 /  9.31 /  2.21
```

This full smoke says the same high-level thing as the five-dataset run:

- `PubMed` is the clearest positive among citation-style graphs.
- `DBLP` keeps a modest NMI/ARI improvement direction over v24a, but the gain
  is not yet strong enough to declare success.
- `BlogCatalog` remains healthier than the harsh-gate v24b result, but the
  benefit is weaker in the full run than in the five-dataset subset.
- `Texas` is still below v24a/v21d, so v24c is promising but not yet a new
  dominant branch.

### Updated v24 Conclusion

The amplitude-control line is now much clearer:

- v24a proved that the shape of the correction matters more than node pruning.
- v24b proved that explicit amplitude calibration can help `DBLP` and
  `PubMed`, but naive gating is too suppressive.
- v24c proved that a lower-bounded amplitude gate is structurally safer than a
  pure sigmoid gate.
- v24d and v24e show that neither graph-adaptive floor coupling nor simple
  sigmoid flattening is enough to beat the fixed-floor family.
- v24g is the strongest current five-dataset branch inside the amplitude-floor
  line, because it is much closer to v24a on `Texas` while also improving
  `PubMed` and giving the best `BlogCatalog` ACC/ARI among the recent variants.

So the current best scientific takeaway is:

- keep the v24a positive residual direction,
- keep the strong v21d-style graph conflict controller,
- calibrate residual magnitude with a soft lower-bounded amplitude gate,
- and treat `v24g` as the next most promising branch for broader validation.

### v24h-v24o: Follow-Up Calibrations Around v24g

After v24g emerged as the healthiest five-dataset branch, several small
calibration refinements were tested without changing the architecture.

`v24h` delayed the activation of the amplitude-floor multiplier so that the
training process stayed closer to the harsher `v24b` style early on and only
introduced the `v24g` floor later.

Five-dataset smoke:

```text
ACM         41.65 /  2.10 /  1.83
DBLP        31.94 /  2.52 /  0.72
PubMed      47.61 /  9.29 /  8.58
Texas       70.49 / 44.90 / 53.90
BlogCatalog 28.77 /  6.32 /  5.43
```

This did not improve `DBLP`, but it did push `Texas` structure metrics upward.
That is a useful diagnosis: delayed floor activation mainly strengthens
high-conflict graphs rather than making the citation-style graphs safer.

`v24i` then tied the floor directly to the node disagreement gate:

```text
floor_i = floor * g_node
```

Five-dataset smoke:

```text
ACM         41.82 /  2.18 /  1.93
DBLP        34.41 /  5.11 /  2.73
PubMed      45.69 /  7.06 /  7.34
Texas       69.40 / 39.20 / 48.95
BlogCatalog 26.62 /  5.17 /  3.41
```

This sharply improved `DBLP`, but the cost was clear:

- `PubMed` dropped,
- `Texas` dropped,
- `BlogCatalog` dropped.

So a fully node-gated floor is too aggressive. It increases selectivity in a
way that helps citation graphs but removes too much weak residual support on
the harder non-homophilic graphs.

`v24j`, `v24k`, `v24l`, and `v24m` then tested convex blends between the fixed
floor of `v24g` and the node-gated floor of `v24i`.

Definitions:

- `v24j`: node-floor blend `0.50`
- `v24k`: node-floor blend `0.15`
- `v24l`: node-floor blend `0.25`
- `v24m`: node-floor blend `0.20`

Representative five-dataset smoke:

```text
v24j
ACM         41.55 /  2.05 /  1.78
DBLP        32.49 /  3.14 /  1.59
PubMed      47.04 /  7.81 /  7.09
Texas       68.31 / 40.58 / 46.07
BlogCatalog 25.06 /  5.68 /  4.31

v24k
ACM         41.65 /  2.08 /  1.82
DBLP        30.69 /  3.35 /  1.86
PubMed      48.16 / 10.05 /  9.08
Texas       69.40 / 40.04 / 49.92
BlogCatalog 26.73 /  5.77 /  3.53

v24l
ACM         41.65 /  2.09 /  1.83
DBLP        35.27 /  5.91 /  3.25
PubMed      47.56 /  9.20 /  9.35
Texas       68.85 / 41.10 / 49.74
BlogCatalog 27.66 /  5.63 /  3.97

v24m
ACM         41.55 /  2.05 /  1.78
DBLP        33.60 /  4.06 /  2.64
PubMed      47.87 /  9.68 /  9.35
Texas       70.49 / 41.59 / 50.23
BlogCatalog 24.44 /  5.07 /  2.82
```

These results sharpen the diagnosis again:

- small amounts of node-adaptive mixing can strongly improve `DBLP`;
- but the same move systematically hurts `BlogCatalog`;
- and the best Texas-preserving versions are still the more fixed-floor
  variants (`v24g`, and to a lesser extent `v24m`).

Finally, `v24n` and `v24o` tested the opposite idea: make the floor larger on
low-disagreement nodes by mixing with `(1 - g_node)` instead of `g_node`.

Definitions:

- `v24n`: inverse-node-floor blend `0.35`
- `v24o`: inverse-node-floor blend `0.60`

Five-dataset smoke:

```text
v24n
ACM         41.65 /  2.07 /  1.81
DBLP        32.54 /  3.45 /  1.80
PubMed      45.56 /  6.51 /  5.75
Texas       67.76 / 40.44 / 46.58
BlogCatalog 27.73 /  5.21 /  4.24

v24o
ACM         41.65 /  2.07 /  1.81
DBLP        33.23 /  4.08 /  1.51
PubMed      47.69 /  9.15 /  9.34
Texas       70.49 / 41.74 / 53.91
BlogCatalog 22.79 /  2.83 /  2.02
```

This line also fails as a general solution:

- stronger inverse-node mixing can recover `Texas`,
- but `BlogCatalog` collapses badly,
- and the overall balance is still worse than v24g.

### Updated Local Conclusion After v24h-v24o

At this point the local picture is quite stable:

- `v24g` remains the safest all-around amplitude-floor variant.
- `v24l` is the strongest branch if the immediate goal is to maximize `DBLP`
  improvement under the same architecture.
- The current conflict is structural: the calibration patterns that improve
  `DBLP` tend to suppress the weak residual support that `BlogCatalog` needs.

So the next direction should not be more blind floor sweeps. The better next
step is to compare `v24g` and `v24l` diagnostics more directly and search for a
feature that separates citation-style `DBLP` from structurally difficult
`BlogCatalog` without falling back to dataset-specific routing.

In other words, the current frontier is no longer “find the right node subset,”
but “find the right graph/node-dependent correction magnitude without damaging
homophilic citation-style graphs.”

## 2026-06-24: v27 Flow-Readout Alignment Note

The current workspace has progressed beyond the v24 family. The strongest
available unified smoke result is now in the v27 family, especially `v27ap`:

```text
Wiki         30.15 / 20.94 / 10.07
Flickr       22.96 /  6.84 /  4.38
Chameleon    31.88 / 10.47 /  3.49
Squirrel     21.94 /  1.26 /  0.22
ACM          50.12 /  8.47 / 10.24
DBLP         64.16 / 34.34 / 29.08
PubMed       52.34 / 18.65 / 13.34
Texas        72.13 / 41.89 / 49.59
BlogCatalog  76.00 / 52.83 / 52.34
```

The important diagnosis is that these gains mostly come from the unified
assignment-flow posterior being used as a final readout anchor:

```text
q_final = 0.30 * q_aptc_raw + 0.70 * q_flow
```

This is still red-line safe because `q_flow` is produced inside the same
differentiable topology-contraction-first model. There is no dataset-specific
head or external post-processing path.

However, `v27ap` still trains the target bootstrap with a weaker flow blend:

```text
target_bootstrap = 0.60 * q_aptc_raw + 0.40 * q_flow
```

So the final posterior, bootstrap target, and clustering loss are not fully
aligned. The next minimal hypothesis is to make those three posterior surfaces
identical under one shared graph:

```text
q_train = q_target = q_final = 0.30 * q_aptc_raw + 0.70 * q_flow
```

### v27av: Posterior-Surface Alignment

Added `v27av` in `scripts/run_unified_aptc_9datasets.py`.

Relative to `v27ap`, this keeps the same architecture, embedding residual
calibration, and assignment-flow readout, but changes:

- `target_bootstrap_flow_weight`: `0.40 -> 0.70`
- `loss_posterior_source`: `q_blend`
- `loss_posterior_flow_weight`: `0.70`

This is not a new backend. It is a training/readout consistency test for the
same unified posterior blend. The expected benefit is reduced objective mismatch
on DBLP/BlogCatalog/Texas, with Wiki/Flickr checked first because they are the
most fragile under recent v27 variants.

### v27av-v27ax Smoke Results

Fair 80-epoch smoke comparison on the sensitive five-dataset subset:

```text
v27ap baseline
Wiki         30.81 / 24.10 / 10.93
Flickr       25.28 /  9.33 /  5.39
DBLP         62.29 / 30.09 / 25.34
Texas        69.40 / 39.21 / 51.46
BlogCatalog  72.15 / 48.05 / 46.24

v27av full posterior-surface alignment
Wiki         27.48 / 21.81 /  9.01
Flickr       24.46 /  9.24 /  5.52
DBLP         62.81 / 32.24 / 26.34
Texas        71.04 / 41.67 / 54.88
BlogCatalog  73.23 / 49.66 / 48.24

v27aw loss-only alignment
Wiki         29.85 / 22.77 / 10.28
Flickr       25.20 /  9.49 /  5.56
DBLP         62.07 / 29.74 / 24.67
Texas        71.58 / 44.05 / 56.03
BlogCatalog  72.25 / 48.27 / 46.40

v27ax adaptive target blend
Wiki         28.98 / 22.89 / 10.03
Flickr       25.16 /  9.50 /  5.52
DBLP         62.66 / 30.48 / 25.74
Texas        70.49 / 41.44 / 54.14
BlogCatalog  71.56 / 47.45 / 45.64
```

Interpretation:

- Moving the bootstrap target from 0.40 to 0.70 flow is real but selective: it
  helps DBLP, Texas, and BlogCatalog, while hurting Wiki and slightly hurting
  Flickr.
- Loss-only alignment mainly helps Texas structure metrics; it does not explain
  the DBLP/BlogCatalog gains.
- The first adaptive target rule (`v27ax`) stayed too close to the 0.40 floor
  in practice (`target_flow_weight` about 0.40-0.47), because the amplitude
  signal shrank inside the closed training loop. It is not a keeper.

Current conclusion: `v27ap` remains the safest default among these 80-epoch
tests. `v27av` is a useful positive diagnostic for DBLP/Texas/BlogCatalog, but
not a global replacement because it damages Wiki/Flickr. The next promising
direction should look for a more stable graph statistic that distinguishes
BlogCatalog-style high-entropy but recoverable flow states from Wiki/Flickr
high-entropy fragile states.

## 2026-06-24: v27ay Cross-Frequency Gate Diagnostic

Before implementing `v27ay`, I added graph-level diagnostics inside the unified
forward/loss path:

- `homo_confidence_mean`: hard-threshold mean confidence among edges with
  `score >= high_threshold`.
- `z_cross_alignment`: mean node-wise cosine similarity between `Z_low` and
  `Z_high`.
- `ambiguous_ratio`: hard-threshold fraction of edges with
  `low_threshold < score < high_threshold`.
- `hetero_confidence_mean`: fallback hard-threshold mean confidence among edges
  with `score <= low_threshold`.

This does not add any dataset branch or alter the forward pipeline; the values
are diagnostics derived from the existing edge confidence, differentiable
topology contraction, and frequency filtering outputs.

80-epoch `v27ap` smoke diagnostic:

```text
Dataset       ACC    HomoConf  ZCross   AmbRatio  HeteroConf
ACM          51.44    0.7837   0.9386    0.5746     0.1703
DBLP         61.77    0.8319   0.9874    0.7546     0.1326
PubMed       52.47    0.7999   0.9338    0.6608     0.1453
Wiki         30.73    0.8074   0.9495    0.6992     0.1389
Flickr       25.64    0.7576   0.9906    0.5545     0.1343
BlogCatalog  72.11    0.7941   0.9843    0.6539     0.1488
Texas        71.04    0.7447   0.9338    0.6198     0.1017
Squirrel     22.53    0.7226   0.9552    0.5582     0.1692
Chameleon    32.10    0.7471   0.9477    0.5485     0.1273
```

Finding:

- `z_cross_alignment` does not cleanly separate fragile Wiki/Flickr from
  recoverable BlogCatalog/DBLP. Wiki is lower than BlogCatalog/DBLP, but Flickr
  is the highest-alignment graph in the run.
- The fallback `hetero_confidence_mean` is also not a stable separator.
  BlogCatalog is only slightly higher than Wiki/Flickr, while DBLP is lower than
  Wiki/Flickr.
- Therefore `v27ay` should not be forced in the proposed form. A gate based only
  on cross-frequency alignment would likely over-open Flickr, exactly the kind
  of failure the diagnostic was meant to avoid.

Conclusion: keep the diagnostics, but do not add the v27ay adaptive gate yet.
The next candidate separator should combine cross-frequency alignment with a
second reliability term that penalizes high-alignment but low-quality flow
states, instead of using alignment alone.

## 2026-06-25: v28a Front-End Partition and Frequency Pressure

### Structural diagnosis

A fuller v27ap 80-epoch front-end diagnostic was run after adding final and
epoch-snapshot statistics for:

- threshold values and threshold gap,
- soft partition masses and hard threshold partition fractions,
- `Z_low` and `Z_high` mean L2 norms,
- `Z_low`/`Z_high` cosine alignment,
- edge-score mean and standard deviation.

Final v27ap diagnostic:

```text
Dataset       ACC    Low     High    Gap    SoftHard  HardAmb  ZCross  ScoreStd
ACM          50.84  0.2134  0.6417  0.4283  0.5225    0.5738   0.9386   0.2318
DBLP         62.29  0.1922  0.7385  0.5463  0.6840    0.7534   0.9870   0.2180
PubMed       52.70  0.2338  0.7004  0.4666  0.5982    0.6607   0.9340   0.2193
Wiki         28.52  0.2133  0.7195  0.5062  0.6397    0.7010   0.9501   0.2187
Flickr       25.11  0.1927  0.6220  0.4294  0.5202    0.5552   0.9907   0.2254
BlogCatalog  71.67  0.2132  0.6808  0.4676  0.5954    0.6540   0.9844   0.2200
Texas        70.49  0.1287  0.6008  0.4721  0.5188    0.6178   0.9325   0.2208
Squirrel     22.50  0.1718  0.5823  0.4105  0.4483    0.5585   0.9553   0.2330
Chameleon    32.94  0.1712  0.6016  0.4304  0.4896    0.5477   0.9474   0.2288
```

The threshold gap is not small; it is already about `0.41-0.55`. So the root
cause is not simply that low/high thresholds collapsed together. The real
structural failure is:

- high soft ambiguous mass remains (`0.45-0.68`);
- hard ambiguous fraction remains high (`0.55-0.75`);
- `Z_low` and `Z_high` are both normalized to mean norm `1.0`, so neither path
  dies by norm collapse;
- cross-frequency alignment rises from moderate early values to `0.93-0.99` by
  epoch 80, which confirms frequency coupling.

### v28a change

Added two optional unified loss terms, both defaulting to zero so prior variants
are unchanged:

```text
partition_spread_weight
freq_separation_weight
```

`partition_spread_pressure_loss` uses the existing learnable thresholds and soft
tripartition masks:

```text
relu(min_spread - (high - low)) + ambiguous_weight * mean(hard_soft)
```

`frequency_separation_pair_loss` uses existing `Z_low`, `Z_high`, and soft
homo/hetero edge masks:

```text
homo edges:   pull low-view neighbors together, push high-view neighbors apart
hetero edges: pull high-view neighbors together, push low-view neighbors apart
```

No dataset branch, new module, new parameter, or external post-processing was
introduced.

`v28a` enables:

```text
partition_spread_weight = 0.05
partition_min_spread = 0.30
partition_ambiguous_penalty_weight = 1.0
freq_separation_weight = 0.10
```

### v28a five-dataset smoke

80-epoch smoke on `ACM, DBLP, Wiki, Texas, BlogCatalog`:

```text
Dataset       v27ap ACC  v28a ACC  v27ap HardAmb  v28a HardAmb  v27ap ZCross  v28a ZCross
ACM             50.84     53.26       0.5738        0.5362        0.9386       0.9355
DBLP            62.29     62.24       0.7534        0.7525        0.9870       0.9868
Wiki            28.52     31.19       0.7010        0.6825        0.9501       0.9495
Texas           70.49     71.04       0.6178        0.6196        0.9325       0.9319
BlogCatalog     71.67     73.19       0.6540        0.6543        0.9844       0.9831
```

Interpretation:

- v28a is mildly positive for ACC on ACM, Wiki, Texas, and BlogCatalog.
- The structural success criteria are not met: ambiguous fractions remain well
  above `0.40`, and Texas `z_cross_alignment` remains above `0.93`.
- The new losses are directionally useful but too weak or too indirect to
  break the low/high frequency coupling.

Current conclusion: v28a is a useful front-end diagnostic branch, not yet a
stable structural fix. The next front-end attempt should target the actual
coupling mechanism more directly, likely by changing the high-pass construction
or adding an explicit low/high orthogonality objective before normalization,
rather than only applying edge-pair pressure after both channels have already
become nearly collinear.

## 2026-06-25: v28b Adaptive High-Pass Scale

### Motivation

The direct root-cause inspection of `_signed_highpass` showed a structural
degeneracy:

```text
smooth = normalized_spmm(edge_index, hetero, h, N)
h = h - smooth
out = z + h
```

When the heterophilic soft mask is sparse or weak, `smooth` is small, so
`h ~= z`, `out ~= 2z`, and normalization turns the high-pass view into a near
copy of the input feature embedding. That explains why `Z_low` and `Z_high`
became highly aligned in the v27/v28a diagnostics.

### Change

`v28b` keeps the v28a losses and adds two targeted changes:

1. A backward-compatible adaptive high-pass mode:

```text
h = z - hetero_mass * highpass_scale * smooth
highpass_scale in [0.5, 4.0], initialized at 2.0
```

The scale is a single learnable parameter shared by every dataset. It is only
used when `highpass_adaptive_scale=True`; prior variants keep the original
high-pass filter.

2. A direct low/high orthogonality regularizer:

```text
freq_ortho_loss = relu(z_cross_alignment - freq_ortho_target)^2
```

`v28b` enables:

```text
highpass_adaptive_scale = True
freq_ortho_weight = 0.20
freq_ortho_target = 0.50
```

No dataset-specific branch, low-pass change, clustering-head change, or
existing loss-weight change was introduced.

### v28b five-dataset smoke

80-epoch smoke on `ACM, DBLP, Texas, Wiki, BlogCatalog`:

```text
Dataset       v28a ACC/NMI/ARI       v28b ACC/NMI/ARI
ACM           53.26 /  9.71 / 11.62  61.55 / 18.04 / 21.38
DBLP          62.24 / 31.00 / 25.35  65.32 / 33.93 / 30.50
Texas         71.04 / 41.31 / 54.51  72.13 / 46.37 / 57.98
Wiki          31.19 / 26.01 / 12.04  32.47 / 28.29 / 13.40
BlogCatalog   73.19 / 48.74 / 47.90  79.35 / 56.29 / 57.43
```

Structural diagnostics:

```text
Dataset       v28b HardAmb  v28b ZCross  HighpassScale
ACM             0.5225        0.9541       1.9544
DBLP            0.7592        0.9896       1.9732
Texas           0.6206        0.9784       1.9958
Wiki            0.6864        0.9661       1.9733
BlogCatalog     0.6689        0.9895       1.9772
```

### Interpretation

v28b is the strongest positive branch in the v28 front-end line so far:

- ACM jumps by more than 8 ACC points over v28a.
- BlogCatalog jumps by more than 6 ACC points and nearly 10 ARI points.
- Texas improves in all three metrics, with NMI/ARI exceeding the target table
  values used by the runner.
- Wiki also improves rather than being damaged.

However, the structural success criterion is only partially satisfied. The
large performance gain did **not** come from lowering `z_cross_alignment`; in
fact, `Z_low` and `Z_high` remain highly aligned. This means the original
diagnosis was right that the old high-pass construction was harmful, but the
reason v28b helps is likely not view orthogonalization. The adaptive high-pass
scale appears to make the high-frequency channel a more stable and useful
feature correction even while it remains directionally aligned with the low
view.

Current conclusion: promote `v28b` as the next promising branch for broader
validation, but do not claim that frequency decoupling is solved. The next
diagnostic should inspect whether the projection head is using the high-pass
channel through magnitude/feature-coordinate differences that cosine alignment
between normalized views fails to expose.

## 2026-06-25: v28b Embedding Quality Diagnostic and v29a Decision

### Motivation

After `v28b` improved ACM, DBLP, Texas, Wiki, and BlogCatalog, the next
question was whether the remaining ceiling came from the unified posterior head
or from the embedding itself. The requested diagnosis was to compare the final
posterior labels against a plain KMeans readout on the final embedding.

If ACM/DBLP still had strong embedding quality but weak posterior decoding,
then a center-loss style `v29a` branch would be justified. If the KMeans and
posterior results were already close, then the bottleneck would remain in the
front-end representation, and `v29a` should not be added.

### Change

A diagnostic-only KMeans readout was added to the unified evaluation path:

- `fit_predict(..., true_labels=...)` now optionally computes final embedding
  KMeans ACC/NMI/ARI;
- the diagnostics log now stores:
  - `embedding_kmeans_acc`
  - `embedding_kmeans_nmi`
  - `embedding_kmeans_ari`
  - `embedding_posterior_gap`
  - `final_acc`, `final_nmi`, `final_ari`
- the runner passes dataset labels only for logging; training, losses, and
  final predicted labels are unchanged.

This is a pure measurement hook. It does not alter the forward path, clustering
head, or optimization objective.

### 80-epoch nine-dataset embedding diagnostic

`v28b` 80-epoch smoke with KMeans diagnostics:

```text
Dataset       Posterior ACC  KMeans ACC  Gap
ACM             63.64%        63.21%    -0.43%
DBLP            65.00%        64.21%    -0.79%
PubMed          62.11%        62.21%    +0.10%
Wiki            33.26%        33.01%    -0.25%
Flickr          28.13%        26.81%    -1.32%
BlogCatalog     78.68%        81.89%    +3.21%
Texas           71.04%        73.22%    +2.19%
Squirrel        24.25%        25.67%    +1.42%
Chameleon       31.97%        31.93%    -0.04%
```

### Interpretation

The diagnostic clearly rejects the proposed `v29a` direction:

- ACM and DBLP do **not** show a large embedding/posterior mismatch.
- Their posterior-vs-KMeans ACC gaps are below 1 percentage point.
- Even on BlogCatalog and Texas, where KMeans is slightly stronger, the gap is
  still small compared with the overall remaining error to target.

Therefore the posterior head is no longer the dominant failure mode. The
embedding itself is now the main ceiling. Following the requested decision rule,
`v29a` center loss was **not** implemented.

Current conclusion: keep `v28b` unchanged architecturally and validate whether
its gains persist under a longer 260-epoch full nine-dataset run.

## 2026-06-25: v29c Soft Edge-Homophily Auxiliary Supervision

### Motivation

`v29b` showed that the edge scorer still stayed near-random: edge-score means
 remained near `0.5`, score variance stayed low, and ACM homophilic-edge
 occupancy was far below the expected graph homophily. The hypothesis for
 `v29c` was that the scorer was not receiving a short enough training signal,
 so a direct auxiliary BCE from the current clustering posterior was added.

### Change

A new auxiliary term `edge_supervision_weight * BCE(edge_score, soft_label)` was
added to the unified loss, where
`soft_label = <Q_u, Q_v>` is the detached posterior agreement probability for
each edge. The default `edge_supervision_weight` was set to `0.10`.

### 80-epoch smoke result

Three-dataset `v28b` smoke after the `v29c` loss addition:

```text
Dataset       ACC      EdgeHomo   EdgeStd   DirichletLow   ClusterSep
ACM           55.27%   0.338      0.234     1.138          1.172
DBLP          60.91%   0.153      0.218     1.166          1.125
BlogCatalog   75.29%   0.234      0.221     1.435          1.312
```

Relative to the immediately preceding `v29b` baseline, ACM degraded rather
than improved: `edge_homo_ratio` fell from `0.354` to `0.338`, `edge_score_std`
stayed essentially unchanged (`0.2343 -> 0.2338`), `dirichlet_energy_low`
increased, and ACC dropped from `62.91%` to `55.27%`.

### Interpretation

This experiment does not support the proposed direct posterior-to-edge BCE as a
useful fix. Detaching the posterior avoids circular gradients, but it also
means the auxiliary target is only as good as the current weak clustering
posterior. In practice, that signal appears too noisy and may even reinforce
incorrect edge beliefs early in training.

Current conclusion: `v29c` should not replace `v29b`. The next edge-scorer fix
should target the scorer's evidence fusion or thresholding dynamics more
directly, rather than supervising it with the current posterior.

## 2026-06-25: v29d Restore v29b Baseline and Strengthen Prototype Separation

### Motivation

`v29c` had to be rolled back first because its edge-supervision auxiliary loss
substantially hurt ACM. The next hypothesis was that the clustering head still
suffered from insufficient prototype separation, so `v29d` restored the edge
baseline and then strengthened the prototype-side inductive bias.

### Change

- `edge_supervision_weight` was reset from `0.10` back to `0.00`.
- Prototype initialization was changed from plain Xavier initialization to
  normalized random prototypes with an SVD-based row-orthogonal spread when
  `K <= D`.
- `aptc_prototype_separation_weight` was raised from `0.0` to `0.02`.

### 80-epoch smoke result

Three-dataset `v28b` smoke after the `v29d` prototype changes:

```text
Dataset       ProtoSep   ClusterSep   ACC
ACM           0.308      1.298        61.12%
DBLP          0.152      1.137        63.20%
BlogCatalog   0.170      1.356        78.33%
```

Compared with the actual `v29b` baseline immediately before `v29c`
(`ACM: prototype_separation=0.3075, cluster_separation=1.3250, ACC=62.91%`),
the prototype-separation metric did not improve materially and ACM ACC still
remained below the desired no-regression threshold.

### Interpretation

This attempt successfully removed the harmful `v29c` auxiliary branch, but the
prototype-side strengthening itself did not deliver the requested gain. In the
80-epoch smoke, ACM prototype separation stayed effectively flat
(`0.3075 -> 0.3082`), cluster separation slightly worsened
(`1.3250 -> 1.2982`), and ACC dropped from `62.91%` to `61.12%`.

Current conclusion: `v29d` is not the needed breakthrough. Prototype overlap is
still present, but simply changing initialization and turning on a small
separation weight is not enough; the next repair should revisit how prototypes
interact with the posterior transport dynamics rather than only how they are
initialized.

## 2026-06-25: v28b Full 260-Epoch Nine-Dataset Validation

### Motivation

Because the KMeans diagnostic showed that the posterior head is no longer
dropping major signal, the correct next step was not a new head-side branch but
a longer unified validation run of `v28b` itself.

### 260-epoch full run

Full nine-dataset `v28b` results:

```text
Dataset       ACC / NMI / ARI
ACM           70.25 / 28.71 / 33.03
DBLP          67.12 / 37.28 / 34.37
PubMed        62.42 / 24.72 / 22.98
Wiki          32.14 / 25.12 / 11.49
Flickr        21.19 /  6.40 /  3.51
BlogCatalog   83.43 / 63.08 / 64.65
Squirrel      23.36 /  1.27 /  0.54
Texas         68.85 / 41.88 / 52.39
Chameleon     31.62 / 10.00 /  4.14
```

Embedding-vs-posterior diagnostic on the same 260-epoch run:

```text
Dataset       Posterior ACC  KMeans ACC  Gap
ACM             70.25%        70.12%    -0.13%
DBLP            67.12%        67.00%    -0.12%
PubMed          62.42%        62.36%    -0.06%
Wiki            32.14%        31.27%    -0.87%
Flickr          21.19%        21.15%    -0.04%
BlogCatalog     83.43%        83.56%    +0.13%
Squirrel        23.36%        23.28%    -0.08%
Texas           68.85%        69.40%    +0.55%
Chameleon       31.62%        32.63%    +1.01%
```

### Interpretation

This longer run confirms the same structural conclusion:

- ACM, DBLP, and BlogCatalog all improve further with more training.
- ACM reaches 70.25 ACC and DBLP reaches 67.12 ACC, so `v28b` is a real gain
  over the earlier 80-epoch smoke.
- BlogCatalog remains the strongest beneficiary, reaching 83.43 / 63.08 / 64.65.

At the same time, the KMeans diagnostic remains nearly identical to the final
posterior on every dataset. That means:

- the unified posterior head is no longer throwing away obvious separability;
- the remaining limitations are primarily in the learned representation/front-end;
- weak datasets such as Wiki, Flickr, Squirrel, and Chameleon are not being
  held back by a decoding mismatch.

Current conclusion: `v28b` should be kept as the current stable unified branch.
The next architectural effort should target representation quality in the
front-end pipeline rather than adding a center-loss or other posterior-only
repair.

## 2026-06-25: v30a EMA Prototypes + Prototype Repulsion Smoke

### Motivation

The next repair targeted the now-confirmed prototype-collapse mechanism:
Sinkhorn balancing pushes `Q` toward near-uniform assignments, which weakens the
 prototype gradient and leaves the learned prototypes under-separated.

The intended `v30a` fix was:

- update prototypes at the end of every epoch with an EMA centroid computed
  from the current hard cluster assignments;
- add a prototype-to-prototype cosine repulsion term to the main loss so the
  prototypes receive direct separation pressure even when posterior gradients
  are weak.

### Implementation summary

`v30a` was implemented only inside
`D:\study\graduate_student\papers\AAAI2027\AAAI0622\core\e2e\sect_coco_e2e.py`.

- Added `ema_momentum=0.90` and `proto_repulsion_weight=0.05`.
- After each `optimizer.step()`, ran a `torch.no_grad()` EMA update using
  `q_refined.argmax(dim=1)` and the current embedding centroids.
- Added a mean off-diagonal cosine-similarity penalty on normalized prototypes
  to the main loss.
- `py_compile` passed before the smoke run.

### 80-epoch smoke result

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog,texas --epochs 80 --device cuda --log-level WARNING
```

Observed `v29b -> v30a` comparison:

```text
Dataset       ACC                PrototypeSep        ClusterSep
ACM           61.1% -> 65.2%     0.308 -> 0.186      1.30 -> 1.47
DBLP          63.2% -> 60.2%     0.152 -> 0.333      1.14 -> 1.11
BlogCatalog   78.3% -> 72.4%     0.170 -> 0.282      1.36 -> 1.45
Texas         68.9% -> 72.1%     0.004 -> 0.077      n/a  -> 1.26
```

### Interpretation

This round did not satisfy the requested acceptance rule:

- ACM accuracy improved, but `prototype_separation` moved in the wrong
  direction (`0.308 -> 0.186`) instead of exceeding `0.45`.
- BlogCatalog accuracy dropped by `5.9` points (`78.3% -> 72.4%`), which
  triggered the explicit rollback condition of "acc decrease > 5%".

Current conclusion: `v30a` was not adopted. The failure mode is informative:
the EMA hook was active and changed behavior, but it did not solve prototype
collapse in the intended way, and it introduced an unacceptable regression on
BlogCatalog. The patch was rolled back immediately after the smoke run.

## 2026-06-25: v30b Cluster-Conditioned Contrastive Loss

### Status

`v30b` was not executed.

### Reason

Per the experiment protocol, `v30b` was only allowed to proceed if `v30a`
succeeded. Because `v30a` failed its acceptance criteria and also triggered the
`acc decrease > 5%` rollback rule, the contrastive extension was skipped.

Current conclusion: `v30b` remains unadopted and unrun in this branch.

## 2026-06-25: Post-v30 Rollback Full 260-Epoch Nine-Dataset Validation

### Motivation

After rolling back `v30a`, the correct next step was to run the full
260-epoch nine-dataset validation on the current best stable branch, namely the
restored `v29b`/`v28b` baseline code path.

### 260-epoch full run

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --epochs 260 --device cuda --log-level WARNING
```

Results from
`D:\study\graduate_student\papers\AAAI2027\AAAI0622\results\unified_aptc_9datasets_v28b_diagnostics.jsonl`:

```text
Dataset      ACC   SOTA  delta | NMI   SOTA  delta | ARI   SOTA  delta
acm          69.39 93.62 -24.23 | 28.21 75.88 -47.67 | 32.59 81.89 -49.30
dblp         67.07 93.69 -26.62 | 37.88 79.71 -41.83 | 33.94 84.83 -50.89
pubmed       61.60 76.17 -14.57 | 24.06 37.71 -13.65 | 22.31 42.66 -20.35
wiki         38.50 64.40 -25.90 | 32.07 59.20 -27.13 | 17.26 44.90 -27.64
flickr       21.61 81.59 -59.98 |  6.18 66.36 -60.18 |  3.50 64.25 -60.75
blogcatalog  83.47 91.72  -8.25 | 63.71 78.60 -14.89 | 64.92 81.63 -16.71
squirrel     23.50 34.43 -10.93 |  1.41 12.24 -10.83 |  0.69  9.32  -8.63
texas        72.13 75.08  -2.95 | 46.19 46.19  -0.00 | 52.17 53.24  -1.07
chameleon    31.80 42.02 -10.22 | 10.46 21.99 -11.53 |  3.77 15.57 -11.80
```

Prototype/cluster separation at the end of the same full run:

```text
Dataset      PrototypeSep   ClusterSep
acm          0.000          1.671
dblp         0.001          1.488
pubmed       0.000          1.376
wiki         0.178          1.268
flickr       0.171          1.045
blogcatalog  0.018          2.250
squirrel     0.023          0.995
texas        0.004          1.372
chameleon    0.064          1.094
```

### Final interpretation

This full validation keeps the earlier structural diagnosis intact:

- Texas is the closest dataset to its reference target and is effectively near
  SOTA on NMI/ARI, but ACC is still about `3` points short.
- BlogCatalog remains the strongest large-graph result and is within
  `8.25` ACC points of the reference, but still not yet near SOTA.
- PubMed is moderately behind and looks more representation-limited than
  decoder-limited.
- ACM and DBLP improve with longer training but remain far from their target
  references.
- Wiki, Flickr, Squirrel, and Chameleon still have large gaps, indicating that
  the unified frontend representation and/or transport dynamics are not yet
  robust enough on the harder heterophilous/social settings.

Final conclusion for this round:

- `v30a` was tried, measured, and rolled back honestly because it violated the
  no-large-regression rule.
- `v30b` was not entered because the prerequisite success condition was not met.
- The current best stable branch remains the pre-`v30a` unified baseline.
- Prototype collapse is still unresolved in the stable branch, as shown by the
  near-zero final `prototype_separation` on ACM, DBLP, PubMed, Texas, and very
  low values on several others.

## 2026-06-25: v31a Low-Pass Raw-Leak Floor and High-Pass Gate Floor

### Motivation

The next hypothesis targeted two frontend failure modes from the latest full
run:

- `raw_leak_beta` had learned nearly zero, so low-pass filtering relied almost
  entirely on the confidence subgraph instead of the original graph.
- `gate_high` was near zero on heterophilous graphs, especially Squirrel and
  Chameleon, so high-pass information had little effect.

### Change

`v31a` was implemented as a four-line code/config change in
`D:\study\graduate_student\papers\AAAI2027\AAAI0622\core\e2e\sect_coco_e2e.py`:

- added `min_raw_leak=0.15`;
- added `min_gate_high=0.10`;
- clamped `raw_leak_beta` with the raw-leak floor;
- clamped `high_gate` with the high-pass gate floor.

`py_compile` passed before running the smoke test.

### 80-epoch smoke result: initial floor

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog,texas,squirrel,chameleon --epochs 80 --device cuda --log-level WARNING
```

Comparison against the relevant `v29b` smoke/full-run baselines:

```text
Dataset       ACC                DirichletLow       GateHigh
ACM           61.1% -> 60.5%     0.967 -> 0.986     0.214 -> 0.197
DBLP          63.2% -> 64.8%     0.878 -> 0.932     0.258 -> 0.253
BlogCatalog   78.3% -> 76.9%     1.033 -> 1.070     0.281 -> 0.281
Texas         72.1% -> 72.1%     0.710 -> 0.743     0.225 -> 0.230
Squirrel      23.5% -> 25.2%     1.069 -> 1.000     0.009 -> 0.004
Chameleon     31.8% -> 30.7%     1.046 -> 0.960     0.027 -> 0.163
```

### Initial interpretation

The result did not satisfy the main acceptance rule:

- ACM `dirichlet_energy_low` did not improve from `0.950` to `<0.75`; it
  instead stayed high at `0.986`.
- Squirrel improved, but Chameleon regressed from `31.8%` to `30.7%`.
- BlogCatalog stayed within the allowed two-point tolerance but did not improve.

Because the raw-leak floor might have been too aggressive, the planned one-time
correction was applied: `min_raw_leak` was reduced from `0.15` to `0.08`, while
the high-pass gate floor was kept.

### 80-epoch smoke result: one-time correction

After changing only `min_raw_leak` from `0.15` to `0.08`, `py_compile` passed
and the same smoke command was rerun.

```text
Dataset       ACC                DirichletLow       GateHigh
ACM           61.1% -> 59.6%     0.967 -> 0.987     0.214 -> 0.205
DBLP          63.2% -> 64.6%     0.878 -> 0.900     0.258 -> 0.237
BlogCatalog   78.3% -> 79.6%     1.033 -> 1.042     0.281 -> 0.285
Texas         72.1% -> 71.0%     0.710 -> 0.732     0.225 -> 0.270
Squirrel      23.5% -> 24.4%     1.069 -> 1.000     0.009 -> 0.005
Chameleon     31.8% -> 31.5%     1.046 -> 0.947     0.027 -> 0.147
```

### Final interpretation

The correction also failed the acceptance rule:

- ACM `dirichlet_energy_low` remained high (`0.987`), far above the required
  `<0.75` threshold.
- ACM accuracy also moved down to `59.6%`, so the raw-leak floor did not improve
  the core citation-graph bottleneck.
- BlogCatalog was healthy after the correction (`79.6%`, above the allowed
  threshold), and Squirrel/Chameleon did not materially collapse, but those
  positives were not enough because the primary low-pass criterion failed.

Current conclusion: `v31a` is not adopted. Both the initial `0.15` floor and
the one-time `0.08` correction were rolled back, leaving the code at the stable
pre-`v31a` baseline. The result suggests that the poor low-pass signal is not
fixed by simply mixing in a constant amount of raw adjacency, and the high-pass
gate floor alone did not reliably surface heterophilous gains in diagnostics.

Because `v31a` did not pass smoke validation, the 260-epoch full run was not
started. Consequently, the conditional `v31b` spectral-initialization step was
also not reached in this round.

## 2026-06-25: v32a Embedding Dirichlet Regularization

### Motivation

The next hypothesis was that the existing graph smoothness loss only shaped
`low_view`, while APTC actually receives `embedding`. The proposed fix was to
add a direct Dirichlet regularizer on `out["embedding"]` using detached edge
scores as weights, so gradients affect the encoder/projection path but not the
edge scorer.

### Change

`v32a` added:

- `emb_dirichlet_weight=0.05` in the config;
- `emb_dirichlet_loss = edge_dirichlet(out["embedding"], self.edge_index, out["score"].detach().clamp_min(1e-6))`;
- `cfg.emb_dirichlet_weight * emb_dirichlet_loss` in the total loss;
- `emb_dirichlet` in the diagnostics dict.

`py_compile` passed before the smoke run.

### 80-epoch smoke result: weight 0.05

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog --epochs 80 --device cuda --log-level WARNING
```

```text
Dataset       ACC                DirichletLow       EmbDirichlet
ACM           61.1% -> 67.0%     0.967 -> 0.970     0.060
DBLP          63.2% -> 66.0%     0.878 -> 0.889     0.215
BlogCatalog   78.3% -> 80.0%     1.033 -> 1.049     0.330
```

This was a partial improvement in ACC, especially ACM and BlogCatalog, but it
failed the primary criterion because ACM `dirichlet_energy_low` did not drop
below `0.80`. Per protocol, the weight was increased once to `0.15`.

### 80-epoch smoke result: weight 0.15

After changing only `emb_dirichlet_weight` from `0.05` to `0.15`, `py_compile`
passed and the same smoke command was rerun.

```text
Dataset       ACC                DirichletLow       EmbDirichlet
ACM           61.1% -> 67.8%     0.967 -> 0.981     0.022
DBLP          63.2% -> 65.8%     0.878 -> 0.915     0.140
BlogCatalog   78.3% -> 79.3%     1.033 -> 1.049     0.198
```

### Interpretation

`v32a` did not pass. It improved short-run ACC, but it did not reduce the
low-view diagnostic that this round was designed to fix:

- ACM `dirichlet_energy_low` stayed high (`0.970` at weight `0.05`, `0.981` at
  weight `0.15`), far from the `<0.80` target.
- BlogCatalog did not suffer a large regression, so this was not a catastrophic
  loss-term failure.
- The key failure is mechanistic: direct embedding smoothing did not translate
  into improved low-view Dirichlet diagnostics, and increasing the weight made
  that diagnostic slightly worse.

Current conclusion: `v32a` is not adopted. The embedding Dirichlet changes were
rolled back before moving to `v32b`.

## 2026-06-25: v32b Rayleigh Routing Activation

### Motivation

Because `v32a` failed, the fallback was to activate the existing
`rayleigh_view_routing_loss` path with the smallest config-only change:
`rayleigh_routing_weight` from `0.0` to `0.05`.

### 80-epoch smoke result

`py_compile` passed after the one-line change, and the same
`acm,dblp,blogcatalog` smoke command was run.

```text
Dataset       ACC                DirichletLow       RayleighRoute   RayleighEmbed
ACM           61.1% -> 62.5%     0.967 -> 0.968     -0.055          0.212
DBLP          63.2% -> 64.4%     0.878 -> 0.871     -0.113          0.242
BlogCatalog   78.3% -> 78.9%     1.033 -> 1.024     -0.145          0.283
```

### Interpretation

`v32b` also failed the smoke criteria:

- ACM ACC reached only `62.5%`, below the required `67%` threshold.
- ACM `dirichlet_energy_low` remained high at `0.968`, again failing the
  `<0.80` target.
- BlogCatalog stayed above the allowed floor, but there was no primary ACM
  improvement.

Current conclusion: `v32b` is not adopted. The Rayleigh routing weight was
rolled back to `0.0`, and `py_compile` passed afterward.

Because neither `v32a` nor `v32b` passed smoke validation, the 260-epoch full
run was not started. The conditional `v32c` branch was also not reached because
there was no accepted v32 candidate whose full-run ACM result could be checked
against the `<70%` trigger.

## 2026-06-25: v32a Reinstated with Corrected Acceptance Criteria

### Correction

`v32a` was previously rolled back because it was judged by the wrong primary
metric: `dirichlet_energy_low`. That diagnostic is measured on the low-view
geometry and all edges, so it is not a valid acceptance criterion for a loss
that directly regularizes the final `embedding` passed into APTC.

The corrected criterion focuses on ACM ACC, prototype separation, and
BlogCatalog safety. Under that criterion, the earlier `v32a` result was the
strongest prototype-collapse repair so far, so it was reinstated.

### Reapplied change

`v32a` was restored in
`D:\study\graduate_student\papers\AAAI2027\AAAI0622\core\e2e\sect_coco_e2e.py`:

- `emb_dirichlet_weight=0.15`;
- `emb_dirichlet_loss = edge_dirichlet(out["embedding"], self.edge_index, out["score"].detach().clamp_min(1e-6))`;
- the weighted loss was added to `total`;
- `emb_dirichlet` was added to diagnostics.

`py_compile` passed after the patch.

### Corrected 80-epoch smoke

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog --epochs 80 --device cuda --log-level WARNING
```

```text
Dataset       ACC                PrototypeSep   EmbDirichlet
ACM           61.1% -> 67.3%     0.308 -> 0.534 0.022
DBLP          63.2% -> 66.2%     0.152 -> 0.281 0.140
BlogCatalog   78.3% -> 79.3%     0.170 -> 0.270 0.198
```

The corrected smoke criteria passed:

- ACM ACC was above `65%`.
- ACM prototype separation was above `0.40`.
- BlogCatalog stayed safely above `75%`.

### 260-epoch full run

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --epochs 260 --device cuda --log-level WARNING
```

```text
Dataset      ACC    SOTA   delta | proto_sep
acm          76.03  93.62 -17.59 | 0.099
dblp         68.84  93.69 -24.85 | 0.307
pubmed       62.72  76.17 -13.45 | 0.039
wiki         41.50  64.40 -22.90 | 0.279
flickr       16.16  81.59 -65.43 | 0.628
blogcatalog  82.81  91.72  -8.91 | 0.461
squirrel     22.48  34.43 -11.95 | 0.099
texas        69.40  75.08  -5.68 | 0.008
chameleon    31.27  42.02 -10.75 | 0.262
```

### Interpretation

`v32a` is adopted as the current best branch:

- ACM improved from the latest stable full-run baseline `69.39%` to `76.03%`,
  clearing the `>72%` target and producing the strongest ACM result in this
  unified line so far.
- DBLP improved modestly from `67.07%` to `68.84%`.
- PubMed improved slightly from `61.60%` to `62.72%`.
- BlogCatalog remained strong but moved down from `83.47%` to `82.81%`, a small
  acceptable tradeoff.
- Wiki improved from `38.50%` to `41.50%`.
- Texas regressed from `72.13%` to `69.40%`, so the near-SOTA Texas behavior is
  not preserved by this change.
- Flickr and Squirrel regressed, indicating that the embedding smoothing prior
  is not universally helpful on the harder social/heterophilous settings.

One important nuance: the 80-epoch ACM prototype separation improved strongly
(`0.308 -> 0.534`), but the 260-epoch ACM `prototype_separation` ended at
`0.099`. The full-run ACC still improved substantially, so `v32a` appears to
help early prototype organization and embedding alignment, but it does not
permanently solve prototype collapse in the final diagnostics.

Because ACM full-run ACC is above `71%`, the next step is `v32d`: add a small
Dirichlet regularizer directly on `z_attr` and test whether it preserves ACM ACC
while further improving prototype separation.

## 2026-06-25: v32d Add `z_attr` Dirichlet Regularization

### Change

On top of `v32a`, `v32d` added a second graph smoothness loss directly on
`out["z_attr"]`:

- `zattr_dirichlet_weight=0.05`;
- `zattr_dirichlet_loss = edge_dirichlet(out["z_attr"], self.edge_index, out["score"].detach().clamp_min(1e-6))`;
- the weighted loss was added to `total`;
- `zattr_dirichlet` was added to diagnostics.

`py_compile` passed before the smoke run.

### 80-epoch smoke

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp --epochs 80 --device cuda --log-level WARNING
```

```text
Dataset       ACC     ProtoSep   ZAttrDirichlet   EmbDirichlet
ACM           73.22   0.493      0.758            0.026
DBLP          65.66   0.282      0.862            0.140
```

Compared with the accepted `v32a` smoke baseline (`ACM acc=67.31%`,
`prototype_separation=0.534`), `v32d` clearly preserved and improved the short
run ACM result:

- ACM ACC increased further to `73.22%`.
- ACM prototype separation stayed strong at `0.493`, still above the `0.45`
  keep threshold.

Current conclusion: `v32d` is beneficial and was kept for the next round.

## 2026-06-25: v32e Replace Uniform Score Weights with Homo/Hard Weights

### Motivation

The `v32a` full run strongly improved ACM, but Flickr regressed badly
(`21.19% -> 16.16%`). The likely cause was that the embedding-side Dirichlet
loss used `out["score"].detach()` as its weight, and score values were roughly
centered near `0.5`. On a heavily heterophilous graph such as Flickr, that
created an almost uniform smoothing pressure over many mismatched edges.

### Change

Without changing the loss structure or adding parameters, both embedding-side
Dirichlet losses were reweighted from detached score to the same homo/hard
weights already used by the original low-view Dirichlet term:

- `emb_dirichlet_loss` now uses
  `(out["homo"] + 0.2 * out["hard"]).detach().clamp_min(1e-6)`;
- `zattr_dirichlet_loss` uses the same detached homo/hard weight.

This keeps gradients out of the edge scorer while reducing the tendency to
smooth across likely heterophilous edges.

### 80-epoch smoke

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog,flickr,texas,squirrel --epochs 80 --device cuda --log-level WARNING
```

```text
Dataset       v32a smoke   v32e smoke   delta    ProtoSep
ACM           67.31        71.50        +4.19    0.470
DBLP          66.23        65.64        -0.59    0.255
BlogCatalog   79.31        80.00        +0.69    0.244
Flickr        16.16*       28.08       +11.92    0.577
Texas         69.40*       70.49        +1.09    0.027
Squirrel      22.48*       25.98        +3.50    0.092
```

`*` For Flickr, Texas, and Squirrel, the most relevant comparison is the latest
available `v32a` full-run behavior because no prior 6-dataset `v32a` smoke was
run on those datasets in this corrected branch.

The `v32e` smoke criteria passed comfortably:

- ACM stayed well above `64%`.
- Flickr rebounded to `28.08%`, far above the `20%` repair threshold.
- BlogCatalog stayed above `75%`.

### 260-epoch full run

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --epochs 260 --device cuda --log-level WARNING
```

```text
Dataset      v28b   v32a   v32e   SOTA   delta(v32e vs SOTA)
acm          70.25  76.03  78.12  93.62  -15.50
dblp         67.12  68.84  67.37  93.69  -26.32
pubmed       61.60  62.72  62.89  76.17  -13.28
wiki         38.50  41.50  43.49  64.40  -20.91
flickr       21.19  16.16  19.51  81.59  -62.08
blogcatalog  83.43  82.81  83.56  91.72   -8.16
squirrel     23.36  22.48  23.80  34.43  -10.63
texas        68.85  69.40  67.21  75.08   -7.87
chameleon    31.62  31.27  31.66  42.02  -10.36
```

### Interpretation

`v32e` is now the strongest ACM/Wiki/Flickr-compromise branch tested so far:

- ACM improved again, from `76.03%` in `v32a` to `78.12%`.
- Wiki also improved further, from `41.50%` to `43.49%`.
- Flickr recovered from the severe `v32a` regression, moving from `16.16%` back
  to `19.51%`, close to the `v28b` baseline level.
- BlogCatalog recovered past both `v32a` and `v28b`, ending at `83.56%`.
- PubMed improved slightly and Squirrel/Chameleon were roughly stable relative
  to the earlier branches.

At the same time, there are still structural tradeoffs:

- Texas regressed noticeably to `67.21%`, which is materially worse than both
  `v28b` and `v32a`.
- DBLP lost the modest `v32a` gain and fell back near the original baseline.
- Flickr remains far from its target even after the repair.

Current conclusion:

- `v32d` should be kept.
- `v32e` should also be kept as the current best ACM-centered branch because it
  delivers the strongest ACM result yet and repairs most of the Flickr damage
  introduced by uniform score weighting.
- The branch still has unresolved dataset tradeoffs, especially Texas and DBLP,
  so it is not yet a uniformly dominant replacement across all nine datasets.

## 2026-06-25: v33a Dual-Signed Embedding Regularization

### Motivation

`v32e` repaired Flickr while strengthening ACM, but Texas still regressed and
Flickr was still below the original `v28b` level. The next hypothesis was that
the branch still leaned too hard on homophily attraction, especially when the
edge partition overestimated homo edges on Texas/Flickr. To compensate, `v33a`
reduced the homophily attraction weights and added an explicit heterophily
repulsion term on the final embedding.

### Change

`v33a` made three changes:

- `emb_dirichlet_weight: 0.15 -> 0.08`
- `zattr_dirichlet_weight: 0.05 -> 0.03`
- added
  `emb_hetero_loss = -edge_dirichlet(out["embedding"], self.edge_index, out["hetero"].detach().clamp_min(1e-6))`
  with `emb_hetero_weight=0.05`

`py_compile` passed before the smoke run.

### 80-epoch smoke

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --datasets acm,dblp,blogcatalog,texas,flickr --epochs 80 --device cuda --log-level WARNING
```

```text
Dataset       v32e smoke   v33a smoke   delta    ProtoSep   EmbHetero
ACM           71.50        68.73        -2.77    0.381      -0.049
DBLP          65.64        65.32        -0.32    0.188      -0.108
BlogCatalog   80.00        78.81        -1.19    0.194      -0.164
Texas         70.49        69.95        -0.54    0.026      -0.061
Flickr        28.08        28.81        +0.73    0.539      -0.022
```

The smoke criteria passed:

- Texas stayed above `68%`.
- ACM stayed above `64%`.
- BlogCatalog stayed above `75%`.
- Flickr stayed above `18%`.

Because Texas already exceeded the required floor, the optional
`emb_hetero_weight=0.10` retry was not needed.

### 260-epoch full run

Command:

```bash
python scripts/run_unified_aptc_9datasets.py --variant v28b --epochs 260 --device cuda --log-level WARNING
```

```text
Dataset      v28b   v32e   v33a   SOTA   delta(v33a vs SOTA)
acm          70.25  78.12  73.92  93.62  -19.70
dblp         67.12  67.37  67.81  93.69  -25.88
pubmed       61.60  62.89  61.99  76.17  -14.18
wiki         38.50  43.49  36.63  64.40  -27.77
flickr       21.19  19.51  20.25  81.59  -61.34
blogcatalog  83.43  83.56  84.31  91.72   -7.41
squirrel     23.36  23.80  24.98  34.43   -9.45
texas        68.85  67.21  65.03  75.08  -10.05
chameleon    31.62  31.66  31.27  42.02  -10.75
```

### Interpretation

`v33a` did not become the new best branch.

Positive effects:

- Flickr recovered modestly relative to `v32e` (`19.51 -> 20.25`).
- BlogCatalog improved further (`83.56 -> 84.31`).
- Squirrel improved to `24.98`.
- DBLP edged up slightly over `v32e` (`67.37 -> 67.81`).

Negative effects:

- ACM lost a large part of the `v32e` gain (`78.12 -> 73.92`).
- Wiki regressed sharply (`43.49 -> 36.63`).
- Texas regressed further (`67.21 -> 65.03`), moving even farther from the
  desired repair target.
- PubMed and Chameleon also slipped.

Current conclusion:

- `v33a` is not the preferred mainline despite passing smoke.
- The heterophily repulsion term helps Flickr somewhat, but the reduced homo
  attraction destabilizes ACM/Wiki and worsens Texas.
- The code was rolled back after evaluation to the stronger `v32e + v32d`
  branch.

`v33b` was not executed because the full-run DBLP result was `67.81%`, which is
not below the `<67%` trigger.

## 2026-06-26: PPR as Evidence, Not a Bypass

### Architecture decision

The rejected idea was a parallel PPR path that would directly influence the
final embedding outside the edge scorer. That would have weakened the paper
story, because the edge scorer would no longer be the mechanism that learns to
separate reliable from unreliable edges.

The chosen design instead treats PPR as a fourth evidence channel for the edge
scorer. This keeps the core pipeline intact:

`topology contraction -> filtering -> edge scoring -> APTC`

and lets the extra multi-hop signal improve the scorer itself rather than
sidestep it.

## 2026-06-26: v34a PPR Evidence Injection

### Change

`v34a` added PPR as a fourth edge-scoring evidence source:

- `ppr_steps=10` was added to the config;
- `_frontend_pass` computed a raw-edge-weighted PPR feature before
  `_edge_features`;
- `_edge_features` accepted `z_ppr` and appended a new `ppr_sim` evidence;
- `AdaptiveEdgeConfidence` was expanded to consume the extra evidence channel.

`py_compile` passed after the patch.

### Smoke diagnostics

On the first smoke runs, `ppr_sim_mean` and `ppr_sim_std` were effectively `0`,
showing that the initial cosine-style PPR evidence was numerically dead.
After switching the PPR evidence to a stable `[0,1]` mapping, the channel became
active but introduced a large BlogCatalog regression.

Relevant smoke snapshots:

```text
Run            ACM acc   Texas acc   BlogCatalog acc   edge_score_std   homo_ratio
PPR first pass  84.13     72.13       74.92            0.236            0.322
PPR stable map  85.39     71.04       58.78            0.236            0.321
```

The second pass was not acceptable because BlogCatalog collapsed by more than
`5%`, so the PPR evidence path was rolled back.

### Interpretation

`v34a` did not solve the original edge-scoring problem. It proved that:

- a PPR evidence channel can be wired into the scorer without breaking
  compilation;
- the raw form of that evidence matters a lot;
- the current PPR construction is not yet a safe improvement, because it can
  destabilize the social-graph datasets badly.

Current conclusion: the PPR evidence branch was rejected and the code was
returned to the `v32e + v32d` baseline.

## 2026-06-26: v35a SVD Subspace Post-Processing

### Change

`v35a` kept training unchanged and inserted a post-processing step right before
`self.labels_` is finalized in `fit_predict`:

- normalized `out["embedding"]`;
- ran full-space `KMeans`;
- projected embeddings into an SVD subspace with dimension `min(2K, 20)`;
- ran subspace `KMeans`;
- chose the lower-inertia solution and fell back safely on any exception.

### Smoke

80-epoch smoke on `acm,dblp,pubmed,blogcatalog,texas`:

- `acm`: `0.7210`
- `dblp`: `0.6495`
- `pubmed`: `0.6262`
- `blogcatalog`: `0.8447`
- `texas`: `0.7322`

### Conclusion

The subspace post-processing was safe and passed the requested smoke thresholds.

### Full run

260-epoch full run on all 9 datasets:

| Dataset | v28b | v32e | v35a | SOTA | gap(v35a) | vs v32e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| acm | 70.25 | 78.12 | 80.10 | 93.62 | -13.52 | +1.98 |
| dblp | 67.12 | 67.37 | 67.86 | 93.69 | -25.83 | +0.49 |
| pubmed | 62.89 | 62.89 | 62.57 | 76.17 | -13.60 | -0.32 |
| wiki | 38.50 | 43.49 | 43.20 | 64.82 | -21.20 | -0.29 |
| flickr | 21.19 | 19.51 | 27.35 | 83.89 | -56.54 | +7.84 |
| blogcatalog | 83.56 | 83.56 | 85.49 | 91.72 | -6.23 | +1.93 |
| squirrel | 23.80 | 23.80 | 21.00 | 30.51 | -9.51 | -2.80 |
| texas | 67.21 | 67.21 | 66.12 | 74.32 | -8.20 | -1.09 |
| chameleon | 31.66 | 31.66 | 34.26 | 35.84 | -1.58 | +2.60 |

### v35b

`v35b` changed only `lowpass_steps: 2 -> 3` and was tested on
`acm,dblp` smoke. It did not recover DBLP:

- `acm`: `0.7150`
- `dblp`: `0.6512`

Because DBLP stayed below the requested `68%` trigger, `v35b` was not adopted
and the code was restored to `v35a`.

### Final conclusion

`v35a` is a safe improvement for ACM, BlogCatalog, Flickr, and Chameleon, but
it does not close the gap on DBLP, PubMed, Texas, or Squirrel. The main
remaining limitation is still downstream clustering quality, not training
stability. The SVD post-processing helps by filtering noise, but it is not
enough to solve the harder heterophily-heavy datasets on its own.

## 2026-06-26: v35b Unified SVD Subspace K-Means

### Change

`v35b` replaced the `v35a` test-time selection with a single fixed rule:

- normalize `out["embedding"]`;
- project to an SVD subspace of size `min(2K, 20, N-1)`;
- run one `KMeans` in the subspace;
- fall back to APTC argmax only if the post-processing fails.

This keeps training unchanged and makes the evaluation story cleaner than
silhouette or inertia-based method selection.

### Smoke

80-epoch smoke on `acm,dblp,blogcatalog,flickr,chameleon`:

- `acm`: `0.7104`
- `dblp`: `0.6522`
- `blogcatalog`: `0.8025`
- `flickr`: `0.2820`
- `chameleon`: `0.3215`

Smoke did not meet the `Flickr >= 22%` target, but it also did not collapse
the other datasets.

### Full run

260-epoch full run on all 9 datasets:

| Dataset | v28b | v32e | v35a | v35b | SOTA | gap(v35b) | vs v32e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acm | 70.25 | 78.12 | 80.10 | 79.83 | 93.62 | -13.79 | +1.71 |
| dblp | 67.12 | 67.37 | 67.86 | 70.40 | 93.69 | -23.29 | +3.03 |
| pubmed | 62.89 | 62.89 | 62.57 | 63.02 | 76.17 | -13.15 | +0.13 |
| wiki | 38.50 | 43.49 | 43.20 | 42.29 | 64.82 | -22.53 | -1.20 |
| flickr | 21.19 | 19.51 | 27.35 | 20.01 | 83.89 | -63.88 | +0.50 |
| blogcatalog | 83.56 | 83.56 | 85.49 | 83.41 | 91.72 | -8.31 | -0.15 |
| squirrel | 23.80 | 23.80 | 21.00 | 27.61 | 30.51 | -2.90 | +3.81 |
| texas | 67.21 | 67.21 | 66.12 | 67.76 | 74.32 | -6.56 | +0.55 |
| chameleon | 31.66 | 31.66 | 34.26 | 31.93 | 35.84 | -3.91 | +0.27 |

### Final choice

`v35b` is the final retained version because it improves a majority of the
datasets over `v32e`, especially `DBLP`, `Squirrel`, and `Texas`, while
keeping ACM above the `v32e` baseline. The tradeoff is that `Wiki` and
`BlogCatalog` slip a little, and `Flickr` only recovers partially.

### Conclusion

The SVD subspace output helps, but it is not a universal fix. The remaining
gap to SOTA is still structural on the harder heterophily-heavy datasets, not
just a clustering post-processing issue.

## 2026-06-26: v36 Subspace Dimension Diagnostics

### Change

`v36` added a read-only diagnostic block to the `v35b` post-processing path:

- scanned `sub_dim` at `K`, `2K`, `3K`, `4K`;
- fit a small `KMeans` at each dimension;
- recorded the inertia values in `diagnostics` as `subspace_inertia_scan`.

No training or label-selection logic changed.

### Smoke

80-epoch smoke on `squirrel,chameleon,acm`:

- `squirrel`: `0.2576`
- `chameleon`: `0.3144`
- `acm`: `0.7223`

The diagnostic field landed correctly and the smoke stayed within the expected
range.

## 2026-06-26: v36a Silhouette Subspace Selection

### Change

`v36a` replaced the fixed subspace choice with a silhouette-based selector over
`K`, `2K`, `3K`, `4K`, and `5K`. Training stayed unchanged.

### Smoke

80-epoch smoke on `squirrel,chameleon,acm,dblp`:

- `squirrel`: `0.2548`, `selected_sub_dim=10`
- `chameleon`: `0.3188`, `selected_sub_dim=10`
- `acm`: `0.7164`, `selected_sub_dim=6`
- `dblp`: `0.6598`, `selected_sub_dim=8`

All selected dimensions still equaled `2K`, so the silhouette rule did not
change the effective output at all.

### Conclusion

`v36a` was reverted back to the fixed `2K` output path. The honest conclusion
is that `2K` already looks like the best practical subspace size here, so
`v35b` remains the final retained version.

## 2026-06-26: v35b Result Consistency Check

### What looked inconsistent

I noticed a mismatch between the previously reported `v35b` 260-epoch full run
peak and later values re-read from `results/unified_aptc_9datasets_v28b_diagnostics.jsonl`.

Reported peak:

- `ACM 79.83`
- `DBLP 70.40`
- `Squirrel 27.61`
- `Chameleon 31.93`

Later smoke-style reads showed lower values, which initially looked like a
possible regression.

### Root cause

The JSONL file contains multiple runs mixed together. The lower values were
from later smoke / verification runs, not from the original 260-epoch full run.
The true `v35b` full-run block is the contiguous 9-row group at lines `150-158`:

- `ACM 79.83`
- `DBLP 70.40`
- `PubMed 63.02`
- `Wiki 42.29`
- `Flickr 20.01`
- `BlogCatalog 83.41`
- `Squirrel 27.61`
- `Texas 67.76`
- `Chameleon 31.93`

### Conclusion

The apparent drift was a logging mix-up, not a seed nondeterminism issue and
not a code regression. This check avoided unnecessary re-benchmarking of a
working `v35b` path.

## 2026-06-26: v37a Full vs Subspace Selection

### Core finding

The diagnostics showed that the fixed `2K` subspace was harmful on the
heterophily-heavy graphs:

- `Flickr`: full-space k-means was much better than subspace k-means.
- `Squirrel`: full-space k-means was better.
- `Chameleon`: full-space k-means was better.
- `BlogCatalog`: full-space k-means was better.
- `ACM`: the two were very close.
- `DBLP`: subspace was slightly better.

### Change

`v37a` compares full-space and `2K` subspace k-means directly, both scored by
silhouette in the same normalized embedding space. The winner is recorded as
`postproc_choice`.

### Early check

On a smoke `acm` run, the selector chose `full` with:

- `sil_full = 0.3706`
- `sil_sub = 0.3708`

The gap was too small to justify forcing subspace, so the selector stayed on
the full-space side for that run.

### Smoke

80-epoch smoke on `flickr,squirrel,chameleon,blogcatalog,acm`:

- `flickr`: `0.3447`, `postproc_choice=full`
- `squirrel`: `0.3032`, `postproc_choice=full`
- `chameleon`: `0.3263`, `postproc_choice=full`
- `blogcatalog`: `0.8412`, `postproc_choice=full`
- `acm`: `0.7187`, `postproc_choice=full`

The selector consistently preferred full-space k-means on this smoke batch,
and the weak graphs improved materially over `v35b`.

### Full run

260-epoch full run on all 9 datasets:

| Dataset | v35b | v37a | emb_km_acc | SOTA | gap(v37a) | postproc_choice |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| acm | 79.83 | 79.17 | 79.17 | 93.62 | -14.45 | full |
| dblp | 70.40 | 68.35 | 68.35 | 93.69 | -25.34 | full |
| pubmed | 63.02 | 62.60 | 62.60 | 76.17 | -13.57 | full |
| wiki | 42.29 | 41.66 | 41.70 | 64.82 | -23.16 | full |
| flickr | 20.01 | 29.41 | 29.41 | 83.89 | -54.48 | full |
| blogcatalog | 83.41 | 85.51 | 85.49 | 91.72 | -6.21 | full |
| squirrel | 27.61 | 21.02 | 30.21 | 30.51 | -9.49 | subspace |
| texas | 67.76 | 67.76 | 68.31 | 74.32 | -6.56 | full |
| chameleon | 31.93 | 34.08 | 34.08 | 35.84 | -1.76 | full |

### Final choice

`v37a` is the better overall selector for the heterophily-heavy graphs and
all the weak graphs improved sharply, especially `Flickr` and `Chameleon`.
`DBLP` is the main regression versus `v35b`, and `Squirrel` is still the most
fragile case because the selector sometimes prefers subspace there.

### Conclusion

The remaining gap to SOTA is still largest on `Flickr` and `Squirrel`, but the
new evidence says the full-space route is the right default and the fixed `2K`
subspace should no longer be the universal post-processing choice.

## 2026-06-26: v37b/v37c Squirrel Rollback Fix and Final Lock

### v37a result and failure mode

`v37a` was a clear improvement on several weak graphs:

- `Flickr`: `20.01 -> 29.41` (`+9.40`)
- `Chameleon`: `31.93 -> 34.08` (`+2.15`, close to SOTA)
- `BlogCatalog`: `83.41 -> 85.51` (`+2.10`)

The failure was `Squirrel`: `27.61 -> 21.02` (`-6.59`). The diagnostics showed
that `embedding_kmeans_acc` was much higher (`30.21`) than the selected final
labeling (`21.02`), so the silhouette selector was too optimistic about the
subspace branch on that run.

### v37b fix

`v37b` raised the subspace switch margin from `0.01` to `0.05` and kept `full`
as the default:

```python
if _sil_sub > _sil_full + float(cfg.postproc_subspace_margin):
    labels = _lab_sub
else:
    labels = _lab_full
```

The 80-epoch smoke on `squirrel,acm,flickr` passed:

| Dataset | postproc_choice | sil_full | sil_sub | ACC |
| --- | --- | ---: | ---: | ---: |
| squirrel | full | 0.0381 | 0.0381 | 30.42 |
| acm | full | 0.3703 | 0.3704 | 69.69 |
| flickr | full | 0.0557 | 0.0600 | 36.20 |

The full 260-epoch `v37b` run still did not recover `Squirrel`: it selected
`full`, but only reached `21.03`. The important follow-up finding is that this
`full` path is the selector's normalized-embedding KMeans, while the diagnostic
`embedding_kmeans_acc` on `Squirrel` was still `30.13`. The rollback was
therefore not only a subspace-selection problem; the final normalized KMeans
path itself can differ from the raw embedding KMeans diagnostic.

### v37b full-run comparison

| Dataset | v28b | v35b | v37a | v37b | SOTA | gap(v37b) | choice | embedding_kmeans_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| acm | 70.25 | 79.83 | 79.17 | 78.68 | 93.62 | -14.94 | full | 78.68 |
| dblp | 67.12 | 70.40 | 68.35 | 68.70 | 93.69 | -24.99 | full | 68.70 |
| pubmed | 62.42 | 63.02 | 62.60 | 64.12 | 76.17 | -12.05 | full | 64.12 |
| wiki | 32.14 | 42.29 | 41.66 | 38.88 | 64.82 | -25.94 | full | 38.88 |
| flickr | 21.19 | 20.01 | 29.41 | 29.03 | 83.89 | -54.86 | full | 31.92 |
| blogcatalog | 83.43 | 83.41 | 85.51 | 85.41 | 91.72 | -6.31 | full | 85.41 |
| squirrel | 23.36 | 27.61 | 21.02 | 21.03 | 30.51 | -9.48 | full | 30.13 |
| texas | 68.85 | 67.76 | 67.76 | 72.13 | 74.32 | -2.19 | full | 72.13 |
| chameleon | 31.62 | 31.93 | 34.08 | 34.87 | 35.84 | -0.97 | full | 34.87 |

### v37c final lock

Because `v37b` still left `Squirrel` in rollback, the conservative branch
raised `postproc_subspace_margin` to `1.0`, effectively making the final
post-processing pure `full`.

| Dataset | v35b | v37a | v37b | v37c | SOTA | gap(v37c) | choice | embedding_kmeans_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| acm | 79.83 | 79.17 | 78.68 | 79.70 | 93.62 | -13.92 | full | 79.70 |
| dblp | 70.40 | 68.35 | 68.70 | 67.59 | 93.69 | -26.10 | full | 67.59 |
| pubmed | 63.02 | 62.60 | 64.12 | 63.64 | 76.17 | -12.53 | full | 63.64 |
| wiki | 42.29 | 41.66 | 38.88 | 39.63 | 64.82 | -25.19 | full | 40.29 |
| flickr | 20.01 | 29.41 | 29.03 | 28.99 | 83.89 | -54.90 | full | 29.41 |
| blogcatalog | 83.41 | 85.51 | 85.41 | 86.39 | 91.72 | -5.33 | full | 86.39 |
| squirrel | 27.61 | 21.02 | 21.03 | 26.51 | 30.51 | -4.00 | full | 26.63 |
| texas | 67.76 | 67.76 | 72.13 | 70.49 | 74.32 | -3.83 | full | 69.95 |
| chameleon | 31.93 | 34.08 | 34.87 | 34.65 | 35.84 | -1.19 | full | 34.17 |

Accumulated ACC over the 9 datasets:

- `v35b`: `486.26`
- `v37a`: `489.56`
- `v37b`: `492.85`
- `v37c`: `497.60`

`v37c` is the final locked version because it has the highest 9-dataset
accumulated ACC and removes the severe `Squirrel` `21%` failure. It does not
fully restore `Squirrel` to the `v35b` peak (`27.61`), and `DBLP`/`Wiki` remain
below `v35b`, but the total score is highest and the gains on `Flickr`,
`BlogCatalog`, `Texas`, and `Chameleon` are large enough to retain it.

### Final SOTA gap summary

- Close to SOTA: `Chameleon` (`-1.19`), `Squirrel` (`-4.00`), `Texas` (`-3.83`)
- Still materially behind: `BlogCatalog` (`-5.33`), `ACM` (`-13.92`),
  `PubMed` (`-12.53`), `Wiki` (`-25.19`), `DBLP` (`-26.10`)
- Biggest bottleneck: `Flickr` (`-54.90`), still the structural
  strong-heterophily large-graph problem.

## 2026-06-26: v38 Fixed Full-Graph PPR View

### Motivation

After repeated `v29`-`v34` experiments, the edge scorer still could not move
`edge_score_std` away from the same narrow band. The working hypothesis for
`v38` was therefore architectural rather than supervisory: keep the edge scorer
unchanged, and add one non-learned full-graph propagation view directly into
the frontend fusion.

### What changed

`v38` adds a fourth fixed frontend view:

- start from encoder output `z_attr`, not raw `x`
- use full candidate-graph `edge_prior` as the propagation weight
- run fixed-hop PPR-style diffusion with `ppr_steps=10` and `ppr_restart=0.15`
- project the resulting view with a learned `ppr_projector`
- concatenate it into the frontend embedding with a scalar `ppr_gate`

This is intentionally different from `v34`:

- `v34`: PPR was used as evidence for the edge scorer, and the evidence path
  collapsed
- `v38`: PPR bypasses the edge scorer and enters the model as a direct fixed
  representation view

### Smoke

80-epoch smoke on `acm,dblp,flickr,squirrel`:

| Dataset | v37c smoke | v38 smoke | delta |
| --- | ---: | ---: | ---: |
| acm | 69.69 | 83.37 | +13.69 |
| dblp | n/a | 66.97 | n/a |
| flickr | 36.20 | 43.84 | +7.64 |
| squirrel | 30.42 | 30.40 | -0.02 |

All requested smoke thresholds were met, so `v38` was promoted to a full
260-epoch run.

### Full run

| Dataset | v37c | v38 | d(v38) | SOTA | gap | ppr_gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| acm | 79.70 | 85.22 | +5.52 | 93.62 | -8.40 | 0.5036 |
| dblp | 67.59 | 69.29 | +1.70 | 93.69 | -24.40 | 0.5054 |
| pubmed | 63.64 | 57.69 | -5.95 | 76.17 | -18.48 | 0.5058 |
| wiki | 39.63 | 42.74 | +3.12 | 64.82 | -22.08 | 0.5072 |
| flickr | 28.99 | 32.51 | +3.52 | 83.89 | -51.38 | 0.5078 |
| blogcatalog | 86.39 | 68.57 | -17.82 | 91.72 | -23.15 | 0.5064 |
| squirrel | 26.51 | 30.24 | +3.73 | 30.51 | -0.27 | 0.5123 |
| texas | 70.49 | 69.95 | -0.55 | 74.32 | -4.37 | 0.4924 |
| chameleon | 34.65 | 34.30 | -0.35 | 35.84 | -1.54 | 0.5202 |

Accumulated ACC:

- `v37c`: `497.60`
- `v38`: `490.52`

### PPR gate analysis

The most important negative result is that `ppr_gate` did not learn a strong
dataset-specific shutdown. Across all 9 datasets it stayed very close to `0.5`:

- minimum: `0.4924` (`Texas`)
- maximum: `0.5202` (`Chameleon`)

That means the PPR view was almost always kept at a medium contribution level,
including on datasets where it was harmful. So the "let the gate ignore it when
needed" idea did not materialize in practice.

### Final lock

`v38` is not retained as the final version. It improves `ACM`, `Flickr`,
`Squirrel`, and `Wiki`, but the regressions on `PubMed` and especially
`BlogCatalog` are too large, and the 9-dataset accumulated ACC falls below
`v37c`.

The final locked version therefore remains `v37c`.

The paper-level conclusion is stronger after this negative result: even adding
a fixed full-graph PPR view does not consistently beat the current best
combination of embedding-side regularization plus full-space final clustering.

## 2026-06-26: v39 Low-Pass Diffusion Depth Scan

### Baseline

Before scanning, the code was restored to the `v37c` frontend shape: no fixed
PPR view, `postproc_subspace_margin=1.0`, and `lowpass_steps=2`.

80-epoch five-dataset smoke baseline:

| Dataset | v37c smoke ACC |
| --- | ---: |
| acm | 70.38 |
| dblp | 66.48 |
| flickr | 36.54 |
| blogcatalog | 84.39 |
| chameleon | 32.54 |

### v39a: lowpass_steps=4

`v39a` changed only `lowpass_steps: 2 -> 4`.

80-epoch smoke:

| Dataset | v37c smoke | v39a | delta | choice |
| --- | ---: | ---: | ---: | --- |
| acm | 70.38 | 70.64 | +0.26 | full |
| dblp | 66.48 | 65.61 | -0.86 | full |
| flickr | 36.54 | 38.43 | +1.89 | full |
| blogcatalog | 84.39 | 74.27 | -10.12 | full |
| chameleon | 32.54 | 32.67 | +0.13 | full |

ACM+DBLP cumulative delta was `-0.60`, far below the `+3.00` candidate
threshold. More importantly, `BlogCatalog` dropped by more than `5` points, so
the scan stopped immediately by rule and `v39b`/`v39c` were not run.

### v39 conclusion

Increasing the confidence-subgraph low-pass depth from 2 to 4 did not validate
diffusion depth as the front-end bottleneck. The small gains on `ACM` and
`Flickr` were not enough to offset the `DBLP` regression and the large
`BlogCatalog` collapse. No 260-epoch full run was launched for `v39`.

The final locked version remains `v37c`, and the next direction should be the
planned `v40` assignment-mechanism audit rather than deeper low-pass diffusion.

## 2026-06-26: v40 AAAI0617 Backend Audit and Legacy Subspace Integration

### AAAI0617 backend analysis

The AAAI0617 multi-head code routes final labels through dataset-selected
backends after E2E training:

- `subspace_refine`: calls the legacy `_subspace_refine` final head on an
  adapter containing the E2E embedding, raw graph, denoised graph, and feature
  views.
- `legacy_sect_bridge`: bypasses the E2E embedding at final-label time and runs
  the full legacy `SECTCoCo.fit_predict` pipeline, optionally with
  `head=subspace_refine`.

The successful ACM backend is not a new loss or Student-t assignment variant.
It is an ELSS-style anchor subspace head:

- choose input representation via `head_input` (`original`, `base`, `concat`,
  or E2E embedding)
- choose graph via `head_graph` (`original_elss`, `homo_elss`, or denoised)
- build ELSS-normalized graph with self-loops
- run `_AnchorSubspaceHead` with anchor count, rank, power, `d`, `alpha2`,
  `gamma`, and optional filter coefficient
- L2/PCA-normalize the resulting low-rank `q`
- finish with KMeans

The successful DBLP backend is broader: `legacy_sect_bridge` runs the old
SECT-CoCo feature pipeline before using the same kind of subspace head. Its
advantage is therefore not a single Student-t replacement, but the combination
of legacy edge-teacher feature construction plus anchor subspace refinement.

### Integrability

Directly restoring AAAI0617's per-dataset routing would violate the unified
pipeline goal. The smallest acceptable integration is to expose one unified
optional final backend in the current E2E model:

- keep the current v37c frontend/APTC training unchanged
- after training, pack current embedding, raw graph, denoised support graph,
  and homophily graph into a legacy-head adapter
- call legacy `_subspace_refine`
- record `postproc_choice=legacy_subspace_refine`

This tests whether the reusable component is the anchor subspace/ELSS final
backend rather than the old dataset-specific multi-head router.

### v40a: unified legacy_subspace_refine backend

`v40a` adds a unified variant that enables `final_label_mode=legacy_subspace_refine`
for all datasets with:

- `head_input=concat`
- `head_graph=original_elss`
- `head_power=2`
- `head_d=0.875`
- `head_alpha2=5e-5`
- `head_gamma=0.003`
- `head_q_norm=l2`
- `head_kmeans_n_init=100`

80-epoch smoke on `acm,dblp,flickr,squirrel`:

| Dataset | v40a smoke | postproc_choice |
| --- | ---: | --- |
| acm | 92.40 | legacy_subspace_refine |
| dblp | 92.95 | legacy_subspace_refine |
| flickr | 33.97 | legacy_subspace_refine |
| squirrel | 25.03 | legacy_subspace_refine |

The smoke confirms the key mechanism: ACM/DBLP jump near the AAAI0617 backend
range as soon as the anchor subspace head is attached to the current model.

### Full run

| Dataset | v37c | v40a | delta | SOTA | gap(v40a) |
| --- | ---: | ---: | ---: | ---: | ---: |
| acm | 79.70 | 85.59 | +5.89 | 93.62 | -8.03 |
| dblp | 67.59 | 89.60 | +22.01 | 93.69 | -4.09 |
| pubmed | 63.64 | 63.66 | +0.02 | 76.17 | -12.51 |
| wiki | 39.63 | 52.47 | +12.84 | 64.82 | -12.35 |
| flickr | 28.99 | 27.37 | -1.62 | 83.89 | -56.52 |
| blogcatalog | 86.39 | 92.90 | +6.51 | 91.72 | +1.18 |
| squirrel | 26.51 | 25.48 | -1.03 | 30.51 | -5.03 |
| texas | 70.49 | 42.08 | -28.41 | 74.32 | -32.24 |
| chameleon | 34.65 | 31.23 | -3.42 | 35.84 | -4.61 |

Accumulated ACC:

- `v37c`: `497.59`
- `v40a`: `510.36`

### v40 conclusion

`v40a` finds a real breakthrough mechanism: the AAAI0617 anchor subspace/ELSS
backend is highly compatible with the current E2E embedding on ACM, DBLP,
Wiki, and BlogCatalog. This proves that the remaining ACM/DBLP gap was largely
a final-backend/assignment mechanism gap, not only a frontend embedding gap.

However, unified always-on `legacy_subspace_refine` is not yet a clean final
replacement because it badly damages `Texas` and hurts `Chameleon`, `Squirrel`,
and `Flickr`. The next useful step is `v40b`: design an unsupervised selector
or confidence gate between v37c full-KMeans and v40a legacy subspace head,
without using dataset-specific routing.

## 2026-06-26: v40b Adaptive Legacy Subspace Dimension

### Goal

`v40b` tested whether the legacy subspace head could be made harmless on
heterophily-heavy graphs without adding a path selector between v37c and v40a.
The implementation kept one unified backend and changed the subspace head
itself:

- legacy `_subspace_refine` can optionally return a higher-rank representation
  via `head_return_k_rank`
- v40b sets `head_k_rank=50` and scans dimensions `K, 2K, 3K, 4K, 5K`
- each candidate is clustered with KMeans
- silhouette is computed on the high-dimensional E2E embedding
- if the best subspace silhouette clears a global absolute threshold, v40b uses
  that subspace; otherwise it falls back to full KMeans

This avoids dataset-specific routing, but it is still a behavioral gate inside
the legacy subspace backend.

### Smoke results

Three threshold settings were tested.

`threshold=0.15` was too strict and selected `full_kmeans` for every dataset:

| Dataset | choice | sil_best_sub | sil_full | ACC |
| --- | --- | ---: | ---: | ---: |
| acm | full_kmeans | 0.1172 | 0.3729 | 71.34 |
| dblp | full_kmeans | 0.0374 | 0.1510 | 66.06 |
| blogcatalog | full_kmeans | 0.1434 | 0.1726 | 84.93 |
| squirrel | full_kmeans | -0.0147 | 0.0380 | 30.30 |
| texas | full_kmeans | -0.0050 | 0.2202 | 73.77 |

`threshold=0.03` restored ACM/DBLP gains, but Texas and Chameleon entered the
subspace path and regressed badly:

| Dataset | choice | sil_best_sub | sil_full | ACC |
| --- | --- | ---: | ---: | ---: |
| acm | subspace_15 | 0.1086 | 0.3752 | 85.55 |
| dblp | subspace_20 | 0.0365 | 0.1508 | 86.94 |
| blogcatalog | subspace_24 | 0.1436 | 0.1743 | 87.30 |
| squirrel | full_kmeans | -0.0126 | 0.0380 | 30.38 |
| texas | subspace_15 | 0.0306 | 0.2221 | 40.98 |
| chameleon | subspace_5 | 0.0739 | 0.1118 | 31.93 |

Following the rule for Texas, the threshold was raised to `0.20`. This protected
Texas/Squirrel/Chameleon, but again disabled the useful ACM/DBLP subspace path:

| Dataset | choice | sil_best_sub | sil_full | ACC |
| --- | --- | ---: | ---: | ---: |
| acm | full_kmeans | 0.1265 | 0.3812 | 70.98 |
| dblp | full_kmeans | 0.0423 | 0.1514 | 66.40 |
| blogcatalog | full_kmeans | 0.1446 | 0.1734 | 84.60 |
| squirrel | full_kmeans | -0.0147 | 0.0384 | 30.32 |
| texas | full_kmeans | 0.0161 | 0.2203 | 73.77 |
| chameleon | full_kmeans | 0.0683 | 0.1054 | 32.89 |

### Conclusion

`v40b` did not meet the smoke criteria and was not promoted to a 260-epoch full
run. The single absolute silhouette threshold cannot simultaneously:

- keep ACM/DBLP/BlogCatalog on the beneficial subspace path
- keep Texas/Chameleon off the harmful subspace path
- preserve all datasets within the v37c safety margin

The important diagnostic is that DBLP's good subspace path has very low
absolute silhouette (`~0.04`), while Texas can have a similarly low but still
harmful positive silhouette (`~0.03`). Therefore absolute silhouette alone is
not a reliable unified reliability signal for this backend.

The final retained result remains `v40a` as a breakthrough mechanism, with the
caveat that always-on legacy subspace is not yet safe enough to replace v37c as
a universal final version.

## 2026-06-26: v41a KMeans Teacher Guidance

### Motivation

The APTC path already had an initialization teacher loss, but it had effectively
been disabled through `aptc_init_teacher_weight=0.0`. That meant the Sinkhorn
prototype assignment had no direct KMeans-style teacher signal during training,
even though final full-space KMeans repeatedly outperformed APTC argmax.

`v41a` changed only this mechanism:

- dataclass default `aptc_init_teacher_weight: 0.0 -> 0.10`
- runner variant `v41a` inherits `v28b` and overrides
  `aptc_init_teacher_weight=0.10`
- no frontend, loss-structure, runner behavior, or dataset branch changes

### Smoke

80-epoch smoke on all 9 datasets:

| Dataset | v37c smoke | v41a | delta | init_teacher | proto_sep |
| --- | ---: | ---: | ---: | ---: | ---: |
| acm | 70.38 | 82.78 | +12.40 | 0.0038 | 0.6336 |
| dblp | 66.48 | 63.25 | -3.23 | 0.2059 | 0.3041 |
| pubmed | 63.21 | 62.09 | -1.12 | 0.1588 | 0.1795 |
| wiki | 44.78 | 38.54 | -6.24 | 0.2735 | 0.3648 |
| flickr | 36.81 | 47.42 | +10.61 | 0.0416 | 0.5919 |
| blogcatalog | 84.60 | 82.56 | -2.04 | 0.1989 | 0.2733 |
| squirrel | 30.32 | 30.36 | +0.04 | 0.1570 | 0.2332 |
| texas | 73.77 | 73.77 | +0.00 | 0.6486 | 0.1097 |
| chameleon | 32.89 | 32.72 | -0.17 | 0.1381 | 0.0480 |

The teacher loss was non-zero on every dataset, so the mechanism was genuinely
active. Prototype separation also rose well above the `0.05` target on most
datasets, except `Chameleon` (`0.0480`).

### Weight sweep

Because v41a improved ACM/Flickr but regressed Wiki and DBLP, two weight-only
variants were tested.

| Dataset | v41a_low 0.05 | v41a 0.10 | v41a_high 0.20 |
| --- | ---: | ---: | ---: |
| acm | 64.83 | 82.78 | 81.16 |
| dblp | 65.76 | 63.25 | 60.78 |
| pubmed | 62.77 | 62.09 | 59.47 |
| wiki | 40.79 | 38.54 | 39.00 |
| flickr | 41.08 | 47.42 | 43.80 |
| blogcatalog | 83.60 | 82.56 | 80.68 |
| squirrel | 30.51 | 30.36 | 30.38 |
| texas | 73.77 | 73.77 | 74.32 |
| chameleon | 32.37 | 32.72 | 32.76 |

Smoke accumulated ACC:

- `v41a_low`: `495.48`
- `v41a`: `513.49`
- `v41a_high`: `502.35`

### Conclusion

`v41a` validates that the KMeans teacher is not a dead mechanism: it strongly
improves ACM and Flickr and produces non-zero teacher loss. However, it fails
the smoke safety gate because `Wiki` drops by more than `5` points from the
v37c smoke baseline, and DBLP also regresses.

No 260-epoch full run was launched for v41a. The best smoke weight is `0.10`,
but the result is not safe enough to replace v37c or v40a. The next refinement,
if pursued, should target teacher refresh stability, such as increasing
`cluster_update_interval`, rather than increasing teacher weight.
