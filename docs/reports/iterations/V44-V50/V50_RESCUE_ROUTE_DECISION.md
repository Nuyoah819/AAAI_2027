# V50 Rescue Route Decision

This document records the rescue-route decision after the V43B-V49A sequence.
It uses `ccf-idea-optimizer` exploratory mode: optimize the research direction
before implementing another variant. No V50 code has been implemented and no
V50 experiment has been run.

## 1. Current Failure Boundary

The latest mechanism chain established:

| Version | Route | Main finding |
| --- | --- | --- |
| V43B | direct embedding/conflict pressure | gate too broad; frontend damaged |
| V44B | pre-HP frequency response | response measurable, but coupling incomplete |
| V45A | edge-local frequency coupling | response gap/corr fail on 3/3 |
| V46A | hard-band scalar penalty | no collapse, but hard band not solved |
| V47A | posterior-guided hard-band CE | targets exist, but band/ACC worsen |
| V48A | topology dynamics audit | masks move, but ACM/Flickr move wrong |
| V49A | orientation/clarity simplex | usage safe, but ACM/Flickr still wrong-direction |

V49A final boundary:

```text
Changing mask geometry is not enough when the orientation signal itself remains
under-specified.
```

After about 40 iterations, continuing to add topology-mask losses or remap the
same edge logit is no longer a credible primary rescue route.

## 2. Baseline Code Signals

### 2.1 S2CAG

Read from:

```text
IJCAI2026/Experiments/comparison/S²CAG/S²CAG
```

Core mechanism:

```text
TF-IDF / feature normalization
graph propagation
randomized SVD or modularity spectral decomposition
SNEM rounding
```

Observed local result records:

| Dataset | Strong local record |
| --- | --- |
| ACM | ACC 93.65 / NMI 75.94 / ARI 81.97 |
| DBLP | ACC 93.54 / NMI 78.76 / ARI 84.29 |
| Wiki | ACC 64.37 / NMI 55.12 / ARI 44.94 |

Interpretation:

```text
The strongest signal is a stable graph-smoothed spectral subspace, not a learned
edge-mask loss.
```

### 2.2 ELSS

Read from:

```text
AAAI0610/ELSS_code
reference_md/Explicit_Low-Rank_Structured_Subspace_Learning_for_Fast_Attributed_Graph_Clustering_IJCAI2026.md
```

Core mechanism:

```text
homophily-aware graph filtering
PageRank-guided anchor sampling
Nyström low-rank approximation
explicit nonlinear low-rank subspace
KMeans on the resulting spectral embedding
```

The code contains dataset-specific hyperparameters, so it cannot be copied as a
final unified method. But the mechanism is highly relevant: it converts
graph-smoothed attributes into an explicit low-rank clustering basis.

### 2.3 AAAI0617 Core

Read from:

```text
AAAI0617/CODE/core/e2e/sect_coco_e2e.py
```

Important historical finding:

```text
legacy/anchor subspace heads were the strongest mechanisms, but dataset-routed
final heads violated the unified-pipeline red line.
```

V40A confirmed this locally in the current project:

```text
legacy subspace refine always-on strongly improved ACM/DBLP/Wiki/BlogCatalog,
but collapsed Texas.
```

Therefore, the rescue should not return to a post-processing selector. It
should internalize the useful subspace principle into the unified training path.

## 3. Reference-Literature Signal

Relevant local reference files point to the same family:

```text
Explicit Low-Rank Structured Subspace Learning for Fast Attributed Graph Clustering
Compactness and Consistency: A Joint Framework for Deep Graph Clustering
Clarifying Confused Nodes via Disentangled Learning
```

Common transferable ideas:

- low-rank compactness removes redundancy and noise;
- local/global view consistency is often more stable than contrastive pressure;
- confusion/ambiguity should be measured as difficulty, not blindly punished;
- strong clustering often emerges from a compact spectral basis plus stable
  assignment, rather than from edge-level separation losses.

## 4. Rejected Next Moves

Do not continue:

- V49A temperature, initialization, or mapping sweep;
- another hard-edge posterior teacher loss;
- another direct hard-band penalty;
- another conflict/frequency response pressure;
- another silhouette or post-processing selector;
- copying S2CAG/ELSS as a non-differentiable final head;
- using dataset-specific ELSS parameters.

## 5. Rescue Insight

The current model spends too much capacity trying to make an edge-level
topology mask become semantically correct. The high-performing baselines suggest
a different bottleneck:

```text
the final clustering basis is not sufficiently anchored to a stable low-rank
graph-attribute subspace.
```

The rescue route should keep the frontend as a representation generator, but
replace the failed mask-loss story with a compact spectral-subspace anchoring
story:

```text
learn embeddings whose cluster readout agrees with a differentiable low-rank
subspace induced by graph-smoothed attributes.
```

This moves the research problem from:

```text
Can topology masks classify edges into homo/hetero/hard?
```

to:

```text
Can a unified end-to-end model preserve the strong spectral-subspace clustering
signal while still learning adaptive frontend representations?
```

## 6. Recommended V50 Route

Recommended name:

```text
v50a_spectral_compactness_anchor
```

Core mechanism:

```text
Build a fixed or stop-gradient spectral compactness anchor from graph-smoothed
features, then train the APTC/embedding readout to align with this anchor through
a unified differentiable objective.
```

First version should be conservative:

- no dataset-specific parameters;
- no final post-processing selector;
- no labels;
- no S2CAG/ELSS final KMeans as the reported output;
- no sweeping over propagation order, anchor count, or rank;
- no replacement of the existing frontend;
- diagnostics must report whether the spectral anchor is actually better than
  current embedding/posterior on the first-stage datasets.

## 7. Candidate Mechanism Forms

### Candidate A: Stop-Gradient Spectral Teacher For Posterior Alignment

Compute a unified spectral basis from graph-smoothed input features, convert it
to a soft cluster distribution, and use it as a stop-gradient teacher for
`q_refined` or `q_main`.

Pros:

- closest to S2CAG/ELSS strength;
- minimal changes to frontend;
- can be audited through teacher quality and posterior gap.

Risk:

- may repeat the teacher-target problem if the spectral teacher is bad on
  heterophily datasets;
- must use uniform hyperparameters and strict gates.

### Candidate B: Differentiable Compactness Regularizer In Spectral Coordinates

Do not create pseudo labels. Instead, compute a spectral anchor space and
regularize the learned embedding so its pairwise or prototype geometry matches
the anchor geometry.

Pros:

- less brittle than hard teacher labels;
- aligns geometry rather than assignments.

Risk:

- pairwise alignment can be expensive;
- if too weak, it may not recover ACC.

### Candidate C: Local/Global Compactness Consistency

Borrow the CoCo idea: produce local and global filtered views, project both into
a compact low-rank space, and align their anchor-similarity distributions.

Pros:

- consistent with graph clustering literature;
- avoids edge-level homo/hetero target ambiguity.

Risk:

- larger implementation scope;
- needs careful evidence to distinguish from existing CoCo-style methods.

## 8. Unique Recommendation

Choose Candidate A first:

```text
v50a_spectral_compactness_anchor
```

Reason:

S2CAG and ELSS show that explicit spectral/low-rank bases are strong on exactly
the datasets where the current end-to-end route struggles most, especially ACM
and DBLP. V40A also showed that subspace heads are powerful inside this project,
but unsafe as a post-processing head. The most direct rescue is therefore to
move the subspace signal from final head selection into a unified training
anchor.

This is not a claim that V50A will improve performance. It is a better
scientific next question:

```text
Is the missing ingredient a stable low-rank clustering basis rather than another
topology-mask calibration loss?
```

## 9. No-Fabrication Status

All V49A numbers come from `V49A_FIRST_SMOKE_VERDICT.md`. S2CAG values cited
above come from local `*_batch.txt` records. ELSS and CoCo mechanisms are
summarized from local code and local reference markdown files. No V50 result is
reported.
