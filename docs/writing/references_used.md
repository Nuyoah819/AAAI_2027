# References Used for CSTC

This file summarizes how the final proposal, **Concordant Spectral Topology Contraction (CSTC)**, is grounded in the local reference library. It does not claim that CSTC has already achieved empirical superiority; it records borrowed ideas, motivations, and technical differences for the AAAI 2027 paper plan.

## Final Method Anchor

CSTC targets attributed graph clustering under attribute-structure mismatch. The method keeps Differentiable Topological Contraction as the core innovation and organizes the model around four technical moves:

1. estimate edge-level attribute-structure concordance;
2. contract reliable homophilic topology while isolating heterophilic and ambiguous associations;
3. decouple low-frequency smoothing and high-frequency correction through soft contracted topologies;
4. produce cluster posteriors through reliability-calibrated transport assignment.

## Literature Mapping

| Reference | Core idea | Why it matters for CSTC | How CSTC uses it | Difference from CSTC |
| --- | --- | --- | --- | --- |
| What is Missing in Homophily: Graph Homophily Disentanglement in GNNs, NeurIPS 2024 | Homophily should be disentangled into label, structural and feature aspects; a single homophily ratio cannot explain GNN behavior | Directly motivates the project problem: attribute-feature similarity and structural connectivity may disagree | CSTC models attribute-structure mismatch at edge level through concordance scores and treats mismatch as a source of noisy associations | The reference is mainly an analysis/metric study for GNN behavior, while CSTC is an end-to-end clustering algorithm |
| Understanding Heterophily in Graph Neural Networks, ICML 2024 | Explains when heterophily helps or hurts GNNs, often through structural and feature interactions | Provides theoretical background for why heterophilic edges are not uniformly harmful | CSTC avoids binary edge deletion and instead produces soft homo/hetero/ambiguous masks | CSTC focuses on unsupervised graph clustering and differentiable topology contraction rather than node classification theory |
| Unsupervised Graph Representation Learning with Edge Heterophily Discrimination, AAAI 2023 | Uses edge heterophily discrimination to improve unsupervised graph representation | Closest conceptual prior for edge-level heterophily awareness without labels | CSTC inherits the importance of edge-level discrimination but converts it into a differentiable contraction and spectral filtering operator | CSTC is designed for clustering, uses soft topology contraction and transport assignment, and does not simply discriminate edges |
| All Roads Lead to Rome: Exploring Edge Distribution Shift in Heterophilic Graph Learning, IJCAI 2025 | Frames heterophilic edge learning as an edge-level distribution/OOD problem | Supports treating heterophilic noise as an edge distribution problem rather than a node-only problem | CSTC uses edge concordance as a continuous reliability signal and handles uncertain edges through soft masks | The reference targets heterophilic GNN node classification; CSTC uses label-free clustering and does not require supervised edge labels |
| Integrating Co-training and Edge Discrimination for Heterophilic GNNs, AAAI 2025 | Combines edge discrimination with co-training signals | Motivates jointly improving representations and edge semantics | CSTC jointly optimizes edge confidence, spectral views and clustering posterior | CSTC avoids supervised co-training assumptions and keeps one unified clustering objective |
| Relation-Aware Learning in Heterogeneous Graphs with Homophily-Heterophily Separation, KDD 2025 | Separates homophily and heterophily relations in heterogeneous graphs | Reinforces the need for relation-type separation rather than raw adjacency smoothing | CSTC performs relation separation on candidate edges through DTC masks | CSTC handles attributed graph clustering in a homogeneous benchmark setting and uses differentiable contraction rather than relation-type modeling |
| Homophily-related Adaptive Hybrid Graph Filtering for Multi-view Graph Clustering, AAAI 2024 | Uses adaptive graph filtering for clustering under varying homophily | Closest filtering-related clustering reference | CSTC uses low-pass and high-pass views but gates them by learned edge concordance and topology contraction | CSTC emphasizes edge-level mismatch and DTC, not only view-level adaptive filtering |
| Disentangling Homophily and Heterophily in Multi-modal Graph Clustering, ACMMM 2025 | Separates homophilic and heterophilic information in multi-modal clustering | Supports the idea that clustering benefits from explicitly separating signal types | CSTC separates low-frequency homophilic smoothing and high-frequency heterophilic correction | CSTC is not multi-modal-specific and treats contraction as a trainable topology operator |
| Compactness and Consistency: A Joint Framework for Deep Graph Clustering, ICLR 2026 | Uses compactness and consistency to stabilize deep graph clustering | Motivates cluster compactness and posterior consistency losses | CSTC includes compactness as a supporting loss and avoids overclaiming it as novel | CSTC's novelty is the contracted topology and mismatch-aware spectral pathway |
| Explicit Low-Rank Structured Subspace Learning for Fast Attributed Graph Clustering, IJCAI 2026 | Low-rank structured subspaces can be effective for attributed graph clustering | Explains why spectral/subspace anchors helped some local variants | CSTC may use low-rank compactness only as a reliability-calibrated auxiliary signal | CSTC does not rely on unconditional low-rank anchor imitation or legacy subspace heads |
| Diffusion-based Graph-Agnostic Clustering, WWW 2025 | Diffusion can produce graph-agnostic clustering representations | Motivates diffusion-like smoothing as a clustering primitive | CSTC uses diffusion-style low-pass propagation on contracted reliable topology | CSTC is not graph-agnostic; it explicitly models graph topology reliability |
| Optimal Transport Graph Contrastive Learning for Heterophilic Text-Attributed Graphs, AAAI 2026 | Uses optimal transport and contrastive learning for heterophilic text-attributed graphs | Supports transport-style objectives under heterophily | CSTC uses Sinkhorn-style balanced transport assignment for cluster posteriors | CSTC's transport is a clustering readout integrated with DTC rather than a contrastive text-graph objective |
| NeurIPS 2025 Hybrid Collaborative Augmentation and Contrastive Sample-Adaptive Differential Awareness for Robust Attributed Graph Clustering | Robust graph clustering via augmentation and contrastive differential awareness | Motivates sample-adaptive robustness and graph clustering baselines | CSTC borrows the need for sample/edge-adaptive reliability, not the augmentation recipe | CSTC focuses on topology contraction and edge-frequency decoupling rather than augmentation-heavy contrastive learning |
| Graph Homophily Enhancer: Rethinking Discrete Features in Heterophilic Graph Learning, ICLR 2026 | Enhances homophily signals in heterophilic settings | Motivates reconstructing more usable topology from noisy observed edges | CSTC reconstructs a soft contracted topology instead of directly enhancing discrete features | CSTC targets continuous attributed clustering and keeps heterophilic correction rather than forcing all edges into homophily |
| Reconciling Homophily and Heterophily in GNNs via Self-supervised Node Encoding, ICLR 2026 | Self-supervised encodings can reconcile homophily and heterophily | Supports using self-supervised training signals under label scarcity | CSTC can use self-supervised reconstruction, contrastive and consistency losses as auxiliary training terms | CSTC's main mechanism is edge-topology contraction, not node encoding alone |
| Clarifying Confused Nodes via Disentangled Learning, TPAMI 2025 | Handles confused nodes by disentangling latent factors | Motivates treating ambiguous nodes/edges explicitly | CSTC keeps an ambiguous edge region rather than forcing every edge into hard homo/hetero bins | CSTC applies this idea to edge topology and clustering posteriors |
| Attribute-Missing Multi-view Graph Clustering, CVPR 2025 | Robust clustering under incomplete views/attributes | Supports robustness to imperfect attributes | CSTC treats attribute-structure mismatch as a structural noise source, not just missing views | CSTC does not center on missing attributes |
| Comprehensive Benchmark for Text-Attributed Heterogeneous Graphs, AAAI 2026 | Benchmarks text-attributed heterogeneous graph methods | Useful for positioning and future benchmark expansion | CSTC should report fair graph clustering protocols and avoid cherry-picking | CSTC currently uses the local nine-dataset attributed graph protocol |
| Efficient High-Quality Clustering for Large-scale Bipartite Graphs, SIGMOD 2024 | Scalable clustering on bipartite graphs | Useful for scalability and clustering evaluation framing | CSTC can cite it when discussing clustering scalability and large graphs | CSTC is not a bipartite-graph-specific method |

## Design Implications

The references support five design choices:

1. Use edge-level evidence because node-level homophily is insufficient for mismatch-heavy graphs.
2. Keep heterophilic information as a different spectral component instead of deleting it.
3. Use soft differentiable masks to avoid brittle thresholding and non-differentiable pruning.
4. Treat compactness and transport as clustering readout mechanisms, not as substitutes for topology modeling.
5. Evaluate with diagnostics that expose edge mask quality, frequency separation and posterior drift.

## Claims That Need Future Verification

The following claims are reasonable hypotheses but still require experiments:

| Claim | Needed evidence |
| --- | --- |
| CSTC improves clustering under attribute-structure mismatch | Main ACC/NMI/ARI tables on nine datasets against graph clustering and heterophily-aware baselines |
| Differentiable topology contraction is the decisive component | Ablation removing DTC or replacing it with hard thresholding / raw adjacency |
| Edge-local spectral decoupling handles heterophilic noise | Diagnostics on high-pass response, homo/hetero masks and edge homophily groups |
| Reliability-calibrated transport reduces late drift | Training curves and ablations against unconditional anchor/self-distillation |
| The unified pipeline avoids dataset-specific behavior | Code audit, diagnostic table showing identical forward/loss/final assignment path |
