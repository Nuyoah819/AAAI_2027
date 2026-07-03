# AAAI 2027 Graph Clustering Idea Review and Selection

本文档使用 `ccf-idea-reviewer` 的严格评审口径，对 `AAAI0622` 中的候选路线进行机制级归纳、评分和方案确定。评审依据来自本地草稿、实现、运行 verdict、失败分析、`频率解耦.md`、`CRITICAL_RED_LINES.md` 与 `reference_md`。本文不引入新的实验结果，不基于测试集事后择优，也不把任何带版本号的草稿名作为最终模型名。

## 1. 评审边界

目标场景是无监督/自监督的 attributed graph clustering。核心科学问题是：属性相似性与图结构连接并不总是一致，二者不匹配会把异配边、跨类伪关联和结构污染混入消息传递，造成频率混叠与聚类后验漂移。

必须保留的核心机制是 Differentiable Topological Contraction，即用可学习、可微、统一的边置信机制将候选边软划分为 homophilic、heterophilic 和 ambiguous 三类，并把该划分直接接入后续频谱过滤和聚类优化。

红线约束如下：不允许数据集专用模块、数据集路由、非端到端核心后处理、测试集事后择优、随机重启择优、使用标签或指标构造训练选择器。

## 2. 候选机制家族

| ID | 机制家族 | 代表材料 | 核心主张 | 主要证据/风险 |
| --- | --- | --- | --- | --- |
| C1 | Edge confidence + DTC + frequency-aware filtering | `README.md`, `README_Algorithm_Evolution.md`, `core/e2e/sect_coco_e2e.py`, V44-V49 系列 | 用属性-结构一致性估计边置信，通过可微拓扑收缩分离同配/异配/困难边，再做低通与高通解耦 | 符合问题主线和红线；但早期后验读出和高频响应稳定性不足，需要更强的边局部频谱约束 |
| C2 | APTC / Sinkhorn / assignment-flow 统一聚类头 | `AdaptivePosteriorTransportHead`, v17-v28 记录 | 用统一的 transport posterior 替代旧多后端路由 | 符合统一 pipeline；但曾出现 embedding 有信号而 posterior 读不出的瓶颈，不能单独作为论文创新核心 |
| C3 | Spectral compactness anchor / low-rank subspace guidance | V50A-V59A | 低秩图平滑 anchor 能给同配图和部分混合图提供强聚类几何 | 在 ACM/DBLP/BlogCatalog 上信号强；但 V50A/V57A/V59A 暴露 anchor reliability 与 long-run drift，不能作为强制 teacher |
| C4 | Self-distillation / phase guard / teacher stabilization | V60A-V63A | 以固定阶段的 q posterior 防止后期漂移 | 能解释 V61/V62 的 drift guard 现象；但 V62 full run 未解决 Flickr/Squirrel，V63 仅 preregistration，无结果，不能作为核心贡献 |
| C5 | Legacy subspace / selector-style head | V40A/V40B, legacy code | 旧 subspace head 在部分数据集极强 | 性能诱人但存在红线风险：容易成为隐式后端选择器，Texas 等异配图安全性差，不宜作为最终论文核心 |
| C6 | Proposed synthesis: Concordant Spectral Topology Contraction | 本文综合方案 | 将 DTC 作为核心，把 attribute-structure mismatch 显式建模为 edge concordance；用边局部频谱解耦和可靠性校准 transport 完成统一聚类 | 保留核心创新，吸收 C1/C2/C3 的可解释部分，弱化 C4 为训练稳定器而非最终选择器，避免 C5 的红线风险 |

## 3. 评分与排名

评分尺度为 1-5。加权总分采用 `ccf-idea-reviewer` rubric：科学问题契合度、创新性、理论可行性、实验可实现性与 AAAI 潜力共同判断。置信度反映本地材料覆盖程度，不等同于分数。

| Rank | Candidate | Problem Fit | Novelty | Theory | Feasibility | AAAI Potential | Weighted Score | Confidence | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | C6 Concordant Spectral Topology Contraction | 5.0 | 4.2 | 4.1 | 4.1 | 4.3 | 4.35 | 4 | Accept-to-develop |
| 2 | C1 DTC + frequency-aware front-end | 4.6 | 4.0 | 3.8 | 4.3 | 4.0 | 4.10 | 4 | Fuse into winner |
| 3 | C2 APTC / transport readout | 3.8 | 3.4 | 3.7 | 4.2 | 3.6 | 3.72 | 4 | Retain as unified readout, not headline |
| 4 | C3 Spectral compactness anchor | 3.6 | 3.5 | 3.3 | 3.8 | 3.4 | 3.52 | 4 | Keep only as reliability-calibrated auxiliary signal |
| 5 | C4 Self-distillation / phase guard | 3.3 | 3.0 | 3.2 | 3.6 | 3.0 | 3.22 | 3 | Optional stability regularizer after validation |
| 6 | C5 Legacy subspace / selector head | 2.5 | 2.6 | 2.4 | 3.2 | 2.3 | 2.58 | 4 | Reject as final route |

## 4. Strict Review Notes

### Field expert

The strongest field-level contribution is not another graph clustering head. It is the formulation of clustering failure as attribute-structure mismatch producing noisy heterophilic associations. This matches recent heterophily literature that separates feature, structural and edge-level homophily, and it is more specific than a generic "robust graph clustering" claim.

Main risk: the method must not look like a pile of modules accumulated from many failed variants. The final paper should center on one insight: edge-level concordance controls which topology should be contracted, filtered, or treated as a high-frequency correction.

### Method expert

DTC is the most defensible mechanism because it produces a continuous edge partition and keeps gradients inside the model. The risky parts are static spectral anchors and teacher snapshots: they can help optimization, but as headline mechanisms they invite reviewer criticism because they resemble self-training heuristics.

Repair condition: express spectral/anchor components as reliability-calibrated auxiliary regularizers, not as final-label selectors or global teachers.

### Experiment expert

The existing framework already supports nine datasets, ACC/NMI/ARI, diagnostics, fixed seed protocol and multiple baselines. The key experiment risk is not feasibility but integrity: historical result logs show many variants. The final paper must preregister the selected model before full reporting and must not select between versions using final test metrics.

Required evidence: module ablations for edge concordance, DTC, edge-local spectral decoupling, transport assignment, and reliability calibration; analysis by edge homophily/mismatch level; diagnostics on homo/hetero/hard mass and high-pass response.

### AC / venue expert

AAAI reviewers are likely to accept a coherent graph clustering method with a clear mechanism and strong heterophily analysis. They are unlikely to reward a post-hoc engineering chain. The winning route should be written as a single model designed from the mismatch problem, not as a history of versions.

### Skeptical prior-art expert

Closest-work risk comes from heterophily edge discrimination, homophily disentanglement, graph filtering, contrastive graph clustering, compactness/consistency clustering and graph structure learning. The novelty delta must be stated narrowly: the paper is not the first to identify heterophilic edges or use graph filters; it proposes a differentiable topology contraction layer for graph clustering that converts attribute-structure concordance into trainable low/high-frequency topology operators.

## 5. Selected Model

Final model name:

```text
Concordant Spectral Topology Contraction for Attributed Graph Clustering
```

Suggested short name:

```text
CSTC
```

The model is a unified end-to-end graph clustering framework:

```text
attribute encoder
-> attribute-structure edge concordance
-> differentiable topological contraction
-> concordance-gated low/high-frequency filtering
-> reliability-calibrated transport assignment
-> clustering posterior
```

Core module names:

| Module | English name | Role |
| --- | --- | --- |
| M1 | Attribute-Structure Concordance Estimator | Estimates whether an edge is semantically and structurally compatible |
| M2 | Differentiable Topological Contraction | Softly contracts reliable homophilic edges and separates heterophilic/ambiguous edges |
| M3 | Concordance-Gated Spectral Decoupling | Performs low-pass aggregation on contracted reliable topology and signed high-pass correction on heterophilic topology |
| M4 | Reliability-Calibrated Transport Assignment | Produces balanced cluster posteriors through Sinkhorn-style transport and topology-aware refinement |
| M5 | Phase-Stable Compactness Regularizer | Optional, label-free training stabilizer; not a selector and not final-label source |

## 6. Why Other Routes Are Not Selected

Legacy subspace heads are rejected as the final mechanism despite strong partial results because they risk reintroducing the very multi-backend behavior the project explicitly forbids. A paper cannot defend a "unified" model if its strongest results depend on an implicit selector or a legacy readout path.

Static spectral anchors are not rejected entirely. They are useful as evidence that compact low-rank geometry matters, but the failure analyses show that stronger anchor agreement can coexist with worse ACC on heterophilic graphs. Therefore CSTC can borrow the idea of compactness, but only through reliability-calibrated regularization and never as unconditional assignment imitation.

Self-distillation and phase locks are also not selected as the central contribution. They are training stabilizers for late drift, not the scientific answer to attribute-structure mismatch. They may appear as an implementation detail or future ablation if preregistered, but the method story should not depend on them.

## 7. Development Recommendation

Build and present CSTC around the following AAAI-level claims:

1. Attribute-structure mismatch can be detected at edge level by combining attribute similarity, raw feature consistency, degree/role similarity and mismatch gaps.
2. Differentiable topology contraction turns uncertain edge semantics into a trainable soft topology rather than a hard edge deletion rule.
3. Low-frequency smoothing should be applied only on contracted reliable topology, while heterophilic associations should contribute through signed high-frequency correction.
4. Balanced transport assignment should read cluster structure from the contracted spectral representation without dataset-specific heads or final-label selection.
5. Reliability calibration should limit auxiliary compactness/teacher signals to nodes and edges where the graph, feature and posterior evidence agree.

Recommendation: `accept-to-develop`, with high development potential and medium-high current readiness. The main unresolved risks are empirical strength on Flickr/Squirrel and novelty positioning against edge-discrimination heterophily literature.
