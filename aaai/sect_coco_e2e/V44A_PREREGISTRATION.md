# v44a_conflict_coupled_topology Preregistration

本文档是 v43a/v43b 失败复盘后的 v44a 预注册方案。它只定义下一机制、诊断指标、实验 gate 和停止条件；不包含任何新实验结果。

## 1. 当前结论

v43a/v43b 的共同结论不是“frontend separation 方向被证伪”，而是“直接对 embedding cosine / edge-level margin 施压”的形式被证伪。

证据：

- v43a 对 DBLP、Wiki、BlogCatalog 有收益：DBLP `0.6475`，Wiki `0.4607`，BlogCatalog `0.8428`。
- v43a 同时伤害 ACM 和 Flickr：ACM `0.7336`，Flickr `0.3394`。
- v43b 试图把 direct embedding pressure 变成 selective conflict margin，但首轮 ACM/DBLP/Flickr 全部未过 gate：ACM `0.6936`，DBLP `0.6406`，Flickr `0.2319`。
- v43b 的 `embedding_posterior_gap` 在三数据集上接近 0，说明失败主要发生在 frontend embedding，不是 posterior/readout 断裂。

因此，下一步不继续设计 v43b-style frontend separation loss，也不做 v43b_soft / strict / high / low / margin / weight 版本。

## 2. v43b 失败证据

| Dataset | ACC | Gate Active | Uncertainty Gate | View Disagreement | Violation Ratio | Overlap Gap | Band Mass | Highpass Energy | Conflict-Energy Corr | Emb-Posterior Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.6936 | 0.7808 | 1.0000 | 0.0506 | 1.0000 | -0.0387 | 0.4991 | 0.5000 | 0.0044 | 0.0000 |
| DBLP | 0.6406 | 0.6862 | 1.0000 | 0.0198 | 1.0000 | -0.0723 | 0.6877 | 0.5000 | 0.0007 | -0.0002 |
| Flickr | 0.2319 | 0.7881 | 1.0000 | 0.0005 | 1.0000 | -0.0532 | 0.4927 | 0.5000 | -0.0009 | -0.0005 |

解释：

- `gate active ratio` 远高于预期 `0.05-0.45`，selective gate 退化为大面积激活。
- `uncertainty_gate_mean=1.0`，不再有选择性。
- Flickr `view_disagreement_mean=0.0005`，但 active ratio 仍 `0.7881`，active 判定与有效冲突强度脱节。
- `violation_ratio=1.0`，被激活边几乎全部违反 margin，loss 变成广泛施压。
- `overlap_gap` 为负，高冲突边 overlap 反而低于低冲突边，margin pressure 没有命中“高冲突高重叠”区域。
- `band_mass` 仍高，hard/ambiguous 区域没有被解决。
- `highpass_energy_mean=0.5` 且 `conflict_energy_corr` 近 0，high-pass branch 仍未承载 conflict signal。

## 3. v44a 唯一机制方向

版本名：

```text
v44a_conflict_coupled_topology
```

核心机制假设：

```text
If topology contraction can reduce hard/ambiguous band mass
and high-pass energy becomes positively coupled with topology conflict,
then frontend embedding quality can improve without direct embedding cosine pressure.
```

中文表述：

在不使用 embedding cosine margin separation 的前提下，如果统一的 topology-band resolution 能降低 hard/ambiguous mass，并让 high-pass branch 与 topology conflict 正相关，则可以修复 v43b 暴露的前端机制断点，同时避免直接压坏 embedding。

## 4. 禁止项

v44a 禁止加入：

- dataset-specific module / branch / head / loss / assigner
- legacy head
- adaptive selector / post-processing selector
- embedding cosine margin loss
- edge-level overlap margin loss
- v43b-style selective conflict gate 改版
- 以 `overlap_gap` 作为直接优化目标
- 参数 sweep 或 full run

必须关闭或保持为 0：

```text
v43b_conflict_margin_weight = 0.0
v43b_band_conflict_weight = 0.0
v43b_highpass_energy_weight = 0.0
ideal_signed_embedding_weight = 0.0
```

## 5. 允许的最小机制

### 5.1 Topology Band Resolution

目标：降低 hard/ambiguous band，使 edge confidence 从中间不确定区向清晰 homo/hetero 分区移动。

建议机制只作用在 topology/frequency 层，不直接作用于 embedding pair cosine。

候选形式：

```text
clear_mass_e = max(homo_e, hetero_e)
band_mass_e = hard_e

score_uncertainty_e =
  1 - abs(score_e - mid_threshold) / threshold_half_width

topology_conflict_e =
  hetero_e + alpha_hard * hard_e

w_conflict_e = stopgrad(
  hard_e * score_uncertainty_e * topology_conflict_e
)

L_band =
  mean(hard_e * w_conflict_e)
  - lambda_clear * mean(clear_mass_e * w_conflict_e)
```

约束：

- `w_conflict_e` 必须 detach。
- 不使用标签。
- 不按数据集设置阈值。
- 不对 embedding cosine 施压。

### 5.2 Conflict-Coupled High-Pass

目标：让 high-pass branch 在高 topology conflict 区域有更强响应，而不是继续表现为 `highpass_energy_mean=0.5` 的占位信号。

建议 node-level conflict：

```text
node_conflict_i =
  mean_{e incident to i} stopgrad(hetero_e + beta * hard_e)

high_energy_i =
  ||Z_high_i||^2 / (||Z_low_i||^2 + ||Z_high_i||^2 + eps)
```

推荐首版目标：

```text
L_hp =
  ReLU(target_corr - corr(node_conflict, high_energy))^2
```

诊断必须同时记录 mean、std、high/low conflict energy gap，避免只把全图 high energy 抬高。

## 6. 首轮实验设计

首轮只允许 3 个数据集：

| Dataset | Role |
| --- | --- |
| ACM | 检查 homophilic graph 是否不再被前端机制误伤 |
| DBLP | 检查 v43a 的 DBLP 收益是否能保留或恢复 |
| Flickr | 检查 heterophilic large graph 中 high-pass/conflict coupling 是否真实生效 |

协议：

```text
epochs = 80
seed = 42
device = cuda
datasets = acm,dblp,flickr
```

命令模板，仅供实现后使用：

```powershell
conda activate aaai-e2e-subspace
python scripts\run_unified_aptc_9datasets.py --variant v44a_conflict_coupled_topology --datasets acm,dblp,flickr --epochs 80 --device cuda --log-level WARNING
```

## 7. 必须新增或保留的 diagnostics

Topology diagnostics：

```text
v44_band_mass
v44_hard_ratio
v44_ambiguous_ratio
v44_clear_mass
v44_decisive_mass
v44_score_uncertainty_mean
v44_score_uncertainty_p90
v44_threshold_gap
v44_low_threshold
v44_high_threshold
```

High-pass conflict diagnostics：

```text
v44_conflict_energy_corr
v44_highpass_energy_mean
v44_highpass_energy_std
v44_node_conflict_mean
v44_node_conflict_std
v44_high_conflict_energy
v44_low_conflict_energy
v44_energy_gap
```

Posterior/readout safety diagnostics：

```text
embedding_kmeans_acc
final_acc
embedding_posterior_gap
view_disagreement_mean
q_low/q_high agreement
legacy_head_used
```

保留 v43b 解释项，但不能作为优化目标：

```text
overlap_gap
high_conflict_overlap
low_conflict_overlap
```

## 8. 首轮 Gate

### 8.1 红线 gate

必须全部满足：

- `legacy_head_used=false`
- no selector
- no post-processing
- no dataset-specific branch
- no embedding cosine margin loss
- `v43b_conflict_margin_weight=0`

任一不满足，立即废弃 v44a。

### 8.2 Posterior/readout safety gate

```text
abs(embedding_posterior_gap) <= 0.02
```

如果 gap 变大，说明 posterior/readout 又开始断裂，不能继续。

### 8.3 Performance gate

必须全部满足：

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

ACC/NMI/ARI 只做 gate，不做 full-run 选择依据。

### 8.4 Topology mechanism gate

相对 v43b：

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.4927
```

且至少 2/3 数据集达到明显下降：

```text
absolute decrease >= 0.03
```

理想首轮目标：

```text
ACM <= 0.4691
DBLP <= 0.6577
Flickr <= 0.4627
```

### 8.5 High-pass mechanism gate

必须至少 2/3 数据集满足：

```text
v44_conflict_energy_corr >= 0.05
v44_highpass_energy_std > 0.02
v44_energy_gap > 0
```

如果 `highpass_energy_mean` 仍接近 0.5，但 std/gap/corr 有效，可以接受。若 mean=0.5 且 std/gap/corr 仍近 0，则失败。

## 9. 停止条件

任一发生即停止，不进入 full run：

- 违反红线。
- `abs(embedding_posterior_gap) > 0.02`。
- ACM `< 0.80`。
- DBLP `< 0.645`。
- Flickr `< 0.45`。
- 三个数据集 `band_mass` 均未下降。
- 三个数据集 `conflict_energy_corr` 仍近 0 或为负。
- high-pass energy 继续表现为恒定占位信号：std 近 0、energy_gap 近 0、corr 近 0。
- 改善只体现在 ACC 偶然波动，diagnostics 没有闭环。

仅允许一次 safety correction：

```text
topology_band_resolution_weight *= 0.5
conflict_highpass_corr_weight *= 0.5
```

触发条件：diagnostics 明显正向，但 ACM/Flickr 出现轻微性能受损。

不允许 correction：

- 调 margin。
- 调 gate center。
- 引入 v43b-style selective edge gate。
- 按数据集调权重。

## 10. 扩展条件

只有 ACM/DBLP/Flickr 首轮全部通过 gate，才允许第二批 smoke：

```text
Wiki, BlogCatalog, Texas
```

第二批目的：

- Wiki：检查 v43a 的 Wiki 收益是否保留。
- BlogCatalog：检查不破坏强大图表现。
- Texas：检查不出现 v40a 式小异配图崩溃。

只有前两批全部通过，才允许完整 9 数据集 80-epoch smoke。只有完整 smoke 通过，才允许考虑 260-epoch full run。

## 11. Result Templates

### 11.1 首轮 gate 表

| Dataset | ACC | NMI | ARI | Emb-KM ACC | Final ACC | Emb-Post Gap | Band Mass | Clear Mass | Corr | HP Mean | HP Std | Energy Gap | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 11.2 v43b 对照表

| Dataset | v43b Band | v44a Band | Delta Band | v43b Corr | v44a Corr | v43b ACC | v44a ACC | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.4991 | TBD | TBD | 0.0044 | TBD | 0.6936 | TBD | TBD |
| DBLP | 0.6877 | TBD | TBD | 0.0007 | TBD | 0.6406 | TBD | TBD |
| Flickr | 0.4927 | TBD | TBD | -0.0009 | TBD | 0.2319 | TBD | TBD |

### 11.3 停止条件记录

| Criterion | Status | Evidence |
| --- | --- | --- |
| No legacy head / selector / postproc | TBD | TBD |
| No embedding cosine margin loss | TBD | TBD |
| `abs(embedding_posterior_gap) <= 0.02` | TBD | TBD |
| ACM ACC >= 0.80 | TBD | TBD |
| DBLP ACC >= 0.645 | TBD | TBD |
| Flickr ACC >= 0.45 | TBD | TBD |
| Band mass non-increase on all 3 | TBD | TBD |
| Band mass clear decrease on at least 2/3 | TBD | TBD |
| Corr positive on at least 2/3 | TBD | TBD |
| High-pass non-constant diagnostics | TBD | TBD |

## 12. Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset | Metrics / diagnostics | Status |
| --- | --- | --- | --- | --- | --- |
| v43b 失败是 embedding margin 形式失败，不是 frontend 方向整体失败 | 为什么不继续调 v43b？ | v43a 正向收益 + v43b gate/violation/overlap 失败 | ACM/DBLP/Flickr/Wiki/BlogCatalog | ACC, gate active, violation ratio, overlap gap, emb-post gap | done |
| hard/ambiguous band 是病灶 | 为什么做 topology band resolution？ | band_mass 在 ACM/DBLP/Flickr 高 | ACM/DBLP/Flickr | band_mass, hard_ratio, ambiguous_ratio, clear_mass | done |
| high-pass 没承载 conflict | 为什么做 conflict-coupled high-pass？ | energy 恒定、corr 近 0 | ACM/DBLP/Flickr | highpass_energy_mean/std, corr, energy_gap | planned |
| v44a 机制闭环有效 | v44a 是否真的解决机制问题？ | band 下降 + corr 转正 + ACC 不坏 | ACM/DBLP/Flickr | gate table | planned |

## 13. No-Fabrication Status

本文档未生成任何新实验结果。所有 v44a 数值均为 TBD，必须由真实运行填入。v43a/v43b/v41f 数值来自现有 `results/` 文件与 diagnostics。

## 14. Next Owner

下一步应先由 `ccf-experiment-designer` 或当前主 agent 对本预注册文档做最终审查；审查通过后再进入代码实现。实现阶段只允许最小加入 topology band resolution 与 conflict-coupled high-pass diagnostics/loss，不允许引入 v43b 参数改版或任何 dataset-specific 逻辑。
