# v44b_pre_normalization_frequency_response Preregistration

本文档是在 `v44a_conflict_coupled_topology` 首轮 smoke 失败后的下一机制预注册。它只定义下一机制、诊断指标、实验 gate 和停止条件；不包含任何新实验结果。

## 1. v44a 结论

v44a 必须停止，不进入第二批 smoke、完整 9 数据集 smoke、260-epoch full run 或权重 sweep。

证据来自 `V44A_FIRST_SMOKE_VERDICT.md`：

| Dataset | ACC | NMI | ARI | Emb-Post Gap | v44 Band | v44 Corr | HP Mean | HP Std | Energy Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.7197 | 0.3594 | 0.3961 | 0.0000 | 0.4944 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| DBLP | 0.6529 | 0.3542 | 0.2953 | 0.0000 | 0.6844 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| Flickr | 0.3622 | 0.1972 | 0.1265 | 0.0000 | 0.5051 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

Gate verdict：

- Red-line gate：PASS。
- Posterior/readout safety gate：PASS，`embedding_posterior_gap=0.0`。
- Performance gate：FAIL，ACM 与 Flickr 未达标。
- Topology gate：FAIL，band mass 未达到 2/3 明显下降，Flickr 反而升高。
- High-pass mechanism gate：FAIL，corr/std/gap 全部为 0 或近 0。

## 2. 失败解释

v44a 的 high-pass mechanism 失败不是单纯权重不足，而是 diagnostic/loss target 选错了能量定义。

当前代码中：

```text
low_view = _diffuse(...)
hetero_view = _signed_highpass(...)
```

而 `_diffuse` 与 `_signed_highpass` 都返回 L2-normalized node view。因此 v44a 使用的

```text
||Z_high_i||^2 / (||Z_low_i||^2 + ||Z_high_i||^2 + eps)
```

在节点级近似恒等于 `0.5`，导致：

```text
v44_highpass_energy_mean = 0.5
v44_highpass_energy_std  ≈ 0
v44_energy_gap           ≈ 0
v44_conflict_energy_corr = 0
```

因此 v44a high-pass correlation loss 没有有效方差可对齐，继续训练、扩展数据集或调权重都不能解决机制问题。

## 3. v44b 唯一机制方向

版本名：

```text
v44b_pre_normalization_frequency_response
```

核心机制假设：

```text
If conflict coupling is applied to a pre-normalization frequency response
instead of post-normalization view norms,
then high-pass conflict diagnostics can become non-constant,
and topology-conflict regions can receive real frequency-side signal
without embedding cosine pressure or dataset-specific routing.
```

中文表述：

不再用归一化后的 `low_view` / `hetero_view` 范数比作为 high-pass energy，而是使用归一化前的 signed residual / high-pass response 幅度作为 conflict-coupled signal。目标是先修复 v44a 暴露的“高通能量恒定占位”问题，再判断该机制是否能改善前端 embedding。

## 4. 禁止项

v44b 禁止加入：

- dataset-specific module / branch / head / loss / assigner
- legacy head
- adaptive selector / post-processing selector
- embedding cosine margin loss
- edge-level overlap margin loss
- v43b-style selective conflict gate 改版
- 以 `overlap_gap` 作为直接优化目标
- 参数 sweep 或 full run
- 使用 post-normalization `||low_view||` / `||hetero_view||` norm ratio 作为 high-pass energy 优化目标

必须关闭或保持为 0：

```text
v43b_conflict_margin_weight = 0.0
v43b_band_conflict_weight = 0.0
v43b_highpass_energy_weight = 0.0
ideal_signed_embedding_weight = 0.0
ideal_band_resolution_weight = 0.0
ideal_highpass_energy_weight = 0.0
v44_conflict_highpass_corr_weight = 0.0  # 禁止沿用 v44a 归一化后能量目标
```

## 5. 允许的最小机制

### 5.1 Pre-normalization Frequency Response Diagnostic

目标：先得到一个非恒定、可解释、与 topology conflict 可比较的 high-pass response。

建议在 `_signed_highpass` 内部或并行 helper 中记录归一化前响应，例如：

```text
smooth_i = hetero-graph normalized aggregation of h_i
signed_residual_i = z_attr_i - hetero_mass * highpass_scale * smooth_i
raw_high_response_i = ||signed_residual_i - z_attr_i||^2
raw_low_response_i  = ||low_raw_i - z_attr_i||^2
```

或更保守地使用当前 high-pass step 中的：

```text
raw_high_response_i = ||hetero_mass * highpass_scale * smooth_i||^2
```

节点级能量定义：

```text
pre_hp_energy_i = raw_high_response_i / (raw_high_response_i + raw_low_response_i + eps)
```

如果首版不安全，也允许只用：

```text
pre_hp_response_i = log1p(raw_high_response_i)
```

约束：

- response 必须来自 normalization 前的 high-pass / residual quantity。
- 不新增独立 high-pass branch。
- 不按数据集设置阈值。
- 不使用标签。
- 不对 embedding pair cosine 施压。

### 5.2 Conflict-Coupled Pre-HP Objective

目标：让 high topology conflict 节点拥有更强 pre-normalization high-pass response，但避免把全图高通能量无差别抬高。

定义：

```text
node_conflict_i = mean_{e incident to i} stopgrad(hetero_e + beta * hard_e)
pre_hp_response_i = normalization-free high-pass response
```

推荐首版目标：

```text
L_pre_hp = ReLU(target_corr - corr(node_conflict, pre_hp_response))^2
```

同时加入 anti-collapse diagnostic，不作为首版强优化目标：

```text
response_std
high_conflict_response - low_conflict_response
response_p90 / response_p10
```

首版不加入 response magnitude maximization，避免模型通过全局放大 high-pass residual 过 gate。

### 5.3 Topology Band Resolution

v44a 的 topology band loss 没有明显奏效，但也不是主要诊断失效点。v44b 可以保留极小权重或关闭该 loss，首版推荐：

```text
v44_topology_band_resolution_weight = 0.0
```

原因：下一步主问题是修复 high-pass response 的可观测性和可优化性；同时动 topology band 与 high-pass response 会混淆失败归因。

## 6. 首轮实验设计

首轮仍只允许 3 个数据集：

| Dataset | Role |
| --- | --- |
| ACM | 检查 homophilic graph 是否不被新 response 机制误伤 |
| DBLP | 检查 v44a 中唯一通过 ACC gate 的数据集是否保持安全 |
| Flickr | 检查 heterophilic large graph 中 high-pass response 是否真实激活 |

协议：

```text
epochs = 80
seed = 42
device = cuda
datasets = acm,dblp,flickr
```

命令模板，仅供实现后使用：

```powershell
conda run -n aaai-e2e-subspace python scripts\run_unified_aptc_9datasets.py --variant v44b_pre_normalization_frequency_response --datasets "acm,dblp,flickr" --epochs 80 --device cuda --log-level WARNING
```

## 7. 必须新增或保留的 diagnostics

Pre-normalization high-pass diagnostics：

```text
v44b_pre_hp_response_mean
v44b_pre_hp_response_std
v44b_pre_hp_response_p10
v44b_pre_hp_response_p90
v44b_pre_hp_response_ratio_p90_p10
v44b_conflict_response_corr
v44b_high_conflict_response
v44b_low_conflict_response
v44b_response_gap
v44b_node_conflict_mean
v44b_node_conflict_std
```

Normalization-degeneracy safety diagnostics：

```text
v44b_postnorm_hp_energy_mean
v44b_postnorm_hp_energy_std
v44b_postnorm_energy_gap
```

这些用于证明 v44b 没有继续优化 v44a 的恒定量。

Topology diagnostics 保留：

```text
v44_band_mass
v44_hard_ratio
v44_ambiguous_ratio
v44_clear_mass
v44_low_threshold
v44_high_threshold
```

Posterior/readout safety diagnostics 保留：

```text
embedding_kmeans_acc
final_acc
embedding_posterior_gap
legacy_head_used
v43b_enabled
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
- no post-processing selector
- no dataset-specific branch
- no embedding cosine margin loss
- `v43b_conflict_margin_weight=0`
- `v44_conflict_highpass_corr_weight=0`，即不再使用 v44a post-normalization high-pass energy loss

任一不满足，立即废弃 v44b。

### 8.2 Response non-degeneracy gate

必须 3/3 数据集满足：

```text
v44b_pre_hp_response_std > 1e-4
v44b_pre_hp_response_p90 > v44b_pre_hp_response_p10
```

至少 2/3 数据集满足：

```text
v44b_response_gap > 0
v44b_conflict_response_corr >= 0.05
```

如果 response 仍恒定或 corr/gap 全为 0，则 v44b 失败。

### 8.3 Posterior/readout safety gate

```text
abs(embedding_posterior_gap) <= 0.02
```

如果 gap 变大，说明 posterior/readout 又开始断裂，不能继续。

### 8.4 Performance gate

首轮必须全部满足：

```text
ACM ACC >= 0.80
DBLP ACC >= 0.645
Flickr ACC >= 0.45
```

如果 mechanism diagnostics 明显通过但 ACM/Flickr 轻微未过，可只记录为 mechanistic partial pass，不允许扩展第二批。

### 8.5 Topology safety gate

v44b 不要求 topology band 明显下降，但要求不能恶化：

```text
ACM band_mass <= 0.4991
DBLP band_mass <= 0.6877
Flickr band_mass <= 0.5051
```

这里 Flickr 使用 v44a 实测上界 `0.5051` 作为 safety ceiling，防止继续恶化。

## 9. 停止条件

任一发生即停止，不进入第二批或 full run：

- 违反红线。
- `abs(embedding_posterior_gap) > 0.02`。
- 3/3 数据集 pre-HP response 仍近似恒定。
- 3/3 数据集 `v44b_conflict_response_corr` 仍近 0 或为负。
- 3/3 数据集 `v44b_response_gap <= 0`。
- ACM `< 0.80`。
- DBLP `< 0.645`。
- Flickr `< 0.45`。
- 改善只体现在 ACC 偶然波动，response diagnostics 没有闭环。

不允许 safety correction：

- 调 margin。
- 调 gate center。
- 引入 v43b-style selective edge gate。
- 按数据集调权重。
- 改用 post-processing selector。

## 10. 扩展条件

只有 ACM/DBLP/Flickr 首轮全部通过 performance gate、posterior gate、response non-degeneracy gate，才允许第二批 smoke：

```text
Wiki, BlogCatalog, Texas
```

第二批目的：

- Wiki：检查 mixed graph 是否不被 high-pass response 过拟合伤害。
- BlogCatalog：检查强大图表现是否保持。
- Texas：检查小异配图是否不出现 v40a 式崩溃。

只有前两批全部通过，才允许完整 9 数据集 80-epoch smoke。只有完整 smoke 通过，才允许考虑 260-epoch full run。

## 11. Result Templates

### 11.1 首轮 gate 表

| Dataset | ACC | NMI | ARI | Emb-KM ACC | Final ACC | Emb-Post Gap | Pre-HP Std | Corr | Response Gap | Band Mass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DBLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Flickr | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 11.2 v44a 对照表

| Dataset | v44a ACC | v44b ACC | v44a HP Std | v44b Pre-HP Std | v44a Corr | v44b Corr | v44a Gap | v44b Gap | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACM | 0.7197 | TBD | 0.0000 | TBD | 0.0000 | TBD | 0.0000 | TBD | TBD |
| DBLP | 0.6529 | TBD | 0.0000 | TBD | 0.0000 | TBD | 0.0000 | TBD | TBD |
| Flickr | 0.3622 | TBD | 0.0000 | TBD | 0.0000 | TBD | 0.0000 | TBD | TBD |

### 11.3 停止条件记录

| Criterion | Status | Evidence |
| --- | --- | --- |
| No legacy head / selector / postproc | TBD | TBD |
| No embedding cosine margin loss | TBD | TBD |
| Post-normalization HP loss disabled | TBD | TBD |
| Pre-HP response non-constant on 3/3 | TBD | TBD |
| Corr positive on at least 2/3 | TBD | TBD |
| Response gap positive on at least 2/3 | TBD | TBD |
| `abs(embedding_posterior_gap) <= 0.02` | TBD | TBD |
| ACM ACC >= 0.80 | TBD | TBD |
| DBLP ACC >= 0.645 | TBD | TBD |
| Flickr ACC >= 0.45 | TBD | TBD |
| Band mass not worse than ceiling | TBD | TBD |

## 12. Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset | Metrics / diagnostics | Status |
| --- | --- | --- | --- | --- | --- |
| v44a failed because post-normalization energy is degenerate | 为什么不继续 v44a 或调权重？ | HP mean=0.5, std=0, gap=0, corr=0 + code path shows L2-normalized views | ACM/DBLP/Flickr | v44 HP diagnostics, `_diffuse`, `_signed_highpass` | done |
| v44b fixes the measured signal | 新 high-pass signal 是否不再恒定？ | pre-HP response std/p10/p90/gap/corr | ACM/DBLP/Flickr | v44b_pre_hp_response_* | planned |
| conflict coupling is meaningful | response 是否集中在 high-conflict topology 区域？ | high-vs-low conflict response gap and corr | ACM/DBLP/Flickr | corr, response_gap | planned |
| mechanism does not reopen readout break | 是否又变成 posterior/readout 问题？ | embedding-posterior gap stays small | ACM/DBLP/Flickr | embedding_posterior_gap | planned |
| mechanism is not just ACC noise | ACC 改善是否有机制闭环？ | performance gate + response diagnostics both pass | ACM/DBLP/Flickr | ACC/NMI/ARI + response diagnostics | planned |

## 13. No-Fabrication Status

本文档未生成任何 v44b 实验结果。所有 v44b 数值均为 TBD，必须由真实运行填入。v44a 数值来自 `V44A_FIRST_SMOKE_VERDICT.md` 与对应 `results/` 文件。

## 14. Next Owner

下一步应先做最小代码实现审查：确认是否能在不破坏统一 forward path 的前提下暴露 pre-normalization high-pass response。实现阶段只允许新增 pre-normalization response diagnostics/loss 与一个 `v44b_pre_normalization_frequency_response` variant；不允许引入 v43b margin、selector、legacy head 或 dataset-specific 逻辑。
