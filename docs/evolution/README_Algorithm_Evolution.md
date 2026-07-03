# SECT-CoCo-E2E Algorithm Evolution Log — Compressed Summary

本文件是原始算法演化日志的压缩版，用于节省上下文 token。它保留主线演化、关键实验结论、有效机制、失败分支、当前最优版本与下一步建议。

---

## 1. 总目标与红线

项目目标是构建一个统一端到端图聚类框架，覆盖 9 个同配/异配数据集：

```text
输入特征 + 图
→ 边置信度评分
→ 可微拓扑收缩
→ 频率感知低/高通滤波
→ APTC 聚类头 / assignment-flow posterior
→ 最终聚类标签
```

核心红线：

1. 不能按数据集切换模块或后端。
2. 不能使用 `if dataset == ...` 的路由。
3. 必须端到端可微，训练阶段梯度应能回传到前端核心参数。
4. 必须保留 edge confidence、topology contraction、frequency filtering、APTC / posterior transport 等核心创新。
5. 目标是尽量接近 AAAI0617 多头模型的 SOTA 表现，但必须用一个统一算法解释。

---

## 2. 最初问题：AAAI0617 结果强，但不是统一算法

旧 AAAI0617 代码效果强，但最终标签来自多个按数据集选择的后端：

- ACM：`subspace_refine`
- DBLP：`legacy_sect_bridge`
- PubMed：`fast_elss`
- Wiki：`wiki_consensus`
- Flickr：`dual_diffusion`
- BlogCatalog：`subspace_refine`
- Texas / Squirrel / Chameleon：KMeans 或 assignment-flow 类变体

这导致两个问题：

1. 论文不可辩护：不是一个统一模型，而是数据集条件后端集合。
2. 非端到端：很多最终 head 在 PyTorch 图外，梯度无法回传到前端。

因此 AAAI0622 主线是：删除多后端路由，构建统一 APTC 后端。

---

## 3. APTC v0–v6：统一后端建立，但性能很差

APTC 初始设计：

```text
H → learnable prototypes C
→ Q0 = softmax(cos(H, C))
→ Sinkhorn balanced transport
→ topology-aware posterior refinement
→ Q*
```

核心思想：

- learnable prototypes
- differentiable Sinkhorn
- learnable cluster prior
- topology refinement
- attr / low / high 三视图 posterior mixing

早期结果很差：

- ACM / DBLP / PubMed 等大幅 collapse。
- Sinkhorn balance 正常，但语义聚类差。
- topology contraction 经常把同配图也判成大量 hetero edge。

v1–v6 尝试过 dynamic prototype、KMeans initialization、teacher posterior、quantile threshold anchor、mask-mass balanced refinement、geometric teacher 等。

主要结论：

- 动态 prototype 从坏 posterior 自举会放大错误。
- teacher posterior 容易冻结错误分配。
- 单纯修 APTC head 不够，前端 edge confidence / topology contraction 才是早期瓶颈。

---

## 4. v7–v8：边置信度校准，解决 score / mask 数值崩溃

v7 加入三类 frontend calibration loss：

1. evidence attention entropy
2. mask diversity
3. structure-attribute consistency

v7 修复了 evidence attention collapse，但 homo mask 仍然太小。

v8 进一步加入：

- logit re-centering
- local ranking loss
- quantile threshold coupling
- evidence attention Dirichlet smoothing

关键有效点：

```text
edge logit 结构性重中心化：
r_hat = (r - mean(r)) / std(r)
score = sigmoid(eta * r_hat)
```

以及：

```text
alpha = (1 - epsilon) * alpha_raw + epsilon / num_sources
```

v8g 成为早期稳定 baseline：

- `score_mean` 稳定在 0.5 左右。
- `edge_logit_mean ≈ 0`。
- mask 不再全 hetero / 全 hard。
- 但性能仍远低于 SOTA。

结论：v8g 修复了前端数值物理问题，但 embedding / posterior readout 仍然弱。

---

## 5. v9–v16：raw topology / subspace / prototype / ranking 多数为负结果

这些版本试图进一步修前端或 prototype 几何。

### v9：raw topology leakage / subspace loss / Rayleigh routing

失败原因：

- raw leakage 太侵入。
- posterior stitching 已经很平滑，但平滑到错误 prototype。
- 问题不在 raw-edge discontinuity，而在 prototype/assignment geometry。

### v10：prototype geometry anchoring

失败原因：

- prototype separation regularizer 不足以改善最终聚类。
- full run 不如 v8g。

### v11–v12：multi-view SVD initialization / prior smoothing

失败原因：

- ACM 短跑有改善。
- PubMed / Texas 等受损。
- prior smoothing 会让 transport 更平静，但不是更正确。

### v13–v14：posterior entropy / view logit calibration

失败原因：

- 直接压 posterior entropy 太粗暴。
- 放大 node-prototype logit contrast 不能修复错误几何。

### v15–v16：raw topology as evidence / ranking teacher

失败原因：

- raw topology 作为直接 evidence 会破坏 Texas / BlogCatalog。
- raw-gated ranking 可以让 `rank_gap` 变好，但最终 partition 不改善。

总诊断：改善 edge ranking 或局部 smoothness 并不足够，瓶颈逐渐转向 embedding → differentiable posterior readout 的结构。

---

## 6. v17–v24：embedding posterior / residual correction 方向

v17 发现关键现象：

```text
final q_refined 很弱
但同一个 embedding 上 KMeans 很强
```

示例诊断：

- ACM：`q_refined ≈ 41%`，embedding KMeans ≈ `80%`
- DBLP：`q_refined ≈ 33%`，embedding KMeans ≈ `68%`
- Texas：`q_refined ≈ 54%`，embedding KMeans ≈ `74%`

这说明前端 embedding 已经含有较强聚类信息，但 APTC posterior 没读出来。

v17–v24 测试 embedding posterior 如何影响 APTC。

主要有效发现：

1. 不能直接加入 `Q_embed` 作为第四 view；它会扰乱原三视图 mixer。
2. 应使用 residual injection：

```text
Q_base = mix(Q_attr, Q_low, Q_high)
Q_mix = normalize(Q_base + gated_residual)
```

3. 更好的形式是 positive residual：

```text
Delta = ReLU(Q_embed - Q_base)
Q_mix = normalize(Q_base + alpha * Delta)
```

4. amplitude gate 需要 floor，不能纯 sigmoid 关死：

```text
g_amp = floor + (1 - floor) * sigmoid(...)
```

v24g 是该线较好的版本：

- 保留 Texas。
- 改善 PubMed。
- BlogCatalog 较健康。
- 但 DBLP / BlogCatalog 之间存在结构性冲突。

结论：embedding correction 有真实信号，但如何统一地调节 residual magnitude 是难点。

---

## 7. v27–v28：assignment-flow readout 与高通修复，出现大幅提升

v27 的重要发现：

```text
q_final = 0.30 * q_aptc_raw + 0.70 * q_flow
```

assignment-flow posterior 作为 final readout anchor 后，结果显著改善。

v27ap 代表结果：

```text
ACM          50.12
DBLP         64.16
PubMed       52.34
Texas        72.13
BlogCatalog  76.00
```

v28 进一步诊断前端频率耦合：

- ambiguous mask 很高。
- `Z_low` / `Z_high` 高度相似。
- high-pass 构造退化为近似 input copy。

v28b 加入 adaptive high-pass scale：

```text
h = z - hetero_mass * highpass_scale * smooth
```

并加低/高频 orthogonality loss。

v28b 260-epoch 结果：

```text
ACM          70.25
DBLP         67.12
PubMed       62.42
BlogCatalog  83.43
Texas        68.85
```

关键结论：

- high-pass 自适应尺度是有效机制。
- 但 `Z_low` / `Z_high` 仍然高 cosine alignment，说明提升不是来自真正正交解耦，而是 high-pass 成为更稳定的 feature correction。

---

## 8. v29–v32：确认瓶颈转向 embedding，并引入 embedding Dirichlet

v29 诊断：

```text
posterior ACC ≈ embedding KMeans ACC
```

说明 APTC posterior 已不再明显丢信号，瓶颈转为 learned representation/front-end。

负结果：

- v29c posterior-to-edge BCE 失败：用当前 posterior 监督 edge score 会强化错误早期信号。
- v30 EMA prototype 失败：ACM 提升但 BlogCatalog 大幅下降，回滚。
- v31 raw leak / high-pass gate floor 失败：低通质量没有改善。

v32a 引入 embedding Dirichlet regularization：

```text
edge_dirichlet(out["embedding"], edge_index, score.detach())
```

后来修正接受标准后，v32a 被重新采用。

v32a full run：

```text
ACM          76.03
DBLP         68.84
PubMed       62.72
Wiki         41.50
BlogCatalog  82.81
```

v32d 又加入 `z_attr` Dirichlet。

v32e 把 Dirichlet 权重从 `score` 改成：

```text
homo + 0.2 * hard
```

减少异配图错误 smoothing。

v32e full run：

```text
ACM          78.12
DBLP         67.37
PubMed       62.89
Wiki         43.49
Flickr       19.51
BlogCatalog  83.56
Texas        67.21
Chameleon    31.66
```

结论：v32e 是 ACM / Wiki / BlogCatalog 较强分支，但 Texas / DBLP 有回退。

---

## 9. v33–v34：hetero repulsion 与 PPR evidence 失败

v33a 加 embedding hetero repulsion：

```text
- edge_dirichlet(embedding, hetero_edges)
```

结果：

- Flickr / BlogCatalog / Squirrel 小幅改善。
- ACM / Wiki / Texas 明显下降。
- 回滚。

v34a 尝试 PPR 作为 edge scorer 第四 evidence channel。

失败原因：

- 初始 PPR evidence 数值死掉。
- 修复后 BlogCatalog 大幅 collapse。
- 回滚。

结论：PPR 作为 scorer evidence 不安全；hetero repulsion 也会破坏同配图和 Texas。

---

## 10. v35–v37：最终输出从 APTC 转向 embedding KMeans / full-space KMeans

v35a 加 SVD subspace post-processing：

- training 不变。
- final labels 用 embedding KMeans / SVD-subspace KMeans。

v35a 提升 ACM、BlogCatalog、Flickr、Chameleon。

v35b 固定 2K SVD subspace，full run：

```text
ACM          79.83
DBLP         70.40
PubMed       63.02
BlogCatalog  83.41
Squirrel     27.61
Texas        67.76
Chameleon    31.93
```

v37 发现 full-space KMeans 对多数异配图更好。

v37c 最终锁定为 full-space KMeans 输出：

```text
postproc_subspace_margin = 1.0
```

v37c full run：

```text
ACM          79.70
DBLP         67.59
PubMed       63.64
Wiki         39.63
Flickr       28.99
BlogCatalog  86.39
Squirrel     26.51
Texas        70.49
Chameleon    34.65
```

累计 ACC：

```text
v35b: 486.26
v37a: 489.56
v37b: 492.85
v37c: 497.60
```

v37c 被锁为当时最稳版本。

---

## 11. v38–v39：PPR view 与 lowpass depth 失败

v38 加固定 full-graph PPR view：

- ACM / Flickr / Squirrel / Wiki 提升。
- PubMed / BlogCatalog 大跌。
- `ppr_gate` 学不会关闭，始终约 0.5。
- 总分低于 v37c。
- 不保留。

v39 扫 lowpass depth：

```text
lowpass_steps: 2 -> 4
```

结果：

- ACM / Flickr 小涨。
- BlogCatalog 大跌。
- 不继续 v39b/v39c。

结论：更深低通不是瓶颈；PPR 直接 view 也不稳定。

---

## 12. v40：AAAI0617 legacy subspace head 审计，发现强机制但统一性有风险

v40 审计旧 AAAI0617 后端：真正强的是 ELSS / anchor subspace head，而不是简单 Student-t 或某个 loss。

v40a 将 legacy `_subspace_refine` 统一接入所有数据集。

v40a full run：

```text
ACM          85.59
DBLP         89.60
PubMed       63.66
Wiki         52.47
Flickr       27.37
BlogCatalog  92.90
Squirrel     25.48
Texas        42.08
Chameleon    31.23
```

巨大提升：

- DBLP 接近 SOTA。
- BlogCatalog 超过目标。
- ACM / Wiki 大幅提升。

但 Texas 崩溃：

```text
Texas 70.49 -> 42.08
```

结论：legacy anchor subspace 是真实突破机制，但 always-on 不安全。需要统一无监督 selector/gate，否则不能作为最终统一版本。

v40b 用 silhouette 选择 subspace / full KMeans，失败：

- 低 absolute silhouette 无法区分 DBLP 的好 subspace 和 Texas 的坏 subspace。
- 因此 v40b 不保留。

---

## 13. v41：KMeans teacher 线，机制有效但 DBLP/Wiki 冲突没解决

v41a 重新打开 KMeans teacher：

```text
aptc_init_teacher_weight = 0.10
```

80-epoch smoke：

- ACM 大涨：`70.38 -> 82.78`
- Flickr 大涨：`36.81 -> 47.42`
- Wiki / DBLP 下降

说明 teacher 是有效机制，但不安全。

v41b 调整 teacher refresh interval：

- `25 -> 50 -> 100`
- Wiki 稍修复。
- DBLP 更差。
- 不保留。

v41c confidence-weighted teacher：用 teacher top1 confidence 加权 KL，改善 Wiki，但 DBLP 仍不足。

v41d adaptive confidence quantile：让 active ratio 从接近 0 恢复到 0.34–0.45，但 DBLP 仍不提升。说明不是 active ratio 问题，而是 confidence 不等于语义可靠性。

v41e margin-based reliability：

```text
teacher_margin = top1 - top2
```

结果：

- Flickr / BlogCatalog 改善。
- DBLP sharp 版有小幅改善。
- Wiki 明显受损。
- full gate 仍失败。

v41f agreement-aware reliability：

```text
teacher_reliability = margin * embedding_knn_agreement
```

结果：

- DBLP agreement 饱和到 1.0，没有额外区分力。
- Wiki agreement 低但不产生干净 teacher subset。
- strict 版伤 Squirrel。
- soft 版伤 Wiki。
- 不保留。

最终 v41 结论：

- KMeans teacher 机制真实有效。
- 但继续调 scalar reliability gate 没意义。
- 下一步如果做 v41g，应改 teacher target，而不只是 weight：

```text
consensus teacher = KMeans teacher + embedding-neighborhood label distribution
```

---

# 当前重要版本定位

## 当前较稳版本：v37c

特点：

- full-space embedding KMeans final output。
- 9 数据集总 ACC 最高于当时 retained 版本。
- 对 Flickr / BlogCatalog / Texas / Chameleon 较好。
- DBLP / Wiki 不如部分 subspace 后端。

v37c full run：

```text
ACM          79.70
DBLP         67.59
PubMed       63.64
Wiki         39.63
Flickr       28.99
BlogCatalog  86.39
Squirrel     26.51
Texas        70.49
Chameleon    34.65
```

## 强机制但不安全版本：v40a

legacy subspace refine always-on：

```text
ACM          85.59
DBLP         89.60
Wiki         52.47
BlogCatalog  92.90
```

但：

```text
Texas        42.08
```

所以不能直接作为统一最终版本。

## Teacher 机制证明版本：v41a / v41c / v41e / v41f

共同结论：

- teacher 可以大幅提升 ACM / Flickr。
- 但 DBLP / Wiki 冲突未解决。
- scalar reliability gating 不够。

---

# 目前最核心的科学结论

1. 最初的 APTC posterior head 确实会丢失 embedding 中的可分性。
2. v28b 以后，posterior 与 embedding KMeans 差距缩小，瓶颈转向 representation/front-end。
3. v32e / v37c 说明 embedding-side smoothing + full-space KMeans 是目前较稳组合。
4. v40a 证明 AAAI0617 的 anchor subspace head 是强机制，但 always-on 会毁 Texas。
5. v41 证明 KMeans teacher 是强机制，但 teacher reliability / target 还不够可靠。
6. 当前最大矛盾：
   - ACM / DBLP / BlogCatalog 喜欢 subspace / teacher / smoothing。
   - Texas / Wiki / Squirrel / 部分 heterophily 图容易被这些机制伤害。
7. 未来关键不是再做 blind sweep，而是找到统一、无监督、可解释的 gate / reliability / consensus mechanism。

---

# 当前最建议的下一步

不建议继续优先调：

- scalar confidence curve
- teacher weight
- refresh interval
- lowpass depth
- PPR evidence
- simple silhouette threshold

更建议做以下方向。

## 方向 A：v41g consensus teacher

把 teacher target 从单一 KMeans soft target 换成局部一致性平滑 target：

```text
teacher_consensus = combine(
  KMeans teacher posterior,
  embedding-kNN label distribution
)
```

目标：

- 保留 KMeans teacher 对 ACM / Flickr 的强信号。
- 减少 Wiki / DBLP 中不可靠 teacher 节点的误导。
- 不增加新 head / dataset routing。
- 仍然在现有 teacher KL 框架内。

## 方向 B：v40c unified selector for legacy subspace

v40a 很强，但 Texas 崩溃。需要比 silhouette 更可靠的 selector，例如组合：

```text
subspace_reliability =
  cluster compactness
+ graph smoothness
+ posterior-flow conflict
+ embedding/full-kmeans agreement
+ subspace-vs-full consistency
```

但必须避免变成 dataset-specific routing。

## 方向 C：Flickr 专项结构诊断，但不能 dataset branch

Flickr 仍是最大缺口：

```text
v37c Flickr ≈ 28.99
SOTA ≈ 83.89
```

说明大规模强异配图仍有结构性失败。应诊断：

- edge score distribution
- hetero mask 是否有效
- high-pass view 是否真的贡献
- full-space KMeans 为什么仍差
- BlogCatalog 可高而 Flickr 极低的统一 graph statistic 是什么

---

# 一句话总括

这份演化日志记录了从“去除旧多后端路由、建立统一 APTC”到“发现 embedding / assignment / subspace / teacher 机制瓶颈”的全过程。当前稳定主线是 v37c full-space embedding KMeans 输出，最强但不安全机制是 v40a legacy anchor subspace head，最有潜力的下一步是 v41g：局部一致性平滑的 KMeans consensus teacher，或 v40c：更可靠的无监督 subspace selector。
