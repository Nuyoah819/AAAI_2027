# 端到端异配感知子空间聚类

这个目录下是一个以 `ELSS` 为主体、融合 `GREET` 前端思想的 PyTorch 原型实现。整体思路不是简单并列拼接两篇论文，而是：

- 保留 `ELSS` 的主线：
  属性图输入 -> PageRank 选锚点 -> Nyström 低秩基 -> ELSS 风格后处理 -> `KMeans` -> 聚类指标。
- 引入 `GREET` 的前端：
  先学习每条边的同配/异配权重，再构造低通同配视图和高通异配视图，生成更适合聚类的节点表示。
- 损失函数改成端到端聚类目标：
  去掉 `InfoNCE`，改为 `subspace smoothness loss + pivot-anchored ranking loss`。

## 文件说明

- `model.py`
  核心模型实现。
  包含边判别器、稀疏双通道传播编码器、可微 Nyström 低秩模块，以及最终的聚类后处理。

- `losses.py`
  损失函数实现。
  包含：
  `subspace_smoothness_loss`
  `pivot_anchored_ranking_loss`

- `utils.py`
  工具函数。
  包括：
  `ELSS/data/*.mat` 数据加载
  稀疏邻接归一化
  PageRank
  结构编码
  锚点采样

- `metrics.py`
  聚类评估指标。
  包含：
  `ACC`
  `NMI`
  `ARI`

- `train_smoke.py`
  小型冒烟测试。
  用一个很小的合成图验证：
  前向是否正常
  反向梯度是否连通
  Nyström 子空间是否还能保持良好的正交性

- `run_pubmed_clustering.py`
  在 `PubMed` 上直接跑聚类实验并输出 `ACC/NMI/ARI`。

- `environment.yml`
  conda 环境配置文件。

## 当前实现重点

### 1. 仍然是 ELSS 主体

虽然前端用了 `GREET` 的边判别思想，但最终任务定义、锚点选取、低秩子空间提取、聚类后处理和评估方式，都是按属性图聚类来组织的，不再走原 `GREET` 的节点分类评估路线。

### 2. 改成了稀疏传播

`PubMed` 图规模是 `19717 x 19717`，如果用稠密邻接矩阵，内存和速度都会很差。所以这里把双通道传播改成了稀疏实现，更适合继续沿着 `ELSS` 的属性图聚类方向扩展。

### 3. Nyström 路径做了数值稳定处理

可微特征分解 / SVD 在训练里比较容易因为特征值过小、重复特征值或病态矩阵而出问题。当前实现加了几层保护：

- 对称化矩阵
- ridge 正则
- 很小的 diagonal jitter
- 特征值下界截断

这样可以减少 `NaN` 梯度和分解失败的概率。

### 4. ELSS 后处理也做了稳定化

一开始直接在 `PubMed` 上做那一步后处理 SVD，会出现不收敛。现在改成了更稳的 Gram 矩阵特征分解版本，目的不变，仍然是在低秩子空间上做 ELSS 风格聚类，只是数值上更稳一些。

## 使用方法

先激活全局 conda 环境：

```powershell
conda activate aaai-e2e-subspace
```

### 1. 跑冒烟测试

```powershell
python aaai/code/train_smoke.py
```

### 2. 在 PubMed 上跑聚类

```powershell
python aaai/code/run_pubmed_clustering.py --dataset pubmed --epochs 5 --eval-every 1 --num-anchors 96 --hidden-dim 32 --disc-hidden-dim 64 --ranking-weight 0.2
```

脚本会输出：

- 每轮训练损失
- 周期性聚类评估结果
- 最优 `ACC/NMI/ARI`

## 当前一组 PubMed 结果

在轻量配置下，我已经跑出过一组结果：

```text
BEST epoch=005 ACC=0.4678 NMI=0.0866 ARI=0.1018
```

这说明流程已经打通，但这还只是第一版结果，后面还有明显的调参空间。

## 后续最值得继续做的事

如果后面继续沿这个方向推进，最优先建议做的是：

1. 系统调参
   重点看：
   `num_anchors`
   `hidden_dim`
   `ranking_weight`
   `high_pass_alpha`
   `gnn_layers`

2. 调整聚类损失
   现在的 `subspace_smoothness_loss` 偏保守，后面可以考虑引入更强的子空间结构约束。

3. 进一步贴近 ELSS 原文中的后处理
   当前版本已经保留了 ELSS 风格主线，但为了稳定性对后处理做了工程化替代，后面可以继续向论文公式靠拢。
