# VDiT Attention Head 功能识别与量化方案

> 状态：基线方案（后续非必要不改动）  
> 建立日期：2026-09-03  
> 适用目标：识别视频生成模型中具有 matching、semantic、position 功能的 attention head。  
> 变更约束：除非实验结果证明定义存在明确问题，或用户明确要求调整，否则不修改本文中的区域定义、核心指标和实验口径。任何调整都应记录原因、旧定义和新定义，避免实验口径漂移。

## 1. 研究目标

本工作的目标不是单纯获得更高的点跟踪分数，而是分析 VDiT 中不同 attention head 所编码的信息类型。

对每个 `(layer, head)`，不直接强制分配唯一类别，而是先构建三维功能画像：

```text
Matching Score / Semantic Score / Position Score
```

原因是一个 head 可能同时具有多种功能。最终分类建立在连续指标、跨视频稳定性和可视化验证之上。

需要区分：

- **Matching head**：响应集中在跨帧真实对应点附近；
- **Semantic head**：响应覆盖同一实例或同一语义类别区域，而不只集中在真实对应点；
- **Position head**：响应主要保留在 query 的原空间坐标附近，即使目标已经移动。

## 2. 数据要求

理想数据同时包含：

1. 连续视频帧；
2. 点级跨帧对应轨迹；
3. 每帧语义分割标签；
4. 最好具有跨帧一致的实例 ID；
5. 可见性或遮挡标注。

优先检查当前 TAP-Vid DAVIS 视频能否与原始 DAVIS 实例 mask 对齐。如果能够对齐，可以同时使用已有点轨迹和实例区域，不必立即引入新数据。

如果需要新的数据，优先选择**视频实例分割数据集**，而不只是普通逐帧语义分割数据集。只有类别 mask 时，可以测量类别语义，但无法可靠区分“同一个具体物体”和“同类别的其他物体”。

## 3. 固定实验条件

比较不同 head 时，以下条件必须保持一致：

- 相同模型和 checkpoint；
- 相同 diffusion step；
- 相同输入视频、分辨率和预处理；
- 相同 layer/head 特征维度处理；
- 相同 query 点集合；
- 相同目标帧集合；
- 相同相似度归一化方法和温度参数；
- 第一轮功能识别使用完整频率通道，不预先删除高频；
- 频率过滤只作为后续功能解释和消融，不与初始 head 分类混在一起。

论文表明最终 denoising step 的特征最好，因此第一版实验固定使用当前 pipeline 已采用的最终 denoising step。

## 4. Query 样本筛选

并非所有点都适合判断 head 功能。有效 query 应满足：

1. query 点在源帧和目标帧均可见；
2. query 点不在 ignore 区域；
3. 尽量避开实例边界，降低 mask 和特征网格对齐误差；
4. 目标实例在两帧中均有有效 mask；
5. 用于区分 position head 时，真实位移必须足够大，使真实对应位置与原坐标邻域不重叠；
6. 按视频、实例和类别均衡采样，避免大物体、背景或高频类别主导结果；
7. 同一实例不应因为面积大或轨迹点多而获得过高权重。

设源帧 query 坐标为 `x_q`，目标帧真实对应坐标为 `x_gt`。用于 position 分析的样本至少满足：

```text
||x_gt - x_q|| > 2r
```

其中 `r` 是对应区域和原位置区域使用的邻域半径。所有坐标统一映射到 attention feature grid 后再计算。

## 5. 跨帧相似度图

对 head `h`，源帧 query patch 为 `q`，目标帧 patch 为 `i`。

为了同时分析模型原生 attention 行为和当前跟踪 descriptor 行为，保留两种相似度图。

### 5.1 原生 Q-K attention logit

```text
z_attn(i) = Q_h(q) · K_h(i) / sqrt(d_h)
```

该指标最接近 attention head 在模型内部实际使用的关系。

### 5.2 K-K descriptor similarity

```text
z_kk(i) = cosine(K_h(q), K_h(i))
```

该指标与当前 HeFT tracking pipeline 使用 Key-Key 特征进行匹配的方式对应。

两种指标必须分开报告，不允许混合后只给出一个无法解释的分数：

- Q-K 主要用于判断 head 的原生功能；
- K-K 用于判断该功能能否转化为有效的 correspondence descriptor；
- 若两者结论一致，功能判断可信度更高；
- 若两者结论不同，应保留差异，而不是人为选择更符合预期的结果。

为了计算 attention mass，对每个目标帧相似度图使用固定温度 `tau`：

```text
p(i) = softmax(z(i) / tau)
```

同时保留原始 `z(i)`，避免结论完全依赖 softmax 温度。

## 6. 目标帧区域定义

针对每个 query，在目标帧 feature grid 上定义以下互斥区域。区域按下列顺序划分，先分配的区域具有更高优先级。

### 6.1 对应区域 C

真实对应点 `x_gt` 周围半径 `r` 的区域：

```text
C = disk(x_gt, r)
```

它表示精确 correspondence。

### 6.2 原位置区域 P

源帧 query 坐标 `x_q` 在目标帧中的同坐标邻域，并排除 C：

```text
P = disk(x_q, r) \ C
```

它用于检测 head 是否不顾目标运动，仍关注原空间位置。

### 6.3 同实例非对应区域 I

目标帧中与 query 属于同一实例的区域，但排除 C 和 P：

```text
I = target_instance_mask \ (C ∪ P)
```

这是识别 instance-level semantic head 的核心区域。因为已经排除了真实对应点附近，单一 matching 峰值不会直接提高该区域的得分。

### 6.4 同类别其他实例区域 G

目标帧中语义类别相同、但实例 ID 不同的区域，并排除前述区域：

```text
G = same_class_other_instances \ (C ∪ P ∪ I)
```

它用于判断 head 是否编码比具体实例更抽象的 category-level semantics。

如果当前帧不存在同类别其他实例，则该 query 不参与 `G` 相关指标，但仍可参与其他指标。

### 6.5 其他区域 O

其余有效、非 ignore 的目标帧区域：

```text
O = valid_grid \ (C ∪ P ∪ I ∪ G)
```

O 是语义和位置指标的主要负样本区域。

## 7. 基础区域统计量

直接求 attention 总量会偏向面积较大的区域，因此必须同时计算 area-normalized density。

对于任意区域 `R`：

### 7.1 Attention mass

```text
Mass(R) = Σ[i ∈ R] p(i)
```

### 7.2 面积归一化密度

```text
Density(R) = Mass(R) / |R|
```

### 7.3 相对密度提升

相对于负样本区域 O：

```text
Lift(R) = log((Density(R) + eps) / (Density(O) + eps))
```

其中 `eps` 仅用于避免数值除零，所有 head 使用同一固定值。

### 7.4 原始相似度差

```text
DeltaZ(R, O) = mean(z(i), i ∈ R) - mean(z(i), i ∈ O)
```

`Lift` 和 `DeltaZ` 必须同时保留：前者反映归一化 attention 分配，后者不依赖 softmax mass。

## 8. Matching 指标

### 8.1 对应区域提升

```text
MatchLift = Lift(C)
```

MatchLift 越高，说明真实对应点邻域相对其他区域获得的 attention 密度越高。

### 8.2 Top-1 命中率

```text
Top1Hit = 1[argmax_i z(i) ∈ C]
```

在所有有效 query 上取平均。

### 8.3 定位误差

```text
PointError = ||argmax_i z(i) - x_gt||
```

同时报告均值和中位数。该指标越低越好。

### 8.4 对应峰相对实例区域的尖锐度

```text
MatchContrast = log((Density(C) + eps) / (Density(I) + eps))
```

若 I 为空，则该 query 不计算此项。

Matching head 的典型表现：

- MatchLift 高；
- Top1Hit 高；
- PointError 低；
- MatchContrast 高；
- 相似度峰集中在 C，而不是平均铺在整个实例区域。

## 9. Semantic 指标

Semantic 指标必须排除 C，避免 matching 峰值被误当作语义响应。

### 9.1 Instance Semantic Lift

```text
InstanceSemLift = Lift(I)
```

它衡量同一实例中、除真实对应点之外的区域，相对其他区域是否仍具有较高相似度。

### 9.2 Category Semantic Lift

```text
CategorySemLift = Lift(G)
```

它衡量不同实例但同类别的区域是否得到更高响应。

### 9.3 Semantic Coverage

仅有平均相似度仍可能被少数异常点影响，因此需要覆盖率指标。

以 O 区域相似度的第 95 百分位数作为当前 query 的背景高响应阈值：

```text
threshold_O = percentile_95({z(i) | i ∈ O})
```

实例覆盖率：

```text
InstanceCoverage = |{i ∈ I : z(i) > threshold_O}| / |I|
```

类别覆盖率：

```text
CategoryCoverage = |{i ∈ G : z(i) > threshold_O}| / |G|
```

Semantic head 应当在同实例或同类别区域形成较大范围的高响应，而不只是单个峰值。

### 9.4 非局部语义比例

```text
SemanticToMatch = log((Density(I) + eps) / (Density(C) + eps))
```

该指标用于区分“整个实例区域响应”和“精确点匹配”：

- Matching head 通常 `Density(C) >> Density(I)`，因此该值较低；
- Semantic head 的 I 区域响应更强、更广，该值相对较高；
- 该指标不能单独使用，必须与 InstanceSemLift 和 Coverage 一起判断。

### 9.5 Semantic head 的判定特征

Semantic head 的典型表现：

- InstanceSemLift 高；
- InstanceCoverage 高；
- 如果存在同类别其他实例，CategorySemLift 或 CategoryCoverage 也可能较高；
- MatchContrast 不应像纯 matching head 那样极端；
- Position 指标不能占主导；
- heatmap 响应覆盖目标实例或同类区域，而不是仅集中在 C。

语义功能分成两个层次分别报告：

```text
Instance semantics：同一个具体物体
Category semantics：不同实例但类别相同
```

不得把两者合并后只报告“semantic”。

## 10. Position 指标

Position 指标只在真实位移足够大的 query 上计算。

### 10.1 原位置提升

```text
PositionLift = Lift(P)
```

### 10.2 原位置相对真实目标的偏置

目标实例区域定义为：

```text
T = C ∪ I
```

位置偏置：

```text
PositionBias = log((Density(P) + eps) / (Density(T) + eps))
```

### 10.3 原位置命中率

```text
OldPositionHit = 1[argmax_i z(i) ∈ P]
```

### 10.4 原位置与真实位置距离对比

```text
d_old = ||argmax_i z(i) - x_q||
d_gt  = ||argmax_i z(i) - x_gt||
```

Position head 通常满足：

- PositionLift 高；
- PositionBias 高；
- OldPositionHit 高；
- `d_old < d_gt`；
- 当物体运动时，峰值仍停留在原坐标附近。

## 11. 三维功能画像与排名

每个 `(layer, head)` 分别汇总 Q-K 和 K-K 两套指标。

### 11.1 Matching Score

主要依据：

```text
MatchLift ↑
Top1Hit ↑
PointError ↓
MatchContrast ↑
```

### 11.2 Semantic Score

主要依据：

```text
InstanceSemLift ↑
InstanceCoverage ↑
SemanticToMatch ↑
CategorySemLift / CategoryCoverage ↑（数据允许时）
```

### 11.3 Position Score

主要依据：

```text
PositionLift ↑
PositionBias ↑
OldPositionHit ↑
d_old - d_gt ↓
```

第一阶段不使用任意手工权重把所有指标压成单个总分。先对每项指标分别排名，并保留三维画像。原因是固定加权和会掩盖混合功能 head，也会让结论依赖未经验证的权重。

需要输出一个用于排序的主指标时，使用：

- Matching 排名主指标：`MatchLift`，以 Top1Hit 和 PointError 作为并列排序依据；
- Instance Semantic 排名主指标：`InstanceSemLift`，以 InstanceCoverage 和 SemanticToMatch 作为并列排序依据；
- Category Semantic 排名主指标：`CategorySemLift`，以 CategoryCoverage 作为并列排序依据；
- Position 排名主指标：`PositionBias`，以 OldPositionHit 作为并列排序依据。

## 12. 跨视频汇总与稳定性

不能仅凭单个 query 或单个视频定义 head 功能。

汇总顺序固定为：

1. query 级指标先在同一实例内平均；
2. 实例级结果在同一视频内平均；
3. 最后对视频做等权平均。

这样可以避免长视频、大实例或轨迹点数量较多的视频占据过高权重。

每项指标至少报告：

- 跨视频均值；
- 跨视频中位数；
- 25%–75% 分位区间；
- 在多少比例的视频中进入全部 heads 的前 20%。

只有在多个视频中排名稳定的 head，才视为稳定的功能 head。

## 13. 分类原则

连续指标是主要结果，离散标签只是方便描述。

### Matching 候选

- MatchLift 排名靠前；
- Top1Hit 高且 PointError 低；
- 在多个视频中保持稳定；
- 可进一步使用 TAP-Vid AJ 验证其实际 matching/tracking 能力。

### Semantic 候选

- InstanceSemLift 与 InstanceCoverage 同时靠前；
- SemanticToMatch 不低，说明响应不只来自对应点尖峰；
- PositionBias 不占主导；
- heatmap 确实覆盖同实例或同类别区域；
- 若 CategorySemLift 也高，则单独标记为 category-semantic head。

### Position 候选

- 在大位移样本上 PositionBias 和 OldPositionHit 靠前；
- 峰值更接近旧坐标而非真实目标坐标；
- 可进一步检查高频 RoPE 通道是否占主导。

如果一个 head 在多类指标上同时较高，应标记为混合功能，例如：

```text
matching-semantic
semantic-position
```

不得为了得到整齐分类而强制给每个 head 唯一标签。

## 14. 可视化验证

每类排名最高的若干 head 必须生成统一格式的可视化：

1. 源帧及 query 点；
2. 目标帧真实对应点；
3. C、P、I、G 区域边界；
4. Q-K attention heatmap；
5. K-K similarity heatmap；
6. argmax、真实对应点和原坐标位置；
7. 各区域 Mass、Density 和 Lift；
8. 不同目标帧上的连续变化。

预期模式：

- Matching：热点跟随真实点移动，且峰值尖锐；
- Semantic：热点覆盖目标实例或同类区域，分布较宽；
- Position：热点停留在原坐标附近；
- 混合 head：表现出两种或多种模式。

可视化用于验证指标是否测到了预期行为，不能代替量化指标。

## 15. 频率分析

完成初始分类后，再分析每个候选 head 的 RoPE 频率特征：

1. 统计 temporal、height、width 各频率段的 Q/K norm；
2. 分别保留低频和高频通道重新计算三维功能画像；
3. 观察删除高频后 Matching Score 是否提升；
4. 观察 Position Score 是否主要依赖高频；
5. 检查 Semantic Score 对不同频率段的依赖。

根据论文，预期：

- Matching head 更依赖低频；
- Position head 更依赖高频；
- Semantic head 的频率特征需要通过本实验补充，不能预先假设。

## 16. 推荐执行顺序

### 阶段 A：小规模验证

- 选择少量具有明显运动、多个物体和清晰 mask 的视频；
- 提取同一目标 layer 的全部 heads；
- 计算 C/P/I/G/O 区域指标；
- 检查 Head 2 是否呈现 matching 模式；
- 检查是否存在稳定的 semantic 和 position 候选；
- 人工查看排名最高和最低 head 的 heatmap，验证指标方向。

### 阶段 B：扩展到所有 layer

- 固定阶段 A 的指标定义和参数；
- 对所有 layers/heads 运行同一评估；
- 输出全模型 head 功能图谱；
- 不根据结果临时更换指标或阈值。

### 阶段 C：跨视频和跨模型验证

- 扩大视频数量；
- 检查同一 head 功能是否跨视频稳定；
- 在其他 VDiT backbone 上使用完全相同的口径；
- 比较功能类型是否普遍存在，但不假设 head index 能跨模型对应。

## 17. 与当前 HeFT 结果的关系

当前 Wan2.1-1.3B 的配置固定使用：

```text
Layer 15 / Head 2
```

当前 30 个 DAVIS 视频上的结果为：

```text
AJ:    45.1828%
δavg:  60.2290%
OA:    84.6866%
```

这说明 Head 2 具有较强的实际 correspondence 能力，但目前只能作为 matching head 的任务性能证据。现有代码尚未通过本文定义的 C/P/I/G/O 指标自动发现 Head 2，也尚未系统识别 semantic head 和 position head。

## 18. 必须避免的错误

1. 不能只统计同类别区域的 attention 总量而不做面积归一化；
2. 不能让真实对应点 C 包含在 semantic 区域 I 中；
3. 不能在目标几乎不移动时判断 position head；
4. 不能只看 argmax 判断 semantic head；
5. 不能只看平均相似度而不看覆盖率；
6. 不能把同实例语义和同类别语义混为一个指标；
7. 不能只用一个视频给 head 定性；
8. 不能混合 Q-K 与 K-K 结果后只输出一个分数；
9. 不能为了符合论文预期而忽略不一致的 head；
10. 不能在看到最终结果后随意修改区域定义、指标或筛选条件。
