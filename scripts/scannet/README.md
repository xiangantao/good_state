# ScanNet / Wan 空间特征扫描

本目录参考 VEGA-3D 的 wan_t2v_adjacent_correspondence_matching.py、video_utils.py 和 wan_t2v_encoder.py，使用 HEFT 本地 Diffusers 权重。无需导入 LLaVA、EmbodiedScan 元数据或联网获取资源。第一版支持 ScanNet .sens v4（JPEG/PNG 彩色、zlib uint16 深度）及同名场景 .txt。

## 运行

在 heft 目录执行，使用已经恢复的 .venv/bin/python。所有模型加载使用 local_files_only=True，程序设置 HF_HUB_OFFLINE 与 TRANSFORMERS_OFFLINE。

```bash
# 五场景数据、采样和几何检查，不加载模型
.venv/bin/python -m scripts.scannet.scan --geometry-only

# 单场景、8帧、单层验证；设备请选当前空闲GPU
.venv/bin/python -m scripts.scannet.scan --scenes scene0707_00 --frames 8 --blocks 20 --device cuda:1

# 五场景、每场景32帧，固定k=300，一次前向捕获多个block
.venv/bin/python -m scripts.scannet.scan --device cuda:1

# 候选层的噪声比较（按需要执行）
.venv/bin/python -m scripts.scannet.scan --blocks 15 20 --timesteps 100 300 500 --device cuda:1

# 正确性测试
.venv/bin/python -m pytest scripts/scannet/test_scannet.py -q
```

默认数据：工作区 datasets/ScanNet/v2/scans_test。默认权重：models/Wan2.1-T2V-1.3B-Diffusers。场景列表可用 --scenes 指定；输入大小 --size H W，默认480×832；评分网格 --grid H W，默认14×14。默认层编号10、12、15、20、25、28均为Python从0编号，block 20表示 model.blocks[20]。这与当前仓库的实现一致，避免把论文的“20th”文字自行解释为索引19。

## 固定实验口径

- **比较范围**：每场景均匀采样最多32帧，只比较采样序列相邻两帧，绝不跨不同场景建立几何对应。采样基于 .sens 中实际存在且位姿有效的帧，不依赖 .txt 里可能不同的 numDepthFrames。不复制帧填满32。
- **RGB-D 标定**：读取 .sens 深度内参与相机到世界位姿；读取 .txt 的 colorToDepthExtrinsics 并求逆，投影深度到彩色相机，再应用与RGB一致的resize/crop；以最近表面z-buffer处理投影冲突。位姿采用深度相机到世界的ScanNet约定。
- **Axis alignment**：存在时应用；测试场景未提供时使用单位变换并记录。整个场景共同的刚性坐标变换不改变点间距离，但体素边界可随轴方向变化，不能将此设置与其他对齐方式的分数混用。
- **无效深度**：排除零值与非有限值，不填补深度空洞。坐标池化仅以有效投影像素加权；有效像素覆盖率默认至少0.1。覆盖率是输出图像像素中有真实深度投影的比例，会受输入分辨率影响，所有层必须固定相同预处理和阈值。物体边界内多个有效表面的平均点仍可能不位于真实表面，14×14/10cm指标只是近似几何探针。
- **HEFT VAE设计**：保留本地 Diffusers 禁用时间下采样的实现；编码器中的因果时间卷积仍然存在。当前单帧扫描通过每次只编码一张图像，避免跨帧 VAE 上下文。加载原始 checkpoint 时，未使用的时间重采样权重是该实现的预期结果；本流程不修改共享 Diffusers 代码。
- **Wan**：每张RGB独立进行单帧VAE编码，取latent mode并做checkpoint定义的mean/std归一化，冻结模型。空文本使用本地T5编码并补零到512。VAE使用float32，Transformer默认bf16；与VEGA原始Wan实现并非逐位复现。
- **噪声**：独立实现并用本地VEGA调度器测试对照其1000步、shift=5的整数timestep最近查找，同时记录实际sigma。k=300不是HEFT 50步调度的第300步。端点也保留参考调度器的实际选择（例如请求0未必意味着严格零噪声）。按seed、场景、原始帧编号生成CPU噪声，同一帧不因batch_size改变噪声；多层共享一次前向。
- **特征**：读取完整DiT block输出，float32池化到14×14。不是Q/K或单head；这些是后续扩展。图像默认480×832，与论文部分实验的720×1280不同，报告应保留此区别。

## 指标

metrics.py 独立于模型。几何映射和负例由固定数据决定，所有层共用。

1. positive_similarity：同体素、同视角内先平均原始特征，再L2归一化，比较两视角体素表征，体素等权。
2. negative_similarity：从共享体素中，为每个视角体素表征选择一个不同且相距至少0.2m的体素作对照，双向固定种子采样。没有可用负例时记null，绝不伪造0。
3. similarity_gap：以上正负相似度差，是诊断量，不是校准后的概率或总分。
4. retrieval_voxel_hit：对目标帧中存在同体素的查询，按余弦相似度在全部有效目标token中找top-1，评价是否同体素。
5. retrieval_hit_distance：同一检索结果的三维距离是否≤0.1m。
6. retrieval_world_error_mean / median：双向可评价查询的三维误差统计。并列最高相似度用分数化命中率和平均距离，常量特征不会因索引恰好对齐而获得虚假完美分数。中位数字段在后续汇总中表示帧对中位数的平均，并非全局查询中位数。

检索时不会先按真值距离过滤候选。无共享体素或无有效token的帧对记录明确status和null分数。覆盖数保存在每对帧的记录中。帧对等权汇总到场景，再对有效场景等权汇总，标准差使用总体标准差；这不是论文附录C的全场景所有跨视角对加权指标。

## 缓存、输出与复查

- cache/scannet/geometry：采样RGB、token世界坐标和覆盖mask；键含数据文件路径/大小/修改时间、采样帧、预处理及几何代码哈希。
- cache/scannet/latents：本地VAE结果，避免噪声扫描重复编码。
- cache/scannet/features：按噪声、种子、模型身份、提取代码、dtype、设备和batch设置隔离，每个block单独保存。默认仅存池化特征；--save-prepool额外保存池化前特征。
- eval/scannet/<配置哈希>：config.json、sampled_frames.json、每帧对JSON、pair_metrics.csv、scene_metrics.csv、summary.csv/json、status.json；错误写入errors.json。
- heft/reports/scannet/<配置哈希>：report.md、层间对比图、共享PCA及几何误差最好/最差帧对可视化。图中并列匹配取第一个索引，仅用于展示；量化指标使用上述并列处理。

同一配置重跑可跳过已有完整评分；部分特征缓存可复用。数据或相关代码改变生成新身份，避免误用旧结果。错误后重跑以status.json为准，历史errors.json可能保留上次失败记录。--no-previews只跳过绘图。完成后若需要重新绘制已跳过的图，应显式运行绘图逻辑，现有扫描不会重算已完成评分。

这是五场景探索。报告保留所有场景的结果、有效对应量和波动，不能用某一个最高分宣称普适3D能力。

## Block 14/15 的 attention head 扫描

```bash
# 两层全部12个head，单场景8帧验证
.venv/bin/python -m scripts.scannet.head_scan --device cuda:7 --scenes scene0707_00 --frames 8

# 五场景、每场景32帧、k=300，默认两层×12heads×QK/KK
.venv/bin/python -m scripts.scannet.head_scan --device cuda:7

# 可选择某些head；编号从0开始
.venv/bin/python -m scripts.scannet.head_scan --device cuda:7 --blocks 14 15 --heads 0 3 7 --modes kk
```

head_extract.py 继承原提取器，复用逐帧VAE、空文本和确定性噪声前向。在本地Wan处理器的 `_heft_capture` 回调读取归一化和RoPE之后的实际Q/K，形状为 `[batch,12,tokens,128]`；每个head独立池化到14×14。不会将完整block输出切片当作head，也不捕获文本交叉注意力。回调在结束或异常时恢复。

KK使用余弦相似度；QK沿用HEFT的 `Q·K/sqrt(128)`。QK双向分别使用 `Q_i,K_j` 和 `Q_j,K_i`，并非转置前向矩阵。分数在池化后的跨帧特征上计算，未softmax，不是原生帧内attention map。RoPE被保留，因此排名包含图像位置编码影响。QK/KK原始分数不可直接比较，几何命中率和距离误差可比较。

同体素分数仍在每个视角内先平均该体素的原始特征；同体素命中率仍在全有效目标token中检索后检查体素归属。没有按真实体素预先限制候选。体素命中与10cm距离命中是两个独立指标，报告均列出；默认报告按体素命中率排序。无共享体素的帧对记缺失，保持原先负例与场景等权口径。

head_scan.py 复用已有几何/VAE缓存，Q/K另存 cache/scannet/head_features；缓存身份含提取代码和本地Wan处理器哈希。默认结果在 eval/scannet_heads/<hash>，报告和图片在 heft/reports/scannet_heads/<hash>。每层/噪声缓存全部12个head，评分按head和mode独立保存；复跑可复用缓存并补齐缺失效果图。完整实验应有240个场景×候选评分文件、48行汇总，以及120张K-PCA图和480张匹配图。

验证：`python -m pytest scripts/scannet/test_scannet.py scripts/scannet/test_heads.py -q`。包含实际本地attention处理器的捕获位置与前向结果不变检查、head空间次序、非对称QK方向、常量特征并列及报告分组测试。

## B15 hidden 通道筛选

`channel_scan.py` 只读取已有层扫描的几何和特征缓存，不加载 Wan。默认复用 `eval/scannet/07c02b354a694662` 的五场景 B15、k=300 数据。首先逐帧对复算原始指标，差值超过 `1e-6` 即停止。缓存身份、文件 SHA256、原始帧编号及评估代码哈希随结果保存。

```bash
# 在 heft 目录；显式指定源码路径，防止恢复后的 editable 安装指向历史 checkout。
export PYTHONPATH="$PWD/src:$PWD/diffusers/src"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.scannet.channel_scan --device cuda:6 --baseline-only
.venv/bin/python -m scripts.scannet.channel_scan --device cuda:6
.venv/bin/python -m pytest -p no:cacheprovider scripts/scannet/test_channels.py scripts/scannet/test_scannet.py -q
```

默认保留维度依次为 `1024,768,512,384,256,128`。以删除单个通道后体素命中率的下降作为重要性，平分时参考正确/错误候选最大余弦的差值，再按原通道编号稳定排序。一次性筛选只排一次；逐档剪枝在当前通道集合内重新消融，每次剪枝后复查组合效果。加速消融使用 float64，最终评分仍使用原始 `evaluate_pair` 的 float32 实现。GPU 缓存分配上限为设备总显存的 4%。

每折在四个场景上排序或拟合 PCA，在第五个场景评分；留出场景不参与通道选择。PCA 使用有效 token 的场景等权协方差，减训练均值、不白化，并提供完整维度仅中心化对照。随机子集采用五个固定种子，在维度间嵌套。所有方法均在变换之后重新归一化；几何、负例、并列处理、帧对到场景的等权聚合保持原样。

结果保存于 `eval/scannet_channels/<hash>`，报告与曲线保存于 `heft/reports/scannet_channels/<hash>`。绘图优先使用 matplotlib，未安装时使用仓库现有 Pillow 依赖。`folds/<留出场景>/masks.json` 保存原始通道编号，`pca.npz` 保存训练均值和投影矩阵，`split.json` 记录拟合场景；消融排序与组合评分另行保存，可断点重跑。

这评估的是筛选流程在新场景上的表现，不等同于已经验证一份用全部五场景拟合的固定通道列表。按汇总结果挑选最佳维度后，还需要额外场景确认。剪枝减少下游描述子的存储和匹配维度，不减少原始 Wan block 前向计算。

## 与点跟踪、OVIS 对齐的多帧扫描

`video_scan.py` 使用与前面两个任务相同的本地 `WanPipeline` 和 `WAN_2_1` 配置：480×832、bf16、50 步调度的索引 49、guidance scale 5、空文本，以及视频 VAE 的 latent sampling。实际 timestep 和 sigma 从 checkpoint 调度器读取，不能将索引 49 与旧实验的 VEGA k=300 混为一谈。

```bash
# 在仓库根目录，先设置上节中的 PYTHONPATH 和离线环境变量。
.venv/bin/python -m scripts.scannet.video_scan --geometry-only
.venv/bin/python -m scripts.scannet.video_scan --device cuda:4 --no-previews

# 单场景验证，默认每景 32 张采样帧、25 帧视频片段。
.venv/bin/python -m scripts.scannet.video_scan \
  --device cuda:4 --scenes scene0707_00 --blocks 15 --no-previews

# 使用相同 HEFT 调度及 VAE sampling 的单帧对照，另存为独立实验。
.venv/bin/python -m scripts.scannet.video_scan --device cuda:4 --chunk-size 1 --no-previews

.venv/bin/python -m pytest -p no:cacheprovider scripts/scannet/test_video.py -q
```

多帧输入为 `[1,C,25,H,W]`，不是将 25 张独立图像放在 batch 维。复用 HEFT 的分块和输入预处理；每个场景独立分块，尾段重复最后一张 RGB 到 25 帧。VAE 在整个片段内保留因果卷积上下文，Transformer 在全部时空 token 上联合注意力。尾段补齐帧会参与上下文，但不进入评分。默认扫描 B14/B15 的完整 block 输出及全部 12 个头的 Q/K，支持 `--blocks`、`--heads` 和 `--modes` 选择。

模型前向只捕获条件分支，hidden 来自完整 block 输出，Q/K 来自原有 HEFT 捕获接口。空间池化逐帧进行，输出 hidden `[采样帧数,1536,14,14]`、Q/K `[采样帧数,所选头数,128,14,14]`；时间顺序与原始帧编号一一对应。每个片段记录实际 Transformer 输入形状、前向次数及 timestep。当前 VAE 编码器按首帧加四帧组处理，因此 `--chunk-size` 要满足 `4n+1`，默认 25。

首轮继续使用各场景原来的 32 张均匀采样帧，采样间隔和时间戳保存在 `sampled_frames.json`。这对齐了视频编码方式，但没有把 ScanNet 的稀疏采样改成原始连续帧，也没有统一不同任务的下游指标。三维几何、14×14 网格、负例、并列处理和场景等权聚合与旧实验一致。

结果位于 `eval/scannet_video/<hash>`，报告位于 `heft/reports/scannet_video/<hash>`，特征使用独立的 `cache/scannet/video_features/<hash>`。原始 `pairs/` 保留所有相邻采样帧对；报告同时提供 `all_pairs` 和 `within_chunk` 两种汇总。后者只评价两帧位于同一视频片段内的帧对。跨片段帧对不应被解释成共享上下文的匹配；和旧结果比较时必须取相同帧对。`pair_metrics.csv` 用 `scope` 区分两种口径，不能把两种 scope 的行混合再次汇总。

通道筛选可以读取新的完整 hidden 缓存，先通过逐帧对基线复算再筛选：

```bash
.venv/bin/python -m scripts.scannet.channel_scan \
  --baseline-run ../eval/scannet_video/RUN_ID --timestep 49 \
  --device cuda:4 --baseline-only
```

将 `RUN_ID` 替换为实际视频扫描目录名。通道筛选目前采用 `all_pairs` 口径，包含分块边界，沿用原始场景留一流程。旧单帧结果选出的 384 维列表尚未验证能迁移到多帧特征。视频模式与旧 k=300 模式同时改变了 VAE 上下文、latent 采样和噪声配置，因此两者的分数差不能全部归因于 Transformer 的跨帧注意力。

### 多帧使用旧实验的噪声点

使用 `--noise-mode legacy` 可以保留视频编码、分块和采样方式，同时按旧 `selected_noise` 查表规则选择 timestep 和 sigma：

```bash
.venv/bin/python -m scripts.scannet.video_scan \
  --noise-mode legacy --timesteps 300 --shift 5 \
  --blocks 14 15 --device cuda:4 --no-previews

.venv/bin/python -m scripts.scannet.channel_scan \
  --baseline-run ../eval/scannet_video/RUN_ID --timestep 300 \
  --dimensions 1024 768 512 384 256 128 --device cuda:4
```

上述设置精确选择实际 timestep 299、sigma `0.299923837184906`、shift 5。`--timesteps 300` 是旧调度的请求值，不是实际 timestep，也不是 pipeline 步索引。该模式将选中的噪声点交给单步 UniPC 调度器，pipeline 捕获索引为 0；旧 1000 点调度只用于确定这个噪声点，不执行其余去噪步骤。

默认 HEFT 模式是 50 步调度中只执行索引 49，legacy 噪声模式是单步调度中执行索引 0。两者实际都只有一轮条件/无条件前向，hidden 和 Q/K 均在首次调度更新前捕获。实际前向的 timestep、调度 sigma、sigma 转换到 latent 精度后的值及此前更新次数均记录在分块 audit 中。加噪继续使用现有 bf16 视频 pipeline，因此调度 sigma 在运算时按 bf16 精度表示。

新的噪声设置使用独立缓存与结果目录；B15 通道筛选需在对应多帧缓存上重新拟合。它与旧单帧 k=300 对齐了噪声点，但视频 VAE 上下文、latent sampling、精度和随机噪声序列仍与旧单帧不同。
