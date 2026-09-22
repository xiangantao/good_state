# 主推表征：Wan原五组 stride8 + JEPA监督256

运行实现位于本目录，**不依赖 `heft-paired-errors`、`paired_errors` 或 CALVIN数据缓存**。
正式训练只更新分类头，Wan、JEPA、ZCA和监督投影全部冻结。

## 表征怎样组成

两个编码器接收相同16帧、256×256 RGB。Wan使用本仓库已有的**不做时间下采样的VAE**：
16帧补最后一帧到17，posterior先裁回16，再采样和加噪；两个噪声分支都完整计算16个时间位置。
五组按下列顺序拼成896通道，保留零基时间位置`1,3,5,7,9,11,13,15`：

| 特征（层/头编号从0开始） | 通道 |
|---|---:|
| t57 B15H2 K |128|
| t57 B13H10 K |128|
| t57 B15H7 Q和K |256|
| t299 B15 hidden，固定ScanNet256通道 |256|
| t299 B15H2 K |128|

沿用固定逐通道中心化/标准化＋组内ZCA；**不采用track仅减均值版本**。
这些ZCA统计原先来自压缩5时间位置的校准，迁移到本版本后未重新拟合。
t57 sigma=0.05763683095574379/shift3；t299 sigma=0.299923837184906/shift5。

JEPA取ViT-L的零基B17和最终LayerNorm输出，每个token拼成2048维：

```text
JEPA2048 → 固定CALVIN通道mean/std → 固定2048×256矩阵 → JEPA256
Wan896：[B,896,8,16,16]
JEPA256：[B,256,8,16,16]  （不做时间插值）
按通道拼接：[B,1152,8,16,16]
按T/H/W展平：[B,2048,1152] / clip
两个segments：[B,4096,1152] / video
```

没有空间池化、补零token、额外可学习适配器或额外位置编码。
Wan取每对RGB位置的后一个位置，JEPA时间token来自两帧tubelet；这是近似对应，
不等同于两模型时间感受野相同。全部编码器计算完成后才取Wan的8个位置。

## JEPA监督256怎样得到

来自已有CALVIN实验：600个片段，400 fit、100 select、100 reserved；方向拟合只用400 fit。
监督是**已观察片段的状态变化，不是预测未来或接触成功标签**。
在采样位置0→7、7→15两个区间，每段56维，共112维：

- 红/蓝/粉三物体的三维位移及相对旋转6D表示；
- 各物体在末端坐标系的区间末相对位置、相对位置变化；
- 末端三维位移、相对旋转6D；夹爪末时刻宽度及宽度变化。

拟合时将JEPA2048通道特征池化到2×2×2，仅作为小读出器的描述子。
按训练数据统计归一化输入/7组监督目标，闭式岭回归beta=0.001得到读出权重。
权重从`[8×2048,112]`整理为`[2048,8×112]`，SVD取左奇异向量前256列，
得到**每个空间/时间位置共用的通道投影**，实际提取时保留8×16×16原生网格。
该256维候选在100 select片段上的既有归一化MSE=3.8076500899。
Wan不参与这次投影拟合/候选评分；没有用SSv2标签重新拟合该投影。

`artifacts/`保存运行必需的mean/std/ZCA/P矩阵、SHA256、fit来源及模型/通道表绑定。
Wan产物从37MB原文件中无损选出实际使用的数组；JEPA产物保持原文件逐字节一致。
以后换JEPA权重必须重新校准/拟合并更新产物绑定，不能直接沿用本投影。

需要复现导出时，在现有CALVIN缓存上运行（不会下载或覆盖原文件）：

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft
PYTHONPATH=.:src:diffusers/src:../vjepa2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m latent_video.wan_jepa.export_projection \
  --cache ../runs/calvin_jepa_proxy_20260921 --output /tmp/reproduced_supervised256.npz
```

## 正式训练、续训、异步评测

使用共享盘上的现有`.venv`、模型和数据；入口不会下载、安装或搬运环境/数据。
额外运行依赖见`requirements.txt`；现有环境已齐全。SciPy只用于离线复现投影，
实际训练不需要导入SciPy或CALVIN数据。
默认8卡DDP，每卡32视频，每卡两路提取，每路16 clips；全局batch256。
这是选定的两卡100步stride8实验实际使用的微批；早期八卡Wan配置是两路各32 clips。
每卡每步32视频×2segments＝64clips：2×16分两轮提取，2×32分一轮提取，
不改变分类头的训练batch。当前融合版本尚未验证2×32的显存及吞吐，因此默认保持2×16。
20 epochs，约660 steps/epoch；LR3e-4、WD0.1、FP16分类头、冻结编码器BF16。
每轮保存不可覆盖的`epoch_XXXX.pt`并更新`latest.pt`，**训练期间不验证**；W&B offline。
`monitoring.jsonl`为每10步窗口，`metrics.jsonl`为每轮统计。终端输出可用tee另存。

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft
bash latent_video/wan_jepa/launch.sh --check
bash latent_video/wan_jepa/launch.sh
```

新训练默认输出`../runs/ssv2_wan_stride8_supervised256_main`。
重新开一轮用`--output-dir /absolute/new/output`，不与旧实验输出混用。

```bash
# 同样8卡，从本主线入口保存的完整epoch恢复
bash latent_video/wan_jepa/launch.sh \
  --resume ../runs/ssv2_wan_stride8_supervised256_main/epoch_0001.pt

# 另一个作业/节点上评测已保存的epoch；rank数可不同，提取配置不变
CUDA_VISIBLE_DEVICES=4,5,6,7 HEFT_NPROC_PER_NODE=4 \
  bash latent_video/wan_jepa/launch.sh \
  --evaluate ../runs/ssv2_wan_stride8_supervised256_main/epoch_0001.pt
```

评测复用官方2segments×3views、概率平均、Top1/Top5，分片不重复覆盖样本；
结果在`output/evaluations/epoch_0001/`。续训要求同一世界大小/协议；支持完整epoch续训。
**旧两卡100步实验checkpoint不直接用于八卡主线续训**：schema、源码来源、世界大小不同。
本入口从随机分类头开始完整训练。没有自动启动全量训练。

## 合入范围与证据

- 新增本目录：独立提取/融合、固定产物、配置、启动、检查、投影复现、测试和文档。
- `classification/train.py`只增加可注入的extractor/encoder以及动态head宽度，
  数据加载、损失、DDP、优化器、checkpoint和评测继续复用主线实现。
- JEPA权重加载复用已在主工作区的`classification/baselines/vjepa2/encoder.py`
  及两个包的`__init__.py`。这三个依赖文件包含在本次主线集成提交中。
- 原`classification/encoder.py`已有他处改动，保持其内容，不覆盖。
- 不合入扫描程序、BCD、PCA融合、track均值实验、数据下载脚本或实验缓存。

选定基线100步末10步：loss3.9015/Top1=16.875%/Top5=37.34375%；
本次仅将这个已选版本独立到主线，没有宣称它优于PCA融合或已通过验证集评测。
`INTEGRATION_CHECKS.md`记录迁移检查及已验证范围。
