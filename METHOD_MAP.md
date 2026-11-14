# 第三章到代码的逐项映射
|论文机制|函数|演示实现与缩小|
|---|---|---|
|exam/clause/fact/concept来源图及不可变offset|`SourceGraph`|四类型，anatomy作为concept命名空间；node key含exam/clause/begin/end；同患者先移除再归一化transition；非训练split报错|
|图像P13×3、q、子句种子与PPR|`encode_image/retrieve`|39概率保留、未提及类别无监督；图像预测而非GT入query；top4事实+全部clauses，重启质量归一化，未收敛明确dense回退；无目标文本查询|
|候选数/token预算、无门后补选|`retrieve/context`|K3/B24包括source/state标记；token不截断；使用固定缓存候选，无门控后重取|
|三状态支持90维缩小接口|`support_prob/hard_gate`|五个d8块+39+5+4+2=90，原稿d256时1330；局部文本条件注意力；冲突优先，composite不能S，未知非negative|
|投影后V权重/零键mask|`projected_attention`|归一化后投影再缩放V；所有解码/纠正/辅助头共用；零权重上下文不能被残差激活|
|A 对齐/代理dm/分类/因果gen|`masked_assertion_loss/patient_aware_dm_loss/patient_aware_contrastive_loss` + `run_demo`A|39维头仅监督已观测0/1/2，3=unmentioned掩蔽；dm列为真实子句嵌入，排除同患者全部exam；按finding/location/side事实键对齐，共同观测断言一致才得分，明确present/absent矛盾整候选得分0；decoder memory仅图像|
|B frozen encoders/cache，softgate|`run_demo`B|实训support与初始decoder；encoder/disease/text冻结，候选PPR无梯度；weak训练配对按原子人工几何关系产生|
|C discrete draft + Q=M0 → M1|`Correction/run_demo`C|draft实际生成后detach，冻结txt编码，原M0查询，KV=原M0(含image/context)+draft；保留mask/weights；tiny单head/GRU替代8headTransformer|
|frozen decoder输入导数、relation/align/keep|`teacher/relation_loss/kept_positions`|adapter及aux训练，decoder权重冻结输入可微；align仅image行；finding/location query不含目标属性；冲突属性mask；keep只覆盖draft内完整正确子句对应参考span|
|一次检索两次生成，无testupdate|`infer`|有检索调用计数、测试state快照；输出所有原始错误，不做成功筛选|

缩小与未实现：真实ResNet50、DART-A、HippoRAG2官方模型未复现；用8维Conv/Text+GRU。事实为明确合成schema，未运行RadGraph/否定parser；固定tauS=.55/tauC=.45/rho=.15为demo配置，不是临床校准，未完成源dev阈值选择。无MIMIC/MS-CXR数据、十基线、完整RadGraph评估或人类盲审。正式数据适配必须从授权的train Findings建立source graph，train/dev/test按患者锁定，模型或解析版本变化重建全部cache；请将真实image、fact schema、encoder/decoder接到对应类，保留权限和梯度合同。
测试核验来源排除/PPR列和、训练记忆限制、冲突与复合候选、投影值衰减与零键、冻结decoder梯度、一次检索两生成、keep跨度、关系属性冲突mask及未提及状态。

## Stage A 边界修订与审计
- `masked_assertion_loss`不索引未提及3，不把它改写为absent；同病种出现多个冲突全局断言时保守掩蔽该分类头，不平均为伪标签。源疾病词汇外目标不进入13头监督。
- `patient_aware_dm_loss`以其他患者的**子句**为候选，分母在softmax前剔除同患者的所有exam。辅助疾病匹配不再复用完整报告相似度；完整报告保留在独立contrastive分支。
- `assertion_agreement`按(finding,location,side)语义键对齐，不依赖clause顺序；忽略unmentioned、缺失键，明确同键present/absent矛盾使整候选为0。左右分别present本身不构成明确矛盾（可能双侧），不同键不产生共同观测分值。
- 目标agreement由离散训练元数据生成并detach；只有正总agreement的query参与该辅助项均值。没有合法候选、全部agreement为0、全部分类标签未提及，均返回连接原计算图的可微零项。
- 分类覆盖/被掩蔽计数、dm活跃/省略query与合法候选数记录在实际Stage A训练history。测试补充未提及、同患者多exam、clause乱序、显式矛盾、全部零/空候选及contrastive同患者负例屏蔽。

## 从一条来源子句走到最终报告

复核固定选取既有输出的第一条记录 `exam_12`，并沿首个返回候选 `p0_c0` 追踪。输入是人工生成的单幅 8×8 几何图，训练患者为 0–9。下述数字来自已保存的合成运行，不能填入真实胸片结果栏。[固定数据](data/sample.json) · [逐例预测](examples/predictions.json) · [运行记录](examples/metrics.json)

1. **来源与召回。** 候选原文为 `left opacity lung present .`，不可变键为 `(exam_0, p0_c0, 0, 27)`，字符区间取左闭右开。图 3a 解释共享概念，式 (1) 保留本条事实的位置。另一个候选 `p6_c0` 的文字完全相同，属于另一检查的另一个事实。查询图像生成 16 个图像 token、8 维全局表示和 13×3 个预测断言分数。已存记录显示排除同患者后有 10 个合格来源，PPR 迭代 72 次收敛，列和最大误差为 0，未走稠密回退。未持久化的种子分值、各结点最终 π 和排序间距无法从该日志逐项复算。[图构建与检索](src/core.py#L95-L138)
2. **门控。** 三个候选均为原子子句。按式 (6)，先查冲突阈值 0.45，再查支持阈值 0.55，否则权重为 0.15×支持概率。数字按落盘精度四舍五入。对首候选，尽管参考文本恰有相同子句，模型仍给出证据不足。比较用的参考文本在推理返回后才附加。[门控与投影](src/core.py#L14-L26) · [支持特征与记忆构造](src/core.py#L63-L78)

| 来源 | 原文 | 支持 / 冲突 / 不足概率 | 判定 | 式 (6) 复算与持久化权重 |
|---|---|---|---|---:|
| p0_c0 | left opacity lung present . | 0.122840 / 0.357754 / 0.519406 | 不足 | 0.0184260 |
| p6_c0 | left opacity lung present . | 0.122840 / 0.357754 / 0.519406 | 不足 | 0.0184260 |
| p1_c1 | left effusion lung absent . | 0.138984 / 0.255301 / 0.605715 | 不足 | 0.0208476 |

3. **读入与修订。** 每条候选含 5 个正文 token 和 2 个标记，因此共占 21 个上下文 token，满足 B=24 与 K=3。37 项记忆权重由 16 个权重为 1 的图像 token 和三组各 7 个候选权重构成。权重作用在归一化与投影后的 V；只要权重大于零，对应键仍可参与 softmax。日志没有记录注意力矩阵，不能把 0.018426 解读为该词的生成概率。首稿为 `right opacity lung . <eos>`；修订后重复 `right opacity lung`，到 14 个 token 截止。合成评估参考则为 `left opacity lung present . right effusion lung absent .`。因此该例能复核检索一次、权重和输入顺序，也让人看到门控后仍可能出现侧别错误。[一次检索与两次解码](src/core.py#L161-L168)

## 正式模型的接入位置

| 环节 | 正式输入、输出与目标 | 演示差异及实现注意点 |
|---|---|---|
| 图像与图文对齐 | 多视图胸片、视图掩膜 → 空间 H、全局 q、13×3 个预测分数。Stage A 使用图文对比、可微候选分布匹配、带缺失掩膜的断言监督与图像条件教师强制生成。 | 目前是单幅几何图、4×4 特征格和 8 维编码。接入正式骨干时应继续排除同患者负例，保留未提及掩膜，并使输入预处理、坐标和多视图掩膜可复查。 |
| 来源库与检索 | 仅用已获准训练分区的 Findings。保留原子/复合标记、原文、字符跨度、断言侧别和定位。训练时在同患者排除后建转移与重启；打包时整句跳过超长项。 | 合成记录预给每条单一关系；当前直接以子句向量作为事实相似度。正式版需实现多事实/复合解析、词表外掩膜和独立序列化事实向量。现有版本指纹绑定源记录和冻结编码器；正式键还须纳入解析器、词表、图权重、检索设置和文本预处理。 |
| 视觉支持 | 图 3b 及式 (4)–(6)。输入五个 256 维向量、39 项预测、5 项侧别、4 项断言和 2 项质量标记，共 1330 维，输出三态。用可信训练配对监督，在独立开发患者上校准阈值。 | 当前维数为 90，两项质量标记固定为 1，阈值在函数中固定。合成标签器先按征象和位置匹配，可能把不同侧别的存在与否认误标冲突，也会将某些单侧互换标为冲突。正式版须先界定断言与侧别的适用范围，再判显式不相容，或逐例复核互斥性；左侧存在与右侧不存在可同时成立。若监督不可靠应排除或按协议处理。 |
| 修订与读出 | 以原记忆 M⁰ 的各行为 Q，以 M⁰ 与生成草稿表示作为 K/V。原候选的键掩膜与投影后值权重在适配器、最终解码和训练期关系读出中均须保持。推理时冻结参数且只取一次候选。 | 当前单头交叉注意力、微型 GRU 和词表替代正式多头解码器。合成关系属性与真实解析器输出不可互换。正式版需在有效视图、零权重键、掩膜和空上下文上保持这些语义，并独立保存草稿与修订稿。 |
| 评估与溯源 | 输出成对报告、候选键、决策、门控权重、版本和失败标志，记录可分阶段的时间。最终关系敏感奖励、冲突传递和正确事实保留分开测量。 | 留存样例没有逐个 PPR 值、注意力或解码 logits。demo 的 seconds 含初始化和训练；不能作为每检查的正式推理时延。真实 MIMIC 与 MS-CXR 的访问、去重、版本与病例选择需另行核验。 |

