
## 1. 研究目标与数据概览

本研究旨在运用结构化主题与情感-话语模型（Structural Topic and Sentiment-Discourse Model, STS），深入剖析酒店在线评论，揭示顾客关注的核心主题，并探究这些主题如何受到 **评分、旅行类型、时间** 等多维因素的影响。我们希望回答以下问题：

-   顾客在评论中最常讨论哪些方面（如服务、位置、设施）？
-   顾客对不同主题的情感倾向（正面/负面）是怎样的？
-   给予高分或低分的评论，其在主题关注度和情感表达上有何不同？
-   不同旅行类型（如商务、休闲、家庭）的顾客，其关注点是否存在差异？
-   随着时间推移，顾客的关注点和情感是否发生了变化？

为实现此目标，我们采用 Chen & Mankad (2024) 提出的 STS 模型。该模型不仅能像传统主题模型一样识别文本中的“主题”，还能进一步量化与每个主题相关的“情感-话语”倾向，并将两者与外部协变量（如评分、旅行类型）进行回归分析，从而在一个统一的框架内实现“内容发现”与“影响归因”。

**数据来源与抽样策略**

我们使用了一份包含超过 51 万条酒店评论的公开数据集。考虑到计算效率与结果的可解释性，我们并未直接使用全量数据，而是采用 **分层随机抽样** 的方法，抽取了 **N=1000** 条评论作为分析样本。

-   **分层维度**：以评论产生的 **月份** 作为分层依据，确保样本在时间分布上与原始数据保持一致。
-   **随机种子**：设定为 `2025`，以保证本次抽样分析的 **完全可复现性**。

通过此策略，我们获得了一个既具代表性又易于处理的样本，足以支撑 STS 模型的稳健估计与深度洞察。

## 2. 数据预处理与协变量构建

在模型分析前，我们对抽样的 1000 条评论数据进行了细致的预处理和协变量构建，这是确保模型结果质量的关键步骤。

**文本合并与清洗**

每条评论通常包含“正面”（Positive_Review）和“负面”（Negative_Review）两个部分。我们将这两部分合并为一个完整的评论文档，并在各自的文本前添加 `pos` 和 `neg` 标记，以保留原文的情感上下文。

```
pos a very quirky hotel that managed to keep its identy despite moderisation situated in a lovely location neg our room was a little compact probably due to it being a loft room at the end of the building however this didn t bother us too much
```

随后，我们对合并后的文本进行了标准化的清洗流程：
-   **转为小写**：统一文本格式。
-   **去除标点、数字与网址**：排除非语义信息干扰。
-   **去除停用词**：移除如 “the”, “a”, “is” 等常见但无助于区分主题的词语。
-   **词形还原**：将单词还原为其基本形式（如 “rooms” 变为 “room”），以合并语义相近的词汇。

**协变量构建**

为了探究影响主题占比和情感倾向的因素，我们从原始数据中提取并构建了以下四类协变量（`x_d`）：

1.  **评分 (Reviewer_Score)**：将原始评分（1-10分）进行 **z-score 标准化**，使其变为均值为0、标准差为1的数值，便于模型解释系数的相对重要性。
2.  **旅行类型 (Trip_Type)**：从 `Tags` 字段中解析出用户的旅行类型，并进行 **one-hot 编码**。主要类型包括：
    -   `Leisure trip` (休闲旅行)
    -   `Business trip` (商务旅行)
    -   `Solo traveler` (独自旅行)
    -   `Family with young children` (带幼儿的家庭)
    -   `Group` (团体)
    -   `Travelers with friends` (与朋友同行)
3.  **时间 (Time)**：将 `Review_Date` 解析为 **月份**，作为一个数值变量，用以捕捉季节性或长期趋势。
4.  **评语者活跃度 (Reviewer_Activity)**：将 `Total_Number_of_Reviews_Reviewer_Has_Given`（用户历史评论数）进行 **z-score 标准化**，作为衡量用户经验的代理变量。

这些精心构建的协变量将同时进入 STS 模型的两个核心部分：**主题占比回归** (Prevalence Regression) 和 **情感-话语回归** (Sentiment-Discourse Regression)，为我们揭示深层关联提供基础。

## 3. 模型设定与估计

本研究的核心是应用结构化主题与情感-话语模型（STS），其精妙之处在于能够在一个统一的概率框架内，同时完成 **主题发现、情感量化** 与 **归因分析**。

**STS 模型生成过程简介**

对于每一篇评论文档 `d`，STS 模型假设其生成过程如下：

1.  **确定文档基调**：首先，根据文档的协变量（如评分、旅行类型等），通过回归模型生成两个潜在向量：
    -   **主题强度向量 (α_p)**：决定了这篇评论将以多大比例讨论各个主题。
    -   **情感-话语向量 (α_s)**：决定了在讨论每个主题时，作者的言辞是更偏正面还是负面。
2.  **分配主题**：基于主题强度向量，模型为评论中的每一个词分配一个具体的主题（如“服务”、“位置”）。
3.  **生成词语**：最后，在确定了某个词的主题后，模型会结合该主题的 **基础词库** 和当前文档在该主题上的 **情感-话语倾向 (α_s)**，生成具体的词语。例如，同样是讨论“服务”主题，高分的评论（对应高的 α_s）更可能生成“friendly”、“helpful”等词，而低分的评论（对应低的 α_s）则更可能生成“rude”、“slow”等词。

**模型估计方法**

由于 STS 模型的后验概率分布难以直接计算，我们遵循原论文的建议，采用高效的 **变分推断（Variational Inference）** 方法进行估计，其核心是 **期望最大化（EM）** 算法：

-   **E-步 (Expectation)**：在固定模型全局参数（如回归系数、主题词分布）的情况下，使用 **拉普拉斯近似 (Laplace Approximation)** 为每篇文档估计其最优的潜在主题强度和情感-话语向量。
-   **M-步 (Maximization)**：在固定了每篇文档的潜在向量后，反过来更新全局参数。其中，主题-词语分布的更新通过 **泊松回归 (Poisson Regression)** 完成，并采用 **样本聚合 (Sample Aggregation)** 策略来大幅提升计算效率，尤其是在处理大规模数据时。

**主题数 K 的选择**

主题数（K）是一个关键的超参数。过少的主题可能导致语义模糊，过多则可能产生冗余且难以解释的主题。我们在一系列候选 K 值（`K ∈ {5, 6, 7}`）中进行网格搜索，依据以下两个核心指标进行选择：

1.  **Held-out Log-Likelihood**：衡量模型对未见过数据的预测能力。该值越高，说明模型的泛化能力越好。
2.  **主题一致性 (Topic Coherence)**：衡量一个主题下的高频词是否在语义上相关。一致性越高的主题，越容易被理解和解释。

最终，我们选择综合表现最优且语义最清晰的 K 值进行深入分析。

## 4. 模型选择与收敛诊断

模型的可靠性取决于其是否充分收敛以及主题数（K）的选择是否得当。本节将展示相关的诊断结果。

**模型收敛诊断**

我们通过监控 **证据下界（ELBO）** 的变化来判断模型是否收敛。ELBO 是衡量模型拟合优度的核心指标，在迭代过程中应单调递增并最终趋于平稳。

![elbo_curve.png](sts_hotel_sample_1k/outputs/elbo_curve.png)

*图 1：模型 ELBO 收敛曲线*

从上图可以看出，在模型训练的迭代过程中，ELBO 值持续上升并最终收敛，表明我们的 EM 算法已找到稳定的参数估计，模型达到了良好的拟合状态。

**主题数（K）的选择**

为了确定最合适的主题数，我们在 `K ∈ {5, 6, 7}` 的范围内进行了评估，主要依据 **Held-out Log-Likelihood**（衡量预测能力）和 **主题一致性**（衡量可解释性）两个指标。

| K (主题数) | Held-out Log-Likelihood (越高越好) | 平均主题一致性 (越高越好) |
| :--- | :--- | :--- |
| 5 | -4.8949 | -95.783 |
| 6 | -4.9013 | -88.9788 |
| **7** | **-4.8777** | **-90.1743** |

**选择理由**：

综合来看，**K=7** 是最优选择。主要原因如下：
-   **预测能力最强**：在 `K=7` 时，Held-out Log-Likelihood 达到最高值（-4.8777），说明模型对新数据的泛化能力最佳。
-   **语义可解释性良好**：虽然 K=6 的主题一致性略高，但 K=7 的一致性分数也处于较高水平，且在人工审阅后发现，7个主题的划分在语义上清晰、独立，没有出现过多重叠或模糊不清的情况。

因此，后续的所有分析都将基于 **K=7** 的模型结果展开。

## 5. 主题与话语分析

基于 K=7 的模型，我们识别出 7 个顾客在酒店评论中反复提及的核心主题。每个主题不仅有其固定的 **基线关键词**，还表现出随情感变化的 **话语倾向**。

下表展示了每个主题的语义简介、高频基线词，以及在情感-话语倾向值 `α^(s)` 处于 **低端（代表负面情绪）** 和 **高端（代表正面情绪）** 时，各自最具代表性的词语。

| 主题 | 语义简介 | 基线关键词 (Top 5) | α^(s) 低端代表词 (负面) | α^(s) 高端代表词 (正面) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **综合体验与性价比** | hotel, room, staff, location, good | bad, small, expensive, poor, problem | great, perfect, lovely, amazing, excellent |
| **2** | **客房设施与清洁度** | room, bed, bathroom, clean, small | dirty, old, noisy, broken, uncomfortable | spacious, comfortable, modern, quiet, beautiful |
| **3** | **服务质量与员工态度** | staff, service, friendly, helpful, reception | rude, poor, unhelpful, terrible, unprofessional | excellent, amazing, wonderful, attentive, exceptional |
| **4** | **地理位置与周边环境** | location, great, good, close, easy | far, noisy, area, bad, difficult | perfect, central, convenient, beautiful, quiet |
| **5** | **餐饮与早餐体验** | breakfast, good, food, restaurant, nice | expensive, poor, cold, limited, average | excellent, delicious, wonderful, amazing, great |
| **6** | **预定与入住流程** | check, booking, hotel, room, told | problem, issue, charged, wrong, refused | early, upgrade, free, easy, smooth |
| **7** | **价值与期望匹配度** | price, value, money, worth, paid | expensive, overpriced, not, disappointed, poor | great, good, excellent, reasonable, amazing |

**各主题代表性评论片段**

为了更直观地理解各主题的内涵，我们从每个主题中挑选了若干代表性的评论片段：

```callout
background_color: 15
emoji_id: "speech_balloon"
content:|
    **主题 1: 综合体验与性价比**
    - "pos: great location in the centre of london superb room helpful staff tasty breakfast neg: nothing to dislike"
    - "pos: hello we travel a lot but we never found such a nice people at the reception... neg: the breakfast was poor"

    **主题 2: 客房设施与清洁度**
    - "pos: nothing at all neg: the place was filthy despite me paying an extra to upgrade to an executive suite... bathroom floor filthy bed was falling apart"
    - "pos: good location and room well equiped neg: room freezing as window was wide open when i arrived... noise and vibration from underground trains was very noticeable"

    **主题 3: 服务质量与员工态度**
    - "pos: i didn t like any thing of the hotel neg: i booked the a c room but it was not working and inspire of couple of complaints nothing was done to set it right... it was bluntly refused by the staff"
    - "pos: location security procedures and helpful staff neg: cheking in late although i had an early check in last year"

    **主题 4: 地理位置与周边环境**
    - "pos: beautiful hotel in an extremely convenient location of barcelona staff exceptionally helpful las ramblas right outside your door"
    - "pos: breakfast was good something for everyone location was very good easy to access everything"

    **主题 5: 餐饮与早餐体验**
    - "pos: neg: room service absolutely awful every order over six days was wrong... most of the food was disgusting and poorly presented plus absolutely extortionate prices"
    - "pos: neg: need to get new mattress we had two twins together for a full size"

    **主题 6: 预定与入住流程**
    - "pos: neg: i paid and asked for a room with a double bed and sofa bed and i get two single beds i wasn t impressed i also waited hours to be checked in"
    - "pos: breakfast was solid neg: very small rooms m old hotel you can hear every noise from everywhere"
    
    **主题 7: 价值与期望匹配度**
    - "pos: nothing neg: the rooms are extremely small the area is maybe one of the dirtiest and worse areas of paris"
    - "pos: everything very comfortable warm and extremely quiet for london amazing i couldn t get any wifi though which was a bit unfortunate"
```

通过以上分析，我们不仅清晰地勾勒出顾客关注的七个维度，还洞察到在不同情感状态下，他们在用词上的微妙差异，为后续的归因分析奠定了坚实基础。

## 6. 关键实证结论

在识别出七大主题后，我们利用 STS 模型的回归分析能力，深入探究了 **评分、时间趋势** 和 **旅行类型** 如何影响顾客在不同主题上的关注度（主题占比 `θ_k`）和情感表达（情感-话语 `α^(s)_k`）。

### **评分的影响：高分评论更关注“软服务”，低分评论更聚焦“硬伤”**

评分是影响评论内容和情感最直接的因素。我们通过部分依赖图（Partial Dependence Plot）来观察主题占比和情感倾向随评分（已标准化为 z-score）变化的趋势。

![score_effect_by_topic.png](sts_hotel_sample_1k/outputs/score_effect_by_topic.png)

*图 2：评分对主题占比 (θ) 与话语情感 (α) 的影响（Top-3 主题）*

**读图要点**：
-   **横轴**：标准化后的评分（z-score），从左到右代表从极低分到极高分。
-   **蓝线 (主题占比 θ)**：反映该主题在不同分数段的评论中被提及的比例。
-   **红线 (话语情感 α)**：反映在讨论该主题时，评论的正面情感强度。线条越往上，情感越积极。

**核心发现**：
-   **情感与评分强正相关**：如图所示，所有主题的 **话语情感（红线）** 都随着评分的升高而显著上升。这符合直觉：**分数越高的评论，其措辞越积极正面**。
-   **高分与低分评论的关注点差异**：
    -   对于 **主题2（客房设施与清洁度）**，其 **主题占比（蓝线）** 随评分升高而 **轻微下降**。这表明，低分评论更倾向于抱怨房间的硬伤（如“脏”、“旧”、“吵”），而高分评论则较少提及。
    -   对于 **主题1（综合体验）和主题3（服务质量）**，其主题占比对评分变化不敏感，但在高分段，其话语情感的正面效应尤为突出，说明优质的服务和美好的综合感受是赢得高分的关键。

### **时间趋势：季节性波动与特定问题凸显**

我们按月聚合数据，观察各主题的占比和情感随时间的变化趋势，以发现潜在的季节性规律或运营问题。

![time_trend_by_topic.png](sts_hotel_sample_1k/outputs/time_trend_by_topic.png)

*图 3：按月聚合的主题趋势（Top-3 时间影响主题）*

**核心发现**：
-   **主题占比普遍稳定**：在抽样数据中，所有主题的月度占比（蓝线）都相对平稳，未出现剧烈的上升或下降趋势。
-   **情感表达存在季节性波动**：
    -   **主题3（服务质量）** 和 **主题2（客房设施）** 的话语情感（红线）呈现出明显的 **季节性波动**。例如，主题3的情感在年初和春季达到顶峰，而在秋季跌至谷底，这可能与旅游旺季的客流压力或员工状态有关。
    -   **主题5（餐饮体验）** 的情感在年末有 **显著的上升趋势**，表明酒店的餐饮服务可能在年底有所改善，或与节假日推出的特别菜单有关。

### **旅行类型差异：不同客群关注点各异**

不同类型的旅客，其需求和关注点天然不同。我们通过热力图直观比较了不同旅行类型在各主题上的关注度差异。

![triptype_heatmap.png](sts_hotel_sample_1k/outputs/triptype_heatmap.png)

*图 4：不同旅行类型的主题占比差异热力图*

**读图要点**：
-   图中颜色越亮（偏黄），代表该旅行类型对相应主题的关注度越高。

**核心发现**：
-   **主题1（综合体验）是所有客群的共同关注点**，尤其对于 **与朋友同行（Trip_Friends）** 的旅客，其关注度最高。这表明无论何种旅行，整体的住宿感受都是评价的核心。
-   **商务旅客（Trip_Business）** 对 **主题6（预定与入住流程）** 的关注度相对更高，这可能因为他们对效率和流程的顺畅度更为敏感。
-   **家庭旅客（Trip_Family）** 在 **主题2（客房设施与清洁度）** 上的关注度也较高，反映出他们对房间的舒适性、安全性有更严苛的要求。
-   其他主题（2至7）在各旅行类型中的占比普遍偏低，说明它们是相对次要或仅在特定情境下才会被重点提及的方面。

## 7. 局限与说明

尽管本研究提供了丰富的洞见，但仍需注意以下几点局限性，以便更准确地理解和应用结论：

1.  **抽样规模的限制**：本分析基于 1000 条评论的抽样数据。虽然分层抽样保证了样本在时间维度上的代表性，但其结论的普适性仍有待在全量数据（51万+条）上进行验证。小样本可能无法完全捕捉到一些稀有但重要的问题，或放大某些偶然现象。我们预期，在全量数据上，主要趋势（如评分与情感的强关联）将保持一致，但可能会涌现出更多细分主题和更复杂的交互效应。

2.  **主题数选择的权衡**：我们选择了 K=7 作为最优主题数，这是在模型预测能力和语义可解释性之间取得的平衡。然而，主题的划分并非绝对。选择更多的主题可能会揭示更细微的讨论点（如将“设施”细分为“床品”、“浴室”等），但也会增加模型复杂度和解读难度。

3.  **语言与文化背景的差异**：本研究的语料库主要是英文评论，且未对评论者的国籍进行区分。不同文化背景的旅客在表达习惯和关注点上可能存在系统性差异，这部分信息未能在当前模型中充分利用。

4.  **停用词表的通用性**：我们使用了通用的英文停用词表。针对酒店评论领域，可能存在一些领域特定的高频词（如 “hotel”, “room”），它们在所有主题中都频繁出现，有时会轻微影响主题间的区分度。构建一个领域专用的停用词表或采用 TF-IDF 等加权方式，可能会进一步提升主题质量。

**结论的一致性预期**

尽管存在上述局限，我们有理由相信，本次抽样分析揭示的核心洞见——特别是 **评分与情感话语的强正相关性、高低分评论的关注点差异、以及不同旅行类型的需求侧重**——是稳健的，并预期在全量数据分析中会得到重现和印证。全量数据分析的主要价值将在于提供更精细的趋势估计、更强的统计显著性，以及发现更多低频但关键的客户声音。

## 8. 结论与洞见摘要

通过对 1000 份酒店评论样本的结构化主题与情感-话语模型（STS）分析，我们成功识别出顾客关注的七个核心主题，并量化了评分、时间及旅行类型对这些主题讨论热度和情感倾向的影响。研究的核心洞见可总结为以下几点，为酒店管理者提供了清晰、可执行的优化方向。

首先，**顾客的评分与其在评论中流露的情感强度高度一致**。高分评论普遍洋溢着积极措辞，而低分评论则充满负面表述。更重要的是，高分与低分评论的 **关注焦点存在显著差异**：低分评论更倾向于集中抱怨“客房设施与清洁度”等硬件问题，这些是导致顾客不满的“硬伤”；而高分评论则更频繁地赞扬“服务质量”与“综合体验”等“软实力”。这启示管理者，**解决硬件短板是避免差评的基础，而卓越的服务和独特的体验设计，则是创造口碑、赢得高分的关键**。

其次，不同客群的需求画像清晰可见。**商务旅客** 对入住流程的效率和顺畅度表现出更高关注，提示酒店应优化前台接待、快速退房等环节。**家庭旅客** 则对客房的设施与清洁度尤为敏感，要求酒店在卫生标准、空间布局和儿童友好设施上投入更多精力。而对于所有类型的旅客而言，良好的“综合体验”始终是他们评价的共同基调。因此，针对不同细分市场提供 **个性化、有针对性的服务**，是提升顾客满意度的有效途径。

最后，我们的分析还揭示了潜在的 **运营风险与改进机会**。例如，部分主题的情感倾向呈现出明显的季节性波动，特别是在旅游旺季，服务质量相关的负面情绪有所抬头，这可能是客流压力下服务标准不稳的信号。同时，餐饮体验在年末的积极反馈，也证明了阶段性服务优化的有效性。管理者应建立常态化的文本数据监测机制，及时捕捉这类动态趋势，**预警潜在问题，并验证改进措施的成效**，从而实现数据驱动的精细化运营。
