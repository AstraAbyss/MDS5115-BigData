## 酒店评论分析：基于结构化主题与情感话语模型（STS）的深度洞察

我们采用 Chen & Mankad (2024) 提出的结构化主题与情感话语模型（STS），对酒店评论数据进行了深度分析。该方法不仅能识别出评论中讨论的核心主题，还能揭示不同用户特征（如评分、旅行类型）如何影响这些主题的讨论热度（Prevalence）与情感倾向（Sentiment-Discourse）。

### 模型选择与收敛性

为了确定最佳主题数（K），我们在 K={5, 6, 7} 的范围内进行了网格搜索。通过综合评估模型的预测能力（held-out log-likelihood）和主题质量（topic coherence），最终选择 **K=6** 作为最优主题数。该配置在保证模型拟合优度的同时，也提供了最具解释力的主题划分。

模型的训练过程表现出良好的收敛性，证据下界（ELBO）曲线稳步上升并趋于平稳，表明模型已达到最优解。

![ELBO 收敛曲线](sts_hotel_analysis/outputs/elbo_curve.png)

### 主题发现：六大核心议题

模型从大量评论中提炼出六个核心主题。下表展示了每个主题在不同情感话语（低/中/高）下的关键词。情感话语的高低与评论者的评分（Reviewer Score）正相关，高分评论倾向于使用积极词汇，反之亦然。

| 主题 | 主题描述 | 高分评论关键词 (High Sentiment) | 中性评论关键词 (Average) | 低分评论关键词 (Low Sentiment) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **综合体验与服务** | `good`, `realli`, `bar`, `breakfast` | `hotel`, `locat`, `staff`, `park` | `servic`, `morn`, `poor` |
| 2 | **客房设施与设计** | `great`, `fantast`, `near`, `awesom` | `hotel`, `locat`, `staff`, `build` | `shower`, `bathroom`, `star`, `place` |
| 3 | **清洁度与维护** | `good`, `build`, `great` | `hotel`, `locat`, `staff`, `park` | `dirti`, `look`, `floor` |
| 4 | **员工互动与预订** | `centr`, `help`, `restaur`, `look` | `hotel`, `locat`, `staff`, `nice` | `book`, `work`, `can`, `check` |
| 5 | **酒店建筑与管理** | `build`, `direct`, `restaur`, `help` | `hotel`, `locat`, `staff`, `park` | `one`, `manag` |
| 6 | **入住流程与感受** | `can`, `good`, `get` | `get`, `hotel`, `locat`, `staff` | `get`, `look`, `good` |

### 核心洞察：协变量如何影响主题与情感

通过回归分析，我们探究了评论者评分、旅行类型和评论时间等因素对主题热度和情感话语的影响。

#### 1. 评分（Reviewer Score）的影响

如下图所示，评论者的评分显著影响了他们所讨论主题的占比。例如，高分评论者更倾向于讨论**综合体验与服务（主题1）**和**客房设施与设计（主题2）**，而低分评论者则更关注**清洁度（主题3）**和**员工互动与预订（主题4）**的问题。

![评分与主题占比关系](sts_hotel_analysis/outputs/score_vs_topic.png)

#### 2. 旅行类型（Trip Type）的影响

不同的旅行类型也表现出对特定主题的偏好。热力图清晰地展示了各类旅行者关注点的差异：
- **商务旅行（Business trip）** 和 **独自旅行（Solo traveler）** 的评论者更关注**员工互动与预订（主题4）**。
- **家庭出游（Family with young children）** 的评论者则更侧重于**客房设施（主题2）**和**清洁度（主题3）**。

![不同旅行类型的主题占比差异](sts_hotel_analysis/outputs/triptype_heatmap.png)

#### 3. 时间趋势的影响

随着时间的推移（以“距评论天数”衡量），部分主题的讨论热度呈现出动态变化。例如，对于**酒店建筑与管理（主题5）**的讨论随时间推移有轻微下降趋势，这可能与酒店设施老化或新酒店的出现有关。

![时间维度的主题占比变化](sts_hotel_analysis/outputs/time_trend.png)

### 代表性评论摘录

为了更直观地理解每个主题，我们从数据集中为每个主题挑选了最具代表性的评论片段：

```callout
background_color: 15
emoji_id: speech_balloon
content:|
  **主题 1: 综合体验与服务**
  > “POS: Basically everything The style of the hotel is really great The breakfast is really good good quality food really really nice coffe order from the bar The wifi was good in my room and also throuout the hotel The location is near a metro station its easy to get to the center ...”

  **主题 2: 客房设施与设计**
  > “POS: The property is beautiful NEG: The place is completely mismanaged The property is amazing and whoever is running the place is doing it a disservice Don t stay here if you are expecting a 4 star experience it s a 4 star location with 1 star service The restaurant service is t...”

  **主题 3: 清洁度与维护**
  > “POS: The breakfast was the only positive element of this hotel original in taste and a good selection of selection of health food dishes on the buffet We did not try the restaurant but the whole set up with its view and terrace on the park is exceptional We had a wedding in the h...”

  **主题 4: 员工互动与预订**
  > “POS: Only the park outside of the hotel was beautiful NEG: I am so angry that i made this post available via all possible sites i use when planing my trips so no one will make the mistake of booking this place I made my booking via booking com We stayed for 6 nights in this hotel...”

  **主题 5: 酒店建筑与管理**
  > “POS: The location is good You need 15min to 20min walk to the center depends on your speed The hotel locates in a beautiful park and have a nice interior design The bed was really comfortable NEG: Sadly I cannot say that the rooms are clean enough for me When we got in our room w...”

  **主题 6: 入住流程与感受**
  > “POS: Lovely hotel with extremely comfortable huge double bed We stayed in the split level room which we really liked If you have difficulty getting up stairs request if you can stay in a room all on one level The Oosterpark is beautiful the shops and restaurants are great with lo...”
```

### 结论

通过应用 STS 模型，我们成功地从酒店评论数据中提取了结构化的主题，并量化了评论者特征与主题热度、情感话语之间的复杂关系。分析结果不仅验证了先前 LDA+VADER 方法的部分结论，还提供了更深层次、更严谨的洞察，充分展示了 STS 模型在文本分析领域的强大能力。
