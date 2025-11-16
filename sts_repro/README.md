# STS Repro — 结构化主题与情感话语模型复现项目

本项目复现论文《A Structural Topic and Sentiment-Discourse Model for Text Analysis》中提出的 STS 模型，提供从原始文本到结果可视化的一条龙流程：数据预处理 → Anchor 初始化 → 变分 E 步（拉普拉斯近似） → 泊松聚合 M 步 → 收敛与输出。

---

## 1. 项目简介与核心方法概述

STS（Structural Topic and Sentiment-Discourse）在传统主题模型基础上引入文档层潜变量：
- **主题流行度 a^(p)**：通过 softmax 给出文档中的主题权重 θ。
- **情感/话语 a^(s)**：按话题对词分布进行“风格”调制，系数由 \(\kappa^{(s)}\) 控制。
- 两者均受文档协变量（如评分、时间、用户属性）驱动的高斯先验 \(\mathcal{N}((x\Gamma)^\top, \Sigma)\)。

推断采用 **拉普拉斯近似 + EM**：
- **E 步**：在 \(f(a)\) 的极值点二阶展开，得到近似高斯后验（均值是模态，协方差近似为负海森逆）。同时计算期望指派 \(\phi\)。
- **M 步**：
  - 用线性回归更新 \(\Gamma\)，用平均协方差 + 残差外积更新 \(\Sigma\)，并做谱裁剪确保正定；
  - 利用“多项式→泊松回归”的等价关系，并引入**样本聚合**（按文档分组汇总计数），用 IRLS 高效估计话题的 \(\kappa^{(t)}, \kappa^{(s)}\)。

---

## 2. 目录结构说明

```
sts_repro/
  run_demo.py                  # 演示脚本：加载数据、运行 EM、生成可视化
  sts/
    __init__.py               # 包初始化
    data.py                   # 分词、词汇表、文档-词计数 (DTM)
    anchor.py                 # 锚点词初始化（SPA + 简单形恢复）
    model.py                  # 生成过程、目标函数、梯度与数值海森
    variational.py            # 变分 E 步（拉普拉斯近似优化 a_d）
    poisson.py                # 样本聚合 + 泊松 IRLS 更新 κ
    em.py                     # EM 主循环、Γ/Σ 更新、收敛监控
    viz.py                    # 结果可视化（Top 词、情感-评分散点）
  tests/
    basic_test.py             # 基础形状与前向计算测试

output/
  top_words.png               # 主题 Top 词图（中位情感水平）
  sentiment_vs_rating.png     # 主题情感-评分关系散点
  sts_workflow.mermaid        # 工作流流程图（Mermaid）
```

---

## 3. 环境依赖与安装步骤

建议使用 Python 3.10 或 3.11。

必需/推荐依赖：
- numpy
- matplotlib
- seaborn
- datasets（可选，用于加载 HuggingFace 的 Yelp Review Full 数据集；无法联网时会自动切换到极小演示集）

安装命令：
```
pip install -U pip setuptools wheel
pip install numpy matplotlib seaborn datasets
```
如需指定镜像源或离线安装，请按企业环境调整 pip 参数。

---

## 4. 数据准备与示例

- 默认演示从 HuggingFace 加载 Yelp Review Full 的前 400 条训练样本；评分（1-5 星）作为文档级协变量（另含截距）。
- 若在线加载失败，脚本将自动使用内嵌的**极小演示样本**（十余条合成文本+评分），用于验证链路可运行。
- 替代路径：
  - 将自有文本数据整理为 `text` 与协变量（如 `stars`、`intercept`）的结构，参考 `run_demo.py` 的 `load_yelp_reviews()` 返回 `Document` 列表即可；
  - 调整 `build_vocabulary()` 的 `max_vocab` 与 `min_df` 以适配文本规模与稀疏度。

---

## 5. 运行方式与脚本说明

在工作区根目录执行：
```
python sts_repro/run_demo.py
```
脚本流程：
1) 加载数据（在线 Yelp 或极小演示集）；
2) 构建词汇表与文档-词计数矩阵（DTM）；
3) Anchor 初始化得到初始话题-词分布 \(\hat{\beta}\)；
4) 运行 EM（变分 E 步 + 泊松聚合 M 步），并监控 ELBO 收敛；
5) 生成两类图表：主题 Top 词与主题情感-评分散点。

所有产出将保存在 `output/` 目录。

---

## 6. 关键配置与超参数

在 `run_demo.py` 中可调整：
- 主题数 **K**：默认 `K=6`
- 词汇表上限 `max_vocab=2000`、文档频率下限 `min_df=5`
- EM 最大迭代 **max_em_iter=8**、收敛阈值 **tol=1e-4**
- 泊松样本聚合组大小 **group_size=30**（在 `EMRunner(..., group_size=30)`）

其它可调参数（位于模块内）：
- `VariationalEstimator(max_iter, tol)`：文档层拉普拉斯近似的迭代与容差；
- `PoissonAggregator(max_iter)`：每次聚合后 IRLS 的迭代步数；
- 数值裁剪范围（如 `eta`、`exp(eta)` 的裁剪，海森反演的抖动等）可按需要扩大或收紧。

这些调整不会改变算法本质，只是帮助在不同数据规模与噪声水平下获得更稳健的收敛。

---

## 7. 期望输出与结果文件说明

- `output/top_words.png`：每个话题在**中位情感水平**下的 Top 词柱状图；用于检视话题可读性与词分布的风格差异。
- `output/sentiment_vs_rating.png`：若干话题的**情感潜变量 vs 评分**散点图；用于观察协变量（评分）对话题层情感的驱动关系。
- `output/sts_workflow.mermaid`：整体流程图文件（Feishu 文档可直接预览）。
- 控制台日志：包含产出路径与基本进度提示。

---

## 8. 常见问题与故障排查

- 数值稳定性：
  - 海森矩阵可能奇异：已加入小抖动与回退步长；必要时增大 `VariationalEstimator.max_iter` 或收紧 `tol`。
  - 泊松 IRLS 溢出：对 `eta` 与 `exp(eta)` 做裁剪，加入岭正则；若仍不稳，减小组大小或词汇表上限。
- 收敛过慢或停滞：
  - 增大 EM 最大迭代或组大小；
  - 检查协变量尺度（如评分是否已合理缩放），避免先验过强或过弱。
- 内存与性能：
  - 大型语料建议减小 `max_vocab`、增大 `group_size`，并考虑分批处理；
  - 可只绘制前若干话题的散点以减少图生成时间。
- 字体与中文显示：
  - 结果图使用 `Noto Sans CJK SC`，若本地环境缺失可改为系统可用中文字体或英文字体。
- 数据加载：
  - 无法访问 HuggingFace 时会自动切换到内置演示集；如需自定义数据，参考 `Document(text, covariates)` 结构即可。

---

## 9. 许可证与致谢

- 许可证：本复现项目默认用于学习与研究目的。若用于商用或二次分发，请添加并遵守相应开源许可证（如 MIT）。
- 致谢：方法来源于论文《A Structural Topic and Sentiment-Discourse Model for Text Analysis》。数据演示使用 Yelp Review Full（如可访问）。

---

## 10. 基础测试

执行：
```
python sts_repro/tests/basic_test.py
```
包含对核心形状、软最大与目标函数前向计算的快速检查，用于验证安装与环境是否正常。
