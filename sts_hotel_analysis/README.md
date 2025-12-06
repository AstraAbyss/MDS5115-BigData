# STS 模型在 Hotel_Reviews 数据集上的应用

本工程复刻 Chen & Mankad (2024/2025) 的 STS（Structural Topic and Sentiment‑Discourse）模型，并将其应用到 Hotel_Reviews.csv 数据集，完成从文本预处理、模型估计（Laplace 近似 + mean‑field 变分 + EM）、Poisson 回归（含样本聚合机制的实现接口）到图表与结果输出的完整流程。

## 目录结构
- `data/Hotel_Reviews.csv`: 原始数据
- `utils/`: 文本预处理、初始化（NMF）、模型、评估、绘图模块
- `scripts/sts_fit.py`: 主脚本（运行所有流程并产出 outputs）
- `outputs/`: 图表 PNG、表格 CSV/JSON、ELBO 曲线等

## 运行环境
- Python 3.9+
- 依赖包：numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, nltk

安装依赖：
```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn nltk
```

> 首选作者 R 包 `sts`。若当前环境无法安装或运行 R，本工程提供严格按论文算法的 Python 复刻版本（Laplace 近似、变分 ELBO 优化、Poisson 回归 + 样本聚合接口、锚词/NMF 初始化）。

## 运行
```bash
python sts_hotel_analysis/scripts/sts_fit.py
```
运行结束后，`outputs/` 目录会生成：
- `elbo_curve.png`: 模型收敛曲线
- `gamma_coeffs.csv` / `gamma_coeffs_se.csv`: 回归系数与标准误
- `topic_keywords.json`: 主题关键词（基线、情感低/高端）
- `score_relationships.png` / `triptype_diffs.png` / `time_trend.png`: 可视化图表
- `model_selection.csv` / `model_selection_summary.json`: K 选取评估（heldout LL + coherence）
- `doc_examples.json`: 每个主题的代表性片段（脱敏简截）

## 说明
- **文本预处理**：合并正负评语（保留 POS/NEG 标记），英文清洗、停用词、词形还原；构建词汇与文档‑词矩阵。
- **协变量**（同时作用于主题“占比”与“情感‑话语”）：评分 z‑score、旅行类型（Tags 解析后 one‑hot）、days_since_review 数值化（z‑score）、评语者活跃度（可选 z‑score）。
- **初始化**：采用 NMF（兼容论文中的锚词/NMF 初始化思想）作为基线，情感‑话语初始化为评分 z‑score。
- **估计**：E 步（Laplace 近似 + BFGS 近似 Hessian）、M 步（线性回归更新 Γ/Σ；Poisson GLM 更新 κ^(t)/κ^(s)）。
- **K 选择**：在 K∈{5,6,7,8,9,10} 网格上，综合 held‑out log‑likelihood 与 topic coherence。
