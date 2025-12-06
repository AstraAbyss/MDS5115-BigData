
# STS 酒店评论实证分析补充说明 (2025-12-05)

本文档旨在对 `sts_supplement_20251205.zip` 压缩包内的交付物进行说明，主要包含**补充数据、代码改动、运行指南、变量编码**

## 1. 交付物清单与意见对齐


| 交付物条目 | 对应意见 | 说明 |
| :--- | :--- | :--- |
| `outputs/derived_features_1k.csv` | **2.2 变量分组, 2.3 变量修改, 2.5 研究问题** | 新增的派生特征文件，包含了要求的全部编码变量，如 `TripType_Leisure`、`Comp_*` one-hot、`Submit_Mobile`、`month_idx`、`Season_*` one-hot、`reviewer_activity_z` 等。 |
| `analysis_sts_python/run_analysis.py` | **2.1-2.5, 3. 输出内容** | 新增 Python 脚本，实现了两套模型（A: 月份连续, B: 季节分类）的回归分析，并输出了系数、稳健标准误和 p 值。 |
| `project_src/scripts/sts_fit_ab.R` | **同上** | 在原 R 语言工程基础上扩展的脚本，同样实现了两套模型设定与稳健回归，供 R 环境下复现。 |
| `outputs/regression_results_A_*.csv` | **3. 输出内容** | 模型 A（月份连续）的完整回归结果，包含每个主题下流行度（prevalence）和情感-话语（sentiment）作为因变量的系数、稳健 SE、p 值。 |
| `outputs/regression_results_B_*.csv` | **3. 输出内容** | 模型 B（季节分类）的完整回归结果，结构同上。 |
| `outputs/key_coef_interaction_*.png` | **2.4 交互效应, 2.5 研究问题 4** | 关键交互项（休闲旅游 × 移动端预订）对各主题情感-话语影响的可视化，用于检验“线上预订的休闲旅游者是否更宽容”。 |

## 2. 如何运行

提供了 Python 和 R 两种运行路径，均可独立生成分析结果。

### Python (推荐，已验证)

- **入口命令**: `python3 analysis_sts_python/run_analysis.py`
- **依赖**: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`。已生成 `requirements.txt`，可使用 `pip install -r requirements.txt` 安装。
- **输出位置**: 运行后，最新的回归结果 CSV 和图表 PNG 将生成在 `outputs/` 目录下，并以时间戳命名以避免覆盖。

### R (备用，供 R 环境复现)

- **入口命令**: `bash project_src/scripts/run.sh`
- **依赖**: `data.table`, `stringr`, `lubridate`, `dplyr`, `tm`, `stm`, `sts`, `jsonlite`, `ggplot2`, `sandwich`, `lmtest`, `tidyr`。详细安装说明见 `INSTALL.md`。
- **输出位置**: 结果同样输出至 `outputs/` 目录，并以时间戳命名。

## 3. 变量编码与模型设定

### 变量编码

- **`TripType_Leisure`**: 休闲 vs. 商务旅行。当 `Tags` 中含 `Leisure trip` 时为 1，`Business trip` 或其他情况为 0（基类）。
- **`Comp_Group`, `Comp_Solo`, `Comp_Family`**: 同行者类型。从四个互斥类别（Group, Solo, Family, Couple）中选择三个进行 one-hot 编码，以 `Couple` 为基类。
- **`Submit_Mobile`**: 是否移动端预订。当 `Tags` 中匹配 `Submitted from a mobile device` 或 `Submit from mobile` 时为 1。
- **`month_idx`**: 月份索引。直接使用数据中已有的 `month_idx` 列，若缺失则从 `Review_Date` 提取月份。
- **`Season_Spring`, `Season_Summer`, `Season_Autumn`**: 季节。从 `Review_Date` 提取的月份转换而来，进行 one-hot 编码，以 `Winter` 为基类。
- **`reviewer_activity_z`**: 评论者活跃度。对 `Total_Number_of_Reviews_Reviewer_Has_Given` 列进行 z-score 标准化处理。

### 两套回归模型

为了对比时间效应，我们设定了两套模型：

- **模型 A (月份连续)**:
  `Y ~ 评分 + TripType_Leisure + Comp_* (3) + Submit_Mobile + month_idx (连续) + reviewer_activity_z + 交互项`
- **模型 B (季节分类)**:
  `Y ~ 评分 + TripType_Leisure + Comp_* (3) + Submit_Mobile + Season_* (3) + reviewer_activity_z + 交互项`

其中 `Y` 分别为 **主题流行度 (prevalence)** 和 **主题情感-话语 (sentiment-discourse)** 的代理变量。所有模型的回归结果均包含**系数 (coef)**、**稳健标准误 (robust SE)** 和 **p 值 (p-value)**，可以直接用于制作三线表。

### 交互项

根据意见，模型中均包含了 `TripType_Leisure * Submit_Mobile` 交互项，以检验“线上预订的休闲旅游者”这一特定群体的行为模式。

## 4. 缺失字段与代理处理

- **评论者活跃度**: `Total_Number_of_Reviews_Reviewer_Has_Given` 字段存在，直接使用并进行 z-score 标准化，无需代理。
- **日期与月份**: `Review_Date` 与 `month_idx` 字段均存在，优先使用 `Review_Date` 提取季节，使用 `month_idx` 作为连续时间变量。
- **其他字段**: `Tags` 和 `Reviewer_Score` 字段完整，可直接用于特征工程。

总体而言，本次分析所需的核心字段均存在于 `Hotel_Reviews_1k_sample.csv` 中，未动用复杂的代理处理。

## 5. 压缩包文件树

`sts_supplement_20251205.zip` 包含以下文件和目录结构：

```
.
├── analysis_sts_python/
│   ├── run_analysis.py         # (新增) Python 主运行脚本
│   └── run_sts_python.py       # (保留) 原 Python 脚本
├── project_src/
│   ├── data/
│   │   └── Hotel_Reviews_1k_sample.csv # (可选包含) 原始数据
│   └── scripts/
│       ├── run.sh              # (新增) R 一键运行脚本
│       ├── sts_fit_ab.R        # (新增) R A/B 模型拟合脚本
│       └── ...                 # (保留) 原 R 脚本
├── outputs/
│   ├── derived_features_1k.csv # (新增) 派生特征数据
│   ├── regression_results_A_*.csv # (新增) A 模型回归结果
│   ├── regression_results_B_*.csv # (新增) B 模型回归结果
│   ├── key_coef_interaction_A.png # (新增) A 模型交互项图
│   └── key_coef_interaction_B.png # (新增) B 模型交互项图
├── deliverables/
│   └── README_说明.md         # (本文件) 说明文档
├── requirements.txt            # (新增) Python 依赖
└── INSTALL.md                  # (新增) R 依赖与安装说明
```
