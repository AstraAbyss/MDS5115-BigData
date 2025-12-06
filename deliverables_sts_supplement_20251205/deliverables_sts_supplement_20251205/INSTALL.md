# 安装与运行说明（R 项目）

本工程基于 R 语言，采用 STS（Structural Topic and Sentiment-Discourse）模型进行主题与情感-话语的联合估计，并在文档层对潜变量进行稳健回归（HC 标准误与 p 值）。

## 依赖包
请在 R 环境中安装以下 CRAN 包：

- data.table
- stringr
- lubridate
- dplyr
- tm
- stm
- sts  （CRAN 上可用）
- jsonlite
- ggplot2
- sandwich  （稳健标准误）
- lmtest     （显著性检验）
- tidyr      （在 sts_plot.R 中使用）

安装示例：
```r
install.packages(c(
  "data.table","stringr","lubridate","dplyr","tm","stm","sts",
  "jsonlite","ggplot2","sandwich","lmtest","tidyr"
))
```

## 数据
- 输入样本数据：`project_src/data/Hotel_Reviews_1k_sample.csv`
  - 若存在大表 `Hotel_Reviews 1.csv`，可自行采样生成；当前流程直接使用 1k 样本。

## 运行
在项目根目录下执行：
```bash
bash project_src/scripts/run.sh
```
该脚本将依次运行：
1. `scripts/sts_fit_ab.R`：分别拟合两套模型（A：月份连续；B：季节三哑变量），并对潜变量（prevalence / sentiment）进行稳健回归，输出系数、标准误与 p 值的 CSV，以及关键交互项系数图。
2. `scripts/sts_plot.R`：基于默认系数文件绘制评分影响与时间趋势等图。

所有输出将写入 `project_src/outputs/`，并带时间戳避免覆盖原有结果。

## 说明
- 文本预处理阶段不再人为加入 "POS:"/"NEG:" 标签，仅将正负面文本直接拼接清洗。
- Tags 的组内互斥编码：
  - 旅游类型：Leisure vs Business（使用单一 one-hot：TripType_Leisure，基类为 Business/其他）
  - 同行类型：Group / Solo / Family（取 3 个 one-hot，基类为 Couple）
  - 订房类型：Submit_Mobile（从标签中识别 "Submitted from a mobile device"）
- 两套时间变量模型：
  - A：`month_idx` 连续变量
  - B：季节哑变量 `Season_Spring`/`Season_Summer`/`Season_Autumn`（基类为冬季）
- 交互项：`TripType_Leisure:Submit_Mobile` 至少纳入；季节交互可按需添加。
