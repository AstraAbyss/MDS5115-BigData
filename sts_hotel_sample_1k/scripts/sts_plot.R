# STS plotting on 1k sample
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(dplyr)
  library(ggplot2)
})

set.seed(2025)

# Paths
args <- commandArgs(trailingOnly = FALSE)
fileArg <- "--file="
script_path <- sub(fileArg, "", args[grep(fileArg, args)])
script_dir <- if (length(script_path) > 0) dirname(script_path) else getwd()
root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
output_dir <- file.path(root_dir, "outputs")
input_sample <- file.path(root_dir, "data", "Hotel_Reviews_1k_sample.csv")
prev_json <- file.path(output_dir, "coeffs_prevalence_sts.json")
sent_json <- file.path(output_dir, "coeffs_sentiment_sts.json")
held_json <- file.path(output_dir, "heldout_ll.json")

# Load coeffs
prev <- fromJSON(prev_json)
sent <- fromJSON(sent_json)
K <- prev$K

# Build Gamma matrices
Gp <- matrix(NA_real_, nrow = length(prev$topics$coefs[[1]]), ncol = K-1)
Gs <- matrix(NA_real_, nrow = length(sent$topics$coefs[[1]]), ncol = K)
row_names <- c("Intercept","Reviewer_Score_z","Trip_Leisure","Trip_Business","Trip_Solo","Trip_Family","Trip_Group","Trip_Friends","Trip_Pet","month_idx","Reviews_Given_z")
colnames(Gp) <- paste0("Topic_", 1:(K-1))
colnames(Gs) <- paste0("Topic_", 1:K)
rownames(Gp) <- row_names
rownames(Gs) <- row_names
for (j in seq_len(K-1)){
  Gp[, j] <- as.numeric(prev$topics$coefs[[j]])
}
for (j in seq_len(K)){
  Gs[, j] <- as.numeric(sent$topics$coefs[[j]])
}

# Load sample data and construct X
DT <- fread(input_sample)
# Trip types
tag_has <- function(s, pat){ if (is.na(s)) FALSE else grepl(pat, s, ignore.case = TRUE) }
DT[, Trip_Leisure := vapply(Tags, tag_has, logical(1), pat = "Leisure trip")]
DT[, Trip_Business := vapply(Tags, tag_has, logical(1), pat = "Business trip")]
DT[, Trip_Solo := vapply(Tags, tag_has, logical(1), pat = "Solo traveler")]
DT[, Trip_Family := vapply(Tags, tag_has, logical(1), pat = "Family with young children")]
DT[, Trip_Group := vapply(Tags, tag_has, logical(1), pat = "Group")]
DT[, Trip_Friends := vapply(Tags, tag_has, logical(1), pat = "Travelers with friends")]
DT[, Trip_Pet := vapply(Tags, tag_has, logical(1), pat = "With a pet")]
DT[, Reviewer_Score_z := as.numeric(scale(Reviewer_Score))]
DT[, Reviews_Given_z := as.numeric(scale(Total_Number_of_Reviews_Reviewer_Has_Given))]
# Month index (fallback from days)
DT[, month_idx := lubridate::month(suppressWarnings(lubridate::mdy(Review_Date)))]
DT[is.na(month_idx), month_idx := lubridate::month(suppressWarnings(lubridate::dmy(Review_Date)))]
parse_days <- function(x){ m <- stringr::str_match(x, "(\\d+)"); if (is.na(m[1,2])) NA_integer_ else as.integer(m[1,2]) }
DT[, days_num := vapply(days_since_review, parse_days, integer(1))]
DT[is.na(month_idx) & !is.na(days_num), month_idx := pmax(1L, pmin(12L, floor(days_num/30)+1L))]
DT[is.na(Reviews_Given_z), Reviews_Given_z := 0]
DT[, month_idx := as.integer(month_idx)]

# Design vector names must match rownames(Gp/Gs)
# Build helper to compute theta and alpha_s for a given rating z and base covariates
softmax <- function(x){ ex <- exp(x - max(x)); ex/sum(ex) }

# Base covariates
base <- list(
  Intercept = 1,
  Reviewer_Score_z = 0,
  Trip_Leisure = TRUE,
  Trip_Business = FALSE,
  Trip_Solo = FALSE,
  Trip_Family = FALSE,
  Trip_Group = FALSE,
  Trip_Friends = FALSE,
  Trip_Pet = FALSE,
  month_idx = median(DT$month_idx, na.rm = TRUE),
  Reviews_Given_z = 0
)

build_x <- function(score_z){
  v <- base
  v$Reviewer_Score_z <- score_z
  # ensure ordering consistent
  x <- unlist(v)[row_names]
  as.numeric(x)
}

# Partial dependence grid over score [-2, 2]
score_grid <- seq(-2, 2, by = 0.1)
pd_list <- lapply(score_grid, function(sz){
  x <- build_x(sz)
  # prevalence alpha_p (K-1) then theta
  ap <- as.numeric(t(x) %*% Gp)
  theta <- softmax(c(ap, 0))
  # sentiment alpha_s (K)
  as <- as.numeric(t(x) %*% Gs)
  data.frame(score_z = sz, topic = seq_len(K), theta = theta, alpha_s = as)
})
pd <- bind_rows(pd_list)

# choose Top-3 topics by absolute rating effect on sentiment (largest |Reviewer_Score_z| coeff in Gs)
rating_effects <- abs(Gs["Reviewer_Score_z", ])
sel_topics <- order(rating_effects, decreasing = TRUE)[1:min(3, K)]
pd_sel <- pd %>% filter(topic %in% sel_topics)

# Plot score_effect_by_topic.png
p1 <- ggplot(pd_sel, aes(x = score_z)) +
  geom_line(aes(y = theta, color = "主题占比 θ"), size = 1.1) +
  geom_line(aes(y = (alpha_s - mean(alpha_s))/sd(alpha_s + 1e-6), color = "话语/情感 α"), size = 1.1, linetype = "dashed") +
  facet_wrap(~ topic, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c("主题占比 θ" = "#1f77b4", "话语/情感 α" = "#d62728")) +
  labs(title = "评分影响的部分依赖图（Top-3 主题）", x = "评分（z-score）", y = "标准化值", color = "曲线") +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")

ggsave(filename = file.path(output_dir, "score_effect_by_topic.png"), p1, width = 12, height = 8, dpi = 200)

# Time trend by month: predict doc-level theta and alpha_s then aggregate
# Build X for all docs
X_df <- data.frame(
  Intercept = 1,
  Reviewer_Score_z = DT$Reviewer_Score_z,
  Trip_Leisure = DT$Trip_Leisure,
  Trip_Business = DT$Trip_Business,
  Trip_Solo = DT$Trip_Solo,
  Trip_Family = DT$Trip_Family,
  Trip_Group = DT$Trip_Group,
  Trip_Friends = DT$Trip_Friends,
  Trip_Pet = DT$Trip_Pet,
  month_idx = DT$month_idx,
  Reviews_Given_z = DT$Reviews_Given_z
)
# order columns
X_mat <- as.matrix(X_df[, row_names])
alpha_p_mat <- X_mat %*% Gp  # N x (K-1)
# construct theta per doc
theta_docs <- t(apply(alpha_p_mat, 1, function(ap){ softmax(c(ap, 0)) }))
alpha_s_docs <- X_mat %*% Gs  # N x K

agg <- data.frame(month_idx = DT$month_idx)
for (k in seq_len(K)){
  agg[[paste0("theta_", k)]] <- theta_docs[, k]
  agg[[paste0("alpha_", k)]] <- alpha_s_docs[, k]
}
agg_dt <- as.data.table(agg)
agg_month <- agg_dt[, lapply(.SD, mean, na.rm = TRUE), by = month_idx, .SDcols = patterns("^(theta_|alpha_)"), ]

# select Top-3 by absolute effect of month on sentiment (proxy: std dev across months of alpha_k)
alpha_cols <- grep("^alpha_", names(agg_month), value = TRUE)
alpha_sd <- sapply(alpha_cols, function(cn) sd(agg_month[[cn]], na.rm = TRUE))
sel_topics2 <- as.integer(gsub("alpha_", "", names(sort(alpha_sd, decreasing = TRUE))[1:min(3, length(alpha_sd))]))

# long format for plotting
library(tidyr)
long_alpha <- agg_month %>% pivot_longer(cols = all_of(alpha_cols), names_to = "alpha_topic", values_to = "alpha")
long_theta <- agg_month %>% pivot_longer(cols = grep("^theta_", names(agg_month), value = TRUE), names_to = "theta_topic", values_to = "theta")
long_alpha$topic <- as.integer(gsub("alpha_", "", long_alpha$alpha_topic))
long_theta$topic <- as.integer(gsub("theta_", "", long_theta$theta_topic))
long <- long_alpha %>% inner_join(long_theta, by = c("month_idx","topic")) %>% filter(topic %in% sel_topics2)

p2 <- ggplot(long, aes(x = month_idx)) +
  geom_smooth(aes(y = theta, color = "主题占比 θ"), method = "loess", se = TRUE) +
  geom_smooth(aes(y = (alpha - mean(alpha))/sd(alpha + 1e-6), color = "话语/情感 α"), method = "loess", se = TRUE, linetype = "dashed") +
  facet_wrap(~ topic, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c("主题占比 θ" = "#1f77b4", "话语/情感 α" = "#d62728")) +
  labs(title = "按月聚合的主题趋势（LOESS）", x = "月份", y = "标准化值", color = "曲线") +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")

ggsave(filename = file.path(output_dir, "time_trend_by_topic.png"), p2, width = 12, height = 8, dpi = 200)

# Trip type heatmap: differences in mean theta by trip type
trip_types <- c("Trip_Leisure","Trip_Business","Trip_Solo","Trip_Family","Trip_Group","Trip_Friends","Trip_Pet")
heat_dt <- data.frame(trip = character(0), topic = integer(0), theta = numeric(0))
for (t in trip_types){
  idx <- which(DT[[t]] == TRUE)
  if (length(idx) < 5) next
  means <- colMeans(theta_docs[idx, , drop = FALSE], na.rm = TRUE)
  heat_dt <- rbind(heat_dt, data.frame(trip = t, topic = seq_len(K), theta = means))
}

p3 <- ggplot(heat_dt, aes(x = factor(topic), y = trip, fill = theta)) +
  geom_tile() +
  scale_fill_viridis_c(option = "C") +
  labs(title = "不同 Trip 类型的主题占比差异（热力图）", x = "主题", y = "Trip 类型", fill = "均值 θ") +
  theme_minimal(base_size = 12)

ggsave(filename = file.path(output_dir, "triptype_heatmap.png"), p3, width = 10, height = 6, dpi = 200)

# Write chart revision notes
rev_path <- file.path(output_dir, "chart_revision.md")
cat(paste0(
"# 图表修订说明\n\n",
"- 评分影响：采用部分依赖（保持 Trip=Leisure、月=样本中位数、其他协变量取均值/众数），在评分 z∈[-2,2] 上计算每主题的 \u03b8（主题占比）与 \u03b1（话语/情感）并绘制连续曲线；为便于比较，\u03b1 进行了标准化。\n",
"- 时间趋势：将每月聚合的 \u03b8 与 \u03b1 用 LOESS 平滑呈现，并附带置信带；选择了时间影响较大的 Top-3 主题以增强可读性。\n",
"- Trip 类型差异：按 Trip 标签分组计算各主题的均值 \u03b8 并以热力图呈现，直观比较休闲/商务/独行/亲子/组团/朋友/携宠等人群的关注差异。\n\n",
"读图要点：蓝色为主题占比（\u03b8），红色虚线为情感/话语（\u03b1）的标准化值；两者同时观察能更清晰地理解“说了什么”和“如何表达”。\n"
), file = rev_path)
