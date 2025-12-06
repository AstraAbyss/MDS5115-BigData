# STS fitting and analysis for Hotel Reviews
# Author: Aime
# Strictly follow Chen & Mankad (2024/2025) STS via R package sts

suppressPackageStartupMessages({
  library(sts)
  library(stm)
  library(tm)
  library(jsonlite)
  library(data.table)
  library(stringr)
  library(lubridate)
  library(ggplot2)
})

# Paths
project_dir <- 'sts_hotel_analysis'
input_csv <- file.path(project_dir, 'data', 'Hotel_Reviews.csv')
outputs_dir <- file.path(project_dir, 'outputs')
if (!dir.exists(outputs_dir)) dir.create(outputs_dir, recursive = TRUE)

# Utility: safe z-score
zscore <- function(x) {
  x <- as.numeric(x)
  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(0, length(x)))
  (x - m) / s
}

# 1) Read data
DT <- fread(input_csv, encoding = 'UTF-8')
# Combine positive & negative reviews; remove placeholders
clean_placeholder <- function(x) {
  x <- gsub("(?i)^\\s*no\\s*positive\\s*$", "", x)
  x <- gsub("(?i)^\\s*no\\s*negative\\s*$", "", x)
  x <- gsub("(?i)\\bno\\s*positive\\b", "", x)
  x <- gsub("(?i)\\bno\\s*negative\\b", "", x)
  x
}
pos <- clean_placeholder(DT$Positive_Review)
neg <- clean_placeholder(DT$Negative_Review)
# Merge with segment markers
text <- paste0('POS: ', pos, ' NEG: ', neg)

# 2) Covariates
# Reviewer score z-score
DT[, Reviewer_Score_z := zscore(Reviewer_Score)]
# Total_Number_of_Reviews_Reviewer_Has_Given z-score (optional)
DT[, Total_Reviews_z := zscore(Total_Number_of_Reviews_Reviewer_Has_Given)]
# days_since_review: extract integer
DT[, days_since_review_num := as.integer(str_extract(days_since_review, '\\d+'))]
DT[is.na(days_since_review_num), days_since_review_num := NA_integer_]
# Tags parsing for Trip_Type one-hot
# Normalize Tags string (remove brackets/quotes)
normalize_tags <- function(s) {
  s <- gsub('\\"', "", s)
  s <- gsub("'", "", s)
  s <- gsub("\\n", " ", s)
  s <- gsub("\\n", " ", s)
  s <- gsub("\\[", "", s)
  s <- gsub("\\]", "", s)
  s <- tolower(s)
  s
}
DT[, Tags_norm := normalize_tags(Tags)]
trip_types <- list(
  Trip_Leisure = 'leisure trip',
  Trip_Business = 'business trip',
  Trip_Solo = 'solo traveler',
  Trip_Family = 'family with young children',
  Trip_Group = 'group',
  Trip_Friends = 'travelers with friends',
  Trip_Pet = 'with a pet'
)
for (nm in names(trip_types)) {
  pat <- trip_types[[nm]]
  DT[, (nm) := as.integer(str_detect(Tags_norm, pat))]
}

# 3) STM-style preprocessing to build corpus for sts
meta <- data.frame(
  Reviewer_Score_z = DT$Reviewer_Score_z,
  Total_Reviews_z = DT$Total_Reviews_z,
  days_since_review_num = DT$days_since_review_num,
  Trip_Leisure = DT$Trip_Leisure,
  Trip_Business = DT$Trip_Business,
  Trip_Solo = DT$Trip_Solo,
  Trip_Family = DT$Trip_Family,
  Trip_Group = DT$Trip_Group,
  Trip_Friends = DT$Trip_Friends,
  Trip_Pet = DT$Trip_Pet,
  doc_text = text,
  stringsAsFactors = FALSE
)

# Use stm::textProcessor for robust English preprocessing
# Lowercase, remove punctuation/numbers/stopwords, apply stemming
temp <- textProcessor(
  documents = meta$doc_text,
  metadata  = meta,
  lowercase = TRUE,
  removestopwords = TRUE,
  removenumbers = TRUE,
  removepunctuation = TRUE,
  stem = TRUE,
  verbose = FALSE
)

out <- prepDocuments(temp$documents, temp$vocab, temp$meta, verbose = FALSE)
# Update meta (after docs drop), ensure numeric safety
out$meta$Reviewer_Score_z <- as.numeric(out$meta$Reviewer_Score_z)
out$meta$Total_Reviews_z  <- as.numeric(out$meta$Total_Reviews_z)
out$meta$days_since_review_num <- as.numeric(out$meta$days_since_review_num)
for (nm in names(trip_types)) out$meta[[nm]] <- as.integer(out$meta[[nm]])

# Build heldout following Roberts et al. (2019) and sts::heldoutLikelihood examples
out_ho <- make.heldout(out$documents, out$vocab)
out_ho$meta <- out$meta

# 4) Grid search K in {5,6,7}
Kgrid <- c(5,6,7)
models <- list()
set.seed(123)
for (K in Kgrid) {
  message(sprintf('Fitting STS for K=%d ...', K))
  mod <- sts(
    prevalence_sentiment = ~ Reviewer_Score_z + days_since_review_num + Total_Reviews_z +
      Trip_Leisure + Trip_Business + Trip_Solo + Trip_Family + Trip_Group + Trip_Friends + Trip_Pet,
    initializationVar    = ~ Reviewer_Score_z,
    corpus = out_ho,
    K = K,
    maxIter = 200,
    convTol = 1e-5,
    initialization = 'anchor',
    kappaEstimation = 'adjusted',
    verbose = FALSE
  )
  # heldout log-likelihood
  ho <- heldoutLikelihood(mod, out_ho$missing)$expected.heldout
  # semantic coherence (UMass) average
  coh <- mean(topicSemanticCoherence(mod, out))
  models[[as.character(K)]] <- list(model = mod, held_ll = ho, coherence = coh, elbo = mod$elbo)
}

# Select best K by combined z-scored heldout + z-scored coherence
sel_tbl <- rbindlist(lapply(names(models), function(k){
  list(K = as.integer(k), held_ll = models[[k]]$held_ll, coherence = models[[k]]$coherence)
}))
# z-score columns
sel_tbl[, held_ll_z := zscore(held_ll)]
sel_tbl[, coherence_z := zscore(coherence)]
sel_tbl[, score := held_ll_z + coherence_z]
sel_tbl <- sel_tbl[order(-score)]
bestK <- sel_tbl$K[1]
final_mod <- models[[as.character(bestK)]]$model

# Save heldout_ll.json with all K and selection reason
heldout_path <- file.path(outputs_dir, 'heldout_ll.json')
sel_json <- list(
  grid = lapply(seq_len(nrow(sel_tbl)), function(i){as.list(sel_tbl[i])}),
  selected_K = bestK,
  reason = sprintf('Selected K=%d for highest combined standardized score (heldout LL + coherence), with heldout_ll=%.3f and mean_coherence=%.3f.',
                   bestK, sel_tbl[1]$held_ll, sel_tbl[1]$coherence)
)
writeLines(toJSON(sel_json, pretty = TRUE, auto_unbox = TRUE, na = 'null'), heldout_path)

# 5) ELBO curve plot for final model
elbo_df <- data.frame(iter = seq_along(final_mod$elbo), elbo = final_mod$elbo)
p <- ggplot(elbo_df, aes(x = iter, y = elbo)) +
  geom_line(color = '#1f77b4', linewidth = 0.8) +
  geom_point(color = '#1f77b4', size = 0.8) +
  theme_minimal(base_size = 12) +
  labs(title = 'ELBO 收敛曲线 (STS)', x = '迭代次数', y = 'ELBO')
ggsave(filename = file.path(outputs_dir, 'elbo_curve.png'), plot = p, width = 8, height = 5, dpi = 200)

# 6) Regression coefficients (Gamma) with uncertainties for prevalence and sentiment
regns <- estimateRegns(final_mod, ~ Reviewer_Score_z + days_since_review_num + Total_Reviews_z +
                         Trip_Leisure + Trip_Business + Trip_Solo + Trip_Family + Trip_Group + Trip_Friends + Trip_Pet,
                       out)
# regns is a list: first K are prevalence, last K are sentiment-discourse
prev_list <- regns[seq_len(bestK)]
sent_list <- regns[seq_len(bestK) + bestK]
# Convert each table to list of rows
to_list_rows <- function(df) {
  df$term <- rownames(df)
  as.list(df)
}
coeffs_prev <- lapply(seq_len(bestK), function(k){
  df <- prev_list[[k]]
  df2 <- as.data.frame(df)
  df2$term <- rownames(df2)
  rows <- lapply(seq_len(nrow(df2)), function(i){
    x <- df2[i, , drop = FALSE]
    list(term = unname(x$term), estimate = unname(x$Estimate), std_error = unname(x$`Std. Error`),
         z_value = unname(x$`z value`), p_value = unname(x$`Pr(>|z|)`))
  })
  list(topic = k, coefficients = rows)
})
coeffs_sent <- lapply(seq_len(bestK), function(k){
  df <- sent_list[[k]]
  df2 <- as.data.frame(df)
  df2$term <- rownames(df2)
  rows <- lapply(seq_len(nrow(df2)), function(i){
    x <- df2[i, , drop = FALSE]
    list(term = unname(x$term), estimate = unname(x$Estimate), std_error = unname(x$`Std. Error`),
         z_value = unname(x$`z value`), p_value = unname(x$`Pr(>|z|)`))
  })
  list(topic = k, coefficients = rows)
})
writeLines(toJSON(coeffs_prev, pretty = TRUE, auto_unbox = TRUE, na = 'null'), file.path(outputs_dir, 'coeffs_prevalence_sts.json'))
writeLines(toJSON(coeffs_sent, pretty = TRUE, auto_unbox = TRUE, na = 'null'), file.path(outputs_dir, 'coeffs_sentiment_sts.json'))

# 7) Topic keywords: baseline and low/high sentiment using printTopWords
# Capture printed output and parse
kw_capture <- capture.output(printTopWords(final_mod, n = 12))
# Parser: lines like "Topic 1 (low/avg/high): word1, word2, ..."
parse_keywords <- function(lines) {
  res <- list()
  current_topic <- NULL
  for (ln in lines) {
    ln <- trimws(ln)
    if (grepl('^Topic', ln)) {
      m <- regmatches(ln, regexec('^Topic\\s+(\\d+):\\s*(.*)$', ln))[[1]]
      if (length(m) >= 3) {
        current_topic <- as.integer(m[2])
        rest <- m[3]
        # Expect three sections separated by ';' or ' | '
        parts <- strsplit(rest, '\\|')[[1]]
        if (length(parts) < 3) {
          parts <- strsplit(rest, ';')[[1]]
        }
        # Fallback: split by '  '
        if (length(parts) < 3) {
          parts <- strsplit(rest, '\\s{2,}')[[1]]
        }
        low <- if (length(parts) >= 1) trimws(gsub('^low:?\\s*', '', parts[1])) else ''
        avg <- if (length(parts) >= 2) trimws(gsub('^avg(?:erage)?:?\\s*', '', parts[2])) else ''
        high<- if (length(parts) >= 3) trimws(gsub('^high:?\\s*', '', parts[3])) else ''
        # words comma-separated
        to_vec <- function(s) {
          ws <- trimws(unlist(strsplit(s, ',')))
          ws[nzchar(ws)]
        }
        res[[length(res)+1]] <- list(topic = current_topic, low = to_vec(low), avg = to_vec(avg), high = to_vec(high))
      }
    }
  }
  res
}
kw_list <- parse_keywords(kw_capture)
writeLines(toJSON(kw_list, pretty = TRUE, auto_unbox = TRUE, na = 'null'), file.path(outputs_dir, 'topic_keywords_sts.json'))

# 8) Examples per topic: use findRepresentativeDocs
examples_path <- file.path(outputs_dir, 'examples_per_topic.txt')
ex_lines <- c()
for (k in seq_len(bestK)) {
  docs <- findRepresentativeDocs(final_mod, out_ho$meta$doc_text, topic = k, n = 3)
  ex_lines <- c(ex_lines, sprintf('=== Topic %d ===', k))
  for (i in seq_along(docs)) {
    # clean and truncate
    txt <- docs[[i]]
    # handle list/vector cases
    if (is.list(txt)) txt <- unlist(txt)
    if (length(txt) > 1) txt <- txt[1]
    txt <- as.character(txt)
    txt <- gsub('\n', ' ', txt)
    txt <- trimws(txt)
    if (nchar(txt) > 280) txt <- paste0(substr(txt, 1, 280), '...')
    ex_lines <- c(ex_lines, paste0(i, ') ', txt))
  }
  ex_lines <- c(ex_lines, '')
}
writeLines(ex_lines, examples_path)

# 9) Figures: score_vs_topic, triptype_heatmap, time_trend
# We will use final_mod$mu as fitted alpha means; derive prevalence proportions via softmax per doc
softmax <- function(mat) {
  # mat: D x K; return D x K
  expm <- exp(mat - apply(mat, 1, max))
  expm / rowSums(expm)
}

# Try to extract prevalence alpha means from final_mod$mu
# final_mod$mu may contain concatenated prevalence and sentiment; attempt to detect
mu <- final_mod$mu
D <- nrow(out$meta)
if (is.matrix(mu)) {
  # If columns are 2K, assume first K are prevalence
  if (ncol(mu) == 2 * bestK) {
    mu_prev <- mu[, seq_len(bestK), drop = FALSE]
    mu_sent <- mu[, seq_len(bestK) + bestK, drop = FALSE]
  } else if (ncol(mu) == bestK) {
    mu_prev <- mu
    mu_sent <- matrix(NA_real_, nrow = D, ncol = bestK)
  } else {
    # Fallback: treat first K as prevalence
    mu_prev <- mu[, seq_len(bestK), drop = FALSE]
    mu_sent <- mu[, seq_len(bestK) + bestK, drop = FALSE]
  }
} else {
  # Try list structure
  mu_prev <- tryCatch({ final_mod$mu$prevalence }, error = function(e) NULL)
  mu_sent <- tryCatch({ final_mod$mu$sentiment }, error = function(e) NULL)
  if (is.null(mu_prev)) mu_prev <- matrix(0, nrow = D, ncol = bestK)
  if (is.null(mu_sent)) mu_sent <- matrix(0, nrow = D, ncol = bestK)
}
# Prevalence proportions per doc-topic
theta_hat <- softmax(mu_prev)
# Build long data frame for plotting
plot_df <- as.data.frame(theta_hat)
colnames(plot_df) <- paste0('Topic_', seq_len(bestK))
# Align meta vectors to D rows
score_vec <- out_ho$meta$Reviewer_Score_z
if (length(score_vec) >= nrow(plot_df)) score_vec <- score_vec[seq_len(nrow(plot_df))]
plot_df$Reviewer_Score_z <- score_vec

days_vec <- out_ho$meta$days_since_review_num
if (length(days_vec) >= nrow(plot_df)) days_vec <- days_vec[seq_len(nrow(plot_df))]
plot_df$days_since_review_num <- days_vec

for (nm in names(trip_types)) {
  v <- out_ho$meta[[nm]]
  if (length(v) >= nrow(plot_df)) v <- v[seq_len(nrow(plot_df))]
  plot_df[[nm]] <- v
}

# Score vs topic prevalence: plot for up to first 4 topics
plot_long <- melt(as.data.table(plot_df), id.vars = c('Reviewer_Score_z'))
colnames(plot_long) <- c('Reviewer_Score_z', 'Topic', 'Prevalence')
ps <- ggplot(plot_long, aes(x = Reviewer_Score_z, y = Prevalence)) +
  geom_point(alpha = 0.5, color = '#2ca02c', size = 1) +
  geom_smooth(method = 'loess', color = '#1f77b4') +
  facet_wrap(~ Topic, ncol = 2) +
  theme_minimal(base_size = 12) + labs(title = '评分与主题占比关系', x = '评分（z-score）', y = '主题占比')
ggsave(filename = file.path(outputs_dir, 'score_vs_topic.png'), plot = ps, width = 9, height = 6, dpi = 200)

# Trip type heatmap: average prevalence by trip types
trip_cols <- names(trip_types)
avg_prev <- lapply(trip_cols, function(tc){
  sub <- plot_df[plot_df[[tc]] == 1, paste0('Topic_', seq_len(bestK)), drop = FALSE]
  colMeans(sub, na.rm = TRUE)
})
avg_prev_mat <- do.call(rbind, avg_prev)
rownames(avg_prev_mat) <- trip_cols
hm_df <- as.data.frame(avg_prev_mat)
hm_df$TripType <- rownames(hm_df)
hm_long <- melt(as.data.table(hm_df), id.vars = 'TripType')
colnames(hm_long) <- c('TripType', 'Topic', 'AvgPrev')
hm <- ggplot(hm_long, aes(x = Topic, y = TripType, fill = AvgPrev)) +
  geom_tile() + scale_fill_viridis_c() + theme_minimal(base_size = 12) +
  labs(title = '不同旅行类型的主题占比差异（热力图）', x = '主题', y = '旅行类型', fill = '平均占比')
ggsave(filename = file.path(outputs_dir, 'triptype_heatmap.png'), plot = hm, width = 8, height = 5.5, dpi = 200)

# Time trend: prevalence vs days_since_review (if available)
plot_df2 <- plot_df[!is.na(plot_df$days_since_review_num), ]
plot_long2 <- melt(as.data.table(plot_df2), id.vars = c('days_since_review_num'))
colnames(plot_long2) <- c('days_since_review_num', 'Topic', 'Prevalence')
pt <- ggplot(plot_long2, aes(x = days_since_review_num, y = Prevalence)) +
  geom_point(alpha = 0.4, color = '#9467bd', size = 1) +
  geom_smooth(method = 'loess', color = '#ff7f0e') +
  facet_wrap(~ Topic, ncol = 2) +
  theme_minimal(base_size = 12) + labs(title = '时间维度的主题占比变化', x = '距评论天数', y = '主题占比')
ggsave(filename = file.path(outputs_dir, 'time_trend.png'), plot = pt, width = 9, height = 6, dpi = 200)

# Save model object for post-processing
saveRDS(final_mod, file = file.path(outputs_dir, 'final_mod.rds'))

message('STS analysis completed. Outputs saved to: ', outputs_dir)
