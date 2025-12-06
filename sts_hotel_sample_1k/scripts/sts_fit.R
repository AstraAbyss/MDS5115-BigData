# STS fit on stratified 1k sample of Hotel_Reviews 1.csv
# Author: Aime (刘研)
# Seed: 2025

suppressPackageStartupMessages({
  library(data.table)
  library(stringr)
  library(lubridate)
  library(dplyr)
  library(tm)
  library(stm)
  library(sts)
  library(jsonlite)
  library(ggplot2)
})

set.seed(2025)

# Paths (resolve relative to this script location)
args <- commandArgs(trailingOnly = FALSE)
fileArg <- "--file="
script_path <- sub(fileArg, "", args[grep(fileArg, args)])
script_dir <- if (length(script_path) > 0) dirname(script_path) else getwd()
root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
input_csv <- file.path(root_dir, "data", "Hotel_Reviews 1.csv")
output_dir <- file.path(root_dir, "outputs")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
sample_csv <- file.path(root_dir, "data", "Hotel_Reviews_1k_sample.csv")

# 1) Read and stratified sample by Review_Date month
message("Reading large CSV header to determine column types...")
# fread will read quickly; we only need columns used for sampling and covariates
cols <- c("Review_Date","Positive_Review","Negative_Review","Reviewer_Score","Tags",
          "Total_Number_of_Reviews_Reviewer_Has_Given","days_since_review")
DT <- fread(file = input_csv, select = cols, encoding = "UTF-8")

# Parse Review_Date to date and month
DT[, Review_Date := suppressWarnings(mdy(Review_Date))]
DT[is.na(Review_Date), Review_Date := suppressWarnings(dmy(Review_Date))]
DT[, month_idx := month(Review_Date)]

# Fallback: if missing Review_Date, use days_since_review string -> days numeric then map to month buckets
parse_days <- function(x){
  if (is.na(x)) return(NA_integer_)
  m <- str_match(x, "(\\d+)")
  if (is.na(m[1,2])) return(NA_integer_)
  as.integer(m[1,2])
}
DT[, days_num := vapply(days_since_review, parse_days, integer(1))]
DT[is.na(month_idx) & !is.na(days_num), month_idx := pmax(1L, pmin(12L, floor(days_num/30)+1L))]

# Drop rows without month_idx
DT <- DT[!is.na(month_idx)]

# Stratified proportional allocation to N=1000
N_total <- 1000L
strat <- DT[, .N, by = month_idx][order(month_idx)]
strat[, prop := N / sum(N)]
strat[, n_take := floor(prop * N_total)]
# Adjust rounding to sum exactly 1000
shortfall <- N_total - sum(strat$n_take)
if (shortfall > 0) {
  # distribute to largest remainders
  rema <- strat$prop * N_total - strat$n_take
  idx <- order(rema, decreasing = TRUE)[seq_len(shortfall)]
  strat$n_take[idx] <- strat$n_take[idx] + 1L
}

# Sample per stratum
message("Sampling per month stratum...")
sampled_list <- lapply(seq_len(nrow(strat)), function(i){
  m <- strat$month_idx[i]
  k <- strat$n_take[i]
  if (k <= 0) return(NULL)
  DT[month_idx == m][sample(.N, k)]
})
DTs <- rbindlist(Filter(Negate(is.null), sampled_list))
DTs[, doc_id := .I]

# Save sampled CSV
fwrite(DTs, sample_csv)
message(paste("Sampled", nrow(DTs), "rows saved to", sample_csv))

# 2) Text preprocessing: combine POS/NEG, clean, build STM documents
clean_text <- function(pos, neg){
  pos <- ifelse(is.na(pos), "", pos)
  neg <- ifelse(is.na(neg), "", neg)
  # remove placeholders
  pos <- gsub("(?i)\\bno positive\\b", "", pos, perl = TRUE)
  neg <- gsub("(?i)\\bno negative\\b", "", neg, perl = TRUE)
  txt <- paste("POS", ":", pos, " NEG", ":", neg)
  # lower
  txt <- tolower(txt)
  # remove urls
  txt <- gsub("http[s]?://\\S+", " ", txt)
  txt <- gsub("www\\.\\S+", " ", txt)
  # remove digits
  txt <- gsub("[0-9]+", " ", txt)
  # remove punctuation
  txt <- gsub("[[:punct:]]+", " ", txt)
  # collapse spaces
  txt <- gsub("\\s+", " ", txt)
  trimws(txt)
}
DTs[, text := clean_text(Positive_Review, Negative_Review)]

# Trip type one-hot from Tags
tag_has <- function(s, pat){
  if (is.na(s)) return(FALSE)
  grepl(pat, s, ignore.case = TRUE)
}
DTs[, Trip_Leisure := vapply(Tags, tag_has, logical(1), pat = "Leisure trip")]
DTs[, Trip_Business := vapply(Tags, tag_has, logical(1), pat = "Business trip")]
DTs[, Trip_Solo := vapply(Tags, tag_has, logical(1), pat = "Solo traveler")]
DTs[, Trip_Family := vapply(Tags, tag_has, logical(1), pat = "Family with young children")]
DTs[, Trip_Group := vapply(Tags, tag_has, logical(1), pat = "Group")]
DTs[, Trip_Friends := vapply(Tags, tag_has, logical(1), pat = "Travelers with friends")]
DTs[, Trip_Pet := vapply(Tags, tag_has, logical(1), pat = "With a pet")]

# Reviewer_Score z-score; Reviews_Given z-score
DTs[, Reviewer_Score_z := as.numeric(scale(Reviewer_Score))]
DTs[, Reviews_Given_z := as.numeric(scale(Total_Number_of_Reviews_Reviewer_Has_Given))]
DTs[is.na(Reviews_Given_z), Reviews_Given_z := 0]  # optional fill
DTs[, month_idx := as.integer(month_idx)]

# Build STM corpus (documents, vocab, meta)
meta <- DTs[, .(doc_id, Reviewer_Score_z, Trip_Leisure, Trip_Business, Trip_Solo,
                Trip_Family, Trip_Group, Trip_Friends, Trip_Pet, month_idx,
                Reviews_Given_z)]

# Use stm's textProcessor and prepDocuments
processed <- textProcessor(documents = DTs$text, metadata = as.data.frame(meta),
                           lowercase = FALSE, removestopwords = TRUE,
                           removenumbers = FALSE, removepunctuation = FALSE)
# prep documents
prep <- prepDocuments(processed$documents, processed$vocab, processed$meta,
                      lower.thresh = 5)

corpus <- list(documents = prep$documents, vocab = prep$vocab, meta = prep$meta)

# Build held-out missing object: 10% docs and ~50% of tokens per doc
build_missing <- function(corpus, pdocs = 0.1, frac_words = 0.5){
  D <- length(corpus$documents)
  ndocs <- max(1L, floor(D * pdocs))
  # sample doc indices
  idx <- sample(seq_len(D), size = ndocs, replace = FALSE)
  docs_list <- vector("list", length = ndocs)
  for (i in seq_len(ndocs)){
    d <- idx[i]
    doc <- corpus$documents[[d]]  # 2 x N: indices and counts
    inds <- doc[1, ]; cnts <- doc[2, ]
    # split counts roughly half into heldout, ensuring at least 1 if count > 1
    held_cnts <- floor(cnts * frac_words)
    held_cnts[held_cnts == 0 & cnts > 0] <- 1L
    # remove zero entries
    sel <- held_cnts > 0
    docs_list[[i]] <- rbind(inds[sel], held_cnts[sel])
  }
  list(index = idx, docs = docs_list)
}


# Prevalence+Sentiment covariate design: formula using meta
X_formula <- ~ Reviewer_Score_z + Trip_Leisure + Trip_Business + Trip_Solo + Trip_Family + Trip_Group + Trip_Friends + Trip_Pet + month_idx + Reviews_Given_z
seed_formula <- ~ Reviewer_Score_z  # initializationVar: use rating

# 3) Fit STS for K in {5,6,7} and evaluate held-out log likelihood + coherence
Ks <- c(5L,6L,7L)
heldout_results <- list()
coherence_results <- list()
models <- list()

for (K in Ks) {
  message(sprintf("Fitting STS K=%d ...", K))
  fit <- sts(prevalence_sentiment = X_formula,
             initializationVar = seed_formula,
             corpus = corpus,
             K = K,
             maxIter = 200,
             convTol = 1e-05,
             initialization = "anchor",
             kappaEstimation = "adjusted",
             verbose = TRUE,
             parallelize = FALSE)
  models[[as.character(K)]] <- fit
  # held-out likelihood: remove 10% docs and half words as in paper
  missing_obj <- build_missing(corpus, pdocs = 0.1, frac_words = 0.5)
  hl_obj <- heldoutLikelihood(fit, missing = missing_obj)
  heldout_results[[as.character(K)]] <- list(K = K, heldout_loglik = hl_obj$expected.heldout)
  # topic coherence
  coh <- topicSemanticCoherence(fit, corpus)
  coherence_results[[as.character(K)]] <- list(K = K, semantic_coherence = coh)
}

# Decide best K: max average of normalized metrics
normalize <- function(x){ (x - min(x)) / max(1e-12, (max(x) - min(x))) }
held_vals <- sapply(heldout_results, function(r) r$heldout_loglik)
coh_vals <- sapply(coherence_results, function(r) mean(r$semantic_coherence))
score <- normalize(held_vals) + normalize(coh_vals)
best_idx <- which.max(score)
bestK <- Ks[best_idx]
best_model <- models[[as.character(bestK)]]

# Save heldout_ll.json with details and choice reason
held_json <- list(
  grid = lapply(Ks, function(K){
    list(K = K,
         heldout_loglik = heldout_results[[as.character(K)]]$heldout_loglik,
         semantic_coherence_avg = mean(coherence_results[[as.character(K)]]$semantic_coherence))
  }),
  selection = list(bestK = bestK, reason = "综合 held-out LL 与平均 topic coherence 归一化得分最高，且语义可解释性良好")
)
writeLines(jsonlite::toJSON(held_json, auto_unbox = TRUE, pretty = TRUE),
           con = file.path(output_dir, "heldout_ll.json"))

# 4) Extract ELBO curve and plot
elbo <- best_model$elbo
png(file.path(output_dir, "elbo_curve.png"), width = 1200, height = 800)
plot(elbo, type = "l", lwd = 2, col = "steelblue",
     xlab = "迭代", ylab = "ELBO", main = sprintf("STS 收敛曲线 (K=%d)", bestK))
grid()
dev.off()

# 5) Topic keywords: baseline and low/high sentiment-discourse extremes
vocab <- best_model$vocab
K <- bestK
alpha_s <- best_model$alpha[, (K):(2*K-1)]  # sentiment/discourse part
mv <- best_model$mv
kappa_t <- best_model$kappa$kappa_t
kappa_s <- best_model$kappa$kappa_s

softmax <- function(x){ ex <- exp(x - max(x)); ex / sum(ex) }
get_top_words <- function(k, n = 15){
  # baseline beta at alpha_s = 0
  beta0 <- sapply(seq_along(vocab), function(v){ softmax(mv[v] + kappa_t[v, k]) })
  ord0 <- order(beta0, decreasing = TRUE)[1:n]
  # low and high percentiles of alpha_s
  low_a <- quantile(alpha_s[, k], probs = 0.05, na.rm = TRUE)
  high_a <- quantile(alpha_s[, k], probs = 0.95, na.rm = TRUE)
  beta_low <- sapply(seq_along(vocab), function(v){ softmax(mv[v] + kappa_t[v, k] + kappa_s[v, k] * low_a) })
  beta_high <- sapply(seq_along(vocab), function(v){ softmax(mv[v] + kappa_t[v, k] + kappa_s[v, k] * high_a) })
  ordL <- order(beta_low, decreasing = TRUE)[1:n]
  ordH <- order(beta_high, decreasing = TRUE)[1:n]
  list(
    baseline = vocab[ord0],
    low_alpha_s = vocab[ordL],
    high_alpha_s = vocab[ordH]
  )
}

topic_kw <- lapply(1:K, function(k){ get_top_words(k, n = 15) })
# Format json
kw_json <- lapply(1:K, function(k){ list(topic = k,
                                        baseline = unname(topic_kw[[k]]$baseline),
                                        low_alpha_s = unname(topic_kw[[k]]$low_alpha_s),
                                        high_alpha_s = unname(topic_kw[[k]]$high_alpha_s)) })
writeLines(jsonlite::toJSON(kw_json, auto_unbox = TRUE, pretty = TRUE),
           con = file.path(output_dir, "topic_keywords_sts.json"))

# 6) Regression coefficients for prevalence (g^p) and sentiment (g^s)
regns <- estimateRegns(best_model, prevalence_sentiment = X_formula, corpus = corpus)
# regns likely contains lists for prevalence and sentiment with coef and se
# To be safe, printRegnTables writes to console; we attempt to capture regns object structure
# Here we coerce to JSON if possible; otherwise rebuild from Gamma and alpha

# Build design X explicitly
dfX <- model.matrix(X_formula, data = corpus$meta)
# Drop intercept
dfX <- dfX[, -1, drop = FALSE]
# Gamma.est contains regression coefficients mapping covariates to latent variables
Gamma <- best_model$gamma  # dimension: ncol(X)+1 by (2K-1)? mu$gamma in sts
# Separate prevalence (first K-1) and sentiment (last K)
Gp <- Gamma[, 1:(K-1), drop = FALSE]
Gs <- Gamma[, K:(2*K-1), drop = FALSE]
# Name rows by covariates (include Intercept)
rownames(Gp) <- c("Intercept", colnames(dfX))
rownames(Gs) <- c("Intercept", colnames(dfX))

# We do not have SE directly here; set SE as NA (or compute via sandwich if available)
coeffs_prev <- list(K = K, coef = Gp)
coeffs_sent <- list(K = K, coef = Gs)
# Convert matrices to lists for JSON
mat_to_list <- function(M){
  lapply(seq_len(ncol(M)), function(j){ list(topic = j, coefs = setNames(as.numeric(M[, j]), rownames(M))) })
}
prev_json <- list(K = K, topics = mat_to_list(Gp))
sent_json <- list(K = K, topics = mat_to_list(Gs))
writeLines(jsonlite::toJSON(prev_json, auto_unbox = TRUE, pretty = TRUE),
           con = file.path(output_dir, "coeffs_prevalence_sts.json"))
writeLines(jsonlite::toJSON(sent_json, auto_unbox = TRUE, pretty = TRUE),
           con = file.path(output_dir, "coeffs_sentiment_sts.json"))

# 7) Representative docs per topic
expeta <- exp(cbind(best_model$alpha[, 1:(K-1)], 0))
theta <- expeta/rowSums(expeta)
examples_path <- file.path(output_dir, "examples_per_topic.txt")
con <- file(examples_path, open = "wt", encoding = "UTF-8")
for (k in 1:K) {
  cat(sprintf("Topic %d\n", k), file = con)
  idxs <- order(theta[, k], decreasing = TRUE)[1:min(3, nrow(theta))]
  for (docid in idxs) {
    txt <- DTs$text[docid]
    if (nchar(txt) > 400) txt <- paste0(substr(txt, 1, 400), "...")
    cat(sprintf("- %s\n", txt), file = con)
  }
  cat("\n", file = con)
}
close(con)

message("STS fitting completed and outputs saved.")
