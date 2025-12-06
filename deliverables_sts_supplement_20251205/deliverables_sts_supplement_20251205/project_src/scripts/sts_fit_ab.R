# STS dual-model fitting (A: month continuous; B: season dummies) with robust SE and p-values
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
  library(sandwich)
  library(lmtest)
})

set.seed(2025)

# Paths
args <- commandArgs(trailingOnly = FALSE)
fileArg <- "--file="
script_path <- sub(fileArg, "", args[grep(fileArg, args)])
script_dir <- if (length(script_path) > 0) dirname(script_path) else getwd()
root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
input_sample <- file.path(root_dir, "data", "Hotel_Reviews_1k_sample.csv")
output_dir <- file.path(root_dir, "outputs")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Read sample
DTs <- fread(file = input_sample, encoding = "UTF-8")
DTs[, doc_id := ifelse(is.na(doc_id), .I, doc_id)]

# Clean text without pos/neg labels
clean_text2 <- function(pos, neg){
  pos <- ifelse(is.na(pos), "", pos)
  neg <- ifelse(is.na(neg), "", neg)
  txt <- paste(pos, neg)
  txt <- tolower(txt)
  txt <- gsub("http[s]?://\\S+", " ", txt)
  txt <- gsub("www\\.\\S+", " ", txt)
  txt <- gsub("[0-9]+", " ", txt)
  txt <- gsub("[[:punct:]]+", " ", txt)
  txt <- gsub("\\s+", " ", txt)
  trimws(txt)
}
DTs[, text := clean_text2(Positive_Review, Negative_Review)]

# Helper: tag detection
has_tag <- function(s, pat){ if (is.na(s)) FALSE else grepl(pat, s, ignore.case = TRUE) }

# Travel type (mutually exclusive group: Leisure vs Business -> 1 one-hot)
DTs[, Trip_Leisure := vapply(Tags, has_tag, logical(1), pat = "Leisure trip")]
DTs[, Trip_Business := vapply(Tags, has_tag, logical(1), pat = "Business trip")]
# Choose 1 one-hot: use Leisure as indicator (baseline = Business/others)
DTs[, TripType_Leisure := Trip_Leisure]

# Companion type group: Group / Solo traveler / Couple / Family with young children -> take 3 one-hot, baseline Couple
DTs[, Comp_Group := vapply(Tags, has_tag, logical(1), pat = "Group")]
DTs[, Comp_Solo := vapply(Tags, has_tag, logical(1), pat = "Solo traveler")]
DTs[, Comp_Family := vapply(Tags, has_tag, logical(1), pat = "Family with young children")]
DTs[, Comp_Couple := vapply(Tags, has_tag, logical(1), pat = "Couple")]
# We'll include Comp_Group, Comp_Solo, Comp_Family as 3 one-hot; baseline = Couple

# Booking type: Submit from mobile phone (dataset uses 'Submitted from a mobile device') -> 1 one-hot
DTs[, Submit_Mobile := vapply(Tags, function(s){
  if (is.na(s)) return(FALSE)
  grepl("Submitted from a mobile device|Submit(ed)? from mobile", s, ignore.case = TRUE)
}, logical(1))]

# Reviewer score z-score
DTs[, Reviewer_Score_z := as.numeric(scale(Reviewer_Score))]

# Reviewer activity: number of reviews given z-score
DTs[, Reviews_Given_z := as.numeric(scale(Total_Number_of_Reviews_Reviewer_Has_Given))]
DTs[is.na(Reviews_Given_z), Reviews_Given_z := 0]

# Month index (already present for many rows, recompute robustly)
DTs[, month_idx := lubridate::month(suppressWarnings(lubridate::mdy(Review_Date)))]
DTs[is.na(month_idx), month_idx := lubridate::month(suppressWarnings(lubridate::dmy(Review_Date)))]
parse_days <- function(x){ m <- stringr::str_match(x, "(\\d+)"); if (is.na(m[1,2])) NA_integer_ else as.integer(m[1,2]) }
DTs[, days_num := vapply(days_since_review, parse_days, integer(1))]
DTs[is.na(month_idx) & !is.na(days_num), month_idx := pmax(1L, pmin(12L, floor(days_num/30)+1L))]
DTs[, month_idx := as.integer(month_idx)]

# Season coding: Winter(Dec-Feb), Spring(Mar-May), Summer(Jun-Aug), Autumn(Sep-Nov)
DTs[, Season := cut(month_idx,
                    breaks = c(0,2,5,8,11,12),
                    labels = c("Winter","Spring","Summer","Autumn","Winter"), include.lowest = TRUE, right = TRUE)]
DTs[is.na(Season) & !is.na(month_idx), Season := ifelse(month_idx %in% c(12,1,2), "Winter",
                                                       ifelse(month_idx %in% 3:5, "Spring",
                                                              ifelse(month_idx %in% 6:8, "Summer",
                                                                     ifelse(month_idx %in% 9:11, "Autumn", NA))))]
# Create 3 one-hot (baseline Winter)
DTs[, Season_Spring := Season == "Spring"]
DTs[, Season_Summer := Season == "Summer"]
DTs[, Season_Autumn := Season == "Autumn"]

# Build STM corpus
meta <- DTs[, .(doc_id,
                Reviewer_Score_z,
                TripType_Leisure,
                Comp_Group, Comp_Solo, Comp_Family,
                Submit_Mobile,
                month_idx,
                Season_Spring, Season_Summer, Season_Autumn,
                Reviews_Given_z)]

processed <- textProcessor(documents = DTs$text, metadata = as.data.frame(meta),
                           lowercase = FALSE, removestopwords = TRUE,
                           removenumbers = FALSE, removepunctuation = FALSE)
prep <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 5)
corpus <- list(documents = prep$documents, vocab = prep$vocab, meta = prep$meta)

# Helper functions for fitting and regression extraction
build_missing <- function(corpus, pdocs = 0.1, frac_words = 0.5){
  D <- length(corpus$documents)
  ndocs <- max(1L, floor(D * pdocs))
  idx <- sample(seq_len(D), size = ndocs, replace = FALSE)
  docs_list <- vector("list", length = ndocs)
  for (i in seq_len(ndocs)){
    d <- idx[i]
    doc <- corpus$documents[[d]]
    inds <- doc[1, ]; cnts <- doc[2, ]
    held_cnts <- floor(cnts * frac_words)
    held_cnts[held_cnts == 0 & cnts > 0] <- 1L
    sel <- held_cnts > 0
    docs_list[[i]] <- rbind(inds[sel], held_cnts[sel])
  }
  list(index = idx, docs = docs_list)
}

fit_sts_with_formula <- function(X_formula, seed_formula, label){
  Ks <- c(5L,6L,7L)
  heldout_results <- list(); coherence_results <- list(); models <- list()
  for (K in Ks) {
    message(sprintf("[%s] Fitting STS K=%d ...", label, K))
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
    missing_obj <- build_missing(corpus, pdocs = 0.1, frac_words = 0.5)
    hl_obj <- heldoutLikelihood(fit, missing = missing_obj)
    heldout_results[[as.character(K)]] <- list(K = K, heldout_loglik = hl_obj$expected.heldout)
    coh <- topicSemanticCoherence(fit, corpus)
    coherence_results[[as.character(K)]] <- list(K = K, semantic_coherence = coh)
  }
  normalize <- function(x){ (x - min(x)) / max(1e-12, (max(x) - min(x))) }
  held_vals <- sapply(heldout_results, function(r) r$heldout_loglik)
  coh_vals <- sapply(coherence_results, function(r) mean(r$semantic_coherence))
  score <- normalize(held_vals) + normalize(coh_vals)
  best_idx <- which.max(score)
  bestK <- Ks[best_idx]
  best_model <- models[[as.character(bestK)]]

  # Save heldout info
  held_json <- list(
    label = label,
    grid = lapply(Ks, function(K){
      list(K = K,
           heldout_loglik = heldout_results[[as.character(K)]]$heldout_loglik,
           semantic_coherence_avg = mean(coherence_results[[as.character(K)]]$semantic_coherence))
    }),
    selection = list(bestK = bestK, reason = "综合 held-out LL 与平均 topic coherence 归一化得分最高")
  )
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  writeLines(jsonlite::toJSON(held_json, auto_unbox = TRUE, pretty = TRUE),
             con = file.path(output_dir, paste0("heldout_ll_", label, "_", ts, ".json")))

  list(model = best_model, K = bestK, label = label)
}

# Robust regression over latent variables
do_robust_reg <- function(best_model, X_formula, label){
  K <- best_model$K
  alpha <- best_model$alpha
  # prevalence latent: 1..(K-1); sentiment: K..(2K-1)
  alpha_p <- alpha[, 1:(K-1), drop = FALSE]
  alpha_s <- alpha[, K:(2*K-1), drop = FALSE]
  dfX <- model.matrix(X_formula, data = corpus$meta)  # includes intercept
  # Add interaction: TripType_Leisure × Submit_Mobile
  # We ensure variables exist in dfX columns
  # If not present in formula, we will add manually
  if (!("TripType_Leisure" %in% colnames(dfX))){
    dfX <- cbind(dfX, TripType_Leisure = as.numeric(corpus$meta$TripType_Leisure))
  }
  if (!("Submit_Mobile" %in% colnames(dfX))){
    dfX <- cbind(dfX, Submit_Mobile = as.numeric(corpus$meta$Submit_Mobile))
  }
  Interact <- as.numeric(corpus$meta$TripType_Leisure) * as.numeric(corpus$meta$Submit_Mobile)
  dfX <- cbind(dfX, `TripType_Leisure:Submit_Mobile` = Interact)

  # Drop duplicate intercepts if present
  # Ensure dfX has Intercept column named '(Intercept)'
  if (!any(colnames(dfX) == "(Intercept)")){
    dfX <- cbind(`(Intercept)` = 1, dfX)
  }

  # Function to regress y on dfX with robust SE
  reg_one <- function(y){
    fit <- lm(y ~ dfX - 1)  # -1 to use provided intercept explicitly
    V <- sandwich::vcovHC(fit, type = "HC1")
    ct <- lmtest::coeftest(fit, vcov. = V)
    tibble::as_tibble(data.frame(var = rownames(ct), coef = ct[,1], std_err = sqrt(diag(V)), p_value = ct[,4]))
  }

  # Prevalence
  prev_list <- lapply(seq_len(ncol(alpha_p)), function(j){
    out <- reg_one(alpha_p[, j])
    out$topic <- j
    out$latent <- "prevalence"
    out
  })
  prev_df <- dplyr::bind_rows(prev_list)

  # Sentiment-discourse
  sent_list <- lapply(seq_len(ncol(alpha_s)), function(j){
    out <- reg_one(alpha_s[, j])
    out$topic <- j
    out$latent <- "sentiment"
    out
  })
  sent_df <- dplyr::bind_rows(sent_list)

  res <- dplyr::bind_rows(prev_df, sent_df)
  res$model_label <- label
  res <- res %>% dplyr::select(model_label, latent, topic, var, coef, std_err, p_value)
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_csv <- file.path(output_dir, paste0("regression_results_", label, "_", ts, ".csv"))
  fwrite(res, out_csv)
  message(sprintf("Saved regression results to %s", out_csv))
  return(out_csv)
}

# Model A: month continuous
X_formula_A <- ~ Reviewer_Score_z + TripType_Leisure + Comp_Group + Comp_Solo + Comp_Family + Submit_Mobile + month_idx + Reviews_Given_z
seed_formula <- ~ Reviewer_Score_z
fitA <- fit_sts_with_formula(X_formula_A, seed_formula, label = "A_month")

csvA <- do_robust_reg(fitA$model, X_formula_A, label = fitA$label)

# Model B: season dummies (3 one-hot, baseline winter)
X_formula_B <- ~ Reviewer_Score_z + TripType_Leisure + Comp_Group + Comp_Solo + Comp_Family + Submit_Mobile + Season_Spring + Season_Summer + Season_Autumn + Reviews_Given_z
fitB <- fit_sts_with_formula(X_formula_B, seed_formula, label = "B_season")

csvB <- do_robust_reg(fitB$model, X_formula_B, label = fitB$label)

# Key coefficient bar plots for interaction term across topics (A and B)
plot_key_bars <- function(csv_path, label){
  df <- fread(csv_path)
  key_var <- "TripType_Leisure:Submit_Mobile"
  df_key <- df[latent == "sentiment" & var == key_var]
  p <- ggplot(df_key, aes(x = factor(topic), y = coef)) +
    geom_col(fill = "#1f77b4") +
    geom_errorbar(aes(ymin = coef - 1.96*std_err, ymax = coef + 1.96*std_err), width = 0.2, color = "#333") +
    labs(title = paste0("关键交互项（休闲×移动端）— 模型 ", label, "：情感/话语系数"), x = "主题", y = "系数（±95%CI）") +
    theme_minimal(base_size = 12)
  ggsave(filename = file.path(output_dir, paste0("key_coef_interaction_", label, ".png")), p, width = 10, height = 6, dpi = 200)
}

plot_key_bars(csvA, "A")
plot_key_bars(csvB, "B")

message("Dual-model STS fitting and robust regressions completed.")
