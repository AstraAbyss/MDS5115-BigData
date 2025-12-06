# Parse captured printTopWords output to JSON
suppressPackageStartupMessages({ library(jsonlite) })
lines <- readLines('sts_hotel_analysis/outputs/topic_keywords_sts.txt')
parse <- function(lines){
  res <- list()
  for (ln in lines) {
    ln <- trimws(ln)
    m <- regmatches(ln, regexec('^Topic[[:space:]]+([0-9]+)[[:space:]]+(Avg|Positive|Negative)[[:space:]]+sentiment-discourse:[[:space:]]*(.*)$', ln))[[1]]
    if (length(m) >= 4) {
      k <- as.integer(m[2])
      type <- m[3]
      words <- trimws(unlist(strsplit(m[4], ',')))
      words <- words[nchar(words) > 0]
      if (is.null(res[[as.character(k)]])) res[[as.character(k)]] <- list(low=NULL, avg=NULL, high=NULL)
      if (type == 'Avg') res[[as.character(k)]]$avg <- words
      else if (type == 'Positive') res[[as.character(k)]]$high <- words
      else if (type == 'Negative') res[[as.character(k)]]$low <- words
    }
  }
  out <- lapply(sort(as.integer(names(res))), function(k){ list(topic=k, low=res[[as.character(k)]]$low, avg=res[[as.character(k)]]$avg, high=res[[as.character(k)]]$high) })
  writeLines(toJSON(out, pretty=TRUE, auto_unbox=TRUE, na='null'), 'sts_hotel_analysis/outputs/topic_keywords_sts.json')
}
parse(lines)
