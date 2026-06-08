#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(pCMF))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop(paste("Missing value for", flag))
  args[[hit + 1]]
}

matrix_path <- get_arg("--matrix")
embedding_path <- get_arg("--embedding")
dimension <- as.integer(get_arg("--dimension"))
seed <- as.integer(get_arg("--seed", "0"))
iter_max <- as.integer(get_arg("--iter-max", "100"))
iter_min <- as.integer(get_arg("--iter-min", "20"))
pseudocount <- as.numeric(get_arg("--pseudocount", "1"))
zero_inflation <- as.logical(get_arg("--zero-inflation", "FALSE"))
sparsity <- as.logical(get_arg("--sparsity", "FALSE"))

if (is.null(matrix_path) || is.null(embedding_path) || is.na(dimension)) {
  stop("Required arguments: --matrix, --embedding, --dimension")
}

set.seed(seed)
x <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
x <- as.matrix(x)
storage.mode(x) <- "double"
x[!is.finite(x)] <- 0
x[x < 0] <- 0
x <- round(x) + pseudocount

fit <- pCMF(
  x,
  K = dimension,
  zero_inflation = zero_inflation,
  sparsity = sparsity,
  verbose = FALSE,
  monitor = FALSE,
  iter_max = iter_max,
  iter_min = iter_min,
  ninit = 1,
  iter_init = min(20, iter_max),
  ncores = 1,
  seed = seed
)

z <- as.data.frame(getU(fit))
rownames(z) <- rownames(x)
colnames(z) <- paste0("pCMF_", seq_len(ncol(z)))
dir.create(dirname(embedding_path), recursive = TRUE, showWarnings = FALSE)
write.csv(z, embedding_path, quote = FALSE)
