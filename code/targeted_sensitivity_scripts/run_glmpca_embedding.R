#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(glmpca))

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
max_iter <- as.integer(get_arg("--max-iter", "25"))
min_iter <- as.integer(get_arg("--min-iter", "5"))

if (is.null(matrix_path) || is.null(embedding_path) || is.na(dimension)) {
  stop("Required arguments: --matrix, --embedding, --dimension")
}

set.seed(seed)
x <- read.csv(matrix_path, row.names = 1, check.names = FALSE)
x <- as.matrix(x)
storage.mode(x) <- "double"
x[!is.finite(x)] <- 0
x[x < 0] <- 0
y <- t(round(x))
keep <- rowSums(y) > 0
y <- y[keep, , drop = FALSE]
if (nrow(y) <= dimension) {
  stop(sprintf("Too few nonzero genes (%d) for GLMPCA dimension %d", nrow(y), dimension))
}

fit <- glmpca(
  y,
  L = dimension,
  fam = "poi",
  ctl = list(maxIter = max_iter, minIter = min_iter, verbose = FALSE)
)

z <- as.data.frame(fit$factors)
rownames(z) <- rownames(x)
colnames(z) <- paste0("GLMPCA_", seq_len(ncol(z)))
dir.create(dirname(embedding_path), recursive = TRUE, showWarnings = FALSE)
write.csv(z, embedding_path, quote = FALSE)
