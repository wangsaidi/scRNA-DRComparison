#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(scGBM))

process_results_fixed <- function(gbm, Y, order.by.deviance = TRUE) {
  M <- gbm$M
  for (m in 1:M) {
    if (gbm$U[1, m] < 0) {
      gbm$U[, m] <- -1 * gbm$U[, m]
      gbm$V[, m] <- -1 * gbm$V[, m]
    }
  }
  u.mean <- colMeans(gbm$U)
  gbm$beta <- gbm$beta + colSums(diag(gbm$D * u.mean) %*% t(gbm$V))
  gbm$U <- scale(gbm$U, center = TRUE, scale = FALSE)
  gbm$alpha[, 1] <- gbm$alpha[, 1] + colSums(diag(gbm$D * u.mean) %*% t(gbm$U))
  gbm$V <- scale(gbm$V, center = TRUE, scale = FALSE)
  gbm$beta <- gbm$beta + mean(gbm$alpha[, 1])
  gbm$alpha[, 1] <- gbm$alpha[, 1] - mean(gbm$alpha[, 1])
  my.order <- seq_len(M)
  dev.diff <- rep(NA_real_, M)
  if (order.by.deviance) {
    dev.full <- sum(Y * log(gbm$W) - gbm$W)
    for (m in 1:M) {
      if (M == 1) {
        Etam <- matrix(gbm$alpha[, 1], nrow = gbm$I, ncol = gbm$J) + matrix(gbm$beta, nrow = gbm$I, ncol = gbm$J)
      } else {
        keep <- setdiff(seq_len(M), m)
        Etam <- matrix(gbm$alpha[, 1], nrow = gbm$I, ncol = gbm$J) + matrix(gbm$beta, nrow = gbm$I, ncol = gbm$J) + gbm$U[, keep, drop = FALSE] %*% diag(gbm$D[keep], nrow = length(keep)) %*% t(gbm$V[, keep, drop = FALSE])
      }
      dev.diff[m] <- dev.full - sum(Y * Etam - exp(Etam))
    }
    my.order <- order(dev.diff, decreasing = TRUE)
    gbm$U <- gbm$U[, my.order, drop = FALSE]
    gbm$V <- gbm$V[, my.order, drop = FALSE]
    gbm$D <- gbm$D[my.order]
  }
  gbm$dev.diff <- dev.diff[my.order]
  gbm$scores <- t(gbm$D * t(gbm$V))
  gbm
}

unlockBinding("process.results", asNamespace("scGBM"))
assign("process.results", process_results_fixed, envir = asNamespace("scGBM"))
lockBinding("process.results", asNamespace("scGBM"))

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
x <- round(x)

# scGBM expects genes x cells.
y <- t(x)

fit <- gbm.sc(
  Y = y,
  M = dimension,
  max.iter = max_iter,
  min.iter = min_iter,
  ncores = 1,
  return.W = TRUE,
  order.by.deviance = FALSE,
  factor.init = "pearson"
)

z <- as.data.frame(fit$scores)
rownames(z) <- rownames(x)
colnames(z) <- paste0("scGBM_", seq_len(ncol(z)))
dir.create(dirname(embedding_path), recursive = TRUE, showWarnings = FALSE)
write.csv(z, embedding_path, quote = FALSE)
