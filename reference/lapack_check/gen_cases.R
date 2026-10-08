# Generates test matrices, runs La.svd(x, nu=0, nv=ncol(x)) and writes
# inputs/outputs as hex floats (sprintf("%a")) to an ndjson file (one case per line).
# usage: Rscript gen_cases.R <out.ndjson> [seed]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[1] else "cases.ndjson"
seed <- if (length(args) >= 2) as.integer(args[2]) else 20240607L
set.seed(seed)

hexv <- function(v) sprintf("%a", as.double(v))

make_case <- function(kind, m, n) {
  x <- switch(kind,
    gauss   = matrix(rnorm(m * n), m, n),
    centered = scale(matrix(rnorm(m * n), m, n), center = TRUE, scale = FALSE),
    offset  = matrix(1e6 + rnorm(m * n), m, n),
    utm     = cbind(matrix(5e5 + runif(m * n, 0, 1e4), m, n)),
    scaled  = scale(matrix(rnorm(m * n) * rep(10^seq(-2, 3, length.out = n), each = m), m, n)),
    collin  = { z <- matrix(rnorm(m * n), m, n)
                if (n >= 2) z[, 2] <- z[, 1] + 1e-9 * rnorm(m)
                if (n >= 3) z[, n] <- rowSums(z[, 1:(n - 1), drop = FALSE]) + 1e-12 * rnorm(m)
                z },
    dups    = { pool <- matrix(rnorm(3 * n), 3, n)
                pool[sample.int(3, m, replace = TRUE), , drop = FALSE] },
    grid    = matrix(sample(0:5, m * n, replace = TRUE), m, n) + 0,
    rankdef = { z <- matrix(rnorm(m * n), m, n)
                if (n >= 3) z[, 3] <- z[, 1] + z[, 2]
                if (n >= 5) z[, 5] <- 0
                z },
    heavy   = matrix(rt(m * n, df = 1.5), m, n),
    tiny    = matrix(rnorm(m * n), m, n) * 1e-200,
    huge    = matrix(rnorm(m * n), m, n) * 1e200,
    mixed   = matrix(rnorm(m * n), m, n) * rep(10^seq(-200, 200, length.out = n), each = m),
    subnorm = matrix(rnorm(m * n), m, n) * 1e-310,
    sparse  = { z <- matrix(rnorm(m * n), m, n); z[runif(m * n) < 0.6] <- 0; z },
    stop("bad kind"))
  x <- matrix(as.double(x), m, n)
  x
}

kinds <- c("gauss", "centered", "offset", "utm", "scaled", "collin", "dups",
           "grid", "rankdef", "heavy", "tiny", "huge", "sparse", "mixed", "subnorm")
ms <- c(3, 4, 5, 10, 50, 200, 1000, 5000)
ns <- c(1, 2, 3, 4, 5, 8, 12)
specs <- list()
for (m in ms) for (n in ns) if (m >= n) for (k in kinds) specs[[length(specs) + 1]] <- list(k, m, n)
# extra: bigger n (up to 25), m near the MNTHR switch, and large m
for (n in c(6, 7, 10, 15, 20, 25)) for (m in c(n, n + 1, ceiling(11 * n / 6) - 1, floor(11 * n / 6), floor(11 * n / 6) + 1, 2 * n, 100))
  if (m >= n) for (k in c("gauss", "centered", "utm")) specs[[length(specs) + 1]] <- list(k, m, n)
for (n in c(2, 5, 12)) for (k in c("gauss", "centered", "utm", "offset")) {
  specs[[length(specs) + 1]] <- list(k, 20000, n)
}
for (k in c("centered", "utm")) specs[[length(specs) + 1]] <- list(k, 100000, 5)
specs[[length(specs) + 1]] <- list("centered", 100000, 12)

con <- file(out, "w")
id <- 0L
for (sp in specs) {
  id <- id + 1L
  x <- make_case(sp[[1]], sp[[2]], sp[[3]])
  s <- La.svd(x, nu = 0, nv = ncol(x))
  rec <- list(id = id, kind = sp[[1]], m = nrow(x), n = ncol(x),
              x = hexv(x), d = hexv(s$d), vt = hexv(s$vt))
  writeLines(toJSON(rec, auto_unbox = TRUE, digits = NA), con)
}
close(con)
cat("wrote", id, "cases to", out, "\n")
