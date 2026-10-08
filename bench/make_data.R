.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
set.seed(42)
outdir <- "data"
if (!dir.exists(outdir)) dir.create(outdir)
hx <- function(m) {
  m <- as.matrix(m)
  out <- matrix(sprintf("%a", m), nrow(m), ncol(m))
  out
}
wr <- function(m, nm, cn) {
  h <- hx(m)
  colnames(h) <- cn
  write.table(h, file.path(outdir, nm), sep = ",", quote = FALSE, row.names = FALSE)
}
ext <- 10000
clustered <- function(n, ncl = 25, sd = 150) {
  centers <- cbind(runif(ncl, 0, ext), runif(ncl, 0, ext))
  id <- sample.int(ncl, n, replace = TRUE)
  cbind(centers[id, 1] + rnorm(n, 0, sd), centers[id, 2] + rnorm(n, 0, sd))
}
for (n in c(500, 1000, 2000, 5000)) {
  nm <- paste0("geo_", n)
  tr <- clustered(n)
  pr <- cbind(runif(5000, 0, ext), runif(5000, 0, ext))
  wr(tr, paste0(nm, "_t.csv"), c("x", "y"))
  wr(pr, paste0(nm, "_p.csv"), c("x", "y"))
}
feat <- function(xy, p = 6) {
  x <- xy[, 1] / ext; y <- xy[, 2] / ext; n <- nrow(xy)
  cbind(sin(3 * x) + rnorm(n, 0, .1), cos(2 * y) + rnorm(n, 0, .1),
        x * y + rnorm(n, 0, .1), exp(-((x - .5)^2 + (y - .5)^2) * 4) + rnorm(n, 0, .1),
        log1p(x + 2 * y) + rnorm(n, 0, .1), sqrt(x^2 + y^2) + rnorm(n, 0, .1))
}
for (n in c(500, 2000)) {
  nm <- paste0("feat_", n)
  tr <- feat(clustered(n))
  pr <- feat(cbind(runif(3000, 0, ext), runif(3000, 0, ext)))
  cn <- paste0("f", 1:6)
  wr(tr, paste0(nm, "_t.csv"), cn)
  wr(pr, paste0(nm, "_p.csv"), cn)
}
