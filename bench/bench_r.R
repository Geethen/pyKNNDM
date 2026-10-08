.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(CAST); library(sf)})
rd <- function(f) {
  d <- read.csv(f, colClasses = "character")
  m <- sapply(d, function(col) as.numeric(col))  # R parses hex floats
  matrix(m, nrow = nrow(d), dimnames = list(NULL, names(d)))
}
cfgs <- list(
  list("geo_500","geographical"), list("geo_1000","geographical"),
  list("geo_2000","geographical"), list("geo_5000","geographical"),
  list("feat_500","feature"), list("feat_2000","feature"))
res <- NULL
for (cf in cfgs) {
  nm <- cf[[1]]; sp <- cf[[2]]
  t <- rd(file.path("data", paste0(nm, "_t.csv")))
  p <- rd(file.path("data", paste0(nm, "_p.csv")))
  n <- nrow(t)
  if (sp == "geographical") {
    tp <- st_as_sf(data.frame(x = t[,1], y = t[,2]), coords = c("x","y"), crs = 3857)
    pp <- st_as_sf(data.frame(x = p[,1], y = p[,2]), coords = c("x","y"), crs = 3857)
  } else { tp <- as.data.frame(t); pp <- as.data.frame(p) }
  skip_km <- FALSE
  for (cl in c("hierarchical", "kmeans")) {
    if (cl == "kmeans" && skip_km) { message("skip ", nm, " ", cl); next }
    reps <- if (n >= 2000) 1 else 3
    tm <- numeric(0); o <- NULL
    for (r in seq_len(reps)) {
      set.seed(1)
      tm[r] <- system.time(o <- CAST::knndm(tp, predpoints = pp, dist_space = sp, clustering = cl, k = 10))[["elapsed"]]
    }
    sec <- median(tm)
    message(nm, " ", cl, " ", sec)
    if (cl == "hierarchical" && n >= 5000 && sec > 300) skip_km <- TRUE
    res <- rbind(res, data.frame(name = nm, space = sp, n = n, clustering = cl,
      seconds = sec, q = o$q, W_hex = sprintf("%a", o$W)))
    write.csv(res, "results_r.csv", row.names = FALSE, quote = FALSE)
  }
}
