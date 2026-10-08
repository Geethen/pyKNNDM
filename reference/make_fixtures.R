.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
library(CAST)
library(sf)
library(jsonlite)

# Output directory: command-line argument or default "tests/fixtures"
args_cli <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args_cli) >= 1 && nzchar(args_cli[1])) args_cli[1] else "tests/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Helper: encode numeric doubles as C99 hex-float strings
hexv <- function(x) sprintf("%a", as.numeric(x))

# -----------------------------------------------------------------------------
# Data Generators
# -----------------------------------------------------------------------------

gen_clustered <- function(n, n_centers = 5, sd = 50, extent = 1000, data_seed = NULL) {
  if (!is.null(data_seed)) set.seed(data_seed)
  centers <- matrix(runif(n_centers * 2, min = 0, max = extent), ncol = 2)
  c_idx <- sample(n_centers, n, replace = TRUE)
  pts <- centers[c_idx, , drop = FALSE] + matrix(rnorm(n * 2, mean = 0, sd = sd), ncol = 2)
  colnames(pts) <- c("x", "y")
  pts
}

gen_uniform <- function(n, extent = 1000, data_seed = NULL) {
  if (!is.null(data_seed)) set.seed(data_seed)
  pts <- matrix(runif(n * 2, min = 0, max = extent), ncol = 2)
  colnames(pts) <- c("x", "y")
  pts
}

gen_grid <- function(n_side, extent = 1000, data_seed = NULL) {
  if (!is.null(data_seed)) set.seed(data_seed)
  vals <- seq(0, extent, length.out = n_side)
  g <- expand.grid(x = vals, y = vals)
  pts <- as.matrix(g)
  colnames(pts) <- c("x", "y")
  pts
}

gen_dupes <- function(n, n_centers = 5, sd = 50, extent = 1000, data_seed = NULL) {
  if (!is.null(data_seed)) set.seed(data_seed)
  pts <- gen_clustered(n, n_centers = n_centers, sd = sd, extent = extent)
  n_dupes <- max(1, round(0.1 * n))
  dupe_sources <- sample((n_dupes + 1):n, n_dupes, replace = TRUE)
  pts[seq_len(n_dupes), ] <- pts[dupe_sources, ]
  pts
}

gen_features <- function(n, p, type = c("clustered", "uniform"), data_seed = NULL, extent = 1000) {
  if (!is.null(data_seed)) set.seed(data_seed)
  type <- match.arg(type)
  pts <- if (type == "clustered") {
    gen_clustered(n, n_centers = 5, sd = 50, extent = extent)
  } else {
    gen_uniform(n, extent = extent)
  }
  cx <- pts[, 1]
  cy <- pts[, 2]

  cols <- vector("list", p)
  if (p >= 1) cols[[1]] <- cx + rnorm(n, sd = 0.5)
  if (p >= 2) cols[[2]] <- cy + rnorm(n, sd = 0.5)
  if (p >= 3) cols[[3]] <- sin(cx / 100) * 100 + cy * 0.1 + rnorm(n, sd = 1)
  if (p >= 4) cols[[4]] <- exp(cx / 1000) * 50 + cos(cy / 100) * 50 + rnorm(n, sd = 1)
  if (p >= 5) cols[[5]] <- (cx * cy) / 1000 + sin((cx + cy) / 200) * 50 + rnorm(n, sd = 1)
  if (p > 5) {
    for (j in 6:p) {
      cols[[j]] <- sin(cx * j / 200) * 50 + cos(cy * j / 200) * 50 + rnorm(n, sd = 1)
    }
  }
  df <- as.data.frame(cols)
  colnames(df) <- paste0("x", seq_len(p))
  df
}

# -----------------------------------------------------------------------------
# Case Definitions
# -----------------------------------------------------------------------------

cases <- list(
  # 1. Geographical, hierarchical: clustered n=50,100,200,400 with k=5,10,10,10
  list(
    name = "geo_hier_clust_50_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 50),
    knndm_args = list(k = 5)
  ),
  list(
    name = "geo_hier_clust_100_k10",
    space = "geographical",
    data_spec = list(type = "clustered", n = 100),
    knndm_args = list(k = 10)
  ),
  list(
    name = "geo_hier_clust_200_k10",
    space = "geographical",
    data_spec = list(type = "clustered", n = 200),
    knndm_args = list(k = 10)
  ),
  list(
    name = "geo_hier_clust_400_k10",
    space = "geographical",
    data_spec = list(type = "clustered", n = 400),
    knndm_args = list(k = 10)
  ),

  # 2. Geographical, hierarchical: uniform n=150 k=5; grid 12x12 k=4; dupes n=120 k=6
  list(
    name = "geo_hier_unif_150_k5",
    space = "geographical",
    data_spec = list(type = "uniform", n = 150),
    knndm_args = list(k = 5)
  ),
  list(
    name = "geo_hier_grid_12x12_k4",
    space = "geographical",
    data_spec = list(type = "grid", n_side = 12),
    knndm_args = list(k = 4)
  ),
  list(
    name = "geo_hier_dupes_120_k6",
    space = "geographical",
    data_spec = list(type = "dupes", n = 120),
    knndm_args = list(k = 6)
  ),

  # 3. Geographical, hierarchical: linkf variants on clustered n=120 k=5
  list(
    name = "geo_hier_linkf_average_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "average")
  ),
  list(
    name = "geo_hier_linkf_complete_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "complete")
  ),
  list(
    name = "geo_hier_linkf_single_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "single")
  ),
  list(
    name = "geo_hier_linkf_ward_D_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "ward.D")
  ),
  list(
    name = "geo_hier_linkf_mcquitty_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "mcquitty")
  ),
  list(
    name = "geo_hier_linkf_median_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "median")
  ),
  list(
    name = "geo_hier_linkf_centroid_120_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 120),
    knndm_args = list(k = 5, linkf = "centroid")
  ),

  # 4. Geographical, hierarchical: maxp=0.3 on clustered n=150 k=5; nk_len=10 on clustered n=150 k=10
  list(
    name = "geo_hier_maxp_03_150_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 150),
    knndm_args = list(k = 5, maxp = 0.3)
  ),
  list(
    name = "geo_hier_nklen_10_150_k10",
    space = "geographical",
    data_spec = list(type = "clustered", n = 150),
    knndm_args = list(k = 10, nk_len = 10)
  ),

  # 5. Geographical, kmeans: clustered n=60 k=5, n=150 k=10, n=300 k=4, uniform n=100 k=3
  list(
    name = "geo_kmeans_clust_60_k5",
    space = "geographical",
    data_spec = list(type = "clustered", n = 60),
    knndm_args = list(k = 5, clustering = "kmeans")
  ),
  list(
    name = "geo_kmeans_clust_150_k10",
    space = "geographical",
    data_spec = list(type = "clustered", n = 150),
    knndm_args = list(k = 10, clustering = "kmeans")
  ),
  list(
    name = "geo_kmeans_clust_300_k4",
    space = "geographical",
    data_spec = list(type = "clustered", n = 300),
    knndm_args = list(k = 4, clustering = "kmeans")
  ),
  list(
    name = "geo_kmeans_unif_100_k3",
    space = "geographical",
    data_spec = list(type = "uniform", n = 100),
    knndm_args = list(k = 3, clustering = "kmeans")
  ),

  # 6. Geographical random-CV cases (jittered predpoints)
  list(
    name = "geo_hier_random_cv_100_k5",
    space = "geographical",
    data_spec = list(type = "random_cv", n = 100),
    knndm_args = list(k = 5)
  ),
  list(
    name = "geo_kmeans_random_cv_100_k5",
    space = "geographical",
    data_spec = list(type = "random_cv", n = 100),
    knndm_args = list(k = 5, clustering = "kmeans")
  ),

  # 7. Geographical test_prop: clustered n=150 test_prop=0.2; n=80 test_prop=0.3 test_tolerance=0.05; kmeans n=100 test_prop=0.25
  list(
    name = "geo_hier_testprop_150",
    space = "geographical",
    data_spec = list(type = "clustered", n = 150),
    knndm_args = list(test_prop = 0.2)
  ),
  list(
    name = "geo_hier_testprop_80_tol",
    space = "geographical",
    data_spec = list(type = "clustered", n = 80),
    knndm_args = list(test_prop = 0.3, test_tolerance = 0.05)
  ),
  list(
    name = "geo_kmeans_testprop_100",
    space = "geographical",
    data_spec = list(type = "clustered", n = 100),
    knndm_args = list(clustering = "kmeans", test_prop = 0.25)
  ),

  # 8. Feature, hierarchical: n=80 p=3 k=5; n=200 p=5 k=10; scale_vars=FALSE n=100 p=3 k=4; linkf="average" n=100 p=4 k=5
  list(
    name = "feat_hier_80_p3_k5",
    space = "feature",
    data_spec = list(type = "feature", n = 80, p = 3),
    knndm_args = list(k = 5)
  ),
  list(
    name = "feat_hier_200_p5_k10",
    space = "feature",
    data_spec = list(type = "feature", n = 200, p = 5),
    knndm_args = list(k = 10)
  ),
  list(
    name = "feat_hier_noscale_100_p3_k4",
    space = "feature",
    data_spec = list(type = "feature", n = 100, p = 3),
    knndm_args = list(k = 4, scale_vars = FALSE)
  ),
  list(
    name = "feat_hier_linkf_average_100_p4_k5",
    space = "feature",
    data_spec = list(type = "feature", n = 100, p = 4),
    knndm_args = list(k = 5, linkf = "average")
  ),

  # 9. Feature, kmeans: n=100 p=3 k=5; n=150 p=4 k=8
  list(
    name = "feat_kmeans_100_p3_k5",
    space = "feature",
    data_spec = list(type = "feature", n = 100, p = 3),
    knndm_args = list(k = 5, clustering = "kmeans")
  ),
  list(
    name = "feat_kmeans_150_p4_k8",
    space = "feature",
    data_spec = list(type = "feature", n = 150, p = 4),
    knndm_args = list(k = 8, clustering = "kmeans")
  ),

  # 10. Feature test_prop=0.2 n=120 p=3; feature random-CV n=90 p=3 k=4
  list(
    name = "feat_hier_testprop_120_p3",
    space = "feature",
    data_spec = list(type = "feature", n = 120, p = 3),
    knndm_args = list(test_prop = 0.2)
  ),
  list(
    name = "feat_hier_random_cv_90_p3_k4",
    space = "feature",
    data_spec = list(type = "feature_random_cv", n = 90, p = 3),
    knndm_args = list(k = 4)
  )
  ,
  # 11. Stress cases: exact ties in PCA ordering, duplicate kmeans centres, larger n
  list(name = "geo_hier_grid_15x15_far_k5", space = "geographical",
       data_spec = list(type = "grid_far", n_side = 15), knndm_args = list(k = 5)),
  list(name = "geo_kmeans_grid_10x10_far_k4", space = "geographical",
       data_spec = list(type = "grid_far", n_side = 10), knndm_args = list(k = 4, clustering = "kmeans")),
  list(name = "geo_kmeans_dupes_150_k5", space = "geographical",
       data_spec = list(type = "dupes", n = 150), knndm_args = list(k = 5, clustering = "kmeans")),
  list(name = "geo_hier_clust_1500_k10", space = "geographical",
       data_spec = list(type = "clustered", n = 1500), knndm_args = list(k = 10)),
  list(name = "feat_hier_600_p8_k10", space = "feature",
       data_spec = list(type = "feature", n = 600, p = 8), knndm_args = list(k = 10)),
  list(name = "feat_kmeans_300_p6_k6", space = "feature",
       data_spec = list(type = "feature", n = 300, p = 6), knndm_args = list(k = 6, clustering = "kmeans")),
  list(name = "geo_hier_grid_12x12_far_single_k4", space = "geographical",
       data_spec = list(type = "grid_far", n_side = 12), knndm_args = list(k = 4, linkf = "single"))
)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

for (i in seq_along(cases)) {
  case <- cases[[i]]
  knn_seed <- 1000 + i
  data_seed <- 2000 + i
  pred_seed <- 3000 + i
  ds <- case$data_spec

  # Generate training and prediction points
  if (case$space == "geographical") {
    n_pred <- 500
    if (ds$type == "clustered") {
      t_mat <- gen_clustered(ds$n, data_seed = data_seed)
      p_mat <- gen_uniform(n_pred, data_seed = pred_seed)
    } else if (ds$type == "uniform") {
      t_mat <- gen_uniform(ds$n, data_seed = data_seed)
      p_mat <- gen_uniform(n_pred, data_seed = pred_seed)
    } else if (ds$type == "grid") {
      t_mat <- gen_grid(ds$n_side, data_seed = data_seed)
      p_mat <- gen_uniform(n_pred, data_seed = pred_seed)
    } else if (ds$type == "grid_far") {
      # training on a regular grid, prediction area 3x larger: forces clustering
      t_mat <- gen_grid(ds$n_side, data_seed = data_seed)
      p_mat <- gen_uniform(n_pred, extent = 3000, data_seed = pred_seed)
    } else if (ds$type == "dupes") {
      t_mat <- gen_dupes(ds$n, data_seed = data_seed)
      p_mat <- gen_uniform(n_pred, data_seed = pred_seed)
    } else if (ds$type == "random_cv") {
      t_mat <- gen_clustered(ds$n, data_seed = data_seed)
      set.seed(pred_seed)
      jitter <- matrix(rnorm(nrow(t_mat) * 2, mean = 0, sd = 1), ncol = 2)
      p_mat <- t_mat + jitter
      colnames(p_mat) <- c("x", "y")
    } else {
      stop("Unknown geographical data type: ", ds$type)
    }

    tpoints <- sf::st_as_sf(as.data.frame(t_mat), coords = c("x", "y"), crs = 3857)
    predpoints <- sf::st_as_sf(as.data.frame(p_mat), coords = c("x", "y"), crs = 3857)

    coords_t <- sf::st_coordinates(tpoints)
    tpoints_json <- list(
      x = I(hexv(coords_t[, 1])),
      y = I(hexv(coords_t[, 2]))
    )
    coords_p <- sf::st_coordinates(predpoints)
    predpoints_json <- list(
      x = I(hexv(coords_p[, 1])),
      y = I(hexv(coords_p[, 2]))
    )
  } else if (case$space == "feature") {
    n_pred <- 400
    if (ds$type == "feature") {
      tpoints <- gen_features(ds$n, ds$p, type = "clustered", data_seed = data_seed)
      predpoints <- gen_features(n_pred, ds$p, type = "uniform", data_seed = pred_seed)
    } else if (ds$type == "feature_random_cv") {
      tpoints <- gen_features(ds$n, ds$p, type = "clustered", data_seed = data_seed)
      set.seed(pred_seed)
      noise <- matrix(rnorm(ds$n * ds$p, mean = 0, sd = 0.01), nrow = ds$n, ncol = ds$p)
      predpoints <- as.data.frame(as.matrix(tpoints) + noise)
      colnames(predpoints) <- colnames(tpoints)
    } else {
      stop("Unknown feature data type: ", ds$type)
    }

    cols_t <- colnames(tpoints)
    tpoints_json <- list(colnames = I(cols_t))
    for (cn in cols_t) {
      tpoints_json[[cn]] <- I(hexv(tpoints[[cn]]))
    }

    cols_p <- colnames(predpoints)
    predpoints_json <- list(colnames = I(cols_p))
    for (cn in cols_p) {
      predpoints_json[[cn]] <- I(hexv(predpoints[[cn]]))
    }
  } else {
    stop("Unknown space: ", case$space)
  }

  # Build arguments for CAST::knndm (only args explicitly set by the case)
  call_args <- list(
    tpoints = tpoints,
    predpoints = predpoints,
    dist_space = case$space
  )
  for (arg_name in names(case$knndm_args)) {
    val <- case$knndm_args[[arg_name]]
    if (!is.null(val)) {
      call_args[[arg_name]] <- val
    }
  }

  # Run knndm with seed, capturing warnings and messages
  captured_messages <- character(0)
  res <- tryCatch(
    suppressWarnings(
      suppressMessages(
        withCallingHandlers(
          {
            set.seed(knn_seed)
            do.call(CAST::knndm, call_args)
          },
          message = function(m) {
            captured_messages <<- c(captured_messages, sub("\n$", "", conditionMessage(m)))
            invokeRestart("muffleMessage")
          },
          warning = function(w) {
            captured_messages <<- c(captured_messages, sub("\n$", "", conditionMessage(w)))
            invokeRestart("muffleWarning")
          }
        )
      )
    ),
    error = function(e) e
  )
  rng_after <- .Random.seed

  # JSON args: full argument structure with NULL for unused
  args_json <- list(
    k = if (!is.null(case$knndm_args$k)) as.integer(case$knndm_args$k) else NULL,
    maxp = if (!is.null(case$knndm_args$maxp)) as.numeric(case$knndm_args$maxp) else NULL,
    clustering = if (!is.null(case$knndm_args$clustering)) as.character(case$knndm_args$clustering) else NULL,
    linkf = if (!is.null(case$knndm_args$linkf)) as.character(case$knndm_args$linkf) else NULL,
    nk_len = if (!is.null(case$knndm_args$nk_len)) as.integer(case$knndm_args$nk_len) else NULL,
    test_prop = if (!is.null(case$knndm_args$test_prop)) as.numeric(case$knndm_args$test_prop) else NULL,
    test_tolerance = if (!is.null(case$knndm_args$test_tolerance)) as.numeric(case$knndm_args$test_tolerance) else NULL,
    scale_vars = if (!is.null(case$knndm_args$scale_vars)) as.logical(case$knndm_args$scale_vars) else NULL
  )

  # Assemble JSON fixture object
  if (inherits(res, "error")) {
    json_obj <- list(
      name = case$name,
      space = case$space,
      args = args_json,
      knndm_seed = as.integer(knn_seed),
      tpoints = tpoints_json,
      predpoints = predpoints_json,
      error = conditionMessage(res),
      messages = I(as.character(captured_messages)),
      rng_after = I(as.integer(rng_after))
    )
    cat(sprintf("%s: ERROR: %s\n", case$name, conditionMessage(res)))
  } else {
    clusters_val <- if (is.character(res$clusters)) {
      I(as.character(res$clusters))
    } else {
      I(as.integer(res$clusters))
    }

    indx_train_val <- if (is.list(res$indx_train)) {
      lapply(res$indx_train, function(v) I(as.integer(v)))
    } else {
      list(I(as.integer(res$indx_train)))
    }

    indx_test_val <- if (is.list(res$indx_test)) {
      lapply(res$indx_test, function(v) I(as.integer(v)))
    } else {
      list(I(as.integer(res$indx_test)))
    }

    q_val <- if (is.numeric(res$q)) as.integer(res$q) else as.character(res$q)

    json_obj <- list(
      name = case$name,
      space = case$space,
      args = args_json,
      knndm_seed = as.integer(knn_seed),
      tpoints = tpoints_json,
      predpoints = predpoints_json,
      clusters = clusters_val,
      indx_train = indx_train_val,
      indx_test = indx_test_val,
      Gij = I(hexv(res$Gij)),
      Gj = I(hexv(res$Gj)),
      Gjstar = I(hexv(res$Gjstar)),
      W = hexv(res$W),
      q = q_val,
      method = as.character(res$method),
      messages = I(as.character(captured_messages)),
      rng_after = I(as.integer(rng_after))
    )
    cat(sprintf("%s: q=%s, W=%.17g\n", case$name, as.character(res$q), as.numeric(res$W)))
  }

  out_path <- file.path(outdir, paste0(case$name, ".json"))
  jsonlite::write_json(json_obj, out_path, auto_unbox = TRUE, digits = NA, null = "null")
}

cat(sprintf("DONE %d\n", length(cases)))
