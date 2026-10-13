.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
library(CAST)
library(sf)
library(jsonlite)

# Output directory: command-line argument or default "tests/fixtures/twcv"
args_cli <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args_cli) >= 1 && nzchar(args_cli[1])) args_cli[1] else "tests/fixtures/twcv"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# R functions copied verbatim from the TWCVpaper repository
script_dir <- {
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(sub("^--file=", "", a[1])) else "reference"
}
source(file.path(script_dir, "twcv_source.R"))

hexv <- function(x) sprintf("%a", as.numeric(x))

# Encode a column: hex strings (NA -> null) for numeric, strings (NA -> null) for character
enc <- function(v) {
  if (is.character(v)) return(I(v))
  h <- hexv(v)
  h[is.na(v)] <- NA_character_
  I(h)
}

gen_clustered <- function(n, n_centers = 5, sd = 50, extent = 1000) {
  centers <- matrix(runif(n_centers * 2, min = 0, max = extent), ncol = 2)
  c_idx <- sample(n_centers, n, replace = TRUE)
  pts <- centers[c_idx, , drop = FALSE] + matrix(rnorm(n * 2, mean = 0, sd = sd), ncol = 2)
  colnames(pts) <- c("x", "y")
  pts
}
gen_uniform <- function(n, extent = 1000) {
  pts <- matrix(runif(n * 2, min = 0, max = extent), ncol = 2)
  colnames(pts) <- c("x", "y")
  pts
}

# Smooth covariates of the coordinates plus noise
cov_num <- function(xy, j, sdn = 0.3) {
  x <- xy[, 1]; y <- xy[, 2]
  base <- switch(j,
    sin(x / 250) + 0.002 * y,
    cos(y / 300) * 2 + 0.001 * x * 0.5,
    (x + y) / 500)
  base + rnorm(nrow(xy), sd = sdn)
}
cov_cat <- function(n, probs) sample(c("a", "b", "c"), n, replace = TRUE, prob = probs)

cases <- list(
  list(name = "twcv_a_dwcv_n200", n = 200, m = 800, k = 5, folds = "knndm",
       covs = character(0), by = 0.2, lambda = 0),
  list(name = "twcv_b_cov1_n300", n = 300, m = 1000, k = 8, folds = "knndm",
       covs = c("c1"), by = 0.2, lambda = 0.2),
  list(name = "twcv_c_cov2_n400", n = 400, m = 1500, k = 10, folds = "knndm",
       covs = c("c1", "c2"), by = 0.1, lambda = 0.5),
  list(name = "twcv_d_cat_n250", n = 250, m = 900, k = 6, folds = "knndm",
       covs = c("c1", "cat"), by = 0.2, lambda = 0.2),
  list(name = "twcv_e_randomcv_na_n150", n = 150, m = 600, k = 5, folds = "random",
       covs = c("c1"), by = 0.2, lambda = 0.2, na_train = 6),
  list(name = "twcv_f_binary_cov_n120", n = 120, m = 400, k = 4, folds = "knndm",
       covs = c("b1"), by = 0.2, lambda = 0.2),
  list(name = "twcv_g_small_n80", n = 80, m = 300, k = 4, folds = "knndm",
       covs = c("c1"), by = 0.25, lambda = 0.1),
  list(name = "twcv_h_pred_na_n200", n = 200, m = 700, k = 5, folds = "knndm",
       covs = c("c1"), by = 0.2, lambda = 0.2, na_pred = 10)
)

for (i in seq_along(cases)) {
  cs <- cases[[i]]
  set.seed(5000 + i)
  t_mat <- gen_clustered(cs$n)
  p_mat <- gen_uniform(cs$m)

  # covariates
  tcov <- list(); pcov <- list()
  for (cn in cs$covs) {
    if (cn == "c1") {
      tcov[[cn]] <- cov_num(t_mat, 1); pcov[[cn]] <- cov_num(p_mat, 1)
    } else if (cn == "c2") {
      tcov[[cn]] <- cov_num(t_mat, 2); pcov[[cn]] <- cov_num(p_mat, 2)
    } else if (cn == "cat") {
      tcov[[cn]] <- cov_cat(cs$n, c(0.6, 0.3, 0.1)); pcov[[cn]] <- cov_cat(cs$m, c(0.2, 0.3, 0.5))
    } else if (cn == "b1") {
      tcov[[cn]] <- as.numeric(cov_num(t_mat, 3) > 1.0); pcov[[cn]] <- as.numeric(cov_num(p_mat, 3) > 1.0)
    }
  }
  if (!is.null(cs$na_train)) {
    na_idx <- sample(cs$n, cs$na_train)
    tcov[[cs$covs[1]]][na_idx] <- NA
  }

  if (!is.null(cs$na_pred)) {
    pcov[[cs$covs[1]]][sample(cs$m, cs$na_pred)] <- NA
  }

  # folds
  if (cs$folds == "knndm") {
    tpoints <- sf::st_as_sf(as.data.frame(t_mat), coords = c("x", "y"), crs = 3857)
    predpoints <- sf::st_as_sf(as.data.frame(p_mat), coords = c("x", "y"), crs = 3857)
    res <- suppressWarnings(suppressMessages(CAST::knndm(tpoints = tpoints, predpoints = predpoints, k = cs$k)))
    folds <- as.integer(res$clusters)
  } else {
    folds <- as.integer(sample(rep(1:cs$k, length.out = cs$n)))
  }

  # synthetic observations / predictions
  obs <- sin(t_mat[, 1] / 300) + cos(t_mat[, 2] / 200) + rnorm(cs$n, sd = 0.2)

  # task descriptors exactly as compute_cv_estimators does
  sample_dat <- data.frame(id = seq_len(cs$n), x = t_mat[, 1], y = t_mat[, 2])
  grid_dat <- data.frame(id = seq_len(cs$m), x = p_mat[, 1], y = p_mat[, 2])
  d_cv <- compute_cv_prediction_distances(sample_dat, folds)
  d_pred <- nearest_neighbor_distance(grid_dat, sample_dat)

  pred_values <- obs + rnorm(cs$n, sd = 0.1) * (0.5 + d_cv / 100)

  loss_df <- data.frame(id = sample_dat$id, d = d_cv)
  grid_tasks <- data.frame(id = grid_dat$id, d = d_pred)
  for (cn in cs$covs) {
    loss_df[[cn]] <- tcov[[cn]]
    grid_tasks[[cn]] <- pcov[[cn]]
  }
  balancing_vars <- c(cs$covs, "d")  # "d" last (Python iterates covariates first)

  bal <- prepare_balanced_tasks_cv(loss_df, grid_tasks, balancing_vars, by = cs$by)
  tw <- compute_twcv_weights(bal$sample_tasks_bal, bal$grid_tasks_bal, balancing_vars, shrink_lambda = cs$lambda)

  err <- obs - pred_values
  ldf <- data.frame(error = err, se = err^2, ae = abs(err))
  est_u <- summarize_losses(ldf)
  est_w <- summarize_losses(ldf, tw$weights)
  dg <- summarize_weight_vector(tw$weights)

  est_json <- function(e) list(bias = hexv(e[["bias"]]), mse = hexv(e[["mse"]]),
                               rmse = hexv(e[["rmse"]]), mae = hexv(e[["mae"]]))

  train_json <- list(x = enc(t_mat[, 1]), y = enc(t_mat[, 2]))
  pred_json <- list(x = enc(p_mat[, 1]), y = enc(p_mat[, 2]))
  for (cn in cs$covs) {
    train_json[[cn]] <- enc(tcov[[cn]])
    pred_json[[cn]] <- enc(pcov[[cn]])
  }

  json_obj <- list(
    name = cs$name,
    args = list(balancing_vars = I(balancing_vars), by = cs$by, shrink_lambda = cs$lambda),
    train = train_json,
    pred = pred_json,
    folds = I(folds),
    obs = enc(obs),
    pred_values = enc(pred_values),
    expected = list(
      d_cv = enc(d_cv),
      d_pred = enc(d_pred),
      weights = enc(tw$weights),
      weights_raw = enc(tw$weights_raw),
      converged = tw$converged,
      iterations = as.integer(tw$iterations),
      unweighted = est_json(est_u),
      weighted = est_json(est_w),
      diagnostics = list(
        n_cases = as.integer(dg$n_cases),
        effective_sample_size = hexv(dg$effective_sample_size),
        ess_fraction = hexv(dg$ess_fraction),
        max_weight = hexv(dg$max_weight),
        q95_weight = hexv(dg$q95_weight),
        q99_weight = hexv(dg$q99_weight),
        gini = hexv(dg$gini)
      )
    )
  )
  out_path <- file.path(outdir, paste0(cs$name, ".json"))
  jsonlite::write_json(json_obj, out_path, auto_unbox = TRUE, digits = NA, null = "null", na = "null")
  cat(sprintf("%s: converged=%s iter=%d ess=%.3f\n", cs$name, tw$converged, tw$iterations, dg$effective_sample_size))
}

# -----------------------------------------------------------------------------
# Numeric primitive fixtures: R-level sqrt (x87 double rounding on Windows),
# quantile(type = 7) and mean()
# -----------------------------------------------------------------------------
make_primitive_fixtures <- function(outdir) {
  set.seed(777)
  xs <- c(0x1.f6bcffb28882fp+1, 2, 3, 5, 0.5, 1e-300, 1e300, 0,
          runif(900, 0, 10), runif(600, 0, 1e6), rexp(500, 0.01))
  w <- jsonlite::write_json
  w(list(x = I(hexv(xs)), sqrt = I(hexv(sqrt(xs)))),
    file.path(outdir, "r_sqrt.json"), auto_unbox = TRUE, digits = NA)

  qcases <- list(
    list(x = c(0, 1, 2), p = c(0.4999999999999999, 0.5, 0.5000000000000001, 0, 1)),
    list(x = c(3, 1, 2, 5, 4), p = seq(0, 1, by = 0.2)),
    list(x = round(rnorm(200), 1), p = seq(0, 1, by = 0.1)),
    list(x = c(rnorm(50), NA, rnorm(50)), p = seq(0, 1, by = 0.05)),
    list(x = runif(37), p = c(0.01, 0.95, 0.99, 0.3333333333333333)),
    list(x = c(5, 5, 5, 5), p = seq(0, 1, by = 0.25)),
    list(x = rexp(1000), p = seq(0, 1, by = 0.1))
  )
  qj <- lapply(qcases, function(cs) list(
    x = I(ifelse(is.na(cs$x), NA_character_, hexv(cs$x))), p = I(hexv(cs$p)),
    q = I(hexv(stats::quantile(cs$x, probs = cs$p, na.rm = TRUE, type = 7, names = FALSE)))))
  w(list(cases = qj), file.path(outdir, "r_quantile.json"), auto_unbox = TRUE, digits = NA, na = "null")

  xmax <- .Machine$double.xmax
  mcases <- list(
    c(xmax, xmax), c(xmax, xmax, xmax / 3), c(xmax, xmax / 7, xmax / 3), c(1, 2, 3), c(0.1, 0.2, 0.3), rnorm(10),
    rnorm(1000), runif(333, 0, 1e10), c(1e16, 1, -1e16, 1), rexp(77)
  )
  mj <- lapply(mcases, function(v) list(x = I(hexv(v)), mean = hexv(mean(v))))
  w(list(cases = mj), file.path(outdir, "r_mean.json"), auto_unbox = TRUE, digits = NA)
}
make_primitive_fixtures(outdir)
cat(sprintf("DONE %d\n", length(cases)))
