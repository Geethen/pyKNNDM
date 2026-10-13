# -----------------------------------------------------------------------------
# Verbatim excerpt of the R functions needed to compute target-weighted CV
# (TWCV) weights, taken from https://github.com/alexanderbrenning/TWCVpaper
# (files R/weights.R, R/weighting_twcv.R, R/cv_designs_simstudy.R,
# R/loss_metrics.R, R/cv_generic.R, R/task_descriptors.R, R/utils.R).
#
# Brenning, A. & Suesse, T. (2026). Aligning Validation with Deployment:
# Target-Weighted Cross-Validation for Spatial Prediction. arXiv:2603.29981.
#
# The original code is licensed under the GNU General Public License v3 (GPL-3);
# this excerpt is distributed under the same license. Function bodies are
# unmodified (roxygen comments omitted); it is used only to generate reference
# fixtures for the Python port in python/knndm/twcv.py.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages(library(FNN))

rake_weights <- function(balance_df,
                         target_margins,
                         base_weights = NULL,
                         max_iter = 500,
                         tol = 1e-6,
                         verbose = 0) {

  n <- nrow(balance_df)
  if (is.null(base_weights)) {
    w <- rep(1, n)
  } else {
    w <- as.numeric(base_weights)
  }

  margin_names <- names(target_margins)

  for (iter in seq_len(max_iter)) {
    w_old <- w

    for (m in margin_names) {
      x <- as.integer(balance_df[[m]])
      levs <- seq_along(target_margins[[m]])
      target_prop <- target_margins[[m]]

      current_totals <- tapply(w, factor(x, levels = levs), sum)
      current_totals[is.na(current_totals)] <- 0

      target_totals <- sum(w) * target_prop

      adj <- rep(1, length(levs))
      ok <- current_totals > 0

      adj[ok] <- target_totals[ok] / current_totals[ok]
      adj[!ok] <- NA_real_

      valid <- !is.na(adj[x])
      w[valid] <- w[valid] * adj[x[valid]]
    }

    rel_change <- max(abs(w - w_old) / pmax(abs(w_old), 1e-12))

    if (verbose >= 2) {
      log_message(verbose, 2, "  raking iter ", iter,
                  ", max relative change = ", format(rel_change, digits = 3))
    }

    if (rel_change < tol) break
  }

  converged <- iter < max_iter || rel_change < tol

  list(
    weights = w,
    converged = converged,
    iterations = iter
  )
}

tabulate_proportions <- function(x, levels = NULL) {
  x <- as.integer(x)
  if (is.null(levels)) levels <- sort(unique(x))
  out <- table(factor(x, levels = levels))
  as.numeric(out) / sum(out)
}

compute_target_margins_generic <- function(grid_tasks_bal, balancing_vars) {
  out <- vector("list", length(balancing_vars))
  names(out) <- balancing_vars

  for (v in balancing_vars) {
    vn <- paste0(v, "_cat")
    if (!(vn %in% names(grid_tasks_bal))) {
      stop("Missing column in grid_tasks_bal: ", vn, call. = FALSE)
    }
    out[[v]] <- tabulate_proportions(grid_tasks_bal[[vn]])
  }

  out
}

normalize_weights <- function(w) {
  w / mean(w)
}

shrink_weights <- function(w, lambda = 0) {
  stopifnot(lambda >= 0, lambda <= 1)
  w <- normalize_weights(w)
  (1 - lambda) * w + lambda
}

compute_twcv_weights <- function(sample_tasks_bal,
                                 grid_tasks_bal,
                                 balancing_vars,
                                 shrink_lambda = 0,
                                 verbose = 0) {
  balance_df <- as.data.frame(
    lapply(balancing_vars, function(v) sample_tasks_bal[[paste0(v, "_cat")]])
  )
  names(balance_df) <- balancing_vars

  target_margins <- compute_target_margins_generic(
    grid_tasks_bal = grid_tasks_bal,
    balancing_vars = balancing_vars
  )

  out <- rake_weights(
    balance_df = balance_df,
    target_margins = target_margins,
    verbose = verbose
  )

  out$weights_raw <- normalize_weights(out$weights)
  out$weights <- shrink_weights(out$weights_raw, lambda = shrink_lambda)
  out$shrink_lambda <- shrink_lambda
  out$balancing_vars <- balancing_vars

  out
}

prepare_balanced_tasks_cv <- function(loss_df,
                                      grid_tasks,
                                      balancing_vars,
                                      by = 0.2) {

  stopifnot(length(balancing_vars) > 0)

  sample_tasks_bal <- prepare_for_balancing(
    df = loss_df,
    vars = balancing_vars,
    ref_df = grid_tasks,
    by = by
  )

  grid_tasks_bal <- prepare_for_balancing(
    df = grid_tasks,
    vars = balancing_vars,
    ref_df = grid_tasks,
    by = by
  )

  list(
    sample_tasks_bal = sample_tasks_bal,
    grid_tasks_bal = grid_tasks_bal
  )
}

prepare_for_balancing <- function(df,
                                  vars,
                                  ref_df,
                                  by = 0.2) {

  df_out <- df

  for (v in vars) {

    if (!(v %in% names(df))) {
      stop("Variable '", v, "' not found in df.", call. = FALSE)
    }
    if (!(v %in% names(ref_df))) {
      stop("Variable '", v, "' not found in ref_df.", call. = FALSE)
    }

    x_ref <- ref_df[[v]]
    x     <- df[[v]]
    out_name <- paste0(v, "_cat")

    # numeric variables
    if (is.numeric(x_ref) && length(unique(stats::na.omit(x_ref))) > 2) {

      probs <- seq(0, 1, by = by)
      qtiles <- stats::quantile(x_ref, probs = probs, na.rm = TRUE, names = FALSE)

      qtiles <- unique(qtiles)

      # degenerate case
      if (length(qtiles) < 2) {
        levs <- paste0(v, "_Q1")
        df_out[[out_name]] <- factor(
          ifelse(is.na(x), NA, levs),
          levels = levs,
          ordered = TRUE
        )
        next
      }

      qtiles[1] <- -Inf
      qtiles[length(qtiles)] <- Inf

      n_bins <- length(qtiles) - 1L
      levs <- paste0(v, "_Q", seq_len(n_bins))

      df_out[[out_name]] <- cut(
        x,
        breaks = qtiles,
        include.lowest = TRUE,
        labels = levs,
        ordered_result = TRUE
      )

    } else {

      ref_levels <- sort(unique(as.character(stats::na.omit(x_ref))))
      x_chr <- as.character(x)

      x_chr[!(x_chr %in% ref_levels) & !is.na(x_chr)] <- NA_character_

      df_out[[out_name]] <- factor(
        x_chr,
        levels = ref_levels,
        ordered = FALSE
      )
    }
  }

  df_out
}

compute_cv_prediction_distances <- function(sample_dat, folds) {
  n <- nrow(sample_dat)
  d <- rep(NA_real_, n)

  check_xy_columns(sample_dat)
  coords <- as.matrix(sample_dat[, c("x", "y")])

  for (f in unique(folds)) {
    test_idx  <- which(folds == f)
    train_idx <- which(folds != f)

    if (length(train_idx) == 0) {
      d[test_idx] <- NA_real_
      next
    }

    dmat <- pairwise_distance_matrix(
      coords1 = coords[test_idx, , drop = FALSE],
      coords2 = coords[train_idx, , drop = FALSE]
    )

    d[test_idx] <- apply(dmat, 1, min)
  }

  d
}

pairwise_distance_matrix <- function(coords1, coords2 = NULL) {
  coords1 <- as.matrix(coords1)

  if (is.null(coords2)) {
    return(as.matrix(stats::dist(coords1)))
  }

  coords2 <- as.matrix(coords2)

  x1 <- coords1[, 1]
  y1 <- coords1[, 2]
  x2 <- coords2[, 1]
  y2 <- coords2[, 2]

  sqrt(outer(x1, x2, "-")^2 + outer(y1, y2, "-")^2)
}

check_xy_columns <- function(dat) {
  if (!all(c("x", "y") %in% names(dat))) {
    stop("Data must contain columns named 'x' and 'y'.", call. = FALSE)
  }
  if (!is.numeric(dat$x) || !is.numeric(dat$y)) {
    stop("Columns 'x' and 'y' must be numeric.", call. = FALSE)
  }
  invisible(TRUE)
}

nearest_neighbor_distance <- function(query_coords,
                                      ref_coords,
                                      exclude_self = FALSE) {
  q <- as.matrix(query_coords[, c("x", "y")])
  r <- as.matrix(ref_coords[, c("x", "y")])

  if (!exclude_self) {
    nn <- FNN::get.knnx(data = r, query = q, k = 1)
    return(as.numeric(nn$nn.dist[, 1]))
  }

  # Self-exclusion only matters when query and reference coordinates coincide.
  same_object <- nrow(q) == nrow(r) &&
    isTRUE(all.equal(q, r, check.attributes = FALSE))

  if (!same_object) {
    nn <- FNN::get.knnx(data = r, query = q, k = 1)
    return(as.numeric(nn$nn.dist[, 1]))
  }

  # For identical point sets, use the second-nearest neighbor.
  nn <- FNN::get.knnx(data = r, query = q, k = 2)
  as.numeric(nn$nn.dist[, 2])
}

weighted_mean_safe <- function(x, w = NULL) {
  if (is.null(w)) {
    return(mean(x, na.rm = TRUE))
  }

  ok <- is.finite(x) & is.finite(w)
  if (!any(ok)) return(NA_real_)

  x <- x[ok]
  w <- w[ok]

  w_sum <- sum(w)
  if (!is.finite(w_sum) || w_sum <= 0) {
    return(NA_real_)
  }

  sum(w * x) / w_sum
}

summarize_losses <- function(loss_df, weights = NULL) {

  if (is.null(weights)) {
    weights <- rep(1, nrow(loss_df))
  }

  stopifnot(length(weights) == nrow(loss_df))

  # normalize weights for numerical stability (optional but useful)
  w_sum <- sum(weights)
  if (is.finite(w_sum) && w_sum > 0) {
    weights <- weights / w_sum
  }

  c(
    bias = weighted_mean_safe(loss_df$error, weights),
    mse  = weighted_mean_safe(loss_df$se, weights),
    rmse = sqrt(weighted_mean_safe(loss_df$se, weights)),
    mae  = weighted_mean_safe(loss_df$ae, weights)
  )
}

effective_sample_size <- function(w) {
  w <- as.numeric(w)
  w <- w[is.finite(w) & w >= 0]
  if (length(w) == 0) return(NA_real_)
  s <- sum(w)
  if (s <= 0) return(NA_real_)
  w <- w / s
  1 / sum(w^2)
}

gini_coefficient <- function(w) {
  w <- as.numeric(w)
  w <- w[is.finite(w) & w >= 0]
  if (length(w) == 0) return(NA_real_)
  if (sum(w) == 0) return(NA_real_)

  w <- sort(w)
  n <- length(w)
  idx <- seq_len(n)

  (2 * sum(idx * w) / (n * sum(w))) - (n + 1) / n
}

q95_weight <- function(w) {
  w <- as.numeric(w)
  w <- w[is.finite(w) & w >= 0]
  if (length(w) == 0) return(NA_real_)
  as.numeric(stats::quantile(w, probs = 0.95, na.rm = TRUE))
}

summarize_weight_vector <- function(w) {
  w <- as.numeric(w)
  w <- w[is.finite(w)]

  if (length(w) == 0) {
    return(data.frame(
      n_cases = 0,
      effective_sample_size = NA_real_,
      ess_fraction = NA_real_,
      max_weight = NA_real_,
      q95_weight = NA_real_,
      q99_weight = NA_real_,
      gini = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  ess <- effective_sample_size(w)

  data.frame(
    n_cases = length(w),
    effective_sample_size = ess,
    ess_fraction = ess / length(w),
    max_weight = max(w),
    q95_weight = q95_weight(w),
    q99_weight = as.numeric(stats::quantile(w, 0.99, na.rm = TRUE)),
    gini = gini_coefficient(w),
    stringsAsFactors = FALSE
  )
}

log_message <- function(verbose = 0, level = 1, ...) {
  if (verbose >= level) {
    timestamp <- format(Sys.time(), "%H:%M:%S")
    message(sprintf("[%s] %s", timestamp, paste(..., collapse = "")))
  }
}

`%||%` <- function(x, y) if (is.null(x)) y else x
