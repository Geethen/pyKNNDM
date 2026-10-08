
#### compute_NND
function (x, y = NULL, dist_space = c("geographical", "feature", 
    "time"), dist_type_label = "sample-to-sample", dist_fun = c("euclidean", 
    "mahalanobis", "gower", "great_circle", "abs_time"), CVtest = NULL, 
    CVtrain = NULL, time_var = NULL, time_unit = "auto", algorithm = "brute") 
{
    dist_space <- match.arg(dist_space)
    dist_fun <- match.arg(dist_fun)
    if (!is.null(y) && !is.null(CVtest)) 
        stop("Currently, `CVtest` and `y` cannot be specified simultaniously.")
    coords_x <- if (dist_space == "geographical") 
        sf::st_coordinates(x)[, 1:2]
    else NULL
    coords_y <- if (dist_space == "geographical" && !is.null(y)) 
        sf::st_coordinates(y)[, 1:2]
    else NULL
    if (dist_space == "geographical") {
        if (dist_fun == "great_circle") {
            if (is.null(CVtest)) {
                distmat <- if (is.null(y)) 
                  sf::st_distance(x)
                else sf::st_distance(y, x)
                units(distmat) <- NULL
                if (is.null(y)) 
                  diag(distmat) <- NA
                min_d <- apply(distmat, 1, min, na.rm = TRUE)
            }
            else {
                min_d <- cv_distances(x, CVtest = CVtest, CVtrain = CVtrain, 
                  dist_fun = dist_fun)
            }
        }
        else {
            if (is.null(y)) {
                if (is.null(CVtest)) {
                  min_d <- FNN::knn.dist(coords_x, k = 1, algorithm = algorithm)
                }
                else {
                  min_d <- cv_distances(coords_x, CVtest = CVtest, 
                    CVtrain = CVtrain, algorithm = algorithm, 
                    dist_fun = dist_fun)
                }
            }
            else {
                min_d <- FNN::knnx.dist(query = coords_y, data = coords_x, 
                  k = 1, algorithm = algorithm)
            }
        }
    }
    else if (dist_space == "feature") {
        df_x <- sf::st_drop_geometry(x)
        df_y <- if (!is.null(y)) 
            sf::st_drop_geometry(y)
        else df_x
        mat_x <- as.matrix(df_x)
        mat_y <- as.matrix(df_y)
        if (dist_fun == "mahalanobis") {
            if (is.null(CVtest)) {
                S <- if (ncol(mat_x) == 1) 
                  matrix(stats::var(mat_x), 1, 1)
                else stats::cov(mat_x)
                S_inv <- MASS::ginv(S)
                min_d <- vapply(seq_len(nrow(mat_y)), function(i) {
                  idx <- if (is.null(y)) 
                    seq_len(nrow(mat_x))[-i]
                  else seq_len(nrow(mat_x))
                  min(vapply(idx, function(j) {
                    diff <- mat_y[i, ] - mat_x[j, ]
                    as.numeric(sqrt(t(diff) %*% S_inv %*% diff))
                  }, numeric(1)))
                }, numeric(1))
            }
            else {
                min_d <- cv_distances(mat_x, CVtest = CVtest, 
                  CVtrain = CVtrain, dist_fun = dist_fun)
            }
        }
        else if (dist_fun == "euclidean") {
            if (is.null(CVtest)) {
                min_d <- if (is.null(y)) 
                  FNN::knn.dist(mat_x, k = 1, algorithm = algorithm)
                else FNN::knnx.dist(query = mat_y, data = mat_x, 
                  k = 1, algorithm = algorithm)
            }
            else {
                min_d <- cv_distances(mat_x, CVtest = CVtest, 
                  CVtrain = CVtrain, algorithm = algorithm, dist_fun = dist_fun)
            }
        }
        else if (dist_fun == "gower") {
            if (is.null(CVtest)) {
                min_d <- if (is.null(y)) {
                  vapply(1:nrow(df_x), function(i) gower::gower_topn(df_x[i, 
                    , drop = FALSE], df_x[-i, , drop = FALSE], 
                    n = 1)$distance[[1]], numeric(1))
                }
                else {
                  gower_res <- gower::gower_topn(df_y, df_x, 
                    n = 1)
                  vapply(gower_res$distance, function(x) x[1], 
                    numeric(1))
                }
            }
            else {
                min_d <- cv_distances(df_x, CVtest = CVtest, 
                  CVtrain = CVtrain, dist_fun = dist_fun)
            }
        }
    }
    else if (dist_space == "time") {
        if (dist_fun == "abs_time") {
            time_x <- sf::st_drop_geometry(x)[, time_var]
            if (is.null(CVtest)) {
                time_y <- if (!is.null(y)) 
                  sf::st_drop_geometry(y)[, time_var]
                else time_x
                dmat <- abs(outer(time_y, time_x, FUN = function(a, 
                  b) as.numeric(difftime(a, b, units = time_unit))))
                if (is.null(y)) 
                  diag(dmat) <- Inf
                min_d <- apply(dmat, 1, min)
            }
            else {
                min_d <- cv_distances(time_x, CVtest = CVtest, 
                  time_unit = time_unit, CVtrain = CVtrain, dist_fun = dist_fun)
            }
        }
    }
    data.frame(dist = min_d, what = factor(dist_type_label), 
        dist_type = dist_space)
}
<bytecode: 0x00000296bfa33738>
<environment: namespace:CAST>

#### cv_distances
function (x, CVtest, CVtrain = NULL, dist_fun = "euclidean", 
    algorithm = "brute", time_unit = "auto") 
{
    n <- if (inherits(x, c("Date", "POSIXct", "POSIXt"))) 
        length(x)
    else nrow(x)
    alldist <- rep(NA, n)
    if (inherits(CVtest, "list")) {
        CVtest_v <- rep(NA_integer_, n)
        for (k in seq_along(CVtest)) CVtest_v[CVtest[[k]]] <- k
    }
    else CVtest_v <- CVtest
    if (dist_fun == "great_circle") {
        distmat <- sf::st_distance(x)
        units(distmat) <- NULL
        diag(distmat) <- NA
    }
    if (dist_fun == "mahalanobis") {
        tr_mat <- as.matrix(x)
        S_inv <- MASS::ginv(stats::cov(tr_mat))
    }
    for (f in unique(CVtest_v)) {
        test_idx <- which(CVtest_v == f)
        if (!is.null(CVtrain)) {
            train_idx <- CVtrain[[f]]
        }
        else {
            train_idx <- which(CVtest_v != f)
        }
        if (inherits(x, c("Date", "POSIXct", "POSIXt"))) {
            tr_train <- x[train_idx]
            tr_test <- x[test_idx]
        }
        else {
            tr_train <- x[train_idx, , drop = FALSE]
            tr_test <- x[test_idx, , drop = FALSE]
        }
        if (dist_fun == "euclidean") {
            alldist[test_idx] <- FNN::knnx.dist(query = tr_test, 
                data = tr_train, k = 1, algorithm = algorithm)
        }
        else if (dist_fun == "great_circle") {
            alldist[test_idx] <- apply(distmat[test_idx, train_idx, 
                drop = FALSE], 1, min)
        }
        else if (dist_fun == "gower") {
            alldist[test_idx] <- c(gower::gower_topn(x[test_idx, 
                , drop = FALSE], x[train_idx, , drop = FALSE], 
                n = 1))$distance[[1]]
        }
        else if (dist_fun == "mahalanobis") {
            alldist[test_idx] <- apply(tr_mat[test_idx, , drop = FALSE], 
                1, function(y) {
                  min(apply(tr_mat[train_idx, , drop = FALSE], 
                    1, function(x) {
                      sqrt(t(y - x) %*% S_inv %*% (y - x))
                    }))
                })
        }
        else if (dist_fun == "abs_time") {
            diffs <- outer(x[test_idx], x[train_idx], function(x, 
                y) abs(as.numeric(difftime(x, y, units = time_unit))))
            alldist[test_idx] <- apply(diffs, 1, min)
        }
    }
    alldist
}
<bytecode: 0x00000296bfb93230>
<environment: namespace:CAST>

#### CreateSpacetimeFolds
function (x, spacevar = NA, timevar = NA, k = 10, class = NA, 
    seed = sample(1:1000, 1)) 
{
    x <- data.frame(x)
    if (!is.na(class)) {
        if (is.numeric(x[, class])) {
            stop("argument class only works for categorical data")
        }
        unit <- unique(x[, c(spacevar, class)])
        unit$CAST_fold <- createFolds(unit[[class]], k = k, list = FALSE)
        key <- interaction(x[[spacevar]], x[[class]], drop = TRUE)
        ukey <- interaction(unit[[spacevar]], unit[[class]], 
            drop = TRUE)
        x$CAST_fold <- unit$CAST_fold[match(key, ukey)]
        spacevar <- "CAST_fold"
    }
    if (!is.na(spacevar)) {
        if (k > length(unique(x[, spacevar]))) {
            k <- length(unique(x[, spacevar]))
            print(paste0("warning: k is higher than number of unique locations. k is set to ", 
                k))
        }
    }
    if (!is.na(timevar)) {
        if (k > length(unique(x[, timevar]))) {
            k <- length(unique(x[, timevar]))
            print(paste0("warning: k is higher than number of unique points in time. k is set to ", 
                k))
        }
    }
    if (!is.na(spacevar)) {
        set.seed(seed)
        spacefolds <- lapply(caret::createFolds(1:length(unique(x[, 
            spacevar])), k), function(y) {
            unique(x[, spacevar])[y]
        })
    }
    if (!is.na(timevar)) {
        set.seed(seed)
        timefolds <- lapply(caret::createFolds(1:length(unique(x[, 
            timevar])), k), function(y) {
            unique(x[, timevar])[y]
        })
    }
    cvindices_train <- list()
    cvindices_test <- list()
    for (i in 1:k) {
        if (!is.na(timevar) & !is.na(spacevar)) {
            cvindices_test[[i]] <- which(x[, spacevar] %in% spacefolds[[i]] & 
                x[, timevar] %in% timefolds[[i]])
            cvindices_train[[i]] <- which(!x[, spacevar] %in% 
                spacefolds[[i]] & !x[, timevar] %in% timefolds[[i]])
        }
        if (is.na(timevar) & !is.na(spacevar)) {
            cvindices_test[[i]] <- which(x[, spacevar] %in% spacefolds[[i]])
            cvindices_train[[i]] <- which(!x[, spacevar] %in% 
                spacefolds[[i]])
        }
        if (!is.na(timevar) & is.na(spacevar)) {
            cvindices_test[[i]] <- which(x[, timevar] %in% timefolds[[i]])
            cvindices_train[[i]] <- which(!x[, timevar] %in% 
                timefolds[[i]])
        }
    }
    result <- list(index = cvindices_train, indexOut = cvindices_test)
    cluster <- do.call(rbind, lapply(seq_along(result$indexOut), 
        function(i) {
            data.frame(Number = result$indexOut[[i]], List = i)
        }))
    x$Number <- seq_len(nrow(x))
    df <- merge(x, cluster, by = "Number", all.x = TRUE)
    result$cluster <- df$List
    return(result)
}
<bytecode: 0x00000296bfbb6708>
<environment: namespace:CAST>

#### ks.test.default
function (x, y, ..., alternative = c("two.sided", "less", "greater"), 
    exact = NULL, simulate.p.value = FALSE, B = 2000) 
{
    alternative <- match.arg(alternative)
    DNAME <- deparse1(substitute(x))
    x <- x[!is.na(x)]
    n <- length(x)
    if (n < 1L) 
        stop("not enough 'x' data")
    if (is.ordered(y)) 
        y <- unclass(y)
    if (is.numeric(y)) {
        args <- list(...)
        if (length(args) > 0L) 
            warning("Parameter(s) ", paste(names(args), collapse = ", "), 
                " ignored")
        DNAME <- paste(DNAME, "and", deparse1(substitute(y)))
        y <- y[!is.na(y)]
        n.x <- as.double(n)
        n.y <- length(y)
        if (n.y < 1L) 
            stop("not enough 'y' data")
        if (is.null(exact)) 
            exact <- (n.x * n.y < 10000)
        if (!simulate.p.value) {
            METHOD <- paste(c("Asymptotic", "Exact")[exact + 
                1L], "two-sample Kolmogorov-Smirnov test")
        }
        else {
            METHOD <- "Monte-Carlo two-sample Kolmogorov-Smirnov test"
        }
        TIES <- FALSE
        w <- c(x, y)
        z <- cumsum(ifelse(order(w) <= n.x, 1/n.x, -1/n.y))
        if (length(unique(w)) < (n.x + n.y)) {
            z <- z[c(which(diff(sort(w)) != 0), n.x + n.y)]
            TIES <- TRUE
            if (!exact && !simulate.p.value) 
                warning("p-value will be approximate in the presence of ties")
        }
        STATISTIC <- switch(alternative, two.sided = max(abs(z)), 
            greater = max(z), less = -min(z))
        nm_alternative <- switch(alternative, two.sided = "two-sided", 
            less = "the CDF of x lies below that of y", greater = "the CDF of x lies above that of y")
        PVAL <- psmirnov(STATISTIC, sizes = c(n.x, n.y), z = if (TIES) 
            w, alternative = alternative, exact = exact, simulate = simulate.p.value, 
            B = B, lower.tail = FALSE)
        if (simulate.p.value) 
            PVAL <- (1 + (PVAL * B))/(B + 1)
    }
    else {
        if (is.character(y)) 
            y <- get(y, mode = "function", envir = parent.frame())
        if (!is.function(y)) 
            stop("'y' must be numeric or a function or a string naming a valid function")
        TIES <- FALSE
        if (length(unique(x)) < n) {
            warning("ties should not be present for the one-sample Kolmogorov-Smirnov test")
            TIES <- TRUE
        }
        if (is.null(exact)) 
            exact <- (n < 100) && !TIES
        METHOD <- paste(c("Asymptotic", "Exact")[exact + 1L], 
            "one-sample Kolmogorov-Smirnov test")
        x <- y(sort(x), ...) - (0:(n - 1))/n
        STATISTIC <- switch(alternative, two.sided = max(c(x, 
            1/n - x)), greater = max(1/n - x), less = max(x))
        PVAL <- pkolmogorov(STATISTIC, n, two.sided = (alternative == 
            "two.sided"), exact = exact, lower.tail = FALSE)
        nm_alternative <- switch(alternative, two.sided = "two-sided", 
            less = "the CDF of x lies below the null hypothesis", 
            greater = "the CDF of x lies above the null hypothesis")
    }
    names(STATISTIC) <- switch(alternative, two.sided = "D", 
        greater = "D^+", less = "D^-")
    PVAL <- min(1, max(0, PVAL))
    RVAL <- list(statistic = STATISTIC, p.value = PVAL, alternative = nm_alternative, 
        method = METHOD, data.name = DNAME, exact = exact)
    class(RVAL) <- c("ks.test", "htest")
    RVAL
}
<bytecode: 0x00000296bfc31ee0>
<environment: namespace:stats>

#### psmirnov
function (q, sizes, z = NULL, alternative = c("two.sided", "less", 
    "greater"), exact = TRUE, simulate = FALSE, B = 2000, lower.tail = TRUE, 
    log.p = FALSE) 
{
    alternative <- match.arg(alternative)
    if (is.numeric(q)) {
        if (!is.double(q)) 
            storage.mode(q) <- "double"
    }
    else stop("argument 'q' must be numeric")
    ret <- q
    i1 <- is.na(q)
    ret[i1] <- NA_real_
    if (any(i2 <- (q <= 0))) {
        p <- 1 - lower.tail
        if (log.p) 
            p <- log(p)
        ret[i2] <- p
    }
    if (any(i3 <- (q > 1))) {
        p <- as.numeric(lower.tail)
        if (log.p) 
            p <- log(p)
        ret[i3] <- p
    }
    IND <- which(!(i1 | i2 | i3))
    if (!length(IND)) 
        return(ret)
    if (length(sizes) != 2L) 
        stop("argument 'sizes' must be a vector of length 2")
    n.x <- sizes[1L]
    n.y <- sizes[2L]
    if (n.x < 1) 
        stop("not enough 'x' data")
    if (n.y < 1) 
        stop("not enough 'y' data")
    n.x <- floor(n.x)
    n.y <- floor(n.y)
    N <- n.x + n.y
    n <- n.x * n.y/(n.x + n.y)
    exact <- exact && !simulate
    if (!exact) {
        ret[IND] <- if (simulate) 
            psmirnov_simul(q[IND], sizes, z, alternative, lower.tail, 
                log.p, B)
        else psmirnov_asymp(q[IND], sizes, alternative, lower.tail, 
            log.p)
        return(ret)
    }
    r <- psmirnov_exact(q[IND], sizes, z, alternative, lower.tail, 
        log.p)
    ret[IND] <- if (all(is.finite(r))) 
        r
    else {
        warning("computation of exact probability failed, returning Monte Carlo approximation")
        psmirnov_simul(q[IND], sizes, z, alternative, lower.tail, 
            log.p, B)
    }
    ret
}
<bytecode: 0x00000296bfc17968>
<environment: namespace:stats>

#### kmeans
function (x, centers, iter.max = 10L, nstart = 1L, algorithm = c("Hartigan-Wong", 
    "Lloyd", "Forgy", "MacQueen"), trace = FALSE) 
{
    .Mimax <- .Machine$integer.max
    do_one <- function(nmeth) {
        switch(nmeth, {
            isteps.Qtran <- as.integer(min(.Mimax, 50 * m))
            iTran <- c(isteps.Qtran, integer(k))
            Z <- .Fortran(C_kmns, x, m, p, centers = centers, 
                as.integer(k), c1 = integer(m), c2 = integer(m), 
                nc = integer(k), double(k), double(k), ncp = integer(k), 
                D = double(m), iTran = iTran, live = integer(k), 
                iter = iter.max, wss = double(k), ifault = as.integer(trace))
            switch(Z$ifault, stop("empty cluster: try a better set of initial centers", 
                call. = FALSE), Z$iter <- max(Z$iter, iter.max + 
                1L), stop("number of cluster centres must lie between 1 and nrow(x)", 
                call. = FALSE), warning(gettextf("Quick-TRANSfer stage steps exceeded maximum (= %d)", 
                isteps.Qtran), call. = FALSE))
        }, {
            Z <- .C(C_kmeans_Lloyd, x, m, p, centers = centers, 
                k, c1 = integer(m), iter = iter.max, nc = integer(k), 
                wss = double(k))
        }, {
            Z <- .C(C_kmeans_MacQueen, x, m, p, centers = as.double(centers), 
                k, c1 = integer(m), iter = iter.max, nc = integer(k), 
                wss = double(k))
        })
        if (m23 <- any(nmeth == c(2L, 3L))) {
            if (any(Z$nc == 0)) 
                warning("empty cluster: try a better set of initial centers", 
                  call. = FALSE)
        }
        if (Z$iter > iter.max) {
            warning(sprintf(ngettext(iter.max, "did not converge in %d iteration", 
                "did not converge in %d iterations"), iter.max), 
                call. = FALSE, domain = NA)
            if (m23) 
                Z$ifault <- 2L
        }
        if (nmeth %in% c(2L, 3L)) {
            if (any(Z$nc == 0)) 
                warning("empty cluster: try a better set of initial centers", 
                  call. = FALSE)
        }
        Z
    }
    x <- as.matrix(x)
    m <- as.integer(nrow(x))
    if (is.na(m)) 
        stop("invalid nrow(x)")
    p <- as.integer(ncol(x))
    if (is.na(p)) 
        stop("invalid ncol(x)")
    if (missing(centers)) 
        stop("'centers' must be a number or a matrix")
    nmeth <- switch(match.arg(algorithm), `Hartigan-Wong` = 1L, 
        Lloyd = 2L, Forgy = 2L, MacQueen = 3L)
    storage.mode(x) <- "double"
    if (length(centers) == 1L) {
        k <- centers
        if (nstart == 1L) 
            centers <- x[sample.int(m, k), , drop = FALSE]
        if (nstart >= 2L || any(duplicated(centers))) {
            cn <- unique(x)
            mm <- nrow(cn)
            if (mm < k) 
                stop("more cluster centers than distinct data points.")
            centers <- cn[sample.int(mm, k), , drop = FALSE]
        }
    }
    else {
        centers <- as.matrix(centers)
        if (any(duplicated(centers))) 
            stop("initial centers are not distinct")
        cn <- NULL
        k <- nrow(centers)
        if (m < k) 
            stop("more cluster centers than data points")
    }
    k <- as.integer(k)
    if (is.na(k)) 
        stop(gettextf("invalid value of %s", "'k'"), domain = NA)
    if (k == 1L) 
        nmeth <- 3L
    iter.max <- as.integer(iter.max)
    if (is.na(iter.max) || iter.max < 1L) 
        stop("'iter.max' must be positive")
    if (ncol(x) != ncol(centers)) 
        stop("must have same number of columns in 'x' and 'centers'")
    storage.mode(centers) <- "double"
    Z <- do_one(nmeth)
    best <- sum(Z$wss)
    if (nstart >= 2L && !is.null(cn)) 
        for (i in 2:nstart) {
            centers <- cn[sample.int(mm, k), , drop = FALSE]
            ZZ <- do_one(nmeth)
            if ((z <- sum(ZZ$wss)) < best) {
                Z <- ZZ
                best <- z
            }
        }
    centers <- matrix(Z$centers, k)
    dimnames(centers) <- list(1L:k, dimnames(x)[[2L]])
    cluster <- Z$c1
    if (!is.null(rn <- rownames(x))) 
        names(cluster) <- rn
    totss <- sum(scale(x, scale = FALSE)^2)
    structure(list(cluster = cluster, centers = centers, totss = totss, 
        withinss = Z$wss, tot.withinss = best, betweenss = totss - 
            best, size = Z$nc, iter = Z$iter, ifault = Z$ifault), 
        class = "kmeans")
}
<bytecode: 0x00000296bfc0aa88>
<environment: namespace:stats>

#### hclust
function (d, method = "complete", members = NULL) 
{
    METHODS <- c("ward.D", "single", "complete", "average", "mcquitty", 
        "median", "centroid", "ward.D2")
    if (method == "ward") {
        message("The \"ward\" method has been renamed to \"ward.D\"; note new \"ward.D2\"")
        method <- "ward.D"
    }
    i.meth <- pmatch(method, METHODS)
    if (is.na(i.meth)) 
        stop("invalid clustering method", paste("", method))
    if (i.meth == -1) 
        stop("ambiguous clustering method", paste("", method))
    n <- as.integer(attr(d, "Size"))
    if (is.null(n)) 
        stop("invalid dissimilarities")
    if (is.na(n) || n > 65536L) 
        stop("size cannot be NA nor exceed 65536")
    if (n < 2) 
        stop("must have n >= 2 objects to cluster")
    len <- as.integer(n * (n - 1)/2)
    if (length(d) != len) 
        (if (length(d) < len) 
            stop
        else warning)("dissimilarities of improper length")
    if (is.null(members)) 
        members <- rep(1, n)
    else if (length(members) != n) 
        stop("invalid length of members")
    storage.mode(d) <- "double"
    hcl <- .Fortran(C_hclust, n = n, len = len, method = as.integer(i.meth), 
        ia = integer(n), ib = integer(n), crit = double(n), members = as.double(members), 
        nn = integer(n), disnn = double(n), diss = d)
    hcass <- .Fortran(C_hcass2, n = n, ia = hcl$ia, ib = hcl$ib, 
        order = integer(n), iia = integer(n), iib = integer(n))
    structure(list(merge = cbind(hcass$iia[1L:(n - 1)], hcass$iib[1L:(n - 
        1)]), height = hcl$crit[1L:(n - 1)], order = hcass$order, 
        labels = attr(d, "Labels"), method = METHODS[i.meth], 
        call = match.call(), dist.method = attr(d, "method")), 
        class = "hclust")
}
<bytecode: 0x00000296bfc56a10>
<environment: namespace:stats>

#### cutree
function (tree, k = NULL, h = NULL) 
{
    if (is.null(n1 <- nrow(tree$merge)) || n1 < 1) 
        stop("invalid 'tree' ('merge' component)")
    n <- n1 + 1
    if (is.null(k) && is.null(h)) 
        stop("either 'k' or 'h' must be specified")
    if (is.null(k)) {
        if (is.unsorted(tree$height)) 
            stop("the 'height' component of 'tree' is not sorted (increasingly)")
        k <- n + 1L - apply(outer(c(tree$height, Inf), h, `>`), 
            2, which.max)
        if (getOption("verbose")) 
            message("cutree(): k(h) = ", k, domain = NA)
    }
    else {
        k <- as.integer(k)
        if (min(k) < 1 || max(k) > n) 
            stop(gettextf("elements of 'k' must be between 1 and %d", 
                n), domain = NA)
    }
    ans <- .Call(C_cutree, tree$merge, k)
    if (length(k) == 1L) {
        ans <- setNames(as.vector(ans), tree$labels)
    }
    else {
        colnames(ans) <- if (!is.null(h)) 
            h
        else k
        rownames(ans) <- tree$labels
    }
    return(ans)
}
<bytecode: 0x00000296bfc5e578>
<environment: namespace:stats>

#### prcomp.default
function (x, retx = TRUE, center = TRUE, scale. = FALSE, tol = NULL, 
    rank. = NULL, ...) 
{
    chkDots(...)
    x <- as.matrix(x)
    x <- scale(x, center = center, scale = scale.)
    cen <- attr(x, "scaled:center")
    sc <- attr(x, "scaled:scale")
    if (any(sc == 0)) 
        stop("cannot rescale a constant/zero column to unit variance")
    n <- nrow(x)
    p <- ncol(x)
    k <- if (!is.null(rank.)) {
        stopifnot(length(rank.) == 1, is.finite(rank.), as.integer(rank.) > 
            0)
        min(as.integer(rank.), n, p)
    }
    else min(n, p)
    s <- svd(x, nu = 0, nv = k)
    j <- seq_len(k)
    s$d <- s$d/sqrt(max(1, n - 1))
    if (!is.null(tol)) {
        rank <- sum(s$d > (s$d[1L] * tol))
        if (rank < k) {
            j <- seq_len(k <- rank)
            s$v <- s$v[, j, drop = FALSE]
        }
    }
    dimnames(s$v) <- list(colnames(x), paste0("PC", j))
    r <- list(sdev = s$d, rotation = s$v, center = cen %||% FALSE, 
        scale = sc %||% FALSE)
    if (retx) 
        r$x <- x %*% s$v
    class(r) <- "prcomp"
    r
}
<bytecode: 0x00000296bfc662d8>
<environment: namespace:stats>

#### predict.prcomp
function (object, newdata, ...) 
{
    chkDots(...)
    if (missing(newdata)) {
        if (!is.null(object$x)) 
            return(object$x)
        else stop("no scores are available: refit with 'retx=TRUE'")
    }
    if (length(dim(newdata)) != 2L) 
        stop("'newdata' must be a matrix or data frame")
    nm <- rownames(object$rotation)
    if (!is.null(nm)) {
        if (!all(nm %in% colnames(newdata))) 
            stop("'newdata' does not have named columns matching one or more of the original columns")
        newdata <- newdata[, nm, drop = FALSE]
    }
    else {
        if (NCOL(newdata) != NROW(object$rotation)) 
            stop("'newdata' does not have the correct number of columns")
    }
    scale(newdata, object$center, object$scale) %*% object$rotation
}
<bytecode: 0x00000296bfc69a50>
<environment: namespace:stats>

#### scale.default
function (x, center = TRUE, scale = TRUE) 
{
    x <- as.matrix(x)
    nc <- ncol(x)
    if (is.logical(center)) {
        if (center) {
            center <- colMeans(x, na.rm = TRUE)
            x <- sweep(x, 2L, center, check.margin = FALSE)
        }
    }
    else {
        if (!is.numeric(center)) 
            center <- as.numeric(center)
        if (length(center) == nc) 
            x <- sweep(x, 2L, center, check.margin = FALSE)
        else stop("length of 'center' must equal the number of columns of 'x'")
    }
    if (is.logical(scale)) {
        if (scale) {
            f <- function(v) {
                v <- v[!is.na(v)]
                sqrt(sum(v^2)/max(1, length(v) - 1L))
            }
            scale <- apply(x, 2L, f)
            x <- sweep(x, 2L, scale, `/`, check.margin = FALSE)
        }
    }
    else {
        if (!is.numeric(scale)) 
            scale <- as.numeric(scale)
        if (length(scale) == nc) 
            x <- sweep(x, 2L, scale, `/`, check.margin = FALSE)
        else stop("length of 'scale' must equal the number of columns of 'x'")
    }
    if (is.numeric(center)) 
        attr(x, "scaled:center") <- center
    if (is.numeric(scale)) 
        attr(x, "scaled:scale") <- scale
    x
}
<bytecode: 0x00000296bfc7f660>
<environment: namespace:base>

#### seq.default
function (from = 1, to = 1, by = ((to - from)/(length.out - 1)), 
    length.out = NULL, along.with = NULL, ...) 
{
    is.logint <- function(.) (is.integer(.) || is.logical(.)) && 
        !is.object(.)
    if ((One <- nargs() == 1L) && !missing(from)) {
        lf <- length(from)
        return(if (mode(from) == "numeric" && lf == 1L) {
            if (!is.finite(from)) stop("'from' must be a finite number")
            1L:from
        } else if (lf) 1L:lf else integer())
    }
    if (!missing(along.with)) {
        length.out <- length(along.with)
        if (One) 
            return(if (length.out) seq_len(length.out) else integer())
        intn1 <- is.integer(length.out)
    }
    else if (!missing(length.out)) {
        len <- length(length.out)
        if (!len) 
            stop(gettextf("'%s' must be of length 1", "length.out"), 
                domain = NA)
        if (len > 1L) {
            warning("first element used of 'length.out' argument")
            length.out <- length.out[1L]
        }
        if (!(intn1 <- is.logint(length.out))) 
            length.out <- as.numeric(ceiling(length.out))
    }
    chkDots(...)
    if (!missing(from) && length(from) != 1L) 
        stop(gettextf("'%s' must be of length 1", "from"), domain = NA)
    if (!missing(to) && length(to) != 1L) 
        stop(gettextf("'%s' must be of length 1", "to"), domain = NA)
    if (!missing(from) && !is.finite(if (is.character(from)) from <- as.numeric(from) else from)) 
        stop("'from' must be a finite number")
    if (!missing(to) && !is.finite(if (is.character(to)) to <- as.numeric(to) else to)) 
        stop("'to' must be a finite number")
    if (is.null(length.out)) 
        if (missing(by)) 
            from:to
        else {
            int <- is.logint(from) && is.logint(to)
            del <- to - if (int) 
                as.double(from)
            else from
            if (del == 0 && to == 0) 
                return(to)
            if (length(by) != 1L) 
                stop(gettextf("'%s' must be of length 1", "by"), 
                  domain = NA)
            if (!is.logint(by)) 
                int <- FALSE
            else if (!int) 
                storage.mode(by) <- "double"
            n <- if (finite.del <- is.finite(del)) 
                del/by
            else to/by - from/by
            if (!is.finite(n)) {
                if (!is.na(by) && by == 0 && del == 0) 
                  return(from)
                stop("invalid '(to - from)/by'")
            }
            if (n < 0L) 
                stop("wrong sign in 'by' argument")
            if (n > .Machine$integer.max) 
                stop("'by' argument is much too small")
            if (finite.del && abs(del)/max(abs(to), abs(from)) < 
                100 * .Machine$double.eps) 
                return(from)
            if (int) {
                n <- as.integer(n)
                if (n >= 2L) 
                  cumsum(rep.int(c(from, by), c(1L, n)))
                else from + (0L:n) * by
            }
            else {
                n <- as.integer(n + 1e-10)
                x <- if (finite.del) 
                  from + (0L:n) * by
                else (from/4 + (0L:n) * (by/4)) * 4
                if (by > 0) 
                  pmin(x, to)
                else pmax(x, to)
            }
        }
    else if (!is.finite(length.out) || length.out < 0L) 
        stop("'length.out' must be a non-negative number")
    else if (length.out == 0L) 
        integer()
    else if (One) 
        seq_len(length.out)
    else if (missing(by)) {
        if (missing(to)) {
            to <- from + (length.out - 1)
            intdel <- intn1 && is.logint(from) && to <= .Machine$integer.max
            if (intdel) 
                storage.mode(to) <- "integer"
        }
        else intdel <- is.logint(to)
        if (missing(from)) {
            from <- to - (length.out - 1)
            if (intdel) {
                intdel <- intn1 && from >= -.Machine$integer.max
                if (intdel) 
                  storage.mode(from) <- "integer"
            }
        }
        else if (intdel) 
            intdel <- is.logint(from)
        if (length.out > 2L) 
            if (from == to) 
                rep.int(from, length.out)
            else {
                n1 <- length.out - 1L
                if (intdel && intn1 && from%%n1 == to%%n1) {
                  by <- to%/%n1 - from%/%n1
                  cumsum(rep.int(c(from, by), c(1L, n1)))
                }
                else {
                  if (intdel) 
                    storage.mode(from) <- "double"
                  del <- to - from
                  if (is.finite(del)) {
                    as.vector(c(from, from + seq_len(length.out - 
                      2L) * (del/n1), to))
                  }
                  else {
                    from <- from/4
                    to <- to/4
                    as.vector(c(from, from + seq_len(length.out - 
                      2L) * ((to - from)/n1), to)) * 4
                  }
                }
            }
        else as.vector(c(from, to))[seq_len(length.out)]
    }
    else if (missing(to)) {
        int <- (intby <- is.logint(by)) && is.logint(from) && 
            (!(nby <- length(by)) || (naby <- is.na(by)) || ((to <- from + 
                (length.out - 1) * by) <= .Machine$integer.max && 
                to >= -.Machine$integer.max))
        if (int && length.out > 2L && nby == 1L && !naby) 
            cumsum(rep.int(c(from, by), c(1L, length.out - 1L)))
        else {
            if (intby && !(int || is.object(from))) 
                storage.mode(by) <- "double"
            from + (0L:(length.out - 1L)) * by
        }
    }
    else if (missing(from)) {
        int <- (intby <- is.logint(by)) && is.logint(to) && (!(nby <- length(by)) || 
            (naby <- is.na(by)) || ((from <- to - (length.out - 
            1) * by) >= -.Machine$integer.max && from <= .Machine$integer.max))
        if (int && length.out > 2L && nby == 1L && !naby) 
            cumsum(rep.int(c(as.integer(from), by), c(1L, length.out - 
                1L)))
        else {
            if (intby && !(int || is.object(to))) 
                storage.mode(by) <- "double"
            to - ((length.out - 1L):0L) * by
        }
    }
    else stop("too many arguments")
}
<bytecode: 0x00000296b046a930>
<environment: namespace:base>

#### wass_stat
function (a, b, power = def_power) 
{
    joint = as.numeric(c(a, b))
    na = length(a)
    nb = length(b)
    labs = c(rep(T, na), rep(F, nb))
    inds = order(joint)
    joint = joint[inds]
    labs = labs[inds]
    presort_stat_fn(joint, labs, power, na)
}
<bytecode: 0x00000296bfd78460>
<environment: 0x00000296bfd58ab8>

#### sample
function (x, size, replace = FALSE, prob = NULL) 
{
    if (length(x) == 1L && is.numeric(x) && is.finite(x) && x >= 
        1) {
        if (missing(size)) 
            size <- x
        sample.int(x, size, replace, prob)
    }
    else {
        if (missing(size)) 
            size <- length(x)
        x[sample.int(length(x), size, replace, prob)]
    }
}
<bytecode: 0x00000296b664d4d8>
<environment: namespace:base>

versions: twosamples 2.0.1  FNN 1.1.4.1  sf 1.1.3 
          GEOS           GDAL         proj.4 GDAL_with_GEOS     USE_PROJ_H 
      "3.14.1"       "3.12.1"        "9.7.1"         "true"         "true" 
          PROJ 
       "9.7.1" 
longdouble: TRUE 16 
[1] "Mersenne-Twister" "Inversion"        "Rejection"       
[1] ""
BLAS 
  "" 
