 [1] "check_knndm_feature" "check_knndm_geo"     "distclust_distmat"  
 [4] "knndm"               "knndm_feature"       "knndm_geo"          
 [7] "nndm"                "nndm_checks_feature" "nndm_checks_geo"    
[10] "plot.knndm"          "plot.nndm"           "print.knndm"        
[13] "print.nndm"          "show.knndm"          "show.nndm"          

#### check_knndm_feature
function (tpoints, predpoints, dist_space, k, maxp, clustering, 
    catVars, dist_fun, test_prop) 
{
    if (!is.null(catVars) && dist_fun != "gower") {
        stop("Only gower distances work with categorical features. Please use dist_fun = 'gower'")
    }
    if (is.null(test_prop)) {
        if (!(maxp < 1 & maxp > 1/k)) {
            stop("maxp must be strictly between 1/k and 1")
        }
    }
    if (is.null(predpoints)) {
        stop("predpoints with predictor data missing")
    }
    if (length(setdiff(names(tpoints), names(predpoints))) > 
        0) {
        stop("tpoints and predpoints need to contain the predictor data and have the same colnames.")
    }
    for (catvar in catVars) {
        if (any(!unique(tpoints[, catvar]) %in% unique(predpoints[, 
            catvar]))) {
            stop(paste0("Some values of factor", catvar, "are only present in training / prediction points.\n                  All factor values in the prediction points must be present in the training points."))
        }
    }
}
<bytecode: 0x000001c5ca9a9dc8>
<environment: namespace:CAST>

#### check_knndm_geo
function (tpoints, predpoints, dist_space, k, maxp, clustering, 
    islonglat, dist_fun, test_prop) 
{
    if (!identical(sf::st_crs(tpoints), sf::st_crs(predpoints))) {
        stop("tpoints and predpoints must have the same CRS")
    }
    if (!(clustering %in% c("kmeans", "hierarchical"))) {
        stop("clustering must be one of `kmeans` or `hierarchical`")
    }
    if (is.null(test_prop)) {
        if (!(maxp < 1 & maxp > 1/k)) {
            stop("maxp must be strictly between 1/k and 1")
        }
    }
    if (isTRUE(islonglat) & clustering == "kmeans") {
        stop("kmeans works in the Euclidean space and therefore can only handle\n         projected coordinates. Please use hierarchical clustering or project your data.")
    }
    if (isTRUE(islonglat) && dist_fun != "great_circle") {
        stop("Only great-circle distances are allowed for lon/lat coordinates. Please use 'great_circle' as 'dist_fun'.")
    }
}
<bytecode: 0x000001c5ca992bd8>
<environment: namespace:CAST>

#### distclust_distmat
function (distm, folds) 
{
    alldist <- rep(NA, length(folds))
    for (f in unique(folds)) {
        alldist[f == folds] <- apply(distm[f == folds, f != folds, 
            drop = FALSE], 1, min)
    }
    alldist
}
<bytecode: 0x000001c5ca9965f0>
<environment: namespace:CAST>

#### knndm
function (tpoints, modeldomain = NULL, predpoints = NULL, dist_space = "geographical", 
    k = 10, maxp = 0.5, clustering = "hierarchical", linkf = "ward.D2", 
    samplesize = 1000, sampling = "regular", dist_fun = "euclidean", 
    algorithm = "brute", scale_vars = TRUE, space = NULL, useMD = NULL, 
    test_prop = NULL, test_tolerance = NULL, nk_len = 100) 
{
    if (!is.null(space)) {
        warning("Argument 'space' is deprecated. Please use 'dist_space' instead.", 
            call. = FALSE)
        dist_space <- space
    }
    if (!is.null(useMD)) {
        warning("Argument 'useMD' is deprecated. Please use 'dist_fun' instead.", 
            call. = FALSE)
    }
    if (dist_space == "geo") 
        dist_space <- "geographical"
    if (!dist_space %in% c("geographical", "feature")) {
        stop("dist_space must be one of 'geographical' or 'feature'")
    }
    if (!(dist_fun %in% c("euclidean", "mahalanobis", "gower", 
        "great_circle"))) {
        stop("dist_fun must be one of 'euclidean', 'mahalanobis', 'gower' or 'great_circle'")
    }
    if (dist_space == "time" && dist_fun != "euclidean") 
        stop("Temporal space only supports euclidean distances.")
    if (dist_space == "feature" && dist_fun == "great_circle") 
        stop("Great-circle distances only work with in geographical space.")
    if (dist_space == "geographical" && dist_fun %in% c("mahalanobis", 
        "gower")) 
        stop("Mahalanobis and Gower distances only work in feature space.")
    if (!is.null(test_prop)) {
        warning("A train/test split will be returned, which is currently experimental.")
    }
    minp <- NULL
    if (!is.null(test_prop)) {
        if (test_prop >= 1 | test_prop <= 0) {
            stop("test_prop must be greater than 0 and smaller than 1")
        }
        if (is.null(test_tolerance)) 
            test_tolerance <- 0.1
        k <- 2
        maxp <- test_prop + test_tolerance
        minp <- test_prop - test_tolerance
        if (maxp <= 0 || minp >= 1 || minp > maxp) {
            stop("Misspecified tolerance. Resulted in infeasible minp/maxp values")
        }
        if (minp <= 0) {
            warning("minp was set to 0.1")
            minp <- 0.1
        }
        else if (maxp >= 1) {
            warning("maxp was set to 0.9")
            maxp <- 0.9
        }
        if (maxp == 1/k) {
            eps <- .Machine$double.eps^0.5
            maxp <- maxp + eps
        }
    }
    else {
        minp <- NULL
    }
    if (!nk_len%%1 == 0) {
        stop("nk_len must be an integer")
    }
    if (is.null(predpoints) & !is.null(modeldomain)) {
        if (!any(c("sfc", "sf", "SpatRaster") %in% class(modeldomain))) {
            stop("modeldomain must be a sf/sfc object or a 'SpatRaster' object.")
        }
        if (any(class(modeldomain) == "SpatRaster")) {
            if (dist_space == "feature") {
                predictor_stack <- modeldomain
            }
            modeldomain[!is.na(modeldomain)] <- 1
            modeldomain <- sf::st_union(sf::st_as_sf(terra::as.polygons(modeldomain, 
                values = FALSE, na.all = TRUE)))
            if (any(c("sfc", "sf") %in% class(tpoints))) {
                modeldomain <- sf::st_transform(modeldomain, 
                  crs = sf::st_crs(tpoints))
            }
        }
        if (!any(class(sf::st_geometry(modeldomain)) %in% c("sfc_POLYGON", 
            "sfc_MULTIPOLYGON"))) {
            stop("modeldomain must be a sf/sfc polygon object.")
        }
        if (!identical(sf::st_crs(tpoints), sf::st_crs(modeldomain)) & 
            dist_space == "geographical") {
            stop("tpoints and modeldomain must have the same CRS")
        }
        message(paste0(samplesize, " prediction points are sampled from the modeldomain"))
        predpoints <- suppressMessages(sf::st_sample(x = modeldomain, 
            size = samplesize, type = sampling))
        sf::st_crs(predpoints) <- sf::st_crs(modeldomain)
        if (dist_space == "feature") {
            message("predictor values are extracted for prediction points")
            predpoints <- terra::extract(predictor_stack, terra::vect(predpoints), 
                ID = FALSE)
        }
    }
    else if (!is.null(predpoints) & dist_space == "geographical") {
        if (!identical(sf::st_crs(tpoints), sf::st_crs(predpoints))) {
            stop("tpoints and predpoints must have the same CRS")
        }
    }
    if (dist_space == "geographical") {
        if (any(class(tpoints) %in% "sfc")) {
            tpoints <- sf::st_sf(geom = tpoints)
        }
        if (any(class(predpoints) %in% "sfc")) {
            predpoints <- sf::st_sf(geom = predpoints)
        }
        if (is.na(sf::st_crs(tpoints))) {
            warning("Missing CRS in training or prediction points. Assuming projected CRS.")
            islonglat <- FALSE
        }
        else {
            islonglat <- sf::st_is_longlat(tpoints)
        }
    }
    else if (dist_space == "feature") {
        if (any(class(tpoints) %in% c("sf", "sfc"))) {
            tpoints <- sf::st_set_geometry(tpoints, NULL)
        }
        if (any(class(predpoints) %in% c("sf", "sfc"))) {
            predpoints <- sf::st_set_geometry(predpoints, NULL)
        }
        catVars <- names(tpoints)[vapply(tpoints, function(z) inherits(z, 
            c("factor", "character")), logical(1))]
        if (length(catVars) == 0) 
            catVars <- NULL
        if (!is.null(catVars)) {
            message(paste0("variable(s) '", catVars, "' is (are) treated as categorical variables"))
        }
        if (any(is.na(predpoints))) {
            message("some prediction points contain NAs, which will be removed")
            predpoints <- stats::na.omit(predpoints)
        }
        if (any(is.na(tpoints))) {
            message("some training points contain NAs, which will be removed")
            tpoints <- stats::na.omit(tpoints)
        }
    }
    if (isTRUE(dist_space == "geographical")) {
        check_knndm_geo(tpoints = tpoints, predpoints = predpoints, 
            dist_space = dist_space, k = k, maxp = maxp, clustering = clustering, 
            dist_fun = dist_fun, test_prop = test_prop, islonglat = islonglat)
        knndm_res <- knndm_geo(tpoints = tpoints, predpoints = predpoints, 
            k = k, maxp = maxp, minp = minp, test_prop = test_prop, 
            clustering = clustering, linkf = linkf, nk_len = nk_len, 
            dist_fun = dist_fun, dist_space = dist_space, algorithm = algorithm)
    }
    else if (isTRUE(dist_space == "feature")) {
        check_knndm_feature(tpoints = tpoints, predpoints = predpoints, 
            dist_space = dist_space, k = k, maxp = maxp, clustering = clustering, 
            dist_fun = dist_fun, test_prop = test_prop, catVars = catVars)
        knndm_res <- knndm_feature(tpoints = tpoints, predpoints = predpoints, 
            k = k, maxp = maxp, minp = minp, test_prop = test_prop, 
            clustering = clustering, linkf = linkf, nk_len = nk_len, 
            dist_fun = dist_fun, dist_space = dist_space, algorithm = algorithm, 
            catVars = catVars, scale_vars = scale_vars)
    }
    knndm_res
}
<bytecode: 0x000001c5caa1c778>
<environment: namespace:CAST>

#### knndm_feature
function (tpoints, predpoints, k, maxp, minp, test_prop, clustering, 
    linkf, nk_len, dist_fun, dist_space, algorithm, catVars, 
    scale_vars) 
{
    if (isTRUE(scale_vars)) {
        if (is.null(catVars)) {
            scale_attr <- attributes(scale(tpoints))
            tpoints <- as.data.frame(scale(tpoints))
            predpoints <- as.data.frame(scale(predpoints, center = scale_attr$`scaled:center`, 
                scale = scale_attr$`scaled:scale`))
        }
        else {
            tpoints_cat <- tpoints[, catVars, drop = FALSE]
            predpoints_cat <- predpoints[, catVars, drop = FALSE]
            tpoints_num <- tpoints[, -which(names(tpoints) %in% 
                catVars), drop = FALSE]
            predpoints_num <- predpoints[, -which(names(predpoints) %in% 
                catVars), drop = FALSE]
            scale_attr <- attributes(scale(tpoints_num))
            tpoints <- as.data.frame(scale(tpoints_num))
            predpoints <- as.data.frame(scale(predpoints_num, 
                center = scale_attr$`scaled:center`, scale = scale_attr$`scaled:scale`))
            tpoints <- as.data.frame(cbind(tpoints, lapply(tpoints_cat, 
                as.factor)))
            predpoints <- as.data.frame(cbind(predpoints, lapply(predpoints_cat, 
                as.factor)))
        }
    }
    if (is.null(catVars)) {
        if (isTRUE(dist_fun == "mahalanobis")) {
            tpoints_mat <- as.matrix(tpoints)
            predpoints_mat <- as.matrix(predpoints)
            if (dim(tpoints_mat)[2] == 1) {
                S <- matrix(stats::var(tpoints_mat), 1, 1)
                tpoints_mat <- as.matrix(tpoints_mat, ncol = 1)
            }
            else {
                S <- stats::cov(tpoints_mat)
            }
            S_inv <- MASS::ginv(S)
            n_rows <- nrow(tpoints_mat)
            distmat <- vapply(seq_len(n_rows), function(i) {
                vapply(seq_len(n_rows), function(j) {
                  diff <- tpoints_mat[i, ] - tpoints_mat[j, ]
                  sqrt(t(diff) %*% S_inv %*% diff)
                }, numeric(1))
            }, numeric(n_rows))
            diag(distmat) <- NA
            Gj <- apply(distmat, 1, min, na.rm = TRUE)
            n_rows_p <- nrow(predpoints_mat)
            n_rows_t <- nrow(tpoints_mat)
            Gij <- vapply(seq_len(n_rows_p), function(i) {
                min(vapply(seq_len(n_rows_t), function(j) {
                  diff <- predpoints_mat[i, ] - tpoints_mat[j, 
                    ]
                  sqrt(t(diff) %*% S_inv %*% diff)
                }, numeric(1)))
            }, numeric(1))
        }
        else {
            Gj <- c(FNN::knn.dist(tpoints, k = 1, algorithm = algorithm))
            Gij <- c(FNN::knnx.dist(query = predpoints, data = tpoints, 
                k = 1, algorithm = algorithm))
        }
    }
    else {
        Gj <- vapply(1:nrow(tpoints), function(i) gower::gower_topn(tpoints[i, 
            ], tpoints[-i, ], n = 1)$distance[[1]], numeric(1))
        Gij <- c(gower::gower_topn(predpoints, tpoints, n = 1)$distance)
    }
    testks <- suppressWarnings(stats::ks.test(Gj, Gij, alternative = "great"))
    if (testks$p.value >= 0.05) {
        if (!is.null(test_prop)) {
            ntest <- floor(test_prop * nrow(tpoints))
            ntrain <- nrow(tpoints) - ntest
            clusters <- c(rep(1, ntrain), rep(2, ntest))
            clust <- sample(clusters, nrow(tpoints))
        }
        else {
            clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), 
                size = nrow(tpoints), replace = F)
        }
        if (is.null(catVars)) {
            if (isTRUE(dist_fun == "mahalanobis")) {
                Gjstar <- cv_distances(tpoints, CVtest = clust, 
                  dist_fun = dist_fun)
            }
            else {
                Gjstar <- cv_distances(tpoints, clust, algorithm = algorithm, 
                  dist_fun = dist_fun)
            }
        }
        else {
            Gjstar <- cv_distances(tpoints, clust, dist_fun = dist_fun)
        }
        k_final <- "random CV"
        W_final <- twosamples::wass_stat(Gjstar, Gij)
        message("Gij <= Gj; a random CV assignment is returned")
    }
    else {
        if (clustering == "hierarchical") {
            if (is.null(catVars)) {
                if (isTRUE(dist_fun == "euclidean")) {
                  distmat <- as.matrix(stats::dist(tpoints, upper = TRUE, 
                    diag = TRUE))
                  diag(distmat) <- NA
                }
            }
            else {
                distmat <- matrix(nrow = nrow(tpoints), ncol = nrow(tpoints))
                for (i in 1:nrow(tpoints)) {
                  trainDist <- gower::gower_dist(tpoints[i, ], 
                    tpoints)
                  trainDist[i] <- NA
                  distmat[i, ] <- trainDist
                }
            }
            hc <- stats::hclust(d = stats::as.dist(distmat), 
                method = linkf)
        }
        clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), 
            log(nrow(tpoints) - 2), length.out = nk_len)))))
        clustgrid$W <- NA
        clustgrid <- clustgrid[!duplicated(clustgrid$nk), ]
        clustgroups <- list()
        if (is.null(catVars)) {
            pcacoords <- stats::prcomp(tpoints, center = TRUE, 
                scale. = FALSE, rank = 1)
        }
        else {
            pcacoords <- PCAmixdata::PCAmix(X.quanti = tpoints[, 
                !(names(tpoints) %in% catVars), drop = FALSE], 
                X.quali = tpoints[, names(tpoints) %in% catVars, 
                  drop = FALSE], graph = FALSE)
        }
        for (nk in clustgrid$nk) {
            if (clustering == "hierarchical") {
                clust_nk <- stats::cutree(hc, k = nk)
            }
            else if (clustering == "kmeans") {
                if (is.null(catVars)) {
                  clust_nk <- tryCatch(stats::kmeans(tpoints, 
                    nk)$cluster, error = function(e) e)
                }
                else {
                  clust_nk <- tryCatch(clustMixType::kproto(tpoints, 
                    nk, verbose = FALSE)$cluster, error = function(e) e)
                }
            }
            if (!inherits(clust_nk, "error")) {
                tabclust <- as.data.frame(table(clust_nk))
                tabclust$clust_k <- NA
                if (is.null(catVars)) {
                  centr_tpoints <- vapply(tabclust$clust_nk, 
                    function(x) {
                      centrpca <- matrix(colMeans(tpoints[clust_nk %in% 
                        x, , drop = FALSE]), nrow = 1)
                      colnames(centrpca) <- colnames(tpoints)
                      return(predict(pcacoords, centrpca))
                    }, numeric(1))
                }
                else {
                  centr_tpoints <- vapply(tabclust$clust_nk, 
                    function(x) {
                      centrpca_num <- matrix(apply(tpoints[clust_nk %in% 
                        x, !(names(tpoints) %in% catVars), drop = FALSE], 
                        2, mean), nrow = 1)
                      centrpca_cat <- matrix(apply(tpoints[clust_nk %in% 
                        x, names(tpoints) %in% catVars, drop = FALSE], 
                        2, function(y) names(which.max(table(y)))), 
                        nrow = 1)
                      colnames(centrpca_num) <- colnames(tpoints[, 
                        !(names(tpoints) %in% catVars), drop = FALSE])
                      colnames(centrpca_cat) <- colnames(tpoints[, 
                        names(tpoints) %in% catVars, drop = FALSE])
                      return(predict(pcacoords, centrpca_num, 
                        centrpca_cat)[, 1])
                    }, numeric(1))
                }
                tabclust$centrpca <- centr_tpoints
                tabclust <- tabclust[order(tabclust$centrpca), 
                  ]
                if (is.null(test_prop)) {
                  clust_i <- 1
                  for (i in 1:nrow(tabclust)) {
                    if (tabclust$Freq[i] >= nrow(tpoints)/k) {
                      tabclust$clust_k[i] <- clust_i
                      clust_i <- clust_i + 1
                    }
                  }
                  rm("clust_i")
                }
                clust_i <- setdiff(1:k, unique(tabclust$clust_k))
                tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, 
                  ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]
                tabclust2 <- data.frame(ID = 1:length(clust_nk), 
                  clust_nk = clust_nk)
                tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
                tabclust2 <- tabclust2[order(tabclust2$ID), ]
                clust_k <- tabclust2$clust_k
                if (is.null(test_prop)) {
                  prop_valid <- !(any(table(clust_k)/length(clust_k) > 
                    maxp))
                }
                else {
                  prop_1 <- mean(clust_k == 1)
                  prop_2 <- mean(clust_k == 2)
                  props <- c(prop_1, prop_2)
                  prop_valid <- props >= minp & props <= maxp
                }
                if (any(prop_valid)) {
                  if (clustering == "kmeans") {
                    if (is.null(catVars)) {
                      if (isTRUE(dist_fun == "mahalanobis")) {
                        Gjstar_i <- cv_distances(tpoints, CVtest = clust_k, 
                          dist_fun = dist_fun)
                      }
                      else {
                        Gjstar_i <- cv_distances(tpoints, CVtest = clust_k, 
                          algorithm = algorithm, dist_fun = dist_fun)
                      }
                    }
                    else {
                      Gjstar_i <- cv_distances(tpoints, CVtest = clust_k, 
                        dist_fun = dist_fun)
                    }
                  }
                  else {
                    Gjstar_i <- distclust_distmat(distmat, clust_k)
                  }
                  clustgrid$W[clustgrid$nk == nk] <- twosamples::wass_stat(Gjstar_i, 
                    Gij)
                  clustgroups[[paste0("nk", nk)]] <- clust_k
                }
            }
        }
        k_final <- clustgrid$nk[which.min(clustgrid$W)]
        W_final <- min(clustgrid$W, na.rm = T)
        clust <- clustgroups[[paste0("nk", k_final)]]
        if (!is.null(test_prop) && is.null(clust)) {
            stop("No valid train/test configurations found in the range test_prop +/- tolerance. Increase tolerance.")
        }
        if (clustering == "kmeans") {
            if (is.null(catVars)) {
                if (isTRUE(dist_fun == "mahalanobis")) {
                  Gjstar <- cv_distances(tpoints, CVtest = clust, 
                    dist_fun = dist_fun)
                }
                else {
                  Gjstar <- cv_distances(tpoints, CVtest = clust, 
                    algorithm = algorithm, dist_fun = dist_fun)
                }
            }
            else {
                Gjstar <- cv_distances(tpoints, CVtest = clust, 
                  dist_fun = dist_fun)
            }
        }
        else {
            Gjstar <- distclust_distmat(distmat, clust)
        }
    }
    if (is.null(test_prop)) {
        cfolds <- CAST::CreateSpacetimeFolds(data.frame(clust = clust), 
            spacevar = "clust", k = k)
    }
    else {
        deviation_1 <- abs((table(clust)[[1]]/length(clust)) - 
            test_prop)
        deviation_2 <- abs((table(clust)[[2]]/length(clust)) - 
            test_prop)
        if (deviation_1 > deviation_2) {
            test_class <- 2
        }
        else {
            test_class <- 1
        }
        clust[clust == test_class] <- "test"
        clust[clust != "test"] <- "train"
        cfolds <- list(indexOut = which(clust == "test"), index = which(clust == 
            "train"))
    }
    res <- list(clusters = clust, indx_train = cfolds$index, 
        indx_test = cfolds$indexOut, Gij = Gij, Gj = Gj, Gjstar = Gjstar, 
        W = W_final, method = clustering, q = k_final, dist_space = "feature")
    class(res) <- c("knndm", "list")
    res
}
<bytecode: 0x000001c5caa345b8>
<environment: namespace:CAST>

#### knndm_geo
function (tpoints, predpoints, k, maxp, minp, test_prop, clustering, 
    linkf, nk_len, dist_fun, dist_space, algorithm) 
{
    tcoords <- sf::st_coordinates(tpoints)[, 1:2]
    if (isTRUE(dist_fun == "great_circle")) {
        distmat <- sf::st_distance(tpoints)
        units(distmat) <- NULL
        diag(distmat) <- NA
        Gj <- apply(distmat, 1, function(x) min(x, na.rm = TRUE))
        Gij <- sf::st_distance(predpoints, tpoints)
        units(Gij) <- NULL
        Gij <- apply(Gij, 1, min)
    }
    else {
        Gj <- compute_NND(tpoints, dist_space = dist_space, dist_fun = dist_fun, 
            algorithm = algorithm)$dist
        Gij <- compute_NND(tpoints, y = predpoints, dist_space = dist_space, 
            dist_fun = dist_fun, algorithm = algorithm)$dist
    }
    testks <- suppressWarnings(stats::ks.test(Gj, Gij, alternative = "great"))
    if (testks$p.value >= 0.05) {
        if (!is.null(test_prop)) {
            ntest <- floor(test_prop * nrow(tpoints))
            ntrain <- nrow(tpoints) - ntest
            clusters <- c(rep(1, ntrain), rep(2, ntest))
            clust <- sample(clusters, nrow(tpoints))
        }
        else {
            clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), 
                size = nrow(tpoints), replace = F)
        }
        if (isTRUE(dist_fun == "great_circle")) {
            Gjstar <- distclust_distmat(distmat, clust)
        }
        else {
            Gjstar <- cv_distances(tcoords, CVtest = clust, algorithm = algorithm, 
                dist_fun = dist_fun)
        }
        k_final <- "random CV"
        W_final <- twosamples::wass_stat(Gjstar, Gij)
        message("Gij <= Gj; a random CV assignment is returned")
    }
    else {
        if (clustering == "hierarchical") {
            if (isTRUE(dist_fun == "euclidean")) {
                distmat <- sf::st_distance(tpoints)
            }
            hc <- stats::hclust(d = stats::as.dist(distmat), 
                method = linkf)
        }
        clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), 
            log(nrow(tpoints) - 2), length.out = nk_len)))))
        clustgrid$W <- NA
        clustgrid <- clustgrid[!duplicated(clustgrid$nk), ]
        clustgroups <- list()
        pcacoords <- stats::prcomp(tcoords, center = TRUE, scale. = FALSE, 
            rank = 1)
        for (nk in clustgrid$nk) {
            if (clustering == "hierarchical") {
                clust_nk <- stats::cutree(hc, k = nk)
            }
            else if (clustering == "kmeans") {
                clust_nk <- stats::kmeans(tcoords, nk)$cluster
            }
            tabclust <- as.data.frame(table(clust_nk))
            tabclust$clust_k <- NA
            centr_tpoints <- vapply(tabclust$clust_nk, function(x) {
                centrpca <- matrix(colMeans(tcoords[clust_nk %in% 
                  x, , drop = FALSE]), nrow = 1)
                colnames(centrpca) <- colnames(tcoords)
                return(predict(pcacoords, centrpca))
            }, numeric(1))
            tabclust$centrpca <- centr_tpoints
            tabclust <- tabclust[order(tabclust$centrpca), ]
            if (is.null(test_prop)) {
                clust_i <- 1
                for (i in 1:nrow(tabclust)) {
                  if (tabclust$Freq[i] >= nrow(tpoints)/k) {
                    tabclust$clust_k[i] <- clust_i
                    clust_i <- clust_i + 1
                  }
                }
                rm("clust_i")
            }
            clust_i <- setdiff(1:k, unique(tabclust$clust_k))
            tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, 
                ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]
            tabclust2 <- data.frame(ID = 1:length(clust_nk), 
                clust_nk = clust_nk)
            tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
            tabclust2 <- tabclust2[order(tabclust2$ID), ]
            clust_k <- tabclust2$clust_k
            if (is.null(test_prop)) {
                prop_valid <- !(any(table(clust_k)/length(clust_k) > 
                  maxp))
            }
            else {
                prop_1 <- mean(clust_k == 1)
                prop_2 <- mean(clust_k == 2)
                props <- c(prop_1, prop_2)
                prop_valid <- props >= minp & props <= maxp
            }
            if (any(prop_valid)) {
                if (isTRUE(dist_fun == "great_circle")) {
                  Gjstar_i <- distclust_distmat(distmat, clust_k)
                }
                else {
                  Gjstar_i <- cv_distances(tcoords, CVtest = clust_k, 
                    algorithm = algorithm, dist_fun = dist_fun)
                }
                clustgrid$W[clustgrid$nk == nk] <- twosamples::wass_stat(Gjstar_i, 
                  Gij)
                clustgroups[[paste0("nk", nk)]] <- clust_k
            }
            if (!any(table(clust_k)/length(clust_k) > maxp)) {
                if (isTRUE(dist_fun == "great_circle")) {
                  Gjstar_i <- distclust_distmat(distmat, clust_k)
                }
                else {
                  Gjstar_i <- cv_distances(tcoords, CVtest = clust_k, 
                    algorithm = algorithm, dist_fun = dist_fun)
                }
                clustgrid$W[clustgrid$nk == nk] <- twosamples::wass_stat(Gjstar_i, 
                  Gij)
                clustgroups[[paste0("nk", nk)]] <- clust_k
            }
        }
        k_final <- clustgrid$nk[which.min(clustgrid$W)]
        W_final <- min(clustgrid$W, na.rm = T)
        clust <- clustgroups[[paste0("nk", k_final)]]
        if (!is.null(test_prop) && is.null(clust)) {
            stop("No valid train/test configurations found in the range test_prop +/- tolerance. Increase tolerance.")
        }
        if (isTRUE(dist_fun == "great_circle")) {
            Gjstar <- distclust_distmat(distmat, clust)
        }
        else {
            Gjstar <- cv_distances(tcoords, CVtest = clust, algorithm = algorithm, 
                dist_fun = dist_fun)
        }
    }
    if (is.null(test_prop)) {
        cfolds <- CAST::CreateSpacetimeFolds(data.frame(clust = clust), 
            spacevar = "clust", k = k)
    }
    else {
        deviation_1 <- abs((table(clust)[[1]]/length(clust)) - 
            test_prop)
        deviation_2 <- abs((table(clust)[[2]]/length(clust)) - 
            test_prop)
        if (deviation_1 > deviation_2) {
            test_class <- 2
        }
        else {
            test_class <- 1
        }
        clust[clust == test_class] <- "test"
        clust[clust != "test"] <- "train"
        cfolds <- list(indexOut = which(clust == "test"), index = which(clust == 
            "train"))
    }
    res <- list(clusters = clust, indx_train = cfolds$index, 
        indx_test = cfolds$indexOut, Gij = Gij, Gj = Gj, Gjstar = Gjstar, 
        W = W_final, method = clustering, q = k_final, dist_space = "geographical")
    class(res) <- c("knndm", "list")
    res
}
<bytecode: 0x000001c5caa9efc8>
<environment: namespace:CAST>

#### nndm
function (tpoints, modeldomain = NULL, predpoints = NULL, space = "geographical", 
    samplesize = 1000, sampling = "regular", phi = "max", min_train = 0.5, 
    algorithm = "brute") 
{
    if (is.null(predpoints) & !is.null(modeldomain)) {
        if (!any(c("sfc", "sf", "SpatRaster") %in% class(modeldomain))) {
            stop("modeldomain must be a sf/sfc object or a 'SpatRaster' object.")
        }
        if (any(class(modeldomain) == "SpatRaster")) {
            if (space == "feature") {
                predictor_stack <- modeldomain
            }
            modeldomain[!is.na(modeldomain)] <- 1
            modeldomain <- sf::st_union(sf::st_as_sf(terra::as.polygons(modeldomain, 
                values = FALSE, na.all = TRUE)))
            if (any(c("sfc", "sf") %in% class(tpoints))) {
                modeldomain <- sf::st_transform(modeldomain, 
                  crs = sf::st_crs(tpoints))
            }
        }
        if (!any(class(sf::st_geometry(modeldomain)) %in% c("sfc_POLYGON", 
            "sfc_MULTIPOLYGON"))) {
            stop("modeldomain must be a sf/sfc polygon object.")
        }
        if (!identical(sf::st_crs(tpoints), sf::st_crs(modeldomain)) & 
            space == "geographical") {
            stop("tpoints and modeldomain must have the same CRS")
        }
        message(paste0(samplesize, " prediction points are sampled from the modeldomain"))
        predpoints <- sf::st_sample(x = modeldomain, size = samplesize, 
            type = sampling)
        sf::st_crs(predpoints) <- sf::st_crs(modeldomain)
        if (space == "feature") {
            message("predictor values are extracted for prediction points")
            predpoints <- terra::extract(predictor_stack, terra::vect(predpoints), 
                ID = FALSE)
        }
    }
    else if (!is.null(predpoints) & space == "geographical") {
        if (!identical(sf::st_crs(tpoints), sf::st_crs(predpoints))) {
            stop("tpoints and predpoints must have the same CRS")
        }
    }
    if (isTRUE(space == "geographical")) {
        if (any(class(tpoints) %in% "sfc")) {
            tpoints <- sf::st_sf(geom = tpoints)
        }
        if (any(class(predpoints) %in% "sfc")) {
            predpoints <- sf::st_sf(geom = predpoints)
        }
        nndm_checks_geo(tpoints, predpoints, phi, min_train)
    }
    else if (isTRUE(space == "feature")) {
        if (any(class(tpoints) %in% c("sf", "sfc"))) {
            tpoints <- sf::st_set_geometry(tpoints, NULL)
        }
        if (any(class(predpoints) %in% c("sf", "sfc"))) {
            predpoints <- sf::st_set_geometry(predpoints, NULL)
        }
        catVars <- names(tpoints)[which(sapply(tpoints, class) %in% 
            c("factor", "character"))]
        if (length(catVars) == 0) {
            catVars <- NULL
        }
        if (!is.null(catVars)) {
            message(paste0("variable(s) '", catVars, "' is (are) treated as categorical variables"))
        }
        if (any(is.na(predpoints))) {
            message("some prediction points contain NAs, which will be removed")
            predpoints <- stats::na.omit(predpoints)
        }
        if (any(is.na(tpoints))) {
            message("some training points contain NAs, which will be removed")
            tpoints <- stats::na.omit(tpoints)
        }
        nndm_checks_feature(tpoints, predpoints, phi, min_train, 
            catVars)
        if (is.null(catVars)) {
            scale_attr <- attributes(scale(tpoints))
            tpoints <- as.data.frame(scale(tpoints))
            predpoints <- as.data.frame(scale(predpoints, center = scale_attr$`scaled:center`, 
                scale = scale_attr$`scaled:scale`))
        }
        else {
            tpoints_cat <- tpoints[, catVars, drop = FALSE]
            predpoints_cat <- predpoints[, catVars, drop = FALSE]
            tpoints_num <- tpoints[, -which(names(tpoints) %in% 
                catVars), drop = FALSE]
            predpoints_num <- predpoints[, -which(names(predpoints) %in% 
                catVars), drop = FALSE]
            scale_attr <- attributes(scale(tpoints_num))
            tpoints <- as.data.frame(scale(tpoints_num))
            predpoints <- as.data.frame(scale(predpoints_num, 
                center = scale_attr$`scaled:center`, scale = scale_attr$`scaled:scale`))
            tpoints <- as.data.frame(cbind(tpoints, lapply(tpoints_cat, 
                as.factor)))
            predpoints <- as.data.frame(cbind(predpoints, lapply(predpoints_cat, 
                as.factor)))
            for (catvar in catVars) {
                tpoints[, catvar] <- droplevels(tpoints[, catvar])
                predpoints[, catvar] <- droplevels(predpoints[, 
                  catvar])
                dvi_train <- predict(caret::dummyVars(paste0("~", 
                  catvar), data = tpoints), tpoints)
                dvi_predpoints <- predict(caret::dummyVars(paste0("~", 
                  catvar), data = predpoints), predpoints)
                tpoints <- data.frame(tpoints, dvi_train)
                predpoints <- data.frame(predpoints, dvi_predpoints)
            }
            tpoints <- tpoints[, -which(names(tpoints) %in% catVars)]
            predpoints <- predpoints[, -which(names(predpoints) %in% 
                catVars)]
        }
    }
    if (isTRUE(space == "geographical")) {
        Gij <- sf::st_distance(predpoints, tpoints)
        units(Gij) <- NULL
        Gij <- apply(Gij, 1, min)
        tdist <- sf::st_distance(tpoints)
        units(tdist) <- NULL
        diag(tdist) <- NA
        Gj <- apply(tdist, 1, function(x) min(x, na.rm = TRUE))
        Gjstar <- Gj
        if (phi == "max") {
            phi <- max(c(Gij, c(tdist)), na.rm = TRUE) + 1e-09
        }
    }
    else if (isTRUE(space == "feature")) {
        if (is.null(catVars)) {
            Gij <- c(FNN::knnx.dist(query = predpoints, data = tpoints, 
                k = 1, algorithm = algorithm))
            tdist <- as.matrix(stats::dist(tpoints, upper = TRUE))
            diag(tdist) <- NA
            Gj <- apply(tdist, 1, function(x) min(x, na.rm = TRUE))
            Gjstar <- Gj
        }
        else {
            Gj <- sapply(1:nrow(tpoints), function(i) gower::gower_topn(tpoints[i, 
                ], tpoints[-i, ], n = 1)$distance[[1]])
            tdist <- matrix(NA, nrow = nrow(tpoints), ncol = nrow(tpoints))
            for (r in 1:nrow(tdist)) {
                tdist[r, ] <- gower::gower_dist(tpoints[r, ], 
                  tpoints)
            }
            diag(tdist) <- NA
            Gj <- apply(tdist, 1, function(x) min(x, na.rm = TRUE))
            Gjstar <- Gj
        }
        if (phi == "max") {
            phi <- max(c(Gij, c(tdist)), na.rm = TRUE) + 1e-09
        }
    }
    rmin <- min(Gjstar)
    jmin <- which.min(Gjstar)[1]
    kmin <- which(tdist[jmin, ] == rmin)
    while (rmin <= phi) {
        if ((sum(Gjstar <= rmin) - 1)/length(Gjstar) >= (sum(Gij <= 
            rmin)/length(Gij)) & sum(!is.na(tdist[jmin, ]))/ncol(tdist) > 
            min_train) {
            tdist[jmin, kmin] <- NA
            Gjstar <- apply(tdist, 1, function(x) min(x, na.rm = TRUE))
            rmin <- min(Gjstar[Gjstar >= rmin])
            jmin <- which(Gjstar == rmin)[1]
            kmin <- which(tdist[jmin, ] == rmin)
        }
        else if (sum(Gjstar > rmin) == 0) {
            break
        }
        else {
            rmin <- min(Gjstar[Gjstar > rmin])
            jmin <- which(Gjstar == rmin)[1]
            kmin <- which(tdist[jmin, ] == rmin)
        }
    }
    indx_train <- list()
    indx_test <- list()
    indx_exclude <- list()
    for (i in 1:nrow(tdist)) {
        indx_train[[i]] <- which(!is.na(tdist[i, ]))
        indx_test[[i]] <- i
        indx_exclude[[i]] <- setdiff(which(is.na(tdist[i, ])), 
            i)
    }
    res <- list(indx_train = indx_train, indx_test = indx_test, 
        indx_exclude = indx_exclude, Gij = Gij, Gj = Gj, Gjstar = Gjstar, 
        phi = phi)
    class(res) <- c("nndm", "list")
    res
}
<bytecode: 0x000001c5caabacf0>
<environment: namespace:CAST>

#### nndm_checks_feature
function (tpoints, predpoints, phi, min_train, catVars) 
{
    if (phi < 0 | (!is.numeric(phi) & phi != "max")) {
        stop("phi must be positive or set to 'max'.")
    }
    if (length(min_train) != 1 | min_train < 0 | min_train > 
        1 | !is.numeric(min_train)) {
        stop("min_train must be a numeric between 0 and 1.")
    }
    if (length(setdiff(names(tpoints), names(predpoints))) > 
        0) {
        stop("tpoints and predpoints need to contain the predictor data and have the same colnames.")
    }
    for (catvar in catVars) {
        if (any(!unique(tpoints[, catvar]) %in% unique(predpoints[, 
            catvar]))) {
            stop(paste0("Some values of factor", catvar, "are only present in training / prediction points.\n                  All factor values in the prediction points must be present in the training points."))
        }
    }
}
<bytecode: 0x000001c5cab2f5e8>
<environment: namespace:CAST>

#### nndm_checks_geo
function (tpoints, predpoints, phi, min_train) 
{
    if (phi < 0 | (!is.numeric(phi) & phi != "max")) {
        stop("phi must be positive or set to 'max'.")
    }
    if (length(min_train) != 1 | min_train < 0 | min_train > 
        1 | !is.numeric(min_train)) {
        stop("min_train must be a numeric between 0 and 1.")
    }
    if (!any(c("sfc", "sf") %in% class(tpoints))) {
        stop("tpoints must be a sf/sfc object.")
    }
    else if (!any(class(sf::st_geometry(tpoints)) %in% c("sfc_POINT"))) {
        stop("tpoints must be a sf/sfc point object.")
    }
    if (!any(c("sfc", "sf") %in% class(predpoints))) {
        stop("predpoints must be a sf/sfc object.")
    }
    else if (!any(class(sf::st_geometry(predpoints)) %in% c("sfc_POINT"))) {
        stop("predpoints must be a sf/sfc point object.")
    }
}
<bytecode: 0x000001c5cab30858>
<environment: namespace:CAST>

#### plot.knndm
function (x, type = "strict", stat = "ecdf", ...) 
{
    Gij_df <- data.frame(r = x$Gij[order(x$Gij)])
    Gij_df$Function <- "1_Gij(r)"
    Gjstar_df <- data.frame(r = x$Gjstar[order(x$Gjstar)])
    Gjstar_df$Function <- "2_Gjstar(r)"
    Gj_df <- data.frame(r = x$Gj[order(x$Gj)])
    Gj_df$Function <- "3_Gj(r)"
    Gplot <- rbind(Gij_df, Gjstar_df, Gj_df)
    myColors <- c("#1B9E77", "#D95F02", "#7570B3")
    if (stat == "ecdf") {
        p <- ggplot2::ggplot(data = Gplot, ggplot2::aes(x = .data[["r"]], 
            group = .data[["Function"]], col = .data[["Function"]])) + 
            ggplot2::geom_vline(xintercept = 0, lwd = 0.1) + 
            ggplot2::geom_hline(yintercept = 0, lwd = 0.1) + 
            ggplot2::geom_hline(yintercept = 1, lwd = 0.1) + 
            ggplot2::stat_ecdf(geom = "step", lwd = 0.8) + ggplot2::theme_bw() + 
            ggplot2::ylab("ECDF") + ggplot2::labs(group = "Distance function") + 
            ggplot2::theme(legend.position = "bottom", legend.text = ggplot2::element_text(size = 10))
        if (type == "strict") {
            p <- p + ggplot2::scale_colour_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c(expression(hat(G)[ij](r)), expression(hat(G)[j]^"*" * 
                  "(r,L)"), expression(hat(G)[j](r))))
        }
        else if (type == "simple") {
            p <- p + ggplot2::scale_colour_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c("prediction-to-sample", "CV-distances", 
                  "sample-to-sample"))
        }
    }
    else if (stat == "density") {
        p <- ggplot2::ggplot(data = Gplot, ggplot2::aes(x = .data[["r"]], 
            group = .data[["Function"]], fill = .data[["Function"]])) + 
            ggplot2::geom_density(adjust = 1.5, alpha = 0.5, 
                stat = stat, lwd = 0.3) + ggplot2::theme_bw() + 
            ggplot2::ylab("Density") + ggplot2::labs(group = "Distance function") + 
            ggplot2::theme(legend.position = "bottom", legend.text = ggplot2::element_text(size = 10))
        if (type == "strict") {
            p <- p + ggplot2::scale_fill_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c(expression(hat(G)[ij](r)), expression(hat(G)[j]^"*" * 
                  "(r,L)"), expression(hat(G)[j](r))))
        }
        else if (type == "simple") {
            p <- p + ggplot2::scale_fill_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c("prediction-to-sample", "CV-distances", 
                  "sample-to-sample"))
        }
    }
    return(p)
}
<bytecode: 0x000001c5cab273f0>
<environment: namespace:CAST>

#### plot.nndm
function (x, type = "strict", stat = "ecdf", ...) 
{
    Gij_df <- data.frame(r = x$Gij[order(x$Gij)])
    Gij_df$val <- 1:nrow(Gij_df)/nrow(Gij_df)
    Gij_df <- Gij_df[Gij_df$r <= x$phi, ]
    Gij_df <- rbind(Gij_df, data.frame(r = 0, val = 0))
    Gij_df <- rbind(Gij_df, data.frame(r = x$phi, val = sum(x$Gij <= 
        x$phi)/length(x$Gij)))
    Gij_df$Function <- "1_Gij(r)"
    Gjstar_df <- data.frame(r = x$Gjstar[order(x$Gjstar)])
    Gjstar_df$val <- 1:nrow(Gjstar_df)/nrow(Gjstar_df)
    Gjstar_df <- Gjstar_df[Gjstar_df$r <= x$phi, ]
    Gjstar_df <- rbind(Gjstar_df, data.frame(r = 0, val = 0))
    Gjstar_df <- rbind(Gjstar_df, data.frame(r = x$phi, val = sum(x$Gjstar <= 
        x$phi)/length(x$Gjstar)))
    Gjstar_df$Function <- "2_Gjstar(r)"
    Gj_df <- data.frame(r = x$Gj[order(x$Gj)])
    Gj_df$val <- 1:nrow(Gj_df)/nrow(Gj_df)
    Gj_df <- Gj_df[Gj_df$r <= x$phi, ]
    Gj_df <- rbind(Gj_df, data.frame(r = 0, val = 0))
    Gj_df <- rbind(Gj_df, data.frame(r = x$phi, val = sum(x$Gj <= 
        x$phi)/length(x$Gj)))
    Gj_df$Function <- "3_Gj(r)"
    if (any(Gj_df$val == 1) & any(Gjstar_df$val == 1) & any(Gij_df$val == 
        1)) {
        Gplot <- rbind(Gij_df, Gjstar_df, Gj_df)
        maxdist <- max(Gplot$r[Gplot$val != 1]) + 1e-09
        Gplot <- Gplot[Gplot$r <= maxdist, ]
        Gplot <- rbind(Gplot, data.frame(r = maxdist, val = 1, 
            Function = c("1_Gij(r)", "2_Gjstar(r)", "3_Gj(r)")))
    }
    else {
        Gplot <- rbind(Gij_df, Gjstar_df, Gj_df)
    }
    myColors <- c("#1B9E77", "#D95F02", "#7570B3")
    if (stat == "ecdf") {
        p <- ggplot2::ggplot(data = Gplot, ggplot2::aes(x = .data[["r"]], 
            group = .data[["Function"]], col = .data[["Function"]])) + 
            ggplot2::geom_vline(xintercept = 0, lwd = 0.1) + 
            ggplot2::geom_hline(yintercept = 0, lwd = 0.1) + 
            ggplot2::geom_hline(yintercept = 1, lwd = 0.1) + 
            ggplot2::stat_ecdf(geom = "step", lwd = 0.8) + ggplot2::theme_bw() + 
            ggplot2::ylab("ECDF") + ggplot2::labs(group = "Distance function") + 
            ggplot2::theme(legend.position = "bottom", legend.text = ggplot2::element_text(size = 10))
        if (type == "strict") {
            p <- p + ggplot2::scale_colour_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c(expression(hat(G)[ij](r)), expression(hat(G)[j]^"*" * 
                  "(r,L)"), expression(hat(G)[j](r))))
        }
        else if (type == "simple") {
            p <- p + ggplot2::scale_colour_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c("prediction-to-sample", "CV-distances", 
                  "sample-to-sample"))
        }
    }
    else if (stat == "density") {
        p <- ggplot2::ggplot(data = Gplot, ggplot2::aes(x = .data[["r"]], 
            group = .data[["Function"]], fill = .data[["Function"]])) + 
            ggplot2::geom_density(adjust = 1.5, alpha = 0.5, 
                stat = stat, lwd = 0.3) + ggplot2::theme_bw() + 
            ggplot2::ylab("Density") + ggplot2::labs(group = "Distance function") + 
            ggplot2::theme(legend.position = "bottom", legend.text = ggplot2::element_text(size = 10))
        if (type == "strict") {
            p <- p + ggplot2::scale_fill_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c(expression(hat(G)[ij](r)), expression(hat(G)[j]^"*" * 
                  "(r,L)"), expression(hat(G)[j](r))))
        }
        else if (type == "simple") {
            p <- p + ggplot2::scale_fill_manual(name = "Distance function", 
                values = c(myColors[2], myColors[3], myColors[1]), 
                labels = c("prediction-to-sample", "CV-distances", 
                  "sample-to-sample"))
        }
    }
    return(p)
}
<bytecode: 0x000001c5cab6a498>
<environment: namespace:CAST>

#### print.knndm
function (x, ...) 
{
    cat(paste0("knndm object\n", "Space: ", x$space, "\n", "Clustering algorithm: ", 
        x$method, "\n", "Intermediate clusters (q): ", x$q, "\n", 
        "W statistic: ", round(x$W, 4), "\n", "Number of folds: ", 
        length(unique(x$clusters)), "\n", "Observations in each fold: "), 
        table(x$clusters), "\n")
    invisible(x)
}
<bytecode: 0x000001c5cab58738>
<environment: namespace:CAST>

#### print.nndm
function (x, ...) 
{
    mean_train <- round(mean(sapply(x$indx_train, length)), 2)
    min_train <- round(min(sapply(x$indx_train, length)), 2)
    cat(paste0("nndm object\n", "Total number of points: ", length(x$Gj), 
        "\n", "Mean number of training points: ", mean_train, 
        "\n", "Minimum number of training points: ", min_train, 
        "\n"))
    invisible(x)
}
<bytecode: 0x000001c5cab80af0>
<environment: namespace:CAST>

#### show.knndm
function (x, ...) 
{
    print.knndm(x)
}
<bytecode: 0x000001c5cab50c40>
<environment: namespace:CAST>

#### show.nndm
function (x, ...) 
{
    print.nndm(x)
}
<bytecode: 0x000001c5cab59fc8>
<environment: namespace:CAST>
