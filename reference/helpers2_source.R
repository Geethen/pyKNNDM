
#### createFolds
function (y, k = 10, list = TRUE, returnTrain = FALSE) 
{
    if (inherits(y, "Surv")) 
        y <- y[, "time"]
    if (is.numeric(y)) {
        cuts <- floor(length(y)/k)
        if (cuts < 2) 
            cuts <- 2
        if (cuts > 5) 
            cuts <- 5
        breaks <- unique(quantile(y, probs = seq(0, 1, length.out = cuts)))
        y <- cut(y, breaks, include.lowest = TRUE)
    }
    if (k < length(y)) {
        y <- factor(as.character(y))
        numInClass <- table(y)
        foldVector <- vector(mode = "integer", length(y))
        for (i in 1:length(numInClass)) {
            min_reps <- numInClass[i]%/%k
            if (min_reps > 0) {
                spares <- numInClass[i]%%k
                seqVector <- rep(1:k, min_reps)
                if (spares > 0) 
                  seqVector <- c(seqVector, sample(1:k, spares))
                foldVector[which(y == names(numInClass)[i])] <- sample(seqVector)
            }
            else {
                foldVector[which(y == names(numInClass)[i])] <- sample(1:k, 
                  size = numInClass[i])
            }
        }
    }
    else foldVector <- seq(along.with = y)
    if (list) {
        out <- split(seq(along.with = y), foldVector)
        names(out) <- paste("Fold", gsub(" ", "0", format(seq(along.with = out))), 
            sep = "")
        if (returnTrain) 
            out <- lapply(out, function(data, y) y[-data], y = seq(along.with = y))
    }
    else out <- foldVector
    out
}
<bytecode: 0x000001ed796aef90>
<environment: namespace:caret>

#### quantile.default
function (x, probs = seq(0, 1, 0.25), na.rm = FALSE, names = TRUE, 
    type = 7L, digits = 7, fuzz = if (type == 7L) 0 else 4 * 
        .Machine$double.eps, ...) 
{
    if (length(type) != 1L || is.na(type) || !is.numeric(type) || 
        !any(type == 1:9)) 
        stop("'type' must be an integer in 1..9")
    if (is.factor(x)) {
        if (is.ordered(x)) {
            if (!any(type == c(1L, 3L))) 
                stop("'type' must be 1 or 3 for ordered factors")
        }
        else stop("(unordered) factors are not allowed")
        lx <- levels(x)
        x <- as.integer(x)
    }
    else {
        if (is.null(x)) 
            x <- numeric()
        lx <- NULL
    }
    if (na.rm) 
        x <- x[!is.na(x)]
    else if (anyNA(x)) 
        stop("missing values and NaN's not allowed if 'na.rm' is FALSE")
    eps <- 100 * .Machine$double.eps
    if (any((p.ok <- !is.na(probs)) & (probs < -eps | probs > 
        1 + eps))) 
        stop("'probs' outside [0,1]")
    probs <- pmax(0, pmin(1, probs))
    stopifnot(is.finite(fuzz), fuzz >= 0, length(fuzz) == 1L)
    floorF <- function(np) floor(np * (1 + fuzz))
    n <- length(x)
    np <- length(probs)
    {
        if (type == 7) {
            index <- 1 + max(n - 1, 0) * probs
            lo <- floorF(index)
            hi <- ceiling(index)
            x <- sort(x, partial = if (n == 0) 
                numeric()
            else unique(c(lo, hi)[p.ok]))
            qs <- x[lo]
            i <- which(!p.ok | (index > lo & x[hi] != qs))
            h <- (index - lo)[i]
            qs[i] <- (1 - h) * qs[i] + h * x[hi[i]]
        }
        else {
            if (type <= 3L) {
                nppm <- if (type == 3L) 
                  n * probs - 0.5
                else n * probs
                j <- floorF(nppm)
                h <- switch(type, !p.ok | (nppm > j), ((nppm > 
                  j) + 1)/2, !p.ok | (nppm != j) | ((j%%2L) == 
                  1L))
            }
            else {
                switch(type - 3L, {
                  a <- 0
                  b <- 1
                }, a <- b <- 0.5, a <- b <- 0, a <- b <- 1, a <- b <- 1/3, 
                  a <- b <- 3/8)
                nppm <- a + probs * (n + 1 - a - b)
                j <- floorF(nppm)
                h <- nppm - j
                if (any(sml <- abs(h) < fuzz, na.rm = TRUE)) 
                  h[sml] <- 0
            }
            x <- sort(x, partial = if (n == 0) 
                numeric()
            else unique(c(1, j[p.ok & j > 0L & j <= n], (j + 
                1)[p.ok & j > 0L & j < n], n)))
            x <- c(x[1L], x[1L], x, x[n], x[n])
            qs <- x[j + 2L]
            qs[!is.na(h) & h == 1] <- x[j + 3L][!is.na(h) & h == 
                1]
            other <- (0 < h) & (h < 1) & (x[j + 2L] != x[j + 
                3L])
            other[is.na(other)] <- TRUE
            if (any(other)) 
                qs[other] <- ((1 - h) * x[j + 2L] + h * x[j + 
                  3L])[other]
        }
    }
    qs[!p.ok] <- probs[!p.ok]
    if (is.character(lx)) 
        qs <- factor(qs, levels = seq_along(lx), labels = lx, 
            ordered = TRUE)
    if (names && np > 0L) {
        stopifnot(is.numeric(digits), digits >= 1)
        names(qs) <- format_perc(probs, digits = digits)
    }
    qs
}
<bytecode: 0x000001ed798067b0>
<environment: namespace:stats>

#### cut.default
function (x, breaks, labels = NULL, include.lowest = FALSE, right = TRUE, 
    dig.lab = 3L, ordered_result = FALSE, ...) 
{
    if (!is.numeric(x)) 
        stop("'x' must be numeric")
    if (length(breaks) == 1L) {
        if (is.na(breaks) || breaks < 2L) 
            stop("invalid number of intervals")
        nb <- as.integer(breaks + 1)
        dx <- diff(rx <- range(x, na.rm = TRUE))
        if (dx == 0) {
            dx <- if (rx[1L] != 0) 
                abs(rx[1L])
            else 1
            breaks <- seq.int(rx[1L] - dx/1000, rx[2L] + dx/1000, 
                length.out = nb)
        }
        else {
            breaks <- seq.int(rx[1L], rx[2L], length.out = nb)
            breaks[c(1L, nb)] <- c(rx[1L] - dx/1000, rx[2L] + 
                dx/1000)
        }
    }
    else nb <- length(breaks <- sort.int(as.double(breaks)))
    if (anyDuplicated(breaks)) 
        stop("'breaks' are not unique")
    codes.only <- FALSE
    if (is.null(labels)) {
        for (dig in dig.lab:max(12L, dig.lab)) {
            ch.br <- formatC(0 + breaks, digits = dig, width = 1L)
            if (ok <- all(ch.br[-1L] != ch.br[-nb])) 
                break
        }
        labels <- if (ok) 
            paste0(if (right) 
                "("
            else "[", ch.br[-nb], ",", ch.br[-1L], if (right) 
                "]"
            else ")")
        else paste0("Range_", seq_len(nb - 1L))
        if (ok && include.lowest) {
            if (right) 
                substr(labels[1L], 1L, 1L) <- "["
            else substring(labels[nb - 1L], nchar(labels[nb - 
                1L], "c")) <- "]"
        }
    }
    else if (is.logical(labels) && !labels) 
        codes.only <- TRUE
    else if (length(labels) != nb - 1L) 
        stop("number of intervals and length of 'labels' differ")
    code <- .bincode(x, breaks, right, include.lowest)
    if (codes.only) 
        code
    else factor(code, seq_along(labels), labels, ordered = ordered_result)
}
<bytecode: 0x000001ed7987bf20>
<environment: namespace:base>

#### presort_stat_fn
function (joint, labs, power, na) 
{
    .Call(`_twosamples_wass_stat_presort`, joint, labs, power, 
        na)
}
<bytecode: 0x000001ed7996dee8>
<environment: namespace:twosamples>
def_power: 1 

#### ks.c psmirnov_exact
function (q, sizes, z = NULL, alternative = c("two.sided", "less", 
    "greater"), lower.tail = TRUE, log.p = FALSE) 
{
    if (!is.null(z)) {
        z <- (diff(sort(z)) != 0)
        z <- if (any(z)) 
            c(0L, z, 1L)
    }
    two.sided <- (alternative == "two.sided")
    if (alternative == "less") 
        sizes <- c(sizes[2L], sizes[1L])
    p <- .Call(C_psmirnov_exact, q, sizes[1L], sizes[2L], z, 
        two.sided, lower.tail)
    if (log.p) 
        log(p)
    else p
}
<bytecode: 0x000001ed79962188>
<environment: namespace:stats>

#### psmirnov_asymp
function (q, sizes, alternative = c("two.sided", "less", "greater"), 
    lower.tail = TRUE, log.p = FALSE) 
{
    alternative <- match.arg(alternative)
    two.sided <- (alternative == "two.sided")
    n <- prod(sizes)/sum(sizes)
    if (two.sided) {
        ret <- .Call(C_pkolmogorov_two_limit, sqrt(n) * q, lower.tail, 
            tol = 1e-06)
        if (log.p) 
            ret <- log(ret)
        return(ret)
    }
    else {
        ret <- -expm1(-2 * n * q^2)
        if (log.p) {
            if (lower.tail) 
                log(ret)
            else log1p(-ret)
        }
        else {
            if (lower.tail) 
                ret
            else 1 - ret
        }
    }
}
<bytecode: 0x000001ed7994a118>
<environment: namespace:stats>

#### knnx.dist
function (data, query, k = 10, algorithm = c("kd_tree", "cover_tree", 
    "CR", "brute")) 
{
    get.knnx(data, query, k, algorithm)$nn.dist
}
<bytecode: 0x000001ed79a55a18>
<environment: namespace:FNN>

#### knn.dist
function (data, k = 10, algorithm = c("kd_tree", "cover_tree", 
    "CR", "brute")) 
{
    get.knn(data, k, algorithm)$nn.dist
}
<bytecode: 0x000001ed79a5ca80>
<environment: namespace:FNN>

#### svd
function (x, nu = min(n, p), nv = min(n, p), LINPACK = FALSE) 
{
    if (!missing(LINPACK)) 
        stop("the LINPACK argument has been defunct since R 3.1.0")
    x <- as.matrix(x)
    if (any(!is.finite(x))) 
        stop("infinite or missing values in 'x'")
    dx <- dim(x)
    n <- dx[1L]
    p <- dx[2L]
    if (!n || !p) 
        stop("a dimension is zero")
    La.res <- La.svd(x, nu, nv)
    res <- list(d = La.res$d)
    if (nu) 
        res$u <- La.res$u
    if (nv) 
        res$v <- if (is.complex(x)) 
            Conj(t(La.res$vt))
        else t(La.res$vt)
    res
}
<bytecode: 0x000001ed79a7a0a8>
<environment: namespace:base>

#### La.svd
function (x, nu = min(n, p), nv = min(n, p)) 
{
    if (!is.logical(x) && !is.numeric(x) && !is.complex(x)) 
        stop("argument to 'La.svd' must be numeric or complex")
    if (any(!is.finite(x))) 
        stop("infinite or missing values in 'x'")
    x <- as.matrix(x)
    n <- nrow(x)
    p <- ncol(x)
    if (!n || !p) 
        stop("a dimension is zero")
    zero <- if (is.complex(x)) 
        0 + (0+0i)
    else 0
    if (nu || nv) {
        np <- min(n, p)
        if (nu <= np && nv <= np) {
            jobu <- "S"
            u <- matrix(zero, n, np)
            vt <- matrix(zero, np, p)
            nu0 <- nv0 <- np
        }
        else {
            jobu <- "A"
            u <- matrix(zero, n, n)
            vt <- matrix(zero, p, p)
            nu0 <- n
            nv0 <- p
        }
    }
    else {
        jobu <- "N"
        u <- matrix(zero, 1L, 1L)
        vt <- matrix(zero, 1L, 1L)
    }
    res <- if (is.complex(x)) 
        .Internal(La_svd_cmplx(jobu, x, double(min(n, p)), u, 
            vt))
    else .Internal(La_svd(jobu, x, double(min(n, p)), u, vt))
    res <- res[c("d", if (nu) "u", if (nv) "vt")]
    if (nu && nu < nu0) 
        res$u <- res$u[, seq_len(min(n, nu)), drop = FALSE]
    if (nv && nv < nv0) 
        res$vt <- res$vt[seq_len(min(p, nv)), , drop = FALSE]
    res
}
<bytecode: 0x000001ed79a51a88>
<environment: namespace:base>
