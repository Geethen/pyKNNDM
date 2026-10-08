"""Verbatim ports of R's ``hclust`` (Fortran HCLUST + HCASS2) and ``cutree``.

Source: R 4.5.0 ``src/library/stats/src/hclust.f`` and ``hclust-utils.c``.
Variable names follow the Fortran so the two can be compared line by line,
including quirks such as ``JJ`` not being reset between NN-list updates.
"""

from __future__ import annotations

import numpy as np

METHODS = ["ward.D", "single", "complete", "average", "mcquitty", "median", "centroid", "ward.D2"]
_INF = 1.0e300


def _method_index(method: str) -> int:
    if method == "ward":
        method = "ward.D"
    matches = [i for i, m in enumerate(METHODS) if m.startswith(method)]
    exact = [i for i, m in enumerate(METHODS) if m == method]
    if exact:
        return exact[0] + 1
    if len(matches) == 1:
        return matches[0] + 1
    if not matches:
        raise ValueError(f"invalid clustering method {method}")
    raise ValueError(f"ambiguous clustering method {method}")


def condensed_from_square(d: np.ndarray) -> np.ndarray:
    """``as.dist(m)``: lower triangle, column-major (= row-major upper triangle)."""
    n = d.shape[0]
    iu = np.triu_indices(n, k=1)
    # as.dist takes m[lower.tri(m)], i.e. m[j, i] for i < j in column order.
    return np.ascontiguousarray(d.T[iu], dtype=float)


def hclust(diss: np.ndarray, n: int, method: str = "complete"):
    """Run HCLUST + HCASS2. ``diss`` is the condensed ``dist`` vector.

    Returns ``(merge, height, order)`` exactly as ``stats::hclust``.
    """
    iopt = _method_index(method)
    diss = np.array(diss, dtype=float)  # working copy (Fortran overwrites it)
    ln = n * (n - 1) // 2
    if diss.shape[0] < ln:
        raise ValueError("dissimilarities of improper length")
    # 1-based helpers
    ia = [0] * (n + 1)
    ib = [0] * (n + 1)
    crit = [0.0] * (n + 1)
    membr = [0.0] + [1.0] * n
    nn = [0] * (n + 1)
    disnn = [0.0] * (n + 1)
    flag = [True] * (n + 1)

    def ioffst(i: int, j: int) -> int:  # returns 0-based index into diss
        return j + (i - 1) * n - (i * (i + 1)) // 2 - 1

    im = jj = jm = 0
    ncl = n
    if iopt == 8:
        diss = diss * diss
    d = diss.tolist()

    for i in range(1, n):
        dmin = _INF
        for j in range(i + 1, n + 1):
            v = d[ioffst(i, j)]
            if dmin > v:
                dmin = v
                jm = j
        nn[i] = jm
        disnn[i] = dmin

    is_ward = iopt == 1 or iopt == 8
    while True:
        dmin = _INF
        for i in range(1, n):
            if flag[i] and disnn[i] < dmin:
                dmin = disnn[i]
                im = i
                jm = nn[i]
        ncl -= 1
        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[n - ncl] = i2
        ib[n - ncl] = j2
        if iopt == 8:
            dmin = float(np.sqrt(dmin))
        crit[n - ncl] = dmin
        flag[j2] = False

        dmin = _INF
        d12 = d[ioffst(i2, j2)]
        for k in range(1, n + 1):
            if flag[k] and k != i2:
                ind1 = ioffst(i2, k) if i2 < k else ioffst(k, i2)
                ind2 = ioffst(j2, k) if j2 < k else ioffst(k, j2)
                if is_ward:
                    v = (membr[i2] + membr[k]) * d[ind1] + (membr[j2] + membr[k]) * d[ind2] - membr[k] * d12
                    v = v / (membr[i2] + membr[j2] + membr[k])
                elif iopt == 2:
                    v = min(d[ind1], d[ind2])
                elif iopt == 3:
                    v = max(d[ind1], d[ind2])
                elif iopt == 4:
                    v = (membr[i2] * d[ind1] + membr[j2] * d[ind2]) / (membr[i2] + membr[j2])
                elif iopt == 5:
                    v = (d[ind1] + d[ind2]) / 2
                elif iopt == 6:
                    v = ((d[ind1] + d[ind2]) - d12 / 2) / 2
                else:  # 7: centroid
                    v = (
                        membr[i2] * d[ind1] + membr[j2] * d[ind2] - membr[i2] * membr[j2] * d12 / (membr[i2] + membr[j2])
                    ) / (membr[i2] + membr[j2])
                d[ind1] = v
                if i2 < k:
                    if v < dmin:
                        dmin = v
                        jj = k
                else:
                    if v < disnn[k]:
                        disnn[k] = v
                        nn[k] = i2
        membr[i2] = membr[i2] + membr[j2]
        disnn[i2] = dmin
        nn[i2] = jj

        for i in range(1, n):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = _INF
                for j in range(i + 1, n + 1):
                    if flag[j]:
                        v = d[ioffst(i, j)]
                        if v < dmin:
                            dmin = v
                            jj = j
                nn[i] = jj
                disnn[i] = dmin
        if ncl <= 1:
            break

    merge, order = hcass2(n, ia, ib)
    height = np.array(crit[1:n], dtype=float)
    return merge, height, order


def hcass2(n: int, ia: list[int], ib: list[int]):
    """HCASS2: convert (ia, ib) to R's merge matrix and leaf order (1-based lists in)."""
    iia = list(ia)
    iib = list(ib)
    for i in range(1, n - 1):
        k = min(ia[i], ib[i])
        for j in range(i + 1, n):
            if ia[j] == k:
                iia[j] = -i
            if ib[j] == k:
                iib[j] = -i
    for i in range(1, n):
        iia[i] = -iia[i]
        iib[i] = -iib[i]
    for i in range(1, n):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            k1 = min(iia[i], iib[i])
            k2 = max(iia[i], iib[i])
            iia[i] = k1
            iib[i] = k2
    iorder = [0] * (n + 2)
    iorder[1] = iia[n - 1]
    iorder[2] = iib[n - 1]
    loc = 2
    for i in range(n - 2, 0, -1):
        for j in range(1, loc + 1):
            if iorder[j] == i:
                iorder[j] = iia[i]
                if j == loc:
                    loc += 1
                    iorder[loc] = iib[i]
                else:
                    loc += 1
                    for kk in range(loc, j + 1, -1):
                        iorder[kk] = iorder[kk - 1]
                    iorder[j + 1] = iib[i]
                break
    order = np.array([-iorder[i] for i in range(1, n + 1)], dtype=np.int64)
    merge = np.array([[iia[i], iib[i]] for i in range(1, n)], dtype=np.int64)
    return merge, order


def cutree(merge: np.ndarray, k: int) -> np.ndarray:
    """``cutree(tree, k)`` for a single k; 1-based cluster labels."""
    n = merge.shape[0] + 1
    if k < 1 or k > n:
        raise ValueError(f"elements of 'k' must be between 1 and {n}")
    if k == n:
        return np.arange(1, n + 1, dtype=np.int64)
    sing = [True] * (n + 1)
    m_nr = [0] * (n + 1)
    ans = [0] * n
    for step in range(1, n):
        m1 = int(merge[step - 1, 0])
        m2 = int(merge[step - 1, 1])
        if m1 < 0 and m2 < 0:
            m_nr[-m1] = m_nr[-m2] = step
            sing[-m1] = sing[-m2] = False
        elif m1 < 0 or m2 < 0:
            if m1 < 0:
                j = -m1
                m1 = m2
            else:
                j = -m2
            for l in range(1, n + 1):
                if m_nr[l] == m1:
                    m_nr[l] = step
            m_nr[j] = step
            sing[j] = False
        else:
            for l in range(1, n + 1):
                if m_nr[l] == m1 or m_nr[l] == m2:
                    m_nr[l] = step
        if k == n - step:
            z = [0] * (n + 1)
            nclust = 0
            for l in range(1, n + 1):
                if sing[l]:
                    nclust += 1
                    ans[l - 1] = nclust
                else:
                    if z[m_nr[l]] == 0:
                        nclust += 1
                        z[m_nr[l]] = nclust
                    ans[l - 1] = z[m_nr[l]]
            break
    return np.array(ans, dtype=np.int64)
