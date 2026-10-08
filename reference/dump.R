library(CAST); ns <- asNamespace("CAST"); fns <- ls(ns)
keep <- grep("knndm|sample2sample|sample2prediction|distclust|nndm", fns, value=TRUE)
print(keep)
for (f in keep) { cat("\n#### ", f, "\n", sep=""); print(get(f, ns)) }
