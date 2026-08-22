# Regression test: grf causal survival forest tree traversal (1-based nodes).
#
# Background: run_csf_hybrid.R originally walked the tree with a 0-based root
# (`nodes[[node_id + 1]]`) which is OFF BY ONE -- grf exposes nodes as a 1-based
# R list, with left_child / right_child being 1-based indices. On deeper trees
# this hit "subscript out of bounds". This test locks in the corrected
# 1-based convention so a future refactor cannot silently reintroduce it.
#
# Run: Rscript R/test_grf_traversal.R   (requires grf)

.libPaths(c('C:/Users/sanje/R/library', .libPaths()))
library(grf)
set.seed(42)

n <- 300; p <- 4
X <- matrix(rnorm(n * p), n, p)
time <- rexp(n, 0.5)
treatment <- rbinom(n, 1, 0.5)
event <- rbinom(n, 1, 0.6)

csf <- causal_survival_forest(X, time, treatment, event == 1,
                              num.trees = 5, honesty = FALSE, horizon = 1.0)
tr <- get_tree(csf, 1)

# --- Convention check: 1-based walk must visit every node without OOB ---
walk <- function(node_id, depth = 0) {
  stopifnot(depth < 1000)                       # no infinite recursion
  stopifnot(node_id >= 1, node_id <= length(tr$nodes))  # no out-of-bounds
  nd <- tr$nodes[[node_id]]
  if (isTRUE(nd$is_leaf)) return()
  walk(nd$left_child, depth + 1)
  walk(nd$right_child, depth + 1)
}

walk(1, 0)  # root is nodes[[1]]

# --- Sanity: root is an internal node (deep enough tree) ---
stopifnot(!isTRUE(tr$nodes[[1]]$is_leaf))

# --- Negative control: the OLD 0-based walk must FAIL (locks the bug) ---
old_walk_oob <- tryCatch({
  walk0 <- function(node_id, depth = 0) {
    stopifnot(depth < 1000)
    idx <- node_id + 1L                      # old (buggy) indexing
    stopifnot(idx >= 1, idx <= length(tr$nodes))
    nd <- tr$nodes[[idx]]
    if (isTRUE(nd$is_leaf)) return()
    walk0(nd$left_child, depth + 1)
    walk0(nd$right_child, depth + 1)
  }
  walk0(0, 0)                                 # old 0-based root
  FALSE                                        # reached end without error
}, error = function(e) TRUE)

cat("1-based traversal: PASS (visited all nodes, no out-of-bounds)\n")
cat("old 0-based walk correctly fails (bug locked):", old_walk_oob, "\n")
stopifnot(old_walk_oob)                        # must be TRUE for the test to pass
cat("TEST PASSED\n")