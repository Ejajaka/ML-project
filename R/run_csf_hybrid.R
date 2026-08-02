# Run CSF using grf, extract tree splits, predict on test data
library(grf)

# Read training data
data <- read.csv("hybrid_input.csv")
X <- as.matrix(data[, grep("^X", names(data)), drop = FALSE])
# Handle column names that might have spaces
colnames(X) <- make.names(colnames(X))
time <- data$time
event <- data$event
treatment <- data$treatment

cat("Data loaded:", nrow(data), "rows,", ncol(X), "covariates\n")

# Fit Causal Survival Forest
csf <- causal_survival_forest(
  X = X, Y = time, W = treatment, D = event,
  num.trees = 500, honesty = TRUE, min.node.size = 5
)
cat("CSF fitted\n")

# Predictions on training data
cate_train <- predict(csf)$predictions
cat("Train CATE range:", round(range(cate_train), 4), "\n")

# Extract tree splits (limit to 100 trees)
all_splits <- data.frame()
for (i in 1:min(csf$num.trees, 100)) {
  tree <- get_tree(csf, i)
  
  extract_node <- function(node_id, conditions) {
    node <- tree$nodes[[node_id + 1]]
    if (node$is_leaf) {
      if (length(conditions) > 0) {
        samples <- node$samples + 1
        if (length(samples) >= 10) {
          all_splits <<- rbind(all_splits, data.frame(
            tree_id = i, node_id = node_id,
            rule = paste(conditions, collapse = " & "),
            n_samples = length(samples),
            cate_mean = mean(cate_train[samples]),
            stringsAsFactors = FALSE
          ))
        }
      }
      return()
    }
    var_name <- colnames(X)[node$split_variable + 1]
    split_val <- node$split_value
    
    left_cond <- paste0(var_name, " <= ", round(split_val, 4))
    extract_node(node$left_child, c(conditions, left_cond))
    
    right_cond <- paste0(var_name, " > ", round(split_val, 4))
    extract_node(node$right_child, c(conditions, right_cond))
  }
  
  extract_node(0, c())
}

cat("Extracted", nrow(all_splits), "rules\n")
write.csv(all_splits, "csf_rules.csv", row.names = FALSE)
write.csv(data.frame(cate_csf = cate_train), "csf_cate.csv", row.names = FALSE)

# Save the model for potential test predictions
saveRDS(csf, "csf_model.rds")

cat("Done. Files saved.\n")
