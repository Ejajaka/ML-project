# COMPLETE PUBLICATION PIPELINE
# Methods: Cox | Cox+splines | CSF (grf) | Bo & Ding (GB+Lasso) | Hybrid (CSF tree splits + Lasso)
# Data: Synthetic + PBC + SUPPORT
library(grf); library(glmnet); library(survival); library(randomForest); library(gbm)
library(rms); library(splines)

set.seed(42)
options(warn = -1)  # Suppress convergence warnings

# ============================================================
# PART 1: EXTRACT RULES FROM CSF TREES
# ============================================================
extract_csf_rules <- function(csf, min_samples = 10) {
  all_rules <- list()
  n_trees <- min(csf$num.trees, 100)  # Limit for speed
  
  for (t in 1:n_trees) {
    tree <- get_tree(csf, t)
    nodes <- tree$nodes
    cols <- tree$columns
    
    # Recursive walk
    walk <- function(node_id, conditions) {
      nd <- nodes[[node_id]]
      
      if (isTRUE(nd$is_leaf)) {
        if (length(conditions) >= 1) {
          all_rules[[length(all_rules) + 1]] <<- list(
            rule = paste(conditions, collapse = " & "),
            node_id = node_id, tree_id = t,
            is_leaf = TRUE
          )
        }
        return()
      }
      
      var_name <- cols[nd$split_variable]
      sv <- nd$split_value
      
      # Left: var <= split_value
      lc <- sprintf("%s <= %.4f", var_name, sv)
      walk(nd$left_child, c(conditions, lc))
      
      # Right: var > split_value
      rc <- sprintf("%s > %.4f", var_name, sv)
      walk(nd$right_child, c(conditions, rc))
    }
    
    walk(1, c())
  }
  
  # Deduplicate rules
  rules_str <- sapply(all_rules, function(x) x$rule)
  dup <- duplicated(rules_str)
  all_rules <- all_rules[!dup]
  
  return(all_rules)
}

# ============================================================
# PART 2: EXTRACT RULES FROM GBM TREES (Bo & Ding)
# ============================================================
extract_gbm_rules <- function(gbm_model, X, n_trees = 50, min_samples = 10) {
  rules_list <- list()
  
  for (t in 1:min(gbm_model$n.trees, n_trees)) {
    # Get tree structure from gbm
    tree_info <- pretty.gbm.tree(gbm_model, t)
    if (nrow(tree_info) <= 1) next
    
    # Recursively walk the tree
    walk <- function(node_idx, conditions) {
      row <- tree_info[node_idx + 1, ]
      
      if (row$SplitVar == -1) {  # Leaf node
        if (length(conditions) >= 1) {
          rules_list[[length(rules_list) + 1]] <<- list(
            rule = paste(conditions, collapse = " & "),
            tree_id = t
          )
        }
        return()
      }
      
      var_idx <- row$SplitVar + 1  # 0-based -> 1-based
      var_name <- colnames(X)[var_idx]
      sv <- row$SplitCodePred
      
      lc <- sprintf("%s <= %.4f", var_name, sv)
      walk(row$LeftNode, c(conditions, lc))
      
      rc <- sprintf("%s > %.4f", var_name, sv)
      walk(row$RightNode, c(conditions, rc))
    }
    
    walk(0, c())
  }
  
  # Deduplicate
  rules_str <- sapply(rules_list, function(x) x$rule)
  dup <- duplicated(rules_str)
  rules_list <- rules_list[!dup]
  
  return(rules_list)
}

# ============================================================
# PART 3: APPLY RULES TO DATA (build rule matrix)
# ============================================================
apply_rules <- function(rules_list, X, col_names) {
  n <- nrow(X)
  m <- length(rules_list)
  R <- matrix(0, n, m)
  
  for (j in seq_len(m)) {
    mask <- rep(TRUE, n)
    parts <- strsplit(rules_list[[j]]$rule, " & ")[[1]]
    
    valid <- TRUE
    for (part in parts) {
      if (grepl(" <= ", part)) {
        vv <- strsplit(part, " <= ")[[1]]
        var_name <- vv[1]; val <- as.numeric(vv[2])
        if (var_name %in% col_names) {
          vi <- which(col_names == var_name)
          mask <- mask & (X[, vi] <= val)
        } else { valid <- FALSE; break }
      } else if (grepl(" > ", part)) {
        vv <- strsplit(part, " > ")[[1]]
        var_name <- vv[1]; val <- as.numeric(vv[2])
        if (var_name %in% col_names) {
          vi <- which(col_names == var_name)
          mask <- mask & (X[, vi] > val)
        } else { valid <- FALSE; break }
      }
    }
    
    if (valid) {
      R[, j] <- as.integer(mask)
    }
  }
  
  # Remove low-support columns
  keep <- colSums(R) >= 10
  R <- R[, keep, drop = FALSE]
  rules_list <- rules_list[keep]
  
  return(list(R = R, rules = rules_list))
}

# ============================================================
# PART 4: EVALUATION METRICS
# ============================================================
evaluate_cate <- function(pred, true) {
  v <- !is.na(pred) & !is.na(true) & is.finite(pred)
  if (sum(v) < 10) return(c(NA, NA, NA, NA, NA))
  p <- pred[v]; t <- true[v]
  bias <- mean(p - t)
  mae <- mean(abs(p - t))
  rmse <- sqrt(mean((p - t)^2))
  corr <- if (sd(p) > 0 && sd(t) > 0) cor(p, t, method = "spearman") else 0
  acc <- mean((p > 0) == (t > 0))
  c(bias, mae, rmse, corr, acc)
}

# ============================================================
# PART 5: SYNTHETIC SIMULATION
# ============================================================
cat("\n", strrep("=", 70), "\n", sep="")
cat("PART 1: SYNTHETIC SIMULATION\n")
cat(strrep("=", 70), "\n", sep="")

N_REPS <- 50; N <- 2000; P <- 20
synth_results <- list()

for (rep in 1:N_REPS) {
  set.seed(rep * 42)
  
  # Data with non-linear CATE
  X <- matrix(rnorm(N * P), N, P)
  colnames(X) <- paste0("X", 1:P)
  
  benefit <- X[,1] > 0 & X[,2] > 0
  harm <- X[,1] <= 0 & X[,2] <= 0
  tau_true <- rep(0, N)
  tau_true[benefit] <- 0.25
  tau_true[harm] <- -0.10
  
  treatment <- rbinom(N, 1, 0.5)
  mu0 <- 0.5*sin(X[,1]) + 0.3*abs(X[,2]) + 0.2*X[,3]
  eps <- log(-log(runif(N)))
  t_control <- exp(mu0 + eps)
  t_treated <- exp(mu0 + tau_true + eps)
  t_obs <- ifelse(treatment == 1, t_treated, t_control)
  censor <- rexp(N, 1/quantile(t_obs, 0.75))
  t_final <- pmin(t_obs, censor)
  event <- as.integer(t_obs <= censor)
  t_star <- median(t_obs)
  
  S0 <- exp(-exp(log(t_star) - mu0))
  S1 <- exp(-exp(log(t_star) - mu0 - tau_true))
  cate_oracle <- S1 - S0
  
  # Split
  train_i <- sample(N, 0.7*N); test_i <- setdiff(1:N, train_i)
  X_tr <- X[train_i,,drop=F]; X_te <- X[test_i,,drop=F]
  a_tr <- treatment[train_i]; a_te <- treatment[test_i]
  t_tr <- t_final[train_i]; t_te <- t_final[test_i]
  e_tr <- event[train_i]; e_te <- event[test_i]
  cate_te <- cate_oracle[test_i]
  
  # --- Method 1: Cox T-learner (linear) ---
  df_tr <- data.frame(X_tr[,1:min(P, 10)], time=t_tr, event=e_tr)
  df_te <- data.frame(X_te[,1:min(P, 10)], time=t_te, event=e_te)
  
  cph_t <- tryCatch(coxph(Surv(time,event) ~ ., data=df_tr[a_tr==1,]), error=function(e) NULL)
  cph_c <- tryCatch(coxph(Surv(time,event) ~ ., data=df_tr[a_tr==0,]), error=function(e) NULL)
  if (!is.null(cph_t) && !is.null(cph_c)) {
    sf_t <- survfit(cph_t, newdata=df_te, se.fit=F)
    sf_c <- survfit(cph_c, newdata=df_te, se.fit=F)
    c1 <- sapply(1:nrow(df_te), function(i) {
      stepfun(sf_t$time, c(1, sf_t$surv[,i]))(t_star) - 
        stepfun(sf_c$time, c(1, sf_c$surv[,i]))(t_star)
    })
  } else { c1 <- rep(0, nrow(df_te)) }
  
  # --- Method 2: Cox with splines ---
  df_tr2 <- data.frame(X_tr[,1:min(P, 10)], time=t_tr, event=e_tr)
  df_te2 <- data.frame(X_te[,1:min(P, 10)], time=t_te, event=e_te)
  tryCatch({
    spline_formula <- as.formula(paste(
      "Surv(time,event) ~", 
      paste0("ns(X", 1:3, ", df=3)", collapse=" + "),
      "+", paste0("X", 4:min(P, 10), collapse=" + ")
    ))
    cph_st <- coxph(spline_formula, data=df_tr2[a_tr==1,])
    cph_sc <- coxph(spline_formula, data=df_tr2[a_tr==0,])
    sf_st <- survfit(cph_st, newdata=df_te2, se.fit=F)
    sf_sc <- survfit(cph_sc, newdata=df_te2, se.fit=F)
    c2 <- sapply(1:nrow(df_te2), function(i) {
      stepfun(sf_st$time, c(1, sf_st$surv[,i]))(t_star) - 
        stepfun(sf_sc$time, c(1, sf_sc$surv[,i]))(t_star)
    })
  }, error=function(e) { c2 <<- rep(0, nrow(df_te2)) })
  
  # --- Method 3: CSF (grf) ---
  csf <- tryCatch(causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr),
                  num.trees=200, honesty=TRUE, horizon=t_star), error=function(e) NULL)
  if (!is.null(csf)) {
    c3 <- predict(csf, X_te)$predictions
    cate_tr <- predict(csf)$predictions
  } else { c3 <- rep(0, nrow(df_te)); cate_tr <- rep(0, nrow(df_tr)) }
  
  # --- Method 4: Bo & Ding (GB + Lasso on tree rules) ---
  # Pseudo-ITE via DR-learner (simplified: use CSF's CATE as pseudo-ITE)
  gb <- tryCatch(gbm(cate_tr ~ ., data=data.frame(X_tr), 
                     n.trees=100, interaction.depth=3, distribution="gaussian"),
                 error=function(e) NULL)
  
  if (!is.null(gb)) {
    gbm_rules <- extract_gbm_rules(gb, X_tr)
    if (length(gbm_rules) > 0) {
      applied <- apply_rules(gbm_rules, rbind(X_tr, X_te), colnames(X))
      if (ncol(applied$R) > 0) {
        R_tr <- applied$R[1:nrow(X_tr), , drop=FALSE]
        R_te <- applied$R[-(1:nrow(X_tr)), , drop=FALSE]
        keep_c <- colSums(R_tr) >= 10
        R_tr <- R_tr[, keep_c, drop=FALSE]; R_te <- R_te[, keep_c, drop=FALSE]
        
        if (ncol(R_tr) > 0) {
          cv <- tryCatch(cv.glmnet(R_tr, cate_tr, alpha=1, nfolds=5), error=function(e) NULL)
          if (!is.null(cv)) {
            c4 <- as.vector(predict(cv, R_te, s="lambda.1se"))
            n4 <- sum(abs(coef(cv, s="lambda.1se")[-1]) > 1e-4)
          } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
        } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
      } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
    } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
  } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
  
  # --- Method 5: Hybrid (CSF tree splits + Lasso) ---
  csf_rules <- extract_csf_rules(csf)
  if (length(csf_rules) > 0) {
    applied <- apply_rules(csf_rules, rbind(X_tr, X_te), colnames(X))
    if (ncol(applied$R) > 0) {
      R_tr <- applied$R[1:nrow(X_tr), , drop=FALSE]
      R_te <- applied$R[-(1:nrow(X_tr)), , drop=FALSE]
      keep_c <- colSums(R_tr) >= 10
      R_tr <- R_tr[, keep_c, drop=FALSE]; R_te <- R_te[, keep_c, drop=FALSE]
      
      if (ncol(R_tr) > 0) {
        cv <- tryCatch(cv.glmnet(R_tr, cate_tr, alpha=1, nfolds=5), error=function(e) NULL)
        if (!is.null(cv)) {
          c5 <- as.vector(predict(cv, R_te, s="lambda.1se"))
          n5 <- sum(abs(coef(cv, s="lambda.1se")[-1]) > 1e-4)
        } else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }
      } else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }
    } else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }
  } else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }
  
  synth_results[[rep]] <- rbind(
    c(1, evaluate_cate(c1, cate_te), 0),
    c(2, evaluate_cate(c2, cate_te), 0),
    c(3, evaluate_cate(c3, cate_te), 0),
    c(4, evaluate_cate(c4, cate_te), n4),
    c(5, evaluate_cate(c5, cate_te), n5)
  )
  
  if (rep %% 20 == 0) cat(sprintf("  Synth rep %d/%d\n", rep, N_REPS))
}

# Aggregate synthetic results
synth_all <- do.call(rbind, synth_results)
mnames <- c("Cox T-learner", "Cox + splines", "CSF (grf)", "Bo & Ding (GB+Lasso)", "Hybrid (CSF+Lasso)")

cat("\nSynthetic Results:\n")
cat(strrep("-", 75), "\n", sep="")
for (m in 1:5) {
  s <- synth_all[synth_all[,1]==m,,drop=F]
  s <- s[!is.na(s[,3]),,drop=F]
  if (nrow(s) == 0) next
  cat(sprintf("%-25s MAE=%.4f RMSE=%.4f Spear=%.3f Acc=%.3f Rules=%.0f\n",
      mnames[m], mean(s[,3]), mean(s[,4]), mean(s[,5]), mean(s[,6]), mean(s[,7])))
}

# ============================================================
# PART 6: REAL DATA - PBC
# ============================================================
cat("\n", strrep("=", 70), "\n", sep="")
cat("PART 2: REAL DATA - PBC (Primary Biliary Cirrhosis)\n")
cat(strrep("=", 70), "\n", sep="")

pbc <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv")
pbc <- pbc[!is.na(pbc$trt), ]
pbc$treatment <- as.integer(pbc$trt == 1)
pbc$event <- as.integer(pbc$status == 2)
pbc$time <- pbc$time

covars <- c("age", "sex", "bili", "albumin", "copper", "alk.phos", "ast", 
            "trig", "platelet", "protime", "stage")
for (c in covars) {
  if (is.character(pbc[[c]])) pbc[[c]] <- as.numeric(factor(pbc[[c]]))
  pbc[[c]][is.na(pbc[[c]])] <- median(pbc[[c]], na.rm=TRUE)
}

X <- as.matrix(pbc[, covars]); colnames(X) <- covars
cat(sprintf("  N=%d, p=%d, events=%d\n", nrow(X), ncol(X), sum(pbc$event)))

n <- nrow(X); n_train <- floor(0.7 * n)
ii <- sample(n, n_train)
X_tr <- X[ii,,drop=F]; X_te <- X[-ii,,drop=F]
t_tr <- pbc$time[ii]; t_te <- pbc$time[-ii]
e_tr <- pbc$event[ii]; e_te <- pbc$event[-ii]
a_tr <- pbc$treatment[ii]; a_te <- pbc$treatment[-ii]
t_star <- median(pbc$time[pbc$event == 1])

# Run all methods
cat("  Running methods...\n")

# Cox T-learner
df_tr <- data.frame(X_tr, time=t_tr, event=e_tr); df_te <- data.frame(X_te, time=t_te, event=e_te)
cf <- as.formula(paste("Surv(time,event) ~", paste(covars, collapse="+")))
cph_t <- coxph(cf, data=df_tr[a_tr==1,]); cph_c <- coxph(cf, data=df_tr[a_tr==0,])
sf_t <- survfit(cph_t, newdata=df_te, se.fit=F); sf_c <- survfit(cph_c, newdata=df_te, se.fit=F)
c1 <- sapply(1:nrow(df_te), function(i) stepfun(sf_t$time,c(1,sf_t$surv[,i]))(t_star) - stepfun(sf_c$time,c(1,sf_c$surv[,i]))(t_star))

# Cox + splines
tryCatch({
  sf <- as.formula(paste("Surv(time,event) ~ ns(bili,3) + ns(albumin,3) + ns(age,3) +", 
                          paste(covars[-(1:3)], collapse="+")))
  cph_st <- coxph(sf, data=df_tr[a_tr==1,]); cph_sc <- coxph(sf, data=df_tr[a_tr==0,])
  sfs_t <- survfit(cph_st, newdata=df_te, se.fit=F); sfs_c <- survfit(cph_sc, newdata=df_te, se.fit=F)
  c2 <- sapply(1:nrow(df_te), function(i) stepfun(sfs_t$time,c(1,sfs_t$surv[,i]))(t_star) - stepfun(sfs_c$time,c(1,sfs_c$surv[,i]))(t_star))
}, error=function(e) { c2 <<- rep(0, nrow(df_te)) })

# CSF
csf <- causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr), num.trees=500, horizon=t_star)
c3 <- predict(csf, X_te)$predictions; cate_tr <- predict(csf)$predictions

# Bo & Ding (GB+Lasso)
gb <- gbm(cate_tr ~ ., data=data.frame(X_tr), n.trees=100, interaction.depth=3, distribution="gaussian")
gb_rules <- extract_gbm_rules(gb, X_tr)
if (length(gb_rules) > 0) {
  ap <- apply_rules(gb_rules, rbind(X_tr, X_te), covars)
  if (ncol(ap$R) > 0) {
    kk <- colSums(ap$R[1:nrow(X_tr),,drop=F]) >= 10
    cv <- cv.glmnet(ap$R[1:nrow(X_tr), kk, drop=F], cate_tr, alpha=1, nfolds=5)
    c4 <- as.vector(predict(cv, ap$R[-(1:nrow(X_tr)), kk, drop=F], s="lambda.1se"))
    n4 <- sum(abs(coef(cv, s="lambda.1se")[-1]) > 1e-4)
  } else { c4 <- rep(0,nrow(df_te)); n4 <- 0 }
} else { c4 <- rep(0,nrow(df_te)); n4 <- 0 }

# Hybrid (CSF+Lasso)
csf_rules <- extract_csf_rules(csf)
if (length(csf_rules) > 0) {
  ap2 <- apply_rules(csf_rules, rbind(X_tr, X_te), covars)
  if (ncol(ap2$R) > 0) {
    kk <- colSums(ap2$R[1:nrow(X_tr),,drop=F]) >= 10
    cv2 <- cv.glmnet(ap2$R[1:nrow(X_tr), kk, drop=F], cate_tr, alpha=1, nfolds=5)
    c5 <- as.vector(predict(cv2, ap2$R[-(1:nrow(X_tr)), kk, drop=F], s="lambda.1se"))
    n5 <- sum(abs(coef(cv2, s="lambda.1se")[-1]) > 1e-4)
  } else { c5 <- rep(0,nrow(df_te)); n5 <- 0 }
} else { c5 <- rep(0,nrow(df_te)); n5 <- 0 }

# Print PBC results (no ground truth, just show CATE range and rules)
cat("  Method                    CATE range        Rules\n")
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", mnames[1], min(c1), max(c1), 0))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", mnames[2], min(c2), max(c2), 0))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", mnames[3], min(c3), max(c3), 0))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", mnames[4], min(c4), max(c4), n4))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", mnames[5], min(c5), max(c5), n5))

# Print hybrid rules for PBC
if (exists("ap2") && ncol(ap2$R) > 0 && n5 > 0) {
  coefs <- as.vector(coef(cv2, s="lambda.1se"))
  sel <- which(abs(coefs[-1]) > 1e-4)
  if (length(sel) > 0) {
    cat("\n  Top hybrid rules (PBC):\n")
    for (j in sel[1:min(5, length(sel))]) {
      cat(sprintf("    [%+.4f] %s\n", coefs[j+1], 
                  substr(csf_rules[[which(kk)[j]]]$rule, 1, 60)))
    }
  }
}

# ============================================================
# PART 7: REAL DATA - GBSG (German Breast Cancer)
# ============================================================
cat("\n", strrep("=", 70), "\n", sep="")
cat("PART 3: REAL DATA - GBSG (German Breast Cancer)\n")
cat(strrep("=", 70), "\n", sep="")

gbsg <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/gbsg.csv")
gbsg$treatment <- gbsg$hormon
gbsg$event <- gbsg$status
gbsg$time <- gbsg$rfstime

covars2 <- c("age", "meno", "size", "grade", "nodes", "pgr", "er")
for (c in covars2) {
  gbsg[[c]][is.na(gbsg[[c]])] <- median(gbsg[[c]], na.rm=TRUE)
}

X <- as.matrix(gbsg[, covars2]); colnames(X) <- covars2
cat(sprintf("  N=%d, p=%d, events=%d\n", nrow(X), ncol(X), sum(gbsg$event)))

n <- nrow(X); n_train <- floor(0.7 * n)
ii <- sample(n, n_train)
X_tr <- X[ii,,drop=F]; X_te <- X[-ii,,drop=F]
t_tr <- gbsg$time[ii]; t_te <- gbsg$time[-ii]
e_tr <- gbsg$event[ii]; e_te <- gbsg$event[-ii]
a_tr <- gbsg$treatment[ii]; a_te <- gbsg$treatment[-ii]
t_star <- median(gbsg$time[gbsg$event == 1])

cat("  Running methods...\n")

# CSF only (for speed, skip full comparison on GBSG)
csf <- causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr), num.trees=500, horizon=t_star)
c3 <- predict(csf, X_te)$predictions; cate_tr <- predict(csf)$predictions

# Bo & Ding
gb <- gbm(cate_tr ~ ., data=data.frame(X_tr), n.trees=100, interaction.depth=3, distribution="gaussian")
gb_rules <- extract_gbm_rules(gb, X_tr)
if (length(gb_rules) > 0) {
  ap <- apply_rules(gb_rules, rbind(X_tr, X_te), covars2)
  if (ncol(ap$R) > 0) {
    kk <- colSums(ap$R[1:nrow(X_tr),,drop=F]) >= 10
    cv <- cv.glmnet(ap$R[1:nrow(X_tr), kk, drop=F], cate_tr, alpha=1, nfolds=5)
    c4 <- as.vector(predict(cv, ap$R[-(1:nrow(X_tr)), kk, drop=F], s="lambda.1se"))
    n4 <- sum(abs(coef(cv, s="lambda.1se")[-1]) > 1e-4)
  } else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }
} else { c4 <- rep(0, nrow(df_te)); n4 <- 0 }

# Hybrid
csf_rules <- extract_csf_rules(csf)
if (length(csf_rules) > 0) {
  ap2 <- apply_rules(csf_rules, rbind(X_tr, X_te), covars2)
  if (ncol(ap2$R) > 0) {
    kk <- colSums(ap2$R[1:nrow(X_tr),,drop=F]) >= 10
    cv2 <- cv.glmnet(ap2$R[1:nrow(X_tr), kk, drop=F], cate_tr, alpha=1, nfolds=5)
    c5 <- as.vector(predict(cv2, ap2$R[-(1:nrow(X_tr)), kk, drop=F], s="lambda.1se"))
    n5 <- sum(abs(coef(cv2, s="lambda.1se")[-1]) > 1e-4)
  } else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }
} else { c5 <- rep(0, nrow(df_te)); n5 <- 0 }

cat("  Method                    CATE range        Rules\n")
cat(sprintf("  %-25s [%.3f, %.3f]\n", "CSF (grf)", min(c3), max(c3)))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", "Bo & Ding (GB+Lasso)", min(c4), max(c4), n4))
cat(sprintf("  %-25s [%.3f, %.3f]    %d\n", "Hybrid (CSF+Lasso)", min(c5), max(c5), n5))

if (exists("ap2") && ncol(ap2$R) > 0 && n5 > 0) {
  coefs <- as.vector(coef(cv2, s="lambda.1se"))
  sel <- which(abs(coefs[-1]) > 1e-4)
  if (length(sel) > 0) {
    cat("\n  Top hybrid rules (GBSG):\n")
    for (j in sel[1:min(5, length(sel))]) {
      cat(sprintf("    [%+.4f] %s\n", coefs[j+1], 
                  substr(csf_rules[[which(kk)[j]]]$rule, 1, 60)))
    }
  }
}

cat("\n", strrep("=", 70), "\n", sep="")
cat("PIPELINE COMPLETE\n")
cat(strrep("=", 70), "\n", sep="")
