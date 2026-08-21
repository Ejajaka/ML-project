# FINAL PUBLICATION PIPELINE
# Cox T-learner | CSF (grf) | RF (Bo & Ding proxy) | Hybrid (CSF+Lasso)
# Simulation: 3 subgroups (benefit, harm, none) + noise variables
library(grf); library(glmnet); library(survival); library(randomForest)

set.seed(42)
N_REPS <- 100; N <- 2000; P <- 20
results_list <- list()
cat(sprintf("Starting: %d reps, n=%d, p=%d (10 signal + 10 noise)\n", N_REPS, N, P))

for (rep in 1:N_REPS) {
  set.seed(rep * 42)
  
  # Generate data
  X <- matrix(rnorm(N * P), N, P)
  colnames(X) <- paste0("X", 1:P)
  
  # 3 subgroups with DIFFERENT treatment effects (benefit / harm / none)
  # Benefit: X1 > 0 & X2 > 0 -> CATE = +0.20 (20% better survival)
  # Harm:    X1 <= 0 & X2 <= 0 -> CATE = -0.12 (12% worse survival)
  # None:    rest -> CATE = 0
  benefit <- X[,1] > 0 & X[,2] > 0
  harm <- X[,1] <= 0 & X[,2] <= 0
  
  # True treatment effect on survival probability scale
  tau_true <- rep(0, N)
  tau_true[benefit] <- 0.20
  tau_true[harm] <- -0.12
  
  treatment <- rbinom(N, 1, 0.5)
  
  # Generate survival outcomes with known CATE
  # log(U) ~ Gumbel(0,1) gives exponential-like survival
  # S0(t) = exp(-exp(log(t) - mu0))
  # S1(t) = exp(-exp(log(t) - mu0 - tau))
  # CATE at t = S1(t) - S0(t) ≈ tau * S0(t) * (1 - S0(t)) * log(S0(t)) / ...
  # For simplicity: directly add tau to the linear predictor
  mu0 <- X[,1] * 0.3 + X[,2] * 0.2 + X[,3] * 0.1 + sin(X[,4]) * 0.5
  eps <- log(-log(runif(N)))  # Gumbel
  
  t_control <- exp(mu0 + eps)
  t_treated <- exp(mu0 + tau_true + eps)  # tau_true shifts log-time
  t_obs <- ifelse(treatment == 1, t_treated, t_control)
  
  # Censoring (mild, ~25%)
  censor <- rexp(N, 1/quantile(t_obs, 0.75))
  t_final <- pmin(t_obs, censor)
  event <- as.integer(t_obs <= censor)
  
  # Oracle CATE at median survival time
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
  
  # --- 1. Cox T-learner (restricted to first 10 vars for stability) ---
  df_tr <- data.frame(X_tr[,1:10], time=t_tr, event=e_tr)
  df_te <- data.frame(X_te[,1:10], time=t_te, event=e_te)
  cph_t <- tryCatch(coxph(Surv(time,event) ~ ., data=df_tr[a_tr==1,]), error=function(e) NULL)
  cph_c <- tryCatch(coxph(Surv(time,event) ~ ., data=df_tr[a_tr==0,]), error=function(e) NULL)
  if (!is.null(cph_t) && !is.null(cph_c)) {
    sf_t <- survfit(cph_t, newdata=df_te, se.fit=F)
    sf_c <- survfit(cph_c, newdata=df_te, se.fit=F)
    cate_cox <- sapply(1:nrow(df_te), function(i) {
      st <- stepfun(sf_t$time, c(1, sf_t$surv[,i]))
      sc <- stepfun(sf_c$time, c(1, sf_c$surv[,i]))
      st(t_star) - sc(t_star)
    })
  } else { cate_cox <- rep(0, nrow(df_te)) }
  
  # --- 2. CSF (grf) ---
  csf <- tryCatch(causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr),
                  num.trees=1000, honesty=TRUE, min.node.size=5,
                  horizon=t_star), error=function(e) NULL)
  if (!is.null(csf)) {
    cate_csf <- predict(csf, X_te)$predictions
    cate_tr <- predict(csf)$predictions
  } else { cate_csf <- rep(0, nrow(df_te)); cate_tr <- rep(0, nrow(df_tr)) }
  
  # --- 3. RF (Bo & Ding proxy) ---
  rf <- randomForest(cate_tr ~ ., data=data.frame(X_tr), ntree=200, maxnodes=20)
  cate_rf <- predict(rf, data.frame(X_te))
  
  # --- 4. Hybrid: CSF top variables + Lasso ---
  # grf::variable_importance returns a VECTOR for a single-output causal
  # survival forest (NOT a matrix). Indexing [,1] always raised an error and
  # silently fell back to rep(0, P), making top_vars constant 1:5 and defeating
  # the "hybrid via variable importance" mechanism. Handle both shapes.
  vi <- tryCatch({
    v <- variable_importance(csf)
    if (is.matrix(v)) v[,1] else as.numeric(v)
  }, error=function(e) rep(0, P))
  if (length(vi) != P) vi <- rep(0, P)
  top_vars <- order(-vi)[1:min(5, P)]
  X_tr_top <- X_tr[, top_vars, drop=FALSE]
  X_te_top <- X_te[, top_vars, drop=FALSE]
  
  make_feat <- function(X) cbind(X, X^2, X[,1]*X[,2], X[,1]*X[,3])
  cv_lasso <- tryCatch(cv.glmnet(make_feat(X_tr_top), cate_tr, alpha=1, nfolds=5), 
                       error=function(e) NULL)
  if (!is.null(cv_lasso)) {
    cate_hyb <- as.vector(predict(cv_lasso, make_feat(X_te_top), s="lambda.1se"))
    n_rules <- sum(abs(as.vector(coef(cv_lasso, s="lambda.1se"))[-1]) > 1e-4)
  } else { cate_hyb <- rep(0, nrow(df_te)); n_rules <- 0 }
  
  # --- Evaluate ---
  eval_fn <- function(pred) {
    v <- !is.na(pred) & !is.na(cate_te) & is.finite(pred)
    if (sum(v) < 10) return(c(NA,NA,NA,NA,NA,NA))
    p <- pred[v]; t <- cate_te[v]
    bias <- mean(p-t)
    mae <- mean(abs(p-t))
    rmse <- sqrt(mean((p-t)^2))
    corr <- if(sd(p)>0 && sd(t)>0) cor(p,t,method="spearman") else 0
    # Recommend treatment if CATE > 0
    acc <- mean((p > 0) == (t > 0))
    # Correlation between predicted and true rank
    c(bias, mae, rmse, corr, acc)
  }
  
  results_list[[rep]] <- rbind(
    c(1, eval_fn(cate_cox), 0),
    c(2, eval_fn(cate_csf), 0),
    c(3, eval_fn(cate_rf), 0),
    c(4, eval_fn(cate_hyb), n_rules)
  )
  
  if (rep %% 20 == 0) cat(sprintf("  Rep %d/%d\n", rep, N_REPS))
}

# Aggregate
all_res <- do.call(rbind, results_list)
methods <- c("Cox T-learner", "CSF (grf)", "RF (Bo & Ding)", "Hybrid (CSF+Lasso)")

cat("\n", strrep("=", 75), "\n", sep="")
cat("PUBLICATION RESULTS (", N_REPS, " reps, n=", N, ", p=", P, ")\n", sep="")
cat(strrep("=", 75), "\n", sep="")

# Compute means and SEs
cat(sprintf("\n%-25s %-12s %-12s %-12s %-10s %-8s\n",
    "Method", "MAE (SE)", "RMSE (SE)", "Spearman (SE)", "Acc (SE)", "Rules"))
cat(strrep("-", 75), "\n", sep="")
for (m in 1:4) {
  sub <- all_res[all_res[,1]==m, , drop=FALSE]
  sub <- sub[!is.na(sub[,3]), , drop=FALSE]
  if (nrow(sub) == 0) next
  
  mae_m <- mean(sub[,3]); mae_s <- sd(sub[,3])/sqrt(nrow(sub))
  rms_m <- mean(sub[,4]); rms_s <- sd(sub[,4])/sqrt(nrow(sub))
  sp_m <- mean(sub[,5]);  sp_s <- sd(sub[,5])/sqrt(nrow(sub))
  acc_m <- mean(sub[,6]); acc_s <- sd(sub[,6])/sqrt(nrow(sub))
  rules_m <- mean(sub[,7])
  
  cat(sprintf("%-25s %.4f (%.4f) %.4f (%.4f) %.3f (%.3f) %.3f (%.3f) %.0f\n",
      methods[m], mae_m, mae_s, rms_m, rms_s, sp_m, sp_s, acc_m, acc_s, rules_m))
}

# Highlight best method
all_means <- sapply(1:4, function(m) {
  sub <- all_res[all_res[,1]==m,,drop=F]
  colMeans(sub[,3:7], na.rm=TRUE)
})
best_mae <- which.min(all_means[1,])
best_acc <- which.max(all_means[4,])
cat("\nBest MAE:", methods[best_mae], "| Best Accuracy:", methods[best_acc])

write.csv(all_res, "paper_final_results.csv", row.names=FALSE)
cat("\n\nSaved to paper_final_results.csv\nDone!\n")
