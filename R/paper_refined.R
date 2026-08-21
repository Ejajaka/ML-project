# REFINED PUBLICATION PIPELINE
# Fixes: (1) Sparsity via max-rules Lasso (2) CATE clipping (3) Bootstrap stability
library(grf); library(glmnet); library(survival); library(gbm)

set.seed(42); options(warn=-1)

# ================ CORE FUNCTIONS ================

# Extract rules from CSF trees
extract_csf_rules <- function(csf, max_trees=50) {
  all_rules <- list()
  for (t in 1:min(csf$num.trees, max_trees)) {
    tree <- get_tree(csf, t)
    walk <- function(nid, conds) {
      nd <- tree$nodes[[nid]]
      if (isTRUE(nd$is_leaf)) {
        if (length(conds) >= 1) 
          all_rules[[length(all_rules)+1]] <<- list(rule=paste(conds,collapse=" & "))
        return()
      }
      vn <- tree$columns[nd$split_variable]; sv <- nd$split_value
      walk(nd$left_child, c(conds, sprintf("%s <= %.4f", vn, sv)))
      walk(nd$right_child, c(conds, sprintf("%s > %.4f", vn, sv)))
    }
    walk(1, c())
  }
  all_rules[!duplicated(sapply(all_rules, `[[`, "rule"))]
}

# Extract rules from GBM trees
extract_gbm_rules <- function(gbm_model, X, n_trees=50) {
  rules_list <- list()
  for (t in 1:min(gbm_model$n.trees, n_trees)) {
    ti <- pretty.gbm.tree(gbm_model, t)
    if (nrow(ti) <= 1) next
    walk <- function(nid, conds) {
      row <- ti[nid+1,]
      if (row$SplitVar == -1) {
        if (length(conds) >= 1) rules_list[[length(rules_list)+1]] <<- list(rule=paste(conds,collapse=" & "))
        return()
      }
      vn <- colnames(X)[row$SplitVar+1]; sv <- row$SplitCodePred
      walk(row$LeftNode, c(conds, sprintf("%s <= %.4f", vn, sv)))
      walk(row$RightNode, c(conds, sprintf("%s > %.4f", vn, sv)))
    }
    walk(0, c())
  }
  rules_list[!duplicated(sapply(rules_list, `[[`, "rule"))]
}

# Apply rules to build matrix
apply_rules <- function(rules, X, col_names, min_support=10) {
  n <- nrow(X); m <- length(rules)
  if (m == 0) return(list(R=matrix(0,n,0), rules=list()))
  R <- matrix(0, n, m)
  valid <- rep(TRUE, m)
  for (j in 1:m) {
    mask <- rep(TRUE, n)
    for (part in strsplit(rules[[j]]$rule, " & ")[[1]]) {
      if (grepl(" <= ", part)) {
        vv <- strsplit(part, " <= ")[[1]]; vi <- which(col_names == vv[1])
        if (length(vi)) mask <- mask & (X[,vi] <= as.numeric(vv[2])) else { valid[j] <- FALSE; break }
      } else if (grepl(" > ", part)) {
        vv <- strsplit(part, " > ")[[1]]; vi <- which(col_names == vv[1])
        if (length(vi)) mask <- mask & (X[,vi] > as.numeric(vv[2])) else { valid[j] <- FALSE; break }
      }
    }
    if (valid[j]) R[,j] <- as.integer(mask)
  }
  R <- R[, valid, drop=FALSE]; rules <- rules[valid]
  keep <- colSums(R) >= min_support
  list(R=R[, keep, drop=FALSE], rules=rules[keep])
}

# FIT SPARSE LASSO with max_rules constraint
fit_sparse_lasso <- function(X, y, max_rules=15, nfolds=5, min_rules=1) {
  if (ncol(X) < 2 || sum(!is.na(y)) < 10) return(list(coef=rep(0,max(1,ncol(X))), n_rules=0, intercept=0, selected=integer()))
  
  X_s <- scale(X); y_m <- mean(y, na.rm=TRUE); y_s <- y - y_m
  cv <- tryCatch(cv.glmnet(X_s, y_s, alpha=1, nfolds=nfolds, standardize=FALSE), error=function(e) NULL)
  if (is.null(cv)) return(list(coef=rep(0,ncol(X)), n_rules=0, intercept=y_m, selected=integer()))
  
  # Start from lambda.min (most variables) and go toward lambda.1se (fewer variables)
  # Pick the one with <= max_rules but >= min_rules non-zero coefficients
  best_i <- length(cv$lambda)  # default: most regularized
  for (i in length(cv$lambda):1) {  # From smallest lambda to largest
    coefs <- as.vector(coef(cv, s=cv$lambda[i]))[-1]
    nz <- sum(abs(coefs) > 1e-5)
    if (nz <= max_rules && nz >= min_rules) {
      best_i <- i
      break
    }
    if (nz >= min_rules) best_i <- i  # keep last one with at least min_rules
  }
  
  m <- glmnet(X_s, y_s, alpha=1, lambda=cv$lambda[best_i], standardize=FALSE)
  cf <- as.vector(coef(m))[-1]
  list(coef=cf, n_rules=sum(abs(cf)>1e-5), intercept=coef(m)[1]+y_m, 
       selected=which(abs(cf)>1e-5), lambda=cv$lambda[best_i])
}

# BOOTSTRAP STABILITY SELECTION
bootstrap_select <- function(R, y, n_boot=20, threshold=0.35) {
  n <- nrow(R); p <- ncol(R)
  if (p == 0 || sum(!is.na(y)) < 10) return(integer())
  sel <- rep(0, p)
  y_obs <- y[!is.na(y)]
  R_obs <- R[!is.na(y), , drop=FALSE]
  for (b in 1:n_boot) {
    idx <- sample(nrow(R_obs), replace=TRUE)
    cv <- tryCatch(cv.glmnet(R_obs[idx,,drop=F], y_obs[idx], alpha=1, nfolds=3), error=function(e) NULL)
    if (!is.null(cv)) {
      sel_v <- abs(as.vector(coef(cv,s=cv$lambda.1se))[-1]) > 1e-5
      sel[sel_v] <- sel[sel_v] + 1
    }
  }
  which(sel / n_boot >= threshold)
}

# CLIP to reasonable range
clip_cate <- function(x) pmax(pmin(x, 1), -1)

# EVALUATION
evaluate <- function(pred, true) {
  v <- !is.na(pred) & !is.na(true) & is.finite(pred)
  if (sum(v) < 10) return(c(NA,NA,NA,NA,NA))
  p <- pred[v]; t <- true[v]
  c(mean(p-t), mean(abs(p-t)), sqrt(mean((p-t)^2)),
    if(sd(p)>0 && sd(t)>0) cor(p,t,method="spearman") else 0,
    mean((p>0)==(t>0)))
}

# ================ PART 1: SYNTHETIC SIMULATION ================
N_REPS <- 20; N <- 1500; P <- 15
cat("\n", strrep("=",70), "\n", sep="")
cat(sprintf("SYNTHETIC SIMULATION (%d reps)\n", N_REPS))
cat(strrep("=",70), "\n", sep="")

synth_res <- list()

for (rep in 1:N_REPS) {
  set.seed(rep*42)
  X <- matrix(rnorm(N*P), N, P); colnames(X) <- paste0("X",1:P)
  benefit <- X[,1]>0 & X[,2]>0; harm <- X[,1]<=0 & X[,2]<=0
  tau <- rep(0,N); tau[benefit] <- 0.25; tau[harm] <- -0.10
  trt <- rbinom(N,1,0.5)
  mu0 <- 0.5*sin(X[,1])+0.3*abs(X[,2])+0.2*X[,3]
  te <- exp(mu0+log(-log(runif(N))))
  tt <- exp(mu0+tau+log(-log(runif(N))))
  to <- ifelse(trt==1,tt,te)
  cen <- rexp(N,1/quantile(to,.75))
  tf <- pmin(to,cen); ev <- as.integer(to<=cen); ts <- median(to)
  cate_or <- exp(-exp(log(ts)-mu0)) - exp(-exp(log(ts)-mu0-tau))
  
  ii <- sample(N,.7*N)
  X_tr <- X[ii,,drop=F]; X_te <- X[-ii,,drop=F]
  a_tr <- trt[ii]; a_te <- trt[-ii]
  t_tr <- tf[ii]; t_te <- tf[-ii]
  e_tr <- ev[ii]; e_te <- ev[-ii]
  cate_te <- cate_or[-ii]
  
  # 1. Cox
  df_tr <- data.frame(X_tr[,1:10], time=t_tr, event=e_tr)
  df_te <- data.frame(X_te[,1:10], time=t_te, event=e_te)
  cf <- as.formula(paste("Surv(time,event) ~", paste0("X",1:min(P,10),collapse="+")))
  cph_t <- tryCatch(coxph(cf, data=df_tr[a_tr==1,]), error=function(e) NULL)
  cph_c <- tryCatch(coxph(cf, data=df_tr[a_tr==0,]), error=function(e) NULL)
  if (!is.null(cph_t) && !is.null(cph_c)) {
    sf_t <- survfit(cph_t, newdata=df_te, se.fit=F)
    sf_c <- survfit(cph_c, newdata=df_te, se.fit=F)
    c1 <- sapply(1:nrow(df_te), function(i) stepfun(sf_t$time,c(1,sf_t$surv[,i]))(ts)-stepfun(sf_c$time,c(1,sf_c$surv[,i]))(ts))
  } else c1 <- rep(0,nrow(df_te))
  
  # 2. CSF
  csf <- tryCatch(causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr), num.trees=100, horizon=ts), error=function(e) NULL)
  if (!is.null(csf)) { c3 <- clip_cate(predict(csf,X_te)$predictions); cate_tr <- clip_cate(predict(csf)$predictions) }
  else { c3 <- rep(0,nrow(df_te)); cate_tr <- rep(0,nrow(df_tr)) }
  
  # 3. Bo & Ding (GB + Lasso)
  gb <- tryCatch(gbm(cate_tr ~ ., data=data.frame(X_tr), n.trees=100, interaction.depth=3, distribution="gaussian"), error=function(e) NULL)
  n4 <- 0; c4 <- rep(0, nrow(df_te))
  if (!is.null(gb)) {
    gr <- extract_gbm_rules(gb, X_tr, 40)
    if (length(gr) > 0) {
      ap <- apply_rules(gr, rbind(X_tr,X_te), colnames(X), 10)
      if (ncol(ap$R) > 0) {
          l <- fit_sparse_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr, max_rules=15, min_rules=2)
          c4 <- l$intercept + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l$coef
        c4 <- clip_cate(as.vector(c4)); n4 <- l$n_rules
      }
    }
  }
  
  # 4. Hybrid (CSF + Lasso)
  n5 <- 0; c5 <- rep(0, nrow(df_te))
  if (!is.null(csf)) {
    cr <- extract_csf_rules(csf, 50)
    if (length(cr) > 0) {
      ap <- apply_rules(cr, rbind(X_tr,X_te), colnames(X), 10)
      if (ncol(ap$R) > 0) {
          l <- fit_sparse_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr, max_rules=15, min_rules=2)
          c5 <- l$intercept + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l$coef
        c5 <- clip_cate(as.vector(c5)); n5 <- l$n_rules
      }
    }
  }
  
  synth_res[[rep]] <- rbind(
    c(1, evaluate(c1,cate_te), 0),
    c(2, evaluate(c3,cate_te), 0),
    c(3, evaluate(c4,cate_te), n4),
    c(4, evaluate(c5,cate_te), n5)
  )
  if (rep %% 10 == 0) cat(sprintf("  Rep %d/%d\n", rep, N_REPS))
}

all_r <- do.call(rbind, synth_res)
mnames <- c("Cox T-learner","CSF (grf)","Bo & Ding (GB+Lasso)","Hybrid (CSF+Lasso)")
cat("\nSynthetic Results:\n", strrep("-",70), "\n", sep="")
for (m in 1:4) {
  s <- all_r[all_r[,1]==m,,drop=F]; s <- s[!is.na(s[,3]),,drop=F]
  if (nrow(s)==0) next
  cat(sprintf("%-25s MAE=%.4f | Acc=%.3f | Rules=%.0f\n",
      mnames[m], mean(s[,3]), mean(s[,6]), mean(s[,7])))
}

# ================ PART 2: REAL DATA - PBC ================
cat("\n", strrep("=",70), "\n", sep="")
cat("REAL DATA: PBC\n")
cat(strrep("=",70), "\n", sep="")

run_real_analysis <- function(name, df, outcome_vars, treatment_col, covars, boot_threshold=0.15, boot_n=15) {
  X <- as.matrix(df[, covars]); colnames(X) <- covars
  Y <- df[[outcome_vars[1]]]; E <- df[[outcome_vars[2]]]
  A <- df[[treatment_col]]
  n <- nrow(X); ts <- median(Y[E==1])
  cat(sprintf("  %s: N=%d, p=%d, events=%d\n", name, n, length(covars), sum(E)))
  
  n_tr <- floor(.7*n); ii <- sample(n, n_tr)
  X_tr <- X[ii,,drop=F]; X_te <- X[-ii,,drop=F]
  t_tr <- Y[ii]; t_te <- Y[-ii]
  e_tr <- E[ii]; e_te <- E[-ii]
  a_tr <- A[ii]; a_te <- A[-ii]
  
  # Cox baseline
  df_tr <- data.frame(X_tr, time=t_tr, event=e_tr)
  df_te <- data.frame(X_te, time=t_te, event=e_te)
  cox_form <- as.formula(paste("Surv(time,event) ~", paste(covars, collapse="+")))
  cph_t <- tryCatch(coxph(cox_form, data=df_tr[a_tr==1,]), error=function(e) NULL)
  cph_c <- tryCatch(coxph(cox_form, data=df_tr[a_tr==0,]), error=function(e) NULL)
  if (!is.null(cph_t) && !is.null(cph_c)) {
    sf_t <- survfit(cph_t, newdata=df_te, se.fit=F)
    sf_c <- survfit(cph_c, newdata=df_te, se.fit=F)
    cox_cate <- sapply(1:nrow(df_te), function(i) stepfun(sf_t$time,c(1,sf_t$surv[,i]))(ts)-stepfun(sf_c$time,c(1,sf_c$surv[,i]))(ts))
    cat(sprintf("  Cox CATE range: [%.3f, %.3f]\n", min(cox_cate), max(cox_cate)))
  }
  
  # CSF
  csf <- causal_survival_forest(X_tr, t_tr, a_tr, as.logical(e_tr), num.trees=200, horizon=ts)
  cate_tr <- clip_cate(predict(csf)$predictions)
  
  # Hybrid with bootstrap stability
  cr <- extract_csf_rules(csf, 100)
  if (length(cr) > 0) {
    ap <- apply_rules(cr, rbind(X_tr,X_te), covars, 10)
    if (ncol(ap$R) > 0) {
      bsel <- bootstrap_select(ap$R[1:nrow(X_tr),,drop=F], cate_tr, n_boot=boot_n, threshold=boot_threshold)
      if (length(bsel) > 0) {
        l <- fit_sparse_lasso(ap$R[1:nrow(X_tr), bsel, drop=F], cate_tr, max_rules=10, min_rules=2)
        cat(sprintf("  Selected %d stable rules\n", l$n_rules))
        if (l$n_rules > 0) {
          cat("  Final rules:\n")
          for (j in seq_along(l$selected)) {
            orig_idx <- bsel[l$selected[j]]
            coef_val <- l$coef[l$selected[j]]
            rule_str <- ap$rules[[orig_idx]]$rule
            cat(sprintf("    [%+.4f] %s\n", coef_val, substr(rule_str, 1, 70)))
          }
        }
      } else cat("  No stable rules found (lower threshold?)\n")
    } else cat("  No rules with sufficient support\n")
  } else cat("  No rules extracted\n")
}

# PBC Data
pbc <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv")
pbc <- pbc[!is.na(pbc$trt),]
pbc$treatment <- as.integer(pbc$trt==1); pbc$event <- as.integer(pbc$status==2)
covars_pbc <- c("age","bili","albumin","copper","alk.phos","ast","trig","platelet","protime","stage")
for (c in covars_pbc) { pbc[[c]][is.na(pbc[[c]])] <- median(pbc[[c]], na.rm=TRUE) }
run_real_analysis("PBC", pbc, c("time","event"), "treatment", covars_pbc, boot_threshold=0.15, boot_n=15)

# ================ PART 3: REAL DATA - GBSG ================
cat("\n", strrep("=",70), "\n", sep="")
cat("REAL DATA: GBSG (German Breast Cancer)\n")
cat(strrep("=",70), "\n", sep="")

gbsg <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/gbsg.csv")
gbsg$treatment <- gbsg$hormon; gbsg$event <- gbsg$status
covars_gbsg <- c("age","meno","size","grade","nodes","pgr","er")
for (c in covars_gbsg) gbsg[[c]][is.na(gbsg[[c]])] <- median(gbsg[[c]], na.rm=TRUE)
run_real_analysis("GBSG", gbsg, c("rfstime","event"), "treatment", covars_gbsg, boot_threshold=0.1, boot_n=20)

cat("\n", strrep("=",70), "\n", sep="")
cat("DONE\n")
cat(strrep("=",70), "\n", sep="")
