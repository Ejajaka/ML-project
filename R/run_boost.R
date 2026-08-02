# BOOSTED HYBRID: CSF rules + bootstrap stability + XGBoost
# New folder, no overwrites. Tests on: synthetic 100 reps + PBC + GBSG
library(grf); library(glmnet); library(survival); library(xgboost)
set.seed(42); options(warn=-1)

# ===================== HELPERS =====================
extract_rules <- function(csf, mt=20) {
  r <- list()
  for (t in 1:min(csf$num.trees, mt)) {
    tr <- get_tree(csf, t)
    walk <- function(n, co) {
      nd <- tr$nodes[[n]]
      if (isTRUE(nd$is_leaf)) { if (length(co)>=1) r[[length(r)+1]] <<- list(rule=paste(co,collapse=' & ')); return() }
      walk(nd$left_child, c(co, sprintf('%s<=%.3f', tr$columns[nd$split_variable], nd$split_value)))
      walk(nd$right_child, c(co, sprintf('%s>%.3f', tr$columns[nd$split_variable], nd$split_value)))
    }
    walk(1, c())
  }
  r[!duplicated(sapply(r, function(x) x$rule))]
}

apply_rules <- function(ru, X, cn, ms=10) {
  n <- nrow(X); m <- length(ru); if (m==0) return(list(R=matrix(0,n,0)))
  R <- matrix(0,n,m); v <- rep(T,m)
  for (j in 1:m) {
    ma <- rep(T,n)
    for (p in strsplit(ru[[j]]$rule,' & ')[[1]]) {
      if (grepl('<=',p)) { vv <- strsplit(p,'<=')[[1]]; vi <- which(cn==trimws(vv[1])); if (length(vi)) ma <- ma&(X[,vi]<=as.numeric(vv[2])) else { v[j]<-F; break } }
      if (grepl('>',p) && !grepl('<=',p)) { vv <- strsplit(p,'>')[[1]]; vi <- which(cn==trimws(vv[1])); if (length(vi)) ma <- ma&(X[,vi]>as.numeric(vv[2])) else { v[j]<-F; break } }
    }
    if (v[j]) R[,j] <- as.integer(ma)
  }
  R <- R[,v,drop=F]; k <- colSums(R)>=ms
  list(R=R[,k,drop=F])
}

# BOOTSTRAP STABLE RULES: run CSF on bootstrap samples, keep rules appearing > threshold
bootstrap_rules <- function(X, t, e, a, ts, n_boot=30, threshold=0.3, n_trees=50, n_extract=15) {
  all_rules <- list()
  for (b in 1:n_boot) {
    idx <- sample(length(t), replace=TRUE)
    csf <- tryCatch(causal_survival_forest(X[idx,,drop=F], t[idx], a[idx], as.logical(e[idx]),
                    num.trees=n_trees, horizon=ts), error=function(e) NULL)
    if (!is.null(csf)) {
      rr <- extract_rules(csf, n_extract)
      all_rules <- c(all_rules, rr)
    }
  }
  if (length(all_rules) == 0) return(list())
  # Count rule frequencies
  rule_strs <- sapply(all_rules, function(x) x$rule)
  freq <- table(rule_strs)
  stable <- names(freq)[freq / n_boot >= threshold]
  if (length(stable) == 0) return(list())
  lapply(stable, function(s) list(rule=s))
}

fit_lasso <- function(X, y, mr=15) {
  if (ncol(X)<2||sum(!is.na(y))<10) return(list(coef=rep(0,max(1,ncol(X))),nr=0,int=0))
  Xs <- scale(X); ym <- mean(y,na.rm=T); ys <- y-ym
  cv <- tryCatch(cv.glmnet(Xs,ys,alpha=1,nfolds=5,standardize=F), error=function(e) NULL)
  if (is.null(cv)) return(list(coef=rep(0,ncol(X)),nr=0,int=ym))
  for (i in length(cv$lambda):1) {
    cf <- as.vector(coef(cv,s=cv$lambda[i]))[-1]
    if (sum(abs(cf)>1e-5)<=mr && sum(abs(cf)>1e-5)>=1) {
      m <- glmnet(Xs,ys,alpha=1,lambda=cv$lambda[i],standardize=F)
      cf <- as.vector(coef(m))[-1]
      return(list(coef=cf,nr=sum(abs(cf)>1e-5),int=coef(m)[1]+ym))
    }
  }
  list(coef=rep(0,ncol(X)),nr=0,int=ym)
}

evaluate <- function(p,t) {
  v <- !is.na(p)&!is.na(t)&is.finite(p); if (sum(v)<10) return(rep(NA,5))
  p<-p[v]; t<-t[v]; c(mean(abs(p-t)), sqrt(mean((p-t)^2)), if(sd(p)>0&&sd(t)>0) cor(p,t,method='spearman') else 0, mean((p>0)==(t>0)))
}

# ===================== PART 1: SYNTHETIC 100 REPS =====================
cat("=== PART 1: SYNTHETIC 50 REPS ===\n\n")
N <- 1500; P <- 15; REPS <- 50
mnames <- c("Cox","CSF","Bo&Ding","Hybrid(Lasso)","BoostedHybrid(XGB)")

run_one <- function(X_tr, X_te, a_tr, a_te, t_tr, t_te, e_tr, e_te, cate_te, ts) {
  nr <- nrow(X_te); res <- rep(NA, 5)
  
  # Cox
  df_tr <- data.frame(X_tr[,1:10], time=t_tr, event=e_tr)
  df_te <- data.frame(X_te[,1:10], time=t_te, event=e_te)
  cf <- as.formula(paste('Surv(time,event)~',paste0('X',1:10,collapse='+')))
  c1 <- tryCatch({
    cph_t <- coxph(cf,data=df_tr[a_tr==1,]); cph_c <- coxph(cf,data=df_tr[a_tr==0,])
    sf_t <- summary(survfit(cph_t,newdata=df_te,se.fit=F),times=ts)$surv
    sf_c <- summary(survfit(cph_c,newdata=df_te,se.fit=F),times=ts)$surv
    if(is.null(sf_t)||is.null(sf_c)) rep(0,nr) else sf_t-sf_c
  }, error=function(e) rep(0,nr))
  
  # CSF
  csf <- tryCatch(causal_survival_forest(X_tr,t_tr,a_tr,as.logical(e_tr),num.trees=100,horizon=ts), error=function(e) NULL)
  if (!is.null(csf)) { c3 <- predict(csf,X_te)$predictions; cate_tr <- predict(csf)$predictions } else { c3 <- rep(0,nr); cate_tr <- rep(0,nrow(X_tr)) }
  
  # CSF + bootstrap stable rules
  bsr <- tryCatch(bootstrap_rules(X_tr, t_tr, e_tr, a_tr, ts, n_boot=10, threshold=0.3, n_trees=30, n_extract=10), error=function(e) list())
  
  # Bo & Ding: GBM + Lasso on rules
  gb <- tryCatch(gbm(cate_tr~.,data=data.frame(X_tr),n.trees=60,interaction.depth=3,distribution='gaussian'), error=function(e) NULL)
  n4 <- 0; c4 <- rep(0,nr)
  if(!is.null(gb)) {
    gr <- extract_rules_from_gbm(gb, X_tr, 15)
    if(length(gr)>0) { ap <- apply_rules(gr, rbind(X_tr,X_te), colnames(X)); if(ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr); c4 <- l$int + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l$coef; c4 <- as.vector(c4); n4 <- l$nr } }
  }
  
  # Hybrid: CSF rules + Lasso
  n5 <- 0; c5 <- rep(0,nr)
  if(!is.null(csf) && length(bsr)>0) {
    cr <- extract_rules(csf, 15)
    if(length(cr)>0) { ap <- apply_rules(cr, rbind(X_tr,X_te), colnames(X)); if(ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr); c5 <- l$int + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l$coef; c5 <- as.vector(c5); n5 <- l$nr } }
  }
  
  # Boosted Hybrid: bootstrap-stable CSF rules + XGBoost
  n6 <- 0; c6 <- rep(0,nr)
  if (length(bsr) > 0) {
    ap <- apply_rules(bsr, rbind(X_tr, X_te), colnames(X))
    if (ncol(ap$R) > 0) {
      n6 <- ncol(ap$R)
      dtrain <- xgb.DMatrix(ap$R[1:nrow(X_tr),,drop=F], label=cate_tr)
      dtest <- xgb.DMatrix(ap$R[-(1:nrow(X_tr)),,drop=F])
      xgb_model <- tryCatch(xgb.train(params=list(max_depth=2, eta=0.1, nthread=1, silent=1),
               data=dtrain, nrounds=50, verbose=0), error=function(e) NULL)
      if (!is.null(xgb_model)) c6 <- predict(xgb_model, dtest)
    }
  }
  
  c(evaluate(c1,cate_te)[1], evaluate(c3,cate_te)[1], evaluate(c4,cate_te)[1], evaluate(c5,cate_te)[1], evaluate(c6,cate_te)[1])
}

# GBM rules helper (need separate because R's scoping)
extract_rules_from_gbm <- function(m, X, nt) {
  r <- list()
  for (t in 1:min(m$n.trees, nt)) {
    ti <- pretty.gbm.tree(m, t); if (nrow(ti)<=1) next
    walk <- function(n, co) {
      row <- ti[n+1,]
      if (row$SplitVar==-1) { if (length(co)>=1) r[[length(r)+1]] <<- list(rule=paste(co,collapse=' & ')); return() }
      walk(row$LeftNode, c(co, sprintf('%s<=%.3f', colnames(X)[row$SplitVar+1], row$SplitCodePred)))
      walk(row$RightNode, c(co, sprintf('%s>%.3f', colnames(X)[row$SplitVar+1], row$SplitCodePred)))
    }
    walk(0, c())
  }
  r[!duplicated(sapply(r, function(x) x$rule))]
}

# Actually run synthetic
synth_mae <- matrix(NA, REPS, 5)
for (rep in 1:REPS) {
  set.seed(rep*42)
  X <- matrix(rnorm(N*P),N,P); colnames(X) <- paste0('X',1:P)
  b <- X[,1]>0&X[,2]>0; h <- X[,1]<=0&X[,2]<=0; tau <- rep(0,N); tau[b]<-0.25; tau[h]<--0.10
  trt <- rbinom(N,1,0.5); mu <- 0.5*sin(X[,1])+0.3*abs(X[,2])+0.2*X[,3]
  to <- ifelse(trt==1, exp(mu+tau+log(-log(runif(N)))), exp(mu+log(-log(runif(N)))))
  cen <- rexp(N,1/quantile(to,.75)); tf <- pmin(to,cen); ev <- as.integer(to<=cen); ts <- median(to)
  co <- exp(-exp(log(ts)-mu)) - exp(-exp(log(ts)-mu-tau))
  set.seed(rep*42+1); ii <- sample(N,.7*N)
  X_tr<-X[ii,,drop=F]; X_te<-X[-ii,,drop=F]; a_tr<-trt[ii]; a_te<-trt[-ii]
  t_tr<-tf[ii]; t_te<-tf[-ii]; e_tr<-ev[ii]; e_te<-ev[-ii]; cate_te<-co[-ii]
  
  synth_mae[rep,] <- run_one(X_tr, X_te, a_tr, a_te, t_tr, t_te, e_tr, e_te, cate_te, ts)
  if (rep%%10==0) cat(sprintf('  Synth %d/%d\n', rep, REPS))
}

cat('\nSynthetic MAE (50 reps):\n')
cat(sprintf('%-25s %s\n','Method','MAE (SE)'))
cat(strrep('-',45),'\n')
for (i in 1:5) {
  v <- synth_mae[!is.na(synth_mae[,i]),i]
  cat(sprintf('%-25s %.4f (%.4f)\n', mnames[i], mean(v), sd(v)/sqrt(length(v))))
}

cat('\n=== DONE ===\n')
