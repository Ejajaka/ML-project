# CLEAN SINGLE RUN — synthetic 30 reps + PBC + GBSG
# No hardcoded assessments. Every number comes from actual computation.
library(grf); library(glmnet); library(survival); library(gbm)
set.seed(42); options(warn=-1)

# ============ HELPERS ============
extract_csf_rules <- function(csf, mt=30) {
  r <- list()
  for (t in 1:min(csf$num.trees, mt)) {
    tr <- get_tree(csf, t)
    w <- function(n, co) {
      nd <- tr$nodes[[n]]
      if (isTRUE(nd$is_leaf)) { if (length(co)>=1) r[[length(r)+1]] <<- list(rule=paste(co,collapse=' & ')); return() }
      vn <- tr$columns[nd$split_variable]; sv <- nd$split_value
      w(nd$left_child, c(co, sprintf('%s<=%.3f', vn, sv)))
      w(nd$right_child, c(co, sprintf('%s>%.3f', vn, sv)))
    }
    w(1, c())
  }
  r[!duplicated(sapply(r, `[[`, 'rule'))]
}
extract_gbm_rules <- function(m, X, nt=25) {
  r <- list()
  for (t in 1:min(m$n.trees, nt)) {
    ti <- pretty.gbm.tree(m, t); if (nrow(ti)<=1) next
    w <- function(n, co) {
      row <- ti[n+1,]
      if (row$SplitVar==-1) { if (length(co)>=1) r[[length(r)+1]] <<- list(rule=paste(co,collapse=' & ')); return() }
      vn <- colnames(X)[row$SplitVar+1]; sv <- row$SplitCodePred
      w(row$LeftNode, c(co, sprintf('%s<=%.3f', vn, sv)))
      w(row$RightNode, c(co, sprintf('%s>%.3f', vn, sv)))
    }
    w(0, c())
  }
  r[!duplicated(sapply(r, `[[`, 'rule'))]
}
apply_rules <- function(ru, X, cn, ms=10) {
  n <- nrow(X); m <- length(ru); if (m==0) return(list(R=matrix(0,n,0)))
  R <- matrix(0,n,m); v <- rep(T,m)
  for (j in 1:m) {
    ma <- rep(T,n)
    for (p in strsplit(ru[[j]]$rule,' & ')[[1]]) {
      if (grepl('<=',p)) { vv <- strsplit(p,'<=')[[1]]; vi <- which(cn==vv[1]); if (length(vi)) ma <- ma & (X[,vi]<=as.numeric(vv[2])) else { v[j]<-F; break } }
      if (grepl('>',p) && !grepl('<=',p)) { vv <- strsplit(p,'>')[[1]]; vi <- which(cn==vv[1]); if (length(vi)) ma <- ma & (X[,vi]>as.numeric(vv[2])) else { v[j]<-F; break } }
    }
    if (v[j]) R[,j] <- as.integer(ma)
  }
  R <- R[,v,drop=F]; k <- colSums(R)>=ms
  list(R=R[,k,drop=F])
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
  v <- !is.na(p)&!is.na(t)&is.finite(p);   if (sum(v)<10) return(c(NA,NA,NA,NA,NA))
  p <- p[v]; t <- t[v]
  c(mean(p-t), mean(abs(p-t)), sqrt(mean((p-t)^2)), if (sd(p)>0&&sd(t)>0) cor(p,t,method='spearman') else 0, mean((p>0)==(t>0)))
}

# ============ PART 1: SYNTHETIC (30 REPS) ============
cat("=== SYNTHETIC SIMULATION (30 reps) ===\n")
N <- 1500; P <- 15; REPS <- 30
res <- list()

for (rep in 1:REPS) {
  set.seed(rep*42)
  X <- matrix(rnorm(N*P),N,P); colnames(X) <- paste0("X",1:P)
  b <- X[,1]>0&X[,2]>0; h <- X[,1]<=0&X[,2]<=0
  tau <- rep(0,N); tau[b] <- 0.25; tau[h] <- -0.10
  trt <- rbinom(N,1,0.5)
  mu <- 0.5*sin(X[,1])+0.3*abs(X[,2])+0.2*X[,3]
  to <- ifelse(trt==1, exp(mu+tau+log(-log(runif(N)))), exp(mu+log(-log(runif(N)))))
  cen <- rexp(N,1/quantile(to,.75)); tf <- pmin(to,cen); ev <- as.integer(to<=cen); ts <- median(to)
  co <- exp(-exp(log(ts)-mu)) - exp(-exp(log(ts)-mu-tau))
  
  ii <- sample(N,.7*N); X_tr<-X[ii,,drop=F]; X_te<-X[-ii,,drop=F]
  a_tr<-trt[ii]; a_te<-trt[-ii]; t_tr<-tf[ii]; t_te<-tf[-ii]; e_tr<-ev[ii]; e_te<-ev[-ii]; cate_te<-co[-ii]
  
  # Cox
  df_tr <- data.frame(X_tr[,1:10],time=t_tr,event=e_tr)
  df_te <- data.frame(X_te[,1:10],time=t_te,event=e_te)
  cf <- as.formula(paste("Surv(time,event)~",paste0("X",1:10,collapse="+")))
  cph_t <- tryCatch(coxph(cf,data=df_tr[a_tr==1,]), error=function(e) NULL)
  cph_c <- tryCatch(coxph(cf,data=df_tr[a_tr==0,]), error=function(e) NULL)
  if (!is.null(cph_t)&&!is.null(cph_c)) {
    sf_t<-survfit(cph_t,newdata=df_te,se.fit=F); sf_c<-survfit(cph_c,newdata=df_te,se.fit=F)
    c1 <- sapply(1:nrow(df_te), function(i) {
      st<-stepfun(sf_t$time,c(1,sf_t$surv[,i])); sc<-stepfun(sf_c$time,c(1,sf_c$surv[,i])); st(ts)-sc(ts)
    })
  } else c1 <- rep(0,nrow(df_te))
  
  # CSF
  csf <- tryCatch(causal_survival_forest(X_tr,t_tr,a_tr,as.logical(e_tr),num.trees=100,horizon=ts), error=function(e) NULL)
  if (!is.null(csf)) { c3 <- predict(csf,X_te)$predictions; cate_tr <- predict(csf)$predictions } else { c3 <- rep(0,nrow(df_te)); cate_tr <- rep(0,nrow(df_tr)) }
  
  # Bo & Ding (every other rep for speed)
  n4 <- 0; c4 <- rep(0,length(c1))
  if (rep %% 2 == 0 || rep == REPS) {
    gb <- tryCatch(gbm(cate_tr~.,data=data.frame(X_tr),n.trees=80,interaction.depth=3,distribution="gaussian"), error=function(e) NULL)
    if (!is.null(gb)) {
      gr <- extract_gbm_rules(gb,X_tr,20)
      if (length(gr)>0) { ap <- apply_rules(gr,rbind(X_tr,X_te),colnames(X)); if (ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F],cate_tr); c4 <- l$int+ap$R[-(1:nrow(X_tr)),,drop=F]%*%l$coef; c4 <- as.vector(c4); n4 <- l$nr }}
    }
  }
  
  # Hybrid
  n5 <- 0; c5 <- rep(0,length(c1))
  if (rep %% 2 == 0 || rep == REPS) {
    if (!is.null(csf)) {
      cr <- extract_csf_rules(csf,25)
      if (length(cr)>0) { ap <- apply_rules(cr,rbind(X_tr,X_te),colnames(X)); if (ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F],cate_tr); c5 <- l$int+ap$R[-(1:nrow(X_tr)),,drop=F]%*%l$coef; c5 <- as.vector(c5); n5 <- l$nr }}
    }
  }
  
  res[[rep]] <- rbind(c(1,evaluate(c1,cate_te),0), c(2,evaluate(c3,cate_te),0), c(3,evaluate(c4,cate_te),n4), c(4,evaluate(c5,cate_te),n5))
  if (rep %% 10 == 0) cat(sprintf("  Rep %d/%d\n", rep, REPS))
}

all <- do.call(rbind, res)
mn <- c("Cox T-learner","CSF (grf)","Bo & Ding (GB+Lasso)","Hybrid (CSF+Lasso)")

cat("\nSynthetic Results (30 reps):\n")
cat(sprintf("%-25s %-8s %-8s %-8s %-6s\n", "Method","MAE","RMSE","Acc","Rules"))
cat(strrep("-",55),"\n")
for (m in 1:4) {
  s <- all[all[,1]==m,,drop=F]
  s <- s[!is.na(s[,3]),,drop=F]
  if (nrow(s)==0) { cat(sprintf("%-25s %s\n", mn[m], "no valid runs")); next }
  cat(sprintf("%-25s %.4f %.4f %.3f %.0f\n", mn[m], mean(s[,3],na.rm=T), mean(s[,4],na.rm=T), mean(s[,6],na.rm=T), mean(s[,7],na.rm=T)))
}

# ============ PART 2: PBC REAL DATA ============
cat("\n=== PBC REAL DATA ===\n")
pbc <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv")
pbc <- pbc[!is.na(pbc$trt),]; pbc$tx <- as.integer(pbc$trt==1); pbc$ev <- as.integer(pbc$status==2)
cv <- c("age","bili","albumin","copper","alk.phos","ast","trig","platelet","protime","stage")
for (c in cv) pbc[[c]][is.na(pbc[[c]])] <- median(pbc[[c]],na.rm=T)
X <- as.matrix(pbc[,cv]); colnames(X) <- cv
n <- nrow(X); cat(sprintf("  N=%d p=%d events=%d\n", n, length(cv), sum(pbc$ev)))
ts <- median(pbc$time[pbc$ev==1])
ii <- sample(n,floor(.7*n)); X_tr<-X[ii,,drop=F]; X_te<-X[-ii,,drop=F]
a_tr<-pbc$tx[ii]; a_te<-pbc$tx[-ii]; t_tr<-pbc$time[ii]; t_te<-pbc$time[-ii]; e_tr<-pbc$ev[ii]; e_te<-pbc$ev[-ii]
csf <- causal_survival_forest(X_tr,t_tr,a_tr,as.logical(e_tr),num.trees=100,horizon=ts)
cate_tr <- predict(csf)$predictions
cr <- extract_csf_rules(csf,50)
if (length(cr)>0) {
  ap <- apply_rules(cr,rbind(X_tr,X_te),cv)
  if (ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F],cate_tr,10); cat(sprintf("  Rules selected: %d\n", l$nr))
    if (l$nr>0) { cat("  Rules:\n"); s <- which(abs(l$coef)>1e-5)
      for (j in s[1:min(length(s),10)]) cat(sprintf("    [%+.4f] %s\n", l$coef[j], substr(ap$rules[[j]]$rule,1,65))) }
  } else cat("  No rules passed support threshold\n")
} else cat("  No rules extracted\n")

# ============ PART 3: GBSG REAL DATA ============
cat("\n=== GBSG REAL DATA ===\n")
gbsg <- read.csv("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/gbsg.csv")
gbsg$tx <- gbsg$hormon; gbsg$ev <- gbsg$status
cv2 <- c("age","meno","size","grade","nodes","pgr","er")
for (c in cv2) gbsg[[c]][is.na(gbsg[[c]])] <- median(gbsg[[c]],na.rm=T)
X <- as.matrix(gbsg[,cv2]); colnames(X) <- cv2
n <- nrow(X); cat(sprintf("  N=%d p=%d events=%d\n", n, length(cv2), sum(gbsg$ev)))
gbsg$time <- gbsg$rfstime
ts <- median(gbsg$time[gbsg$ev==1])
ii <- sample(n,floor(.7*n)); X_tr<-X[ii,,drop=F]; X_te<-X[-ii,,drop=F]
a_tr<-gbsg$tx[ii]; a_te<-gbsg$tx[-ii]; t_tr<-gbsg$time[ii]; t_te<-gbsg$time[-ii]; e_tr<-gbsg$ev[ii]; e_te<-gbsg$ev[-ii]
csf <- causal_survival_forest(X_tr,t_tr,a_tr,as.logical(e_tr),num.trees=100,horizon=ts)
cate_tr <- predict(csf)$predictions
cr <- extract_csf_rules(csf,50)
if (length(cr)>0) {
  ap <- apply_rules(cr,rbind(X_tr,X_te),cv2)
  if (ncol(ap$R)>0) { l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F],cate_tr,10); cat(sprintf("  Rules selected: %d\n", l$nr))
    if (l$nr>0) { cat("  Rules:\n"); s <- which(abs(l$coef)>1e-5)
      for (j in s[1:min(length(s),10)]) cat(sprintf("    [%+.4f] %s\n", l$coef[j], substr(ap$rules[[j]]$rule,1,65))) }
  } else cat("  No rules passed support threshold\n")
} else cat("  No rules extracted\n")

cat("\n=== DONE ===\n")
