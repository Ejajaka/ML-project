library(grf); library(glmnet); library(gbm)
sup <- read.csv('../data/support_clean.csv'); set.seed(123)
X <- as.matrix(sup[,!colnames(sup) %in% c('death','d.time')]); n <- nrow(X); cn <- colnames(X)
trt <- rbinom(n,1,0.5)
am <- median(X[,'age']); bm <- median(X[,'bili']); mm <- median(X[,'meanbp'])
s1 <- X[,'age']>am & X[,'bili']>bm; s2 <- X[,'age']<=am & X[,'meanbp']>mm
tau <- rep(0,n); tau[s1]<-0.30; tau[s2]<-0.12
cat(sprintf('SUPPORT: N=%d | H1=%d H2=%d null=%d\n',n,sum(s1),sum(s2),sum(!s1&!s2)))
bh <- exp(0.3*scale(X[,'age'])[,1]+0.2*scale(X[,'bili'])[,1]-0.2*scale(X[,'meanbp'])[,1])
t_obs <- ifelse(trt==1, rexp(n,bh*exp(-tau*1.5)), rexp(n,bh))
cen <- rexp(n,0.15); tf <- pmin(t_obs,cen); ev <- as.integer(t_obs<=cen); ts <- 1
cate_o <- exp(-bh*exp(-tau*1.5)) - exp(-bh)
ii <- sample(n,5000); X_tr<-X[ii,,drop=F]; X_te<-X[-ii,,drop=F]
a_tr<-trt[ii]; a_te<-trt[-ii]; t_tr<-tf[ii]; t_te<-tf[-ii]
e_tr<-ev[ii]; e_te<-ev[-ii]; cate_te<-cate_o[-ii]

cat('CSF...\n')
csf <- causal_survival_forest(X_tr,t_tr,a_tr,as.logical(e_tr),num.trees=100,horizon=ts)
c3 <- predict(csf,X_te)$predictions
cate_tr <- predict(csf)$predictions

# Helpers
extract_rules <- function(csf, mt=25) {
  r <- list()
  for (t in 1:min(csf$num.trees, mt)) {
    tree <- get_tree(csf, t)
    walk <- function(nid, conds) {
      nd <- tree$nodes[[nid]]
      if (isTRUE(nd$is_leaf)) {
        if (length(conds) >= 1) r[[length(r)+1]] <<- list(rule=paste(conds,collapse=' & '))
        return()
      }
      vn <- tree$columns[nd$split_variable]
      sv <- nd$split_value
      walk(nd$left_child, c(conds, sprintf('%s <= %.4f', vn, sv)))
      walk(nd$right_child, c(conds, sprintf('%s > %.4f', vn, sv)))
    }
    walk(1, c())
  }
  r[!duplicated(sapply(r, `[[`, 'rule'))]
}
extract_gbm_rules <- function(m, X, nt=25) {
  r <- list()
  for (t in 1:min(m$n.trees, nt)) {
    ti <- pretty.gbm.tree(m, t)
    if (nrow(ti) <= 1) next
    walk <- function(nid, conds) {
      row <- ti[nid+1, ]
      if (row$SplitVar == -1) {
        if (length(conds) >= 1) r[[length(r)+1]] <<- list(rule=paste(conds,collapse=' & '))
        return()
      }
      vn <- colnames(X)[row$SplitVar+1]
      sv <- row$SplitCodePred
      walk(row$LeftNode, c(conds, sprintf('%s <= %.4f', vn, sv)))
      walk(row$RightNode, c(conds, sprintf('%s > %.4f', vn, sv)))
    }
    walk(0, c())
  }
  r[!duplicated(sapply(r, `[[`, 'rule'))]
}
apply_rules <- function(rules, X, cn, ms=10) {
  n <- nrow(X); m <- length(rules)
  if (m == 0) return(list(R=matrix(0,n,0)))
  R <- matrix(0, n, m); valid <- rep(TRUE, m)
  for (j in 1:m) {
    mask <- rep(TRUE, n)
    for (p in strsplit(rules[[j]]$rule, ' & ')[[1]]) {
      if (grepl('<=', p)) {
        vv <- strsplit(p, '<=')[[1]]
        vi <- which(cn == vv[1])
        if (length(vi)) mask <- mask & (X[,vi] <= as.numeric(vv[2])) else { valid[j] <- FALSE; break }
      }
      if (grepl('>', p) && !grepl('<=', p)) {
        vv <- strsplit(p, '>')[[1]]
        vi <- which(cn == vv[1])
        if (length(vi)) mask <- mask & (X[,vi] > as.numeric(vv[2])) else { valid[j] <- FALSE; break }
      }
    }
    if (valid[j]) R[,j] <- as.integer(mask)
  }
  R <- R[, valid, drop=FALSE]; keep <- colSums(R) >= ms
  list(R=R[, keep, drop=FALSE])
}
fit_lasso <- function(X, y, max_r=15) {
  if (ncol(X) < 2 || sum(!is.na(y)) < 10) return(list(coef=rep(0,max(1,ncol(X))), nr=0, int=0))
  Xs <- scale(X); ym <- mean(y, na.rm=TRUE); ys <- y - ym
  cv <- tryCatch(cv.glmnet(Xs, ys, alpha=1, nfolds=5, standardize=FALSE), error=function(e) NULL)
  if (is.null(cv)) return(list(coef=rep(0,ncol(X)), nr=0, int=ym))
  for (i in length(cv$lambda):1) {
    cf <- as.vector(coef(cv, s=cv$lambda[i]))[-1]
    if (sum(abs(cf) > 1e-5) <= max_r && sum(abs(cf) > 1e-5) >= 1) {
      m <- glmnet(Xs, ys, alpha=1, lambda=cv$lambda[i], standardize=FALSE)
      cf <- as.vector(coef(m))[-1]
      return(list(coef=cf, nr=sum(abs(cf) > 1e-5), int=coef(m)[1] + ym))
    }
  }
  list(coef=rep(0,ncol(X)), nr=0, int=ym)
}
evaluate <- function(pred, true) {
  v <- !is.na(pred) & !is.na(true) & is.finite(pred)
  if (sum(v) < 10) return(c(NA,NA,NA,NA))
  p <- pred[v]; t <- true[v]
  c(mean(abs(p-t)), sqrt(mean((p-t)^2)),
    if (sd(p) > 0 && sd(t) > 0) cor(p, t, method='spearman') else 0,
    mean((p > 0) == (t > 0)))
}

# Bo & Ding
cat('Bo & Ding...\n')
gb <- gbm(cate_tr ~ ., data=data.frame(X_tr), n.trees=80, interaction.depth=3, distribution='gaussian')
gr <- extract_gbm_rules(gb, X_tr, 25)
n4 <- 0; c4 <- rep(0, length(c3))
if (length(gr) > 0) {
  ap <- apply_rules(gr, rbind(X_tr, X_te), cn)
  if (ncol(ap$R) > 0) {
    l <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr)
    c4 <- l$int + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l$coef
    c4 <- as.vector(c4); n4 <- l$nr
  }
}
cat(sprintf('  B&D: %d rules\n', n4))

# Hybrid
cat('Hybrid...\n')
cr <- extract_rules(csf, 25)
n5 <- 0; c5 <- rep(0, length(c3))
if (length(cr) > 0) {
  ap <- apply_rules(cr, rbind(X_tr, X_te), cn)
  if (ncol(ap$R) > 0) {
    l2 <- fit_lasso(ap$R[1:nrow(X_tr),,drop=F], cate_tr)
    c5 <- l2$int + ap$R[-(1:nrow(X_tr)),,drop=F] %*% l2$coef
    c5 <- as.vector(c5); n5 <- l2$nr
  }
}
cat(sprintf('  Hybrid: %d rules\n', n5))

# Results
cat('\n', strrep('=', 70), '\n', sep='')
cat('SUPPORT SEMI-SYNTHETIC (n=9105)\n')
cat(strrep('=', 70), '\n\n', sep='')
e3 <- evaluate(c3, cate_te); e4 <- evaluate(c4, cate_te); e5 <- evaluate(c5, cate_te)
cat(sprintf('%-28s %-10s %-10s %-10s %-8s\n', 'Method', 'MAE', 'RMSE', 'Spearman', 'Rules'))
cat(strrep('-', 68), '\n', sep='')
cat(sprintf('%-28s %.4f    %.4f    %.3f    %d\n', 'CSF (grf)', e3[1], e3[2], e3[3], 0))
cat(sprintf('%-28s %.4f    %.4f    %.3f    %d\n', 'Bo & Ding (GB+Lasso)', e4[1], e4[2], e4[3], n4))
cat(sprintf('%-28s %.4f    %.4f    %.3f    %d\n', 'Hybrid (CSF+Lasso)', e5[1], e5[2], e5[3], n5))

if (n5 > 0) {
  cat('\nTop Hybrid rules:\n')
  sel <- which(abs(l2$coef) > 1e-5)
  for (j in sel[1:min(5, length(sel))]) {
    cat(sprintf('  [%+.4f] %s\n', l2$coef[j], substr(cr[[j]]$rule, 1, 70)))
  }
}
cat('\nDone.\n')
