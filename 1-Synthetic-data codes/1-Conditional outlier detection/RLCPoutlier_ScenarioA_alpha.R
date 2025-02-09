


########## LCP outlier Scenario A different n------------

library(kernlab)
library(MASS)
library("ks")
library(foreach)
library(randomForest)
library(doParallel)
#require(tidyverse)
require(ggplot2)
library(ggpubr)
library(caret)
library(rgl)
library(grf)
library(isotree)
library(doSNOW)
library(e1071)
library(nnet)
library(kknn)
library(qrnn)
library(ggsci)


setwd(dirname(rstudioapi::getSourceEditorContext()$path))

d <- 10
n <- 2000
N <- 2000
h <- (n)^(-1/(2+1))
alpha <- 0.15
beta <- 0.5*c(1, -1, 1, -1, 1, rep(0, d-6))



################ CQR-QNN --------------

nr <- 500
cl <- makeCluster(12)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_CQRNN <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","qrnn"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    t <- runif(2*n+N)
    X <- matrix(runif((2*n+N)*(d-1), -1, 1), ncol = d-1)
    Z <- (3+2*sin(2*pi*t))*rnorm(2*n+N)
    Y <- X%*%beta + Z
    
    # modelmean <- randomForest(y~., data = data.frame(y = Y[1:n], X = X[1:n,], t = t[1:n]), ntree = 100)
    # mX <- predict(modelmean, data.frame(t = t, X = X))
    # data <- data.frame(t = t, X = X, y = Y, mX = mX)
    data <- data.frame(t = t, X = X, y = Y)
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    outlier <- sample(1:N, 0.1*N)
    datatest$y[outlier] <- datatest$y[outlier] + 3*(3+1.5*sin(2*pi*datatest$t[outlier]))*ifelse(runif(length(outlier))>0.5, 1, -1)
    
    plot(datatest$t, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
    plot(datatrain$t,datatrain$y)
    
    
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      #rand <- runif(2)
      #s_sam[i,] <- c(datatest$s.1[i], datatest$s.2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    
    
    modelQR1 <- qrnn.fit(x = as.matrix(datatrain[, 1:d]), y = as.matrix(datatrain$y), n.hidden = 5, tau = 0.1, iter.max = 1000)
    modelQR2 <- qrnn.fit(x = as.matrix(datatrain[, 1:d]), y = as.matrix(datatrain$y), n.hidden = 5, tau = 0.9, iter.max = 1000)
    
    mXQR1 <- qrnn.predict(as.matrix(datatest[, 1:d]),modelQR1)
    mXQRcal1 <- qrnn.predict(as.matrix(datacal[, 1:d]),modelQR1)
    
    mXQR2 <- qrnn.predict(as.matrix(datatest[, 1:d]),modelQR2)
    mXQRcal2 <- qrnn.predict(as.matrix(datacal[, 1:d]),modelQR2)
    
    RQR <- apply(cbind(datatest$y-mXQR2, mXQR1-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal2, mXQRcal1-datacal$y), 1, max)
    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
    
    #### CP ---
    
    pvalues_CQR <- (apply(IndQR, 1, sum))/(n+1)
    rej_CQR <- sort(pvalues_CQR)<((1:length(pvalues_CQR))/length(pvalues_CQR))*alpha
    rejnum_CQR <- max(which(rej_CQR==T))
    reject_CQR <- which(pvalues_CQR<=sort(pvalues_CQR)[rejnum_CQR])
    if(rejnum_CQR==-Inf){
      FDP_CQR <- 0
      POWER_CQR <- 0
    }else{
      FDP_CQR <- sum(ifelse(reject_CQR%in%outlier, 0, 1))/rejnum_CQR
      POWER_CQR <- sum(ifelse(reject_CQR%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP',Alg = 'CQR-QNN', alpha = alpha))
    
    
    ###---Weighted_CP---###
    WQR <- weight*IndQR
    pvalues <- (apply(WQR, 1, sum))/(apply(weight, 1, sum))
    pvalues[is.na(pvalues)] <- 1
    
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(WQR, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - WQR[, n+1] + weight[, n+1]*ifelse(RQR<=RQR[j], 1, 0))/sum_weight
      pvalues_j[is.na(pvalues_j)] <- 1
      pvalues_j[j] <- 0
      rej_j <- sort(pvalues_j)<((1:length(pvalues_j))/length(pvalues_j))*alpha
      rejnum_j <- max(which(rej_j==T))
      Rtild[j] <- rejnum_j
    }
    S <- alpha*Rtild/N
    R1 <- which(pvalues<=S)
    xi <- runif(N)
    R <- 0
    for (r in 1:length(R1)) {
      if(sum(ifelse(pvalues<=S&xi*Rtild<=r, 1, 0))>=r){
        R <- r
      }
    }
    reject <- which(pvalues<=S&xi*Rtild<=R)
    rejnum <- length(reject)
    
    if(rejnum==0){
      FDP <- 0
      POWER <- 0
    }else{
      FDP <- sum(ifelse(reject%in%outlier, 0, 1))/rejnum
      POWER <- sum(ifelse(reject%in%outlier, 1, 0))/length(outlier)
    }
    
    plot(datatest$t, RQR, col = ifelse(1:N%in%reject_CQR, 'red', 'gray'))
    plot(datatest$t, RQR, col = ifelse(1:N%in%reject, 'red', 'gray'))
    
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP-od', Alg = 'CQR-QNN', alpha = alpha))
  }
  result
  
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_CQRNN

Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Resultdraw$Method <- factor(Result$Method,levels=c('LCP','CP'))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "alpha") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1



############# CQR-QRF ------------------------

nr <- 500
cl <- makeCluster(12)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_CQRRF <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","qrnn"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    t <- runif(2*n+N)
    X <- matrix(runif((2*n+N)*(d-1), -1, 1), ncol = d-1)
    Z <- (3+2*sin(2*pi*t))*rnorm(2*n+N)
    Y <- X%*%beta + Z
    
    # modelmean <- randomForest(y~., data = data.frame(y = Y[1:n], X = X[1:n,], t = t[1:n]), ntree = 100)
    # mX <- predict(modelmean, data.frame(t = t, X = X))
    # data <- data.frame(t = t, X = X, y = Y, mX = mX)
    data <- data.frame(t = t, X = X, y = Y)
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    outlier <- sample(1:N, 0.1*N)
    datatest$y[outlier] <- datatest$y[outlier] + 3*(3+1.5*sin(2*pi*datatest$t[outlier]))*ifelse(runif(length(outlier))>0.5, 1, -1)
    
    plot(datatest$t, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
    plot(datatrain$t,datatrain$y)
    
    
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      #rand <- runif(2)
      #s_sam[i,] <- c(datatest$s.1[i], datatest$s.2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    modelQR <- quantile_forest(datatrain[, 1:d], datatrain$y, quantiles = c(0.1, 0.9))
    mXQR <- predict(modelQR, datatest[, 1:d])$predictions
    mXQRcal <- predict(modelQR, datacal[, 1:d])$predictions
    RQR <- apply(cbind(datatest$y-mXQR[, 2], mXQR[, 1]-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal[, 2], mXQRcal[, 1]-datacal$y), 1, max)
    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
    
    #### CP ---
    
    pvalues_CQR <- (apply(IndQR, 1, sum))/(n+1)
    rej_CQR <- sort(pvalues_CQR)<((1:length(pvalues_CQR))/length(pvalues_CQR))*alpha
    rejnum_CQR <- max(which(rej_CQR==T))
    reject_CQR <- which(pvalues_CQR<=sort(pvalues_CQR)[rejnum_CQR])
    if(rejnum_CQR==-Inf){
      FDP_CQR <- 0
      POWER_CQR <- 0
    }else{
      FDP_CQR <- sum(ifelse(reject_CQR%in%outlier, 0, 1))/rejnum_CQR
      POWER_CQR <- sum(ifelse(reject_CQR%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP',Alg = 'CQR-QRF', alpha = alpha))
    
    
    ###---Weighted_CP---###
    WQR <- weight*IndQR
    pvalues <- (apply(WQR, 1, sum))/(apply(weight, 1, sum))
    pvalues[is.na(pvalues)] <- 1
    
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(WQR, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - WQR[, n+1] + weight[, n+1]*ifelse(RQR<=RQR[j], 1, 0))/sum_weight
      pvalues_j[is.na(pvalues_j)] <- 1
      pvalues_j[j] <- 0
      rej_j <- sort(pvalues_j)<((1:length(pvalues_j))/length(pvalues_j))*alpha
      rejnum_j <- max(which(rej_j==T))
      Rtild[j] <- rejnum_j
    }
    S <- alpha*Rtild/N
    R1 <- which(pvalues<=S)
    xi <- runif(N)
    R <- 0
    for (r in 1:length(R1)) {
      if(sum(ifelse(pvalues<=S&xi*Rtild<=r, 1, 0))>=r){
        R <- r
      }
    }
    reject <- which(pvalues<=S&xi*Rtild<=R)
    rejnum <- length(reject)
    
    if(rejnum==0){
      FDP <- 0
      POWER <- 0
    }else{
      FDP <- sum(ifelse(reject%in%outlier, 0, 1))/rejnum
      POWER <- sum(ifelse(reject%in%outlier, 1, 0))/length(outlier)
    }
    
    plot(datatest$t, RQR, col = ifelse(1:N%in%reject_CQR, 'red', 'gray'))
    plot(datatest$t, RQR, col = ifelse(1:N%in%reject, 'red', 'gray'))
    
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP-od', Alg = 'CQR-QRF', alpha = alpha))
  }
  result
  
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_CQRRF

Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Resultdraw$Method <- factor(Result$Method,levels=c('LCP','CP'))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "alpha") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1

##################### Res-RF ---------------------


nr <- 500
cl <- makeCluster(12)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_resRF <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    t <- runif(2*n+N)
    X <- matrix(runif((2*n+N)*(d-1), -1, 1), ncol = d-1)
    Z <- (3+2*sin(2*pi*t))*rnorm(2*n+N)
    Y <- X%*%beta + Z
    
    modelmean <- randomForest(y~., data = data.frame(y = Y[1:n], X = X[1:n,], t = t[1:n]), ntree = 100)
    mX <- predict(modelmean, data.frame(t = t, X = X))
    data <- data.frame(t = t, X = X, y = Y, mX = mX)
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    outlier <- sample(1:N, 0.1*N)
    datatest$y[outlier] <- datatest$y[outlier] + 3*(3+1.5*sin(2*pi*datatest$t[outlier]))*ifelse(runif(length(outlier))>0.5, 1, -1)
    
    plot(datatest$t, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
    plot(datatrain$t,datatrain$y)
    
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      #rand <- runif(2)
      #s_sam[i,] <- c(datatest$s.1[i], datatest$s.2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(abs(datatest$y-datatest$mX)<abs(datacal$y[j]-datacal$mX[j]), 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    
    #### CP ------------ ###
    pvalues_CP <- (apply(IndMat, 1, sum))/(n+1)
    rej_CP <- sort(pvalues_CP)<((1:length(pvalues_CP))/length(pvalues_CP))*alpha
    rejnum_CP <- max(which(rej_CP==T))
    reject_CP <- which(pvalues_CP<=sort(pvalues_CP)[rejnum_CP])
    if(rejnum_CP==-Inf){
      FDP_CP <- 0
      POWER_CP <- 0
    }else{
      FDP_CP <- sum(ifelse(reject_CP%in%outlier, 0, 1))/rejnum_CP
      POWER_CP <- sum(ifelse(reject_CP%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', Alg = 'Res-RF', alpha = alpha))
    
    
    
    
    ###---Weighted_CP---###
    
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    pvalues[is.na(pvalues)] <- 1
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(abs(datatest$y-datatest$mX)<abs(datacal$y[j]-datacal$mX[j]), 1, 0))/sum_weight
      pvalues_j[j] <- 0
      rej_j <- sort(pvalues_j)<((1:length(pvalues_j))/length(pvalues_j))*alpha
      rejnum_j <- max(which(rej_j==T))
      Rtild[j] <- rejnum_j
    }
    S <- alpha*Rtild/N
    R1 <- which(pvalues<=S)
    xi <- runif(N)
    R <- 0
    for (r in 1:length(R1)) {
      if(sum(ifelse(pvalues<=S&xi*Rtild<=r, 1, 0))>=r){
        R <- r
      }
    }
    reject <- which(pvalues<=S&xi*Rtild<=R)
    rejnum <- length(reject)
    
    if(rejnum==0){
      FDP <- 0
      POWER <- 0
    }else{
      FDP <- sum(ifelse(reject%in%outlier, 0, 1))/rejnum
      POWER <- sum(ifelse(reject%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP-od', Alg = 'Res-RF', alpha = alpha))
    
  }
  result
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_resRF

Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Resultdraw$Method <- factor(Result$Method,levels=c('LCP','CP'))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "alpha") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1


########### Res-SVM ---------------


nr <- 500
cl <- makeCluster(12)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_resSVM <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","e1071"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    t <- runif(2*n+N)
    X <- matrix(runif((2*n+N)*(d-1), -1, 1), ncol = d-1)
    Z <- (3+2*sin(2*pi*t))*rnorm(2*n+N)
    Y <- X%*%beta + Z
    
    #modelmean <- randomForest(y~., data = data.frame(y = Y[1:n], X = X[1:n,], t = t[1:n]), ntree = 100)
    modelmean <- svm(y ~ ., data = data.frame(y = Y[1:n],X = X[1:n,], t = t[1:n]))
    mX <- predict(modelmean, data.frame(t = t, X = X))
    data <- data.frame(t = t, X = X, y = Y, mX = mX)
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    outlier <- sample(1:N, 0.1*N)
    datatest$y[outlier] <- datatest$y[outlier] + 3*(3+1.5*sin(2*pi*datatest$t[outlier]))*ifelse(runif(length(outlier))>0.5, 1, -1)
    
    plot(datatest$t, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
    plot(datatrain$t,datatrain$y)
    
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      #rand <- runif(2)
      #s_sam[i,] <- c(datatest$s.1[i], datatest$s.2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(abs(datatest$y-datatest$mX)<abs(datacal$y[j]-datacal$mX[j]), 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    
    #### CP ------------ ###
    pvalues_CP <- (apply(IndMat, 1, sum))/(n+1)
    rej_CP <- sort(pvalues_CP)<((1:length(pvalues_CP))/length(pvalues_CP))*alpha
    rejnum_CP <- max(which(rej_CP==T))
    reject_CP <- which(pvalues_CP<=sort(pvalues_CP)[rejnum_CP])
    if(rejnum_CP==-Inf){
      FDP_CP <- 0
      POWER_CP <- 0
    }else{
      FDP_CP <- sum(ifelse(reject_CP%in%outlier, 0, 1))/rejnum_CP
      POWER_CP <- sum(ifelse(reject_CP%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', Alg = 'Res-SVM', alpha = alpha))
    
    
    
    
    ###---Weighted_CP---###
    
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    pvalues[is.na(pvalues)] <- 1
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(abs(datatest$y-datatest$mX)<abs(datacal$y[j]-datacal$mX[j]), 1, 0))/sum_weight
      pvalues_j[j] <- 0
      rej_j <- sort(pvalues_j)<((1:length(pvalues_j))/length(pvalues_j))*alpha
      rejnum_j <- max(which(rej_j==T))
      Rtild[j] <- rejnum_j
    }
    S <- alpha*Rtild/N
    R1 <- which(pvalues<=S)
    xi <- runif(N)
    R <- 0
    for (r in 1:length(R1)) {
      if(sum(ifelse(pvalues<=S&xi*Rtild<=r, 1, 0))>=r){
        R <- r
      }
    }
    reject <- which(pvalues<=S&xi*Rtild<=R)
    rejnum <- length(reject)
    
    if(rejnum==0){
      FDP <- 0
      POWER <- 0
    }else{
      FDP <- sum(ifelse(reject%in%outlier, 0, 1))/rejnum
      POWER <- sum(ifelse(reject%in%outlier, 1, 0))/length(outlier)
    }
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP-od', Alg = 'Res-SVM', alpha = alpha))
    
  }
  result
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_resSVM 

Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Resultdraw$Method <- factor(Result$Method,levels=c('LCP','CP'))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "alpha") +
  theme_bw() +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T) +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 12),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(.~Type, scales = "free")
P1


########### plots ---------------

Result_all <- data.frame(rbind(Result_CQRNN,Result_CQRRF,Result_resRF,Result_resSVM))

#write.csv(Result_all,"Outlier_ScenarioA_n_500times.csv")

Result_all$Method <- factor(Result_all$Method, levels = c('LCP-od', 'CP'))
Result_all$Alg <- factor(Result_all$Alg, levels = c('CQR-QNN', 'CQR-QRF','Res-RF','Res-SVM'))
Result_all$alphadraw <- factor(Result_all$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))


Result1.new <- Result_all

Result1.tidy <- data.frame(quant = c(Result1.new$FDP-Result1.new$alpha, Result1.new$POWER), 
                           Method = c(Result1.new$Method, Result1.new$Method), 
                           ndraw = c(Result1.new$ndraw, Result1.new$ndraw), 
                           type = c(rep('FDP above nominal', nrow(Result1.new)), 
                                    rep('Power', nrow(Result1.new))), 
                           hline = c(rep(0, nrow(Result1.new)), rep(NA, nrow(Result1.new))),
                           Alg = c(Result1.new$Alg,Result1.new$Alg))


Result1.tidy$Method <- factor(Result1.tidy$Method,levels = c('LCP-od','CP'))



pdf(file="Outlier_ScenarioA_alpha.pdf",
    width=12,height=7)
P1 <- ggplot(Result1.tidy, aes(x = alphadraw, y = quant, color=Method)) +
  geom_boxplot(alpha=0.8) +
  scale_y_continuous(limits = c(-0.2, 0.8),
                     breaks = seq(0,1,0.2))+
  scale_x_discrete(name = TeX("$alpha$")) +
  ylab("") +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF", "#20854EFF", "#6F99ADFF"))+
  facet_grid(type~Alg, scales = "free_y")+
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") +
  facet_grid(type~Alg,scales = "free_y")
P1
dev.off()
