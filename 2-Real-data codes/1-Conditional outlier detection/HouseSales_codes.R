

########### HouseSales codes --------------------


# https://www.kaggle.com/datasets/harlfoxem/housesalesprediction


library(kernlab)
library(MASS)
library("ks")
library(foreach)
library(randomForest)
library(doParallel)
require(tidyverse)
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
library(dplyr)
library(stringr)
library(qrnn)

library(quantreg)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

data <- read.csv("kc_house_data.csv")
head(data)
colnames(data)

summary(data)
dim(data)
cols_with_na <- apply(data, 2, function(x) any(is.na(x)))

data1<- data [, !cols_with_na]
data1 <- data1[,-c(1,2,17)]
colnames(data1)
dim(data1)
summary(data1)
#data1$price <- log(data1$price)

### calculate conditional quantile ----

# 
conditional_quantile <- function(data, prob) {
  data %>%
    group_by(lat) %>%
    summarise(quantile = quantile(price, probs = prob)) %>%
    ungroup()
}

# 
condquan <- conditional_quantile(data1, 0.9)
print(condquan,n=20)


# 
get_conditional_quantile <- function(s1_value, result_df) {
  filtered_result <- result_df %>%
    filter(lat == s1_value)
  
  if (nrow(filtered_result) == 0) {
    return(NA)
  } else {
    return(filtered_result$quantile)
  }
}

# 
s1_value <- data1$lat[1]
conditional_quantile_value <- get_conditional_quantile(s1_value, condquan)
print(conditional_quantile_value)


######### calculate conditional variance -------

conditional_variance <- function(data) {
  data %>%
    group_by(lat) %>%
    filter(n() > 1) %>% # 
    summarise(variance = var(price, na.rm = TRUE)) %>%
    ungroup()
}

condvar <- conditional_variance(data1)
print(condvar,n=40)




get_conditional_variance <- function(s1_value, result_df) {
  filtered_result <- result_df %>%
    filter(abs(lat - s1_value)==0)
  
  if (nrow(filtered_result) == 0) {
    return(0)
  } else {
    return(filtered_result$variance)
  }
}


s1_value <- data1$lat[5]
conditional_variance_value <- get_conditional_variance(s1_value, condvar)
print(conditional_variance_value)



############## quantile-besed outliers -----------



############ CQR score ---------

d <- dim(data1)[2] # dimension of covariates
n <- 3000
N <- 3000
h <- (n)^(-1/(2+1))*0.5

nr <- 500
cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_CQR <-  foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","e1071","dplyr","nnet","qrnn"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  
  X <- data1[,-c(1,15)]
  Y <- data1[,1]
  s <- data1[,c(15)]
  
  
  data2 <- data.frame(s=s, X = X, y = Y)
  head(data2)
  names(data2) <- c("lat",
                    "X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16",
                    "y")
  data2 <- data2[sample(nrow(data2)), ]
  datacal <- data2[(n+1):(2*n),]
  datatest <- data2[(2*n+1):(2*n+N),]
  datatrain <- data2[1:n,]
  
  outlier <- sample(1:N, 0.1*N)
  
  #datatest$y[outlier] <- datatest$y[outlier] + get_conditional_quantile(datatest$s.s1[outlier], datatest$s.s2[outlier], result)*ifelse(runif(length(outlier))>0.5, 1, -1)
  
  for(i in outlier){
    datatest$y[i] <- datatest$y[i] +
      0.75*get_conditional_quantile(datatest$lat[i], condquan)*ifelse(runif(1)>0.5, 1, -1)
  }
  
  
  # for(i in outlier){
  #   datatest$y[i] <- datatest$y[i] + 
  #     2*sqrt(get_conditional_variance(datatest$lat[i], condvar))*ifelse(runif(1)>0.5, 2, -1)
  # }
  
  plot(datatest$lat, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
  plot(datatrain$lat,datatrain$y)
  
  
  for (alpha in c(0.15, 0.2)) {
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      # rand <- runif(2)
      # s_sam[i,] <- c(datatest$s1[i], datatest$s2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s1[j], s_sam[, 2]-datacal$s2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    
   
    ###---CP CQR QNN---###
    
    modelQR1 <- qrnn.fit(x = as.matrix(datatrain[, 1:(d-1)]), y = as.matrix(datatrain$y), n.hidden = 10, tau = 0.1, iter.max = 1000)
    modelQR2 <- qrnn.fit(x = as.matrix(datatrain[, 1:(d-1)]), y = as.matrix(datatrain$y), n.hidden = 10, tau = 0.9, iter.max = 1000)
    
    
    
    mXQR1 <- qrnn.predict(as.matrix(datatest[, 1:(d-1)]),modelQR1)
    mXQRcal1 <- qrnn.predict(as.matrix(datacal[, 1:(d-1)]),modelQR1)
    
    mXQR2 <- qrnn.predict(as.matrix(datatest[, 1:(d-1)]),modelQR2)
    mXQRcal2 <- qrnn.predict(as.matrix(datacal[, 1:(d-1)]),modelQR2)
    
    RQR <- apply(cbind(datatest$y-mXQR2, mXQR1-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal2, mXQRcal1-datacal$y), 1, max)
    
    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
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
    data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CQR', alpha = alpha)
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP(CQR-QNN)', alpha = alpha))
    
    
    ###---Weighted_CP CQR QRF---###
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
    
    
    
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP(CQR-QNN)', alpha = alpha))
    
    
    
    ############CP  CQR (QRF) -------------
    
    modelQR <- quantile_forest(datatrain[, 1:(d-1)], datatrain$y, quantiles = c(0.1, 0.9))
    mXQR <- predict(modelQR, datatest[, 1:(d-1)])$predictions
    mXQRcal <- predict(modelQR, datacal[, 1:(d-1)])$predictions
    RQR <- apply(cbind(datatest$y-mXQR[, 2], mXQR[, 1]-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal[, 2], mXQRcal[, 1]-datacal$y), 1, max)

    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
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
    data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CQR', alpha = alpha)
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP(CQR-QRF)', alpha = alpha))
    
    
    ###---Weighted_CP CQR QRF---###
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
    
   
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP(CQR-QRF)', alpha = alpha))
    
    
    
  }
  result
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_CQR%>%
  group_by(alpha,Method)%>%
  dplyr::summarize(FDR = mean(FDP), sdFDP = sd(FDP),
                   Power = mean(POWER), sdPOWER = sd(POWER))
pp


################ variance-based outliers ---------------


############ CQR score ---------

d <- dim(data1)[2] # dimension of covariates
n <- 3000
N <- 3000
h <- (n)^(-1/(2+1))*0.5

nr <- 500
cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_CQR <-  foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","e1071","dplyr","nnet","qrnn"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  
  X <- data1[,-c(1,15)]
  Y <- data1[,1]
  s <- data1[,c(15)]
  
  
  data2 <- data.frame(s=s, X = X, y = Y)
  head(data2)
  names(data2) <- c("lat",
                    "X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16",
                    "y")
  data2 <- data2[sample(nrow(data2)), ]
  datacal <- data2[(n+1):(2*n),]
  datatest <- data2[(2*n+1):(2*n+N),]
  datatrain <- data2[1:n,]
  
  outlier <- sample(1:N, 0.1*N)
  
  #datatest$y[outlier] <- datatest$y[outlier] + get_conditional_quantile(datatest$s.s1[outlier], datatest$s.s2[outlier], result)*ifelse(runif(length(outlier))>0.5, 1, -1)
  
  # for(i in outlier){
  #   datatest$y[i] <- datatest$y[i] +
  #     0.75*get_conditional_quantile(datatest$lat[i], condquan)*ifelse(runif(1)>0.5, 1, -1)
  # }
  

  for(i in outlier){
    datatest$y[i] <- datatest$y[i] +
      2*sqrt(get_conditional_variance(datatest$lat[i], condvar))*ifelse(runif(1)>0.5, 2, -1)
  }
  
  plot(datatest$lat, datatest$y, col = ifelse(1:N%in%outlier, 'red', 'gray'))
  plot(datatrain$lat,datatrain$y)
  
  
  for (alpha in c(0.15, 0.2)) {
    s_sam <- matrix(0, ncol = 1, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:1]), (h^2)*diag(1))
      # rand <- runif(2)
      # s_sam[i,] <- c(datatest$s1[i], datatest$s2[i]) + h*sqrt(rand[1])*c(cos(2*pi*rand[2]), sin(2*pi*rand[2]))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      diffmat <- matrix(0, nrow = N, ncol = 2)
      for (k in 1:1) {
        diffmat[, k] <- s_sam[, k] - datacal[j, k]
      }
      #weight[, j] <- ifelse(apply((cbind(s_sam[, 1]-datacal$s1[j], s_sam[, 2]-datacal$s2[j]))^2, 1, sum)<=h^2, 1, 0)
      weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    }
    diffmat <- matrix(0, nrow = N, ncol = 1)
    for (k in 1:1) {
      diffmat[, k] <- s_sam[, k] - datatest[, k]
    }
    weight[, n+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
    
    
    
    
    ###---CP CQR QNN---###
    
    modelQR1 <- qrnn.fit(x = as.matrix(datatrain[, 1:(d-1)]), y = as.matrix(datatrain$y), n.hidden = 10, tau = 0.1, iter.max = 1000)
    modelQR2 <- qrnn.fit(x = as.matrix(datatrain[, 1:(d-1)]), y = as.matrix(datatrain$y), n.hidden = 10, tau = 0.9, iter.max = 1000)
    
    
    
    mXQR1 <- qrnn.predict(as.matrix(datatest[, 1:(d-1)]),modelQR1)
    mXQRcal1 <- qrnn.predict(as.matrix(datacal[, 1:(d-1)]),modelQR1)
    
    mXQR2 <- qrnn.predict(as.matrix(datatest[, 1:(d-1)]),modelQR2)
    mXQRcal2 <- qrnn.predict(as.matrix(datacal[, 1:(d-1)]),modelQR2)
    
    RQR <- apply(cbind(datatest$y-mXQR2, mXQR1-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal2, mXQRcal1-datacal$y), 1, max)
    
    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
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
    data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CQR', alpha = alpha)
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP(CQR-QNN)', alpha = alpha))
    
    
    ###---Weighted_CP CQR QRF---###
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
    
    
    
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP(CQR-QNN)', alpha = alpha))
    
    
    
    ############CP  CQR (QRF) -------------
    
    modelQR <- quantile_forest(datatrain[, 1:(d-1)], datatrain$y, quantiles = c(0.1, 0.9))
    mXQR <- predict(modelQR, datatest[, 1:(d-1)])$predictions
    mXQRcal <- predict(modelQR, datacal[, 1:(d-1)])$predictions
    RQR <- apply(cbind(datatest$y-mXQR[, 2], mXQR[, 1]-datatest$y), 1, max)
    RQRcal <- apply(cbind(datacal$y-mXQRcal[, 2], mXQRcal[, 1]-datacal$y), 1, max)
    
    IndQR <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndQR[,j] <- ifelse(RQR<RQRcal[j], 1, 0)
    }
    IndQR[, n+1] <- runif(N)
    
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
    data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CQR', alpha = alpha)
    result <- rbind(result, data.frame(FDP = FDP_CQR, POWER = POWER_CQR, Method = 'CP(CQR-QRF)', alpha = alpha))
    
    
    ###---Weighted_CP CQR QRF---###
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
    
    
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'LCP(CQR-QRF)', alpha = alpha))
    
    
    
  }
  result
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_CQR%>%
  group_by(alpha,Method)%>%
  dplyr::summarize(FDR = mean(FDP), sdFDP = sd(FDP),
                   Power = mean(POWER), sdPOWER = sd(POWER))
pp

