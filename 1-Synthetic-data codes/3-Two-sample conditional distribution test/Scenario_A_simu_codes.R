

#####  two-sample conditional distribution test -----------

# library(kernlab)
# library(MASS)
# library("ks")
# library(foreach)
# library(randomForest)
# library(doParallel)
# require(tidyverse)
# require(ggplot2)
# library(ggpubr)
# library(caret)
# library(rgl)
# library(grf)
# library(isotree)
# library(doSNOW)
# library(e1071)
# library(nnet)
# library(kknn)
# 
# 
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))


library(splines)
library(plyr); library(dplyr)
library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(tidyverse)
library(ggplot2)
library(mvtnorm)
library(doSNOW)
library(nnet)
library(e1071)
library(ggsci)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
#setwd("/home/ll2120210104/RLCP0723")



d <- 5
narray <- c(200, 400, 600, 800, 1000)

H <- function(x){
  return(exp(-sum(x^2)/(2*h^2)))
}

# H <- function(x){
#   return(ifelse(sum(x^2)<=h^2, 1, 0))
# }

DRX <- function(x){
  return(exp((sum(x^2)-sum((x-c(1,1,-1,-1,0))^2))/2))
}

beta <- c(1, 1, 1, -1, -1)
#beta[(6:d)] <- 0





######### Linear Logistic regression null ------------------


nr <- 500
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_LL_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) #+ 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      
      # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) {
        for (j in 1:K) {
          classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          # 
          # classX_cross <- nnet(label~., data = data.frame(rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),]))[,-(d+1)],size = 5, maxit = 200, linout = FALSE)
          # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),])/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),]))
          
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_LL_null%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp



######### Linear Logistic Regression alter ---------------


nr <- 500
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_LL_alter <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) + 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      
      # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) {
        for (j in 1:K) {
          classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_LL_alter%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp



########### Random Forest null -----------------


nr <- 500
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_RF_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) #+ 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      
      classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) {
        for (j in 1:K) {
          #classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_RF_null%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp


########### Random Forest alter -----------------


nr <- 20
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_RF_alter <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) + 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      
      classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) {
        for (j in 1:K) {
          #classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_RF_alter%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp


######### Neural Network null ----------------

nr <- 500
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_NN_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) #+ 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      class <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train)),size = 5, maxit = 200, linout = FALSE)
      V1 <- (1 - predict(class, data1test))/predict(class, data1test)
      V2 <- (1 - predict(class, data2test))/predict(class, data2test)
      
      # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      classX <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train))[-(d+1)],size = 5, maxit = 200, linout = FALSE)
      g1 <- predict(classX, data1test)/(1 - predict(classX, data1test))
      g2 <- predict(classX, data2test)/(1 - predict(classX, data2test))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0.02, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) { 
        for (j in 1:K) {
          #classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          
          classX_cross <- nnet(label~., data = data.frame(rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),]))[,-(d+1)],size = 5, maxit = 200, linout = FALSE)
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),])/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),]))
          prob <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),])
          prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
          gamma1 <- prob/(1 - prob)
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_NN_null%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant,na.rm=T), sdQuant = sd(quant,na.rm=T))
pp



######### Neural Network alter ---------------


nr <- 500
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_NN_alter <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  for (n in narray) {
    h <- (n/2)^(-1/(2+d))
    rej <- 0
    rej_debias <- 0
    rej_weight <- 0
    T_vec <- rep(0, ns)
    for (k in 1:ns) {
      X1 <- mvrnorm(n, rep(0, d), diag(d))
      X2 <- mvrnorm(n, c(1, 1, -1, -1, 0), diag(d))
      Y1 <- X1%*%beta + rnorm(n)
      Y2 <- X2%*%beta + rnorm(n) + 0.5#0.8*(1 - 0.5*apply(X2^2, 1, sum)/d)
      
      data1train <- data.frame(x = X1[1:(n/2),], y = Y1[1:(n/2)], label = factor(1))
      data2train <- data.frame(x = X2[1:(n/2),], y = Y2[1:(n/2)], label = factor(2))
      data1test <- data.frame(x = X1[1:(n/2)+n/2,], y = Y1[1:(n/2)+n/2], label = factor(1))
      data2test <- data.frame(x = X2[1:(n/2)+n/2,], y = Y2[1:(n/2)+n/2], label = factor(2))
      
      
      # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
      # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
      # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
      # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
      # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
      # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
      class <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train)),size = 5, maxit = 200, linout = FALSE)
      V1 <- (1 - predict(class, data1test))/predict(class, data1test)
      V2 <- (1 - predict(class, data2test))/predict(class, data2test)
      
      # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
      # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
      # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
      # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
      # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
      # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
      classX <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train))[-(d+1)],size = 5, maxit = 200, linout = FALSE)
      g1 <- predict(classX, data1test)/(1 - predict(classX, data1test))
      g2 <- predict(classX, data2test)/(1 - predict(classX, data2test))
      
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      Vg1[is.na(Vg1)] <- Inf
      Vg2[is.na(Vg2)] <- Inf
      
      
      #### Chen & Lei, 2024 debiased statistics-------------
      
      K <- 5
      n1 <- n/(2*K)
      
      a <- matrix(0, n/2, n/2)
      xi <- runif(n/2)
      for (i in 1:(n/2)) {
        for (j in 1:(n/2)) {
          a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
        }
      }
      
      gamma <- matrix(0, n/2, n/2)
      alphamat1 <- matrix(0, n/2, n/2)
      alphamat2 <- matrix(0, n/2, n/2)
      for (i in 1:K) {
        for (j in 1:K) {
          #classX_cross <- glm(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], family = "binomial")
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response')/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'response'))
          # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
          # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
          
          classX_cross <- nnet(label~., data = data.frame(rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),]))[,-(d+1)],size = 5, maxit = 200, linout = FALSE)
          #gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),])/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),]))
          prob <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),])
          prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
          gamma1 <- prob/(1 - prob)
          
          gamma[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- gamma1
          astar <- apply(a[-((n1*(i-1)+1):(n1*i)),-((n1*(j-1)+1):(n1*j))], 1, mean)
          alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n1*(i-1)+1):(n1*i)),1:d]), ntree = 100)
          alpha1 <- predict(alphamodel, data.frame(x = data1test[(n1*(i-1)+1):(n1*i),1:d]))
          alpha2 <- predict(alphamodel, data.frame(x = data2test[(n1*(j-1)+1):(n1*j),1:d]))
          alphamat1[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- alpha1
          alphamat2[(n1*(i-1)+1):(n1*i),(n1*(j-1)+1):(n1*j)] <- matrix(rep(alpha2, n1), nrow = n1, ncol = n1, byrow = T)
        }
      }
      
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      
      psi <- gamma*a + alphamat2 - alphamat1*gamma
      theta <- mean(psi[ok1, ok2])
      sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
      T_hat_debias <- sqrt(n)*(0.5 - theta)/sqrt(sigma2)
      
      rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
      
      
      #### Hu & Lei, 2023 ----------
      ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
      ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
      g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
      g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
      V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
      V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
      Vg1 <- as.numeric(V1*g1)
      Vg2 <- as.numeric(V2*g2)
      
      
      Indicator <- matrix(0, length(Vg1), length(Vg2))
      rand <- runif(length(Vg2))
      for (j in 1:length(Vg2)) {
        Indicator[, j] <- ifelse(Vg1<Vg2[j], 1, 0) + ifelse(Vg1==Vg2[j], 1, 0)*rand[j]
      }
      Fn <- ecdf(Vg2)
      Fn_func <- function(x){
        return(sum(Vg2<x)/length(Vg2))
      }
      Fn_ <- 1 - sapply(Vg1, Fn_func)
      Fnhat <- 1 - Fn(Vg1)
      
      var_hat <- var(g1*(Fnhat + Fn_)/2)
      U <- rep(0, length(Vg2))
      for (j in 1:length(Vg2)) {
        U[j] <- sum(Indicator[, j]*g1)/sum(g1)
      }
      T_hat <- sqrt(length(Vg1))*(1/2-mean(U))/(sqrt(var_hat+length(Vg1)/(12*length(Vg2))+var(g1)/4-cov(g1, g1*(Fnhat + Fn_)/2)))
      rej <- rej + as.numeric(pnorm(T_hat)>0.95)
      T_vec[k] <- T_hat
      
      
      #### proposed ----------------
      X1 <- X1[n/2+which(ok1),]
      X2 <- X2[n/2+which(ok2),]
      
      Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
      for (j in 1:length(Vg1)) {
        Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
      }
      T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
      Kernel <- Kernel*(1/2-Indicator)
      var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(n/2)
      
      T_weight <- T_weight/sqrt(var_weight)
      rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
    }
    
    rate <- rej/ns
    rate_debias <- rej_debias/ns
    rate_weight <- rej_weight/ns
    result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n))
    result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n))
    result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)

pp <- Result_NN_alter%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp


########## plots the results -------------

Result_all <- data.frame(rbind(Result_LL_null,Result_RF_null,Result_NN_null,
                               Result_LL_alter,Result_RF_alter,Result_NN_alter))
dim(Result_all)
Result_all$Alg <- c(rep('LL',nrow(Result_LL_null)),rep('RF',nrow(Result_RF_null)),rep('NN',nrow(Result_NN_null)),
                    rep('LL',nrow(Result_LL_alter)),rep('RF',nrow(Result_RF_alter)),rep('NN',nrow(Result_NN_alter)))

Result_all$type <- c(rep('Type I error',nrow(Result_LL_null)+nrow(Result_RF_null)+nrow(Result_NN_null)),
                     rep('Power',nrow(Result_LL_alter)+nrow(Result_RF_alter)+nrow(Result_NN_alter)))
Result_all$type <- factor(Result_all$type, levels = c('Type I error','Power'))

Result_all$Method <- factor(Result_all$Method, levels = c('weight','ori','debias'))
Result_all$hline <- c(rep(0.05,nrow(Result_LL_null)+nrow(Result_RF_null)+nrow(Result_NN_null)),
                      rep(NA,nrow(Result_LL_alter)+nrow(Result_RF_alter)+nrow(Result_NN_alter)))
head(Result_all)

write.csv(Result_all,"ModelA_500times.csv")
#Result_all <- read.csv("ModelA_20.csv")[,-1]
Result_all$n <- factor(Result_all$n, levels = c('200','400','600','800','1000'))

pp <- Result_all%>%
  group_by(Method, n,Alg,type,hline)%>%
  dplyr::summarize(Quant = mean(quant, na.rm = TRUE), sdQuant = sd(quant, na.rm = TRUE))
pp

class(pp)


dev.off()

pdf(file="ModelA.pdf",
    width=8,height=6)
p1 <- ggplot(data = pp,aes(x=n,y=Quant,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.0)+geom_ribbon(aes(ymin = Quant - sdQuant,ymax = Quant + sdQuant),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("n")+
  ylab("")+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF"))+
  facet_grid(type~Alg, scales = "free")+
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

p1
dev.off()
#write.csv(pp,"ModelAdraw.csv")
dev.new()


