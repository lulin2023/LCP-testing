

###########  case 1 ----------------------------


##### airfoil data ----------------

library(splines)
library(plyr); library(dplyr)
library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(mvtnorm)
library(doSNOW)
library(nnet)
library(e1071)
library(ggsci)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))



H <- function(x){
  return(exp(-sum(x^2)/(2*h^2)))
}


## calculate the cross entropy error
## y:label, 0 or 1; p: estimated probability for label 1
centropy <- function(p, y){
  return(mean(-y*log(p)-(1-y)*log(1-p)))
}


gerror <- function(g1, g2){
  sum(abs(g1/sum(g1)-g2/sum(g2)))
}



### case 1 -----------------------

dat <- read.table("airfoil.txt")
dim(dat)
colnames(dat) <- c("Frequency",
                   "Angle",
                   "Chord",
                   "Velocity",
                   "Suction",
                   "Sound")

dat.x <- as.matrix(dat[,1:5])
dat.y <- as.numeric(dat[,6])
dat.x[,1] = log(dat.x[,1]) # Log transform
dat.x[,5] = log(dat.x[,5]) # Log transform
N <- nrow(dat.x); d <- ncol(dat.x)

n2 <- round(N/2); n1 <- N - n2
n12 <- floor(n1/2); n11 <- n1-n12
n22 <- floor(n2/2); n21 <- n2-n22

n12
n11
n22
n21


summary(dat.x)

######## Linear Logistic -----

nr <- 2
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_LL_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  
  n2 <- round(N/2); n1 <- N - n2
  n12 <- floor(n1/2); n11 <- n1-n12
  n22 <- floor(n2/2); n21 <- n2-n22  
  
  h <- (n2/2)^(-1/(2+d))
  rej <- 0
  rej_debias <- 0
  rej_weight <- 0
  
  T_vec <- rep(0, ns)
  T_debias_vec <- rep(0,ns)
  T_weight_vec <- rep(0,ns)
  
  for (k in 1:ns) {
    
    X_scale <- scale(dat.x)
    
    #sampling
    index <- sample(1:N, size = n1)
    X1 <- x1 <- dat.x[index,]; X2 <- x2 <- dat.x[-index,]
    Y1 <- y1 <- dat.y[index]; Y2 <- y2 <- dat.y[-index]
    
    
    x1_scale <- X_scale[index,]
    x2_scale <- X_scale[-index,]
    
    n1 <- n11+n12; n2 <- n21+n22
    index1 <- sample(1:n1, size = n11)
    x11 <- x1[index1,]; x12 <- x1[-index1,]
    y11 <- y1[index1]; y12 <- y1[-index1]
    index2 <- sample(1:n2, size = n21)
    x21 <- x2[index2,]; x22 <- x2[-index2,]
    y21 <- y2[index2]; y22 <- y2[-index2]
    
    
    
    data1train <- data.frame(x = x11, y = y11, label = factor(0))
    data2train <- data.frame(x = x21, y = y21, label = factor(1))
    data1test <- data.frame(x = x12, y = y12, label = factor(0))
    data2test <- data.frame(x = x22, y = y22, label = factor(1))
    
    g12.orac <- rep(1, n12); g22.orac <- rep(1, n22)
    
    # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
    # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
    # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
    class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
    prob.joint1 <- predict(class, data1test, type = 'response')
    prob.joint1[prob.joint1<0.01] <- 0.01; prob.joint1[prob.joint1>0.99] <- 0.99
    V1 <- (1-prob.joint1)/prob.joint1
    
    prob.joint2 <- predict(class, data2test, type = 'response')
    prob.joint2[prob.joint2<0.01] <- 0.01; prob.joint2[prob.joint2>0.99] <- 0.99
    V2 <- (1-prob.joint2)/prob.joint2
    
    
    # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
    # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
    
    # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
    # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
    # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
    classX <- glm(label~., data = rbind(data1train, data2train)[,-(d+1)], family = "binomial")
    prob.marginal1 <- predict(classX, data1test, type = 'response')
    prob.marginal1[prob.marginal1<0.01] <- 0.01; prob.marginal1[prob.marginal1>0.99] <- 0.99
    g1 <- prob.marginal1/(1-prob.marginal1)
    
    prob.marginal2 <- predict(classX, data2test, type = 'response')
    prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
    g2 <- prob.marginal2/(1-prob.marginal2)
    
    
    g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
    g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
    cerror12.ll.marginal <- mean(prob.marginal1>0.5)
    cerror22.ll.marginal <- mean(prob.marginal2<0.5)
    centropy.ll.marginal <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, n12), rep(1, n22)))
    error.ll <- gerror(g12.est.ll, g12.orac)
    
    # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
    # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
    
    Vg1 <- as.numeric(V1*g1)
    Vg2 <- as.numeric(V2*g2)
    Vg1[is.na(Vg1)] <- Inf
    Vg2[is.na(Vg2)] <- Inf
    
    
    #### Chen & Lei, 2024 debiased statistics-------------
    
    K <- 5
    n0 <- floor(n2/(2*K))
    
    a <- matrix(0, length(Vg1), length(Vg2))
    xi <- runif(length(Vg2))
    for (i in 1:(length(Vg1))) {
      for (j in 1:(length(Vg2))) {
        a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
      }
    }
    
    gamma <- matrix(0.02, length(Vg1), length(Vg2))
    alphamat1 <- matrix(0, length(Vg1), length(Vg2))
    alphamat2 <- matrix(0, length(Vg1), length(Vg2))
    for (i in 1:K) {
      for (j in 1:K) {
        classX_cross <- glm(label~., data = rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),])[,-(d+1)], family = "binomial")
        prob <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),], type = 'response')
        prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
        gamma1 <- prob/(1 - prob)
        # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
        # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
        
        gamma[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- gamma1
        astar <- apply(a[-((n0*(i-1)+1):(n0*i)),-((n0*(j-1)+1):(n0*j))], 1, mean)
        alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n0*(i-1)+1):(n0*i)),1:d]), ntree = 100)
        alpha1 <- predict(alphamodel, data.frame(x = data1test[(n0*(i-1)+1):(n0*i),1:d]))
        alpha2 <- predict(alphamodel, data.frame(x = data2test[(n0*(j-1)+1):(n0*j),1:d]))
        alphamat1[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- alpha1
        alphamat2[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- matrix(rep(alpha2, n0), nrow = n0, ncol = n0, byrow = T)
      }
    }
    
    ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
    ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
    
    psi <- gamma*a + alphamat2 - alphamat1*gamma
    theta <- mean(psi[ok1, ok2])
    sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
    T_hat_debias <- sqrt(length(Vg1)+length(Vg2))*(0.5 - theta)/sqrt(sigma2)
    T_debias_vec[k] <- T_hat_debias 
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
    # X1 <- X1[n11+which(ok1),]
    # X2 <- X2[n21+which(ok2),]
    # 
    # X1 <- scale(X1)
    # X2 <- scale(X2)
    X1 <- x1_scale[-index1,]
    X2 <- x2_scale[-index2,]
    
    Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
    for (j in 1:length(Vg1)) {
      Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
    }
    T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
    Kernel <- Kernel*(1/2-Indicator)
    var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(length(Vg1))
    
    T_weight <- T_weight/sqrt(var_weight)
    T_weight_vec[k] <- T_weight
    rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
  }
  
  rate <- rej/ns
  rate_debias <- rej_debias/ns
  rate_weight <- rej_weight/ns
  
  T_vec.ave <- mean(T_vec,na.rm=T)
  T_debias_vec.ave <- mean(T_debias_vec,na.rm=T)
  T_weight_vec.ave <- mean(T_weight_vec,na.rm=T)
  
  
  result <- rbind(result, data.frame(quant = rate, Method = 'ori',  error.ll= cerror12.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias',  error.ll= cerror12.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', error.ll= cerror12.ll.marginal, n = n2))
  
  return(result)
}
close(pb)
stopCluster(cl)


#### Null
Result0 <- Result_LL_null
#Result0 <- rbind(Result0, Result)
pp <- Result0%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant),
                   err=mean(error.ll))
pp



######## Random forest -----------

nr <- 20
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_RF_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  
  n2 <- round(N/2); n1 <- N - n2
  n12 <- floor(n1/2); n11 <- n1-n12
  n22 <- floor(n2/2); n21 <- n2-n22  
  
  h <- (n2/2)^(-1/(2+d))
  rej <- 0
  rej_debias <- 0
  rej_weight <- 0
  T_vec <- rep(0, ns)
  for (k in 1:ns) {
    
    
    X_scale <- scale(dat.x)
    
    #sampling
    index <- sample(1:N, size = n1)
    X1 <- x1 <- dat.x[index,]; X2 <- x2 <- dat.x[-index,]
    Y1 <- y1 <- dat.y[index]; Y2 <- y2 <- dat.y[-index]
    
    
    x1_scale <- X_scale[index,]
    x2_scale <- X_scale[-index,]
    
    n1 <- n11+n12; n2 <- n21+n22
    index1 <- sample(1:n1, size = n11)
    x11 <- x1[index1,]; x12 <- x1[-index1,]
    y11 <- y1[index1]; y12 <- y1[-index1]
    index2 <- sample(1:n2, size = n21)
    x21 <- x2[index2,]; x22 <- x2[-index2,]
    y21 <- y2[index2]; y22 <- y2[-index2]
    
    
    
    data1train <- data.frame(x = x11, y = y11, label = factor(1))
    data2train <- data.frame(x = x21, y = y21, label = factor(2))
    data1test <- data.frame(x = x12, y = y12, label = factor(1))
    data2test <- data.frame(x = x22, y = y22, label = factor(2))
    
    
    
    # class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
    # V1 <- (1 - predict(class, data1test, type = 'prob')[, 2])/predict(class, data1test, type = 'prob')[, 2]
    # V2 <- (1 - predict(class, data2test, type = 'prob')[, 2])/predict(class, data2test, type = 'prob')[, 2]
    class <- randomForest(label~., data = rbind(data1train, data2train), ntree = 100)
    prob.joint1 <- predict(class, data1test, type = 'prob')[, 2]
    prob.joint1[prob.joint1<0.01] <- 0.01; prob.joint1[prob.joint1>0.99] <- 0.99
    V1 <- (1-prob.joint1)/prob.joint1
    
    prob.joint2 <- predict(class, data2test, type = 'prob')[, 2]
    prob.joint2[prob.joint2<0.01] <- 0.01; prob.joint2[prob.joint2>0.99] <- 0.99
    V2 <- (1-prob.joint2)/prob.joint2
    
    
    # V1 <- (1 - predict(class, data1test, type = 'response'))/predict(class, data1test, type = 'response')
    # V2 <- (1 - predict(class, data2test, type = 'response'))/predict(class, data2test, type = 'response')
    
    # classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
    # g1 <- predict(classX, data1test, type = 'prob')[, 2]/(1 - predict(classX, data1test, type = 'prob')[, 2])
    # g2 <- predict(classX, data2test, type = 'prob')[, 2]/(1 - predict(classX, data2test, type = 'prob')[, 2])
    classX <- randomForest(label~., data = rbind(data1train, data2train)[,-(d+1)], ntree = 100)
    prob.marginal1 <- predict(classX, data1test, type = 'prob')[, 2]
    prob.marginal1[prob.marginal1<0.01] <- 0.01; prob.marginal1[prob.marginal1>0.99] <- 0.99
    g1 <- prob.marginal1/(1-prob.marginal1)
    
    prob.marginal2 <- predict(classX, data2test, type = 'prob')[, 2]
    prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
    g2 <- prob.marginal2/(1-prob.marginal2)
    
    # g1 <- predict(classX, data1test, type = 'response')/(1 - predict(classX, data1test, type = 'response'))
    # g2 <- predict(classX, data2test, type = 'response')/(1 - predict(classX, data2test, type = 'response'))
    
    g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
    g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
    cerror12.ll.marginal <- mean(prob.marginal1>0.5)
    cerror22.ll.marginal <- mean(prob.marginal2<0.5)
    centropy.ll.marginal <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, n12), rep(1, n22)))
    error.ll <- gerror(g12.est.ll, g12.orac)
    
    
    
    Vg1 <- as.numeric(V1*g1)
    Vg2 <- as.numeric(V2*g2)
    Vg1[is.na(Vg1)] <- Inf
    Vg2[is.na(Vg2)] <- Inf
    
    
    #### Chen & Lei, 2024 debiased statistics-------------
    
    K <- 5
    n0 <- floor(n2/(2*K))
    
    a <- matrix(0, length(Vg1), length(Vg2))
    xi <- runif(length(Vg2))
    for (i in 1:(length(Vg1))) {
      for (j in 1:(length(Vg2))) {
        a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
      }
    }
    
    gamma <- matrix(0.02, length(Vg1), length(Vg2))
    alphamat1 <- matrix(0, length(Vg1), length(Vg2))
    alphamat2 <- matrix(0, length(Vg1), length(Vg2))
    for (i in 1:K) {
      for (j in 1:K) {
        classX_cross <- randomForest(label~., data = rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),])[,-(d+1)], ntree = 100)
        prob <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),], type = 'prob')[, 2]
        prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
        gamma1 <- prob/(1 - prob)
        # classX_cross <- randomForest(label~., data = rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),])[,-(d+1)], ntree = 100)
        # gamma1 <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),], type = 'prob')[, 2])
        
        gamma[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- gamma1
        astar <- apply(a[-((n0*(i-1)+1):(n0*i)),-((n0*(j-1)+1):(n0*j))], 1, mean)
        alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n0*(i-1)+1):(n0*i)),1:d]), ntree = 100)
        alpha1 <- predict(alphamodel, data.frame(x = data1test[(n0*(i-1)+1):(n0*i),1:d]))
        alpha2 <- predict(alphamodel, data.frame(x = data2test[(n0*(j-1)+1):(n0*j),1:d]))
        alphamat1[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- alpha1
        alphamat2[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- matrix(rep(alpha2, n0), nrow = n0, ncol = n0, byrow = T)
      }
    }
    
    ok1 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
    ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
    
    psi <- gamma*a + alphamat2 - alphamat1*gamma
    theta <- mean(psi[ok1, ok2])
    sigma2 <- 2*mean((apply(psi[ok1, ok2], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok1, ok2], 2, mean) - 0.5)^2)
    T_hat_debias <- sqrt(length(Vg1)+length(Vg2))*(0.5 - theta)/sqrt(sigma2)
    
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
    X1 <- x1_scale[-index1,]
    X2 <- x2_scale[-index2,]
    
    Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
    for (j in 1:length(Vg1)) {
      Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
    }
    T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
    Kernel <- Kernel*(1/2-Indicator)
    var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(length(Vg1))
    
    T_weight <- T_weight/sqrt(var_weight)
    rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
  }
  
  rate <- rej/ns
  rate_debias <- rej_debias/ns
  rate_weight <- rej_weight/ns
  result <- rbind(result, data.frame(quant = rate, Method = 'ori', n = n2))
  result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias', n = n2))
  result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', n = n2))
  
  return(result)
}
close(pb)
stopCluster(cl)


Result0 <- Result_RF_null
#Result0 <- rbind(Result0, Result)
pp <- Result0%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant))
pp



########### Neural network --------------


nr <- 10
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_NN_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  
  n2 <- round(N/2); n1 <- N - n2
  n12 <- floor(n1/2); n11 <- n1-n12
  n22 <- floor(n2/2); n21 <- n2-n22  
  
  h <- (n2/2)^(-1/(2+d))
  rej <- 0
  rej_debias <- 0
  rej_weight <- 0
  T_vec <- rep(0, ns)
  for (k in 1:ns) {
    
    
    X_scale <- scale(dat.x)
    
    #sampling
    index <- sample(1:N, size = n1)
    X1 <- x1 <- dat.x[index,]; X2 <- x2 <- dat.x[-index,]
    Y1 <- y1 <- dat.y[index]; Y2 <- y2 <- dat.y[-index]
    
    
    x1_scale <- X_scale[index,]
    x2_scale <- X_scale[-index,]
    
    n1 <- n11+n12; n2 <- n21+n22
    index1 <- sample(1:n1, size = n11)
    x11 <- x1[index1,]; x12 <- x1[-index1,]
    y11 <- y1[index1]; y12 <- y1[-index1]
    index2 <- sample(1:n2, size = n21)
    x21 <- x2[index2,]; x22 <- x2[-index2,]
    y21 <- y2[index2]; y22 <- y2[-index2]
    
    
    g12.orac <- rep(1, n12); g22.orac <- rep(1, n22)
    
    data1train <- data.frame(x = x11, y = y11, label = factor(1))
    data2train <- data.frame(x = x21, y = y21, label = factor(2))
    data1test <- data.frame(x = x12, y = y12, label = factor(1))
    data2test <- data.frame(x = x22, y = y22, label = factor(2))
    
    
    class <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train)),size = 10, maxit = 200, linout = FALSE)
    prob.joint1 <- predict(class, data1test)
    prob.joint1[prob.joint1<0.01] <- 0.01; prob.joint1[prob.joint1>0.99] <- 0.99
    V1 <- (1-prob.joint1)/prob.joint1
    
    prob.joint2 <- predict(class, data2test)
    prob.joint2[prob.joint2<0.01] <- 0.01; prob.joint2[prob.joint2>0.99] <- 0.99
    V2 <- (1-prob.joint2)/prob.joint2
    
    classX <- nnet(label ~ ., data = data.frame(rbind(data1train, data2train))[-(d+1)],size = 10, maxit = 200, linout = FALSE)
    prob.marginal1 <- predict(classX, data1test)
    prob.marginal1[prob.marginal1<0.01] <- 0.01; prob.marginal1[prob.marginal1>0.99] <- 0.99
    g1 <- prob.marginal1/(1-prob.marginal1)
    
    prob.marginal2 <- predict(classX, data2test)
    prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
    g2 <- prob.marginal2/(1-prob.marginal2)
    
    
    g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
    g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
    cerror12.ll.marginal <- mean(prob.marginal1>0.5)
    cerror22.ll.marginal <- mean(prob.marginal2<0.5)
    centropy.ll.marginal <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, n12), rep(1, n22)))
    error.ll <- gerror(g12.est.ll, g12.orac)
    cerror12.ll.marginal
    
    
    # g1 <- predict(classX, data1test)/(1 - predict(classX, data1test))
    # g2 <- predict(classX, data2test)/(1 - predict(classX, data2test))
    
    ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
    ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
    g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
    g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
    V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
    V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
    Vg1 <- as.numeric(V1*g1)
    Vg2 <- as.numeric(V2*g2)
    
    
    #### Chen & Lei, 2024 debiased statistics-------------
    
    K <- 5
    n0 <- floor(n2/(2*K))
    
    a <- matrix(0, length(Vg1), length(Vg2))
    xi <- runif(length(Vg2))
    for (i in 1:(length(Vg1))) {
      for (j in 1:(length(Vg2))) {
        a[i, j] <- ifelse(Vg1[i]<Vg2[j], 1, 0) + xi[j]*ifelse(Vg1[i]==Vg2[j], 1, 0)
      }
    }
    
    gamma <- matrix(0.02, length(Vg1), length(Vg2))
    alphamat1 <- matrix(0, length(Vg1), length(Vg2))
    alphamat2 <- matrix(0, length(Vg1), length(Vg2))
    for (i in 1:K) {
      for (j in 1:K) {
        classX_cross <- nnet(label~., data = data.frame(rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),]))[,-(d+1)],size = 5, maxit = 200, linout = FALSE)
        prob <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),])
        prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
        gamma1 <- prob/(1 - prob)
        
        
        gamma[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- gamma1
        astar <- apply(a[-((n0*(i-1)+1):(n0*i)),-((n0*(j-1)+1):(n0*j))], 1, mean)
        alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n0*(i-1)+1):(n0*i)),1:d]), ntree = 100)
        alpha1 <- predict(alphamodel, data.frame(x = data1test[(n0*(i-1)+1):(n0*i),1:d]))
        alpha2 <- predict(alphamodel, data.frame(x = data2test[(n0*(j-1)+1):(n0*j),1:d]))
        alphamat1[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- alpha1
        alphamat2[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- matrix(rep(alpha2, n0), nrow = n0, ncol = n0, byrow = T)
      }
    }
    
    ok11 <- g1<100&g1>0.01&V1<100&V1>0.01&apply(gamma, 1, max)<100&apply(gamma, 1, min)>0.01
    ok22 <- g2<100&g2>0.01&V2<100&V2>0.01
    
    psi <- gamma*a + alphamat2 - alphamat1*gamma
    theta <- mean(psi[ok11, ok22])
    sigma2 <- 2*mean((apply(psi[ok11, ok22], 1, mean) - 0.5)^2) + 2*mean((apply(psi[ok11, ok22], 2, mean) - 0.5)^2)
    T_hat_debias <- sqrt(length(Vg1)+length(Vg2))*(0.5 - theta)/sqrt(sigma2)
    
    rej_debias <- rej_debias + as.numeric(pnorm(T_hat_debias)>0.95)
    
    
    #### Hu & Lei, 2023 ----------
    # ok1 <- g1<100&g1>0.01&V1<100&V1>0.01
    # ok2 <- g2<100&g2>0.01&V2<100&V2>0.01
    # g1 <- as.numeric(g1[ok1]*sum(ok1)/sum(ok2))
    # g2 <- as.numeric(g2[ok2]*sum(ok1)/sum(ok2))
    # V1 <- as.numeric(V1[ok1]*sum(ok2)/sum(ok1))
    # V2 <- as.numeric(V2[ok2]*sum(ok2)/sum(ok1))
    # Vg1 <- as.numeric(V1*g1)
    # Vg2 <- as.numeric(V2*g2)
    
    
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
    # X1 <- X1[n11+which(ok1),]
    # X2 <- X2[n21+which(ok2),]
    # 
    # X1 <- scale(X1)
    # X2 <- scale(X2)
    X1 <- x1_scale[-index1,]
    X2 <- x2_scale[-index2,]
    
    Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
    for (j in 1:length(Vg1)) {
      Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H)
    }
    T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
    Kernel <- Kernel*(1/2-Indicator)
    var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(length(Vg1))
    
    T_weight <- T_weight/sqrt(var_weight)
    rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
  }
  
  rate <- rej/ns
  rate_debias <- rej_debias/ns
  rate_weight <- rej_weight/ns
  result <- rbind(result, data.frame(quant = rate, Method = 'ori',  error.ll= cerror12.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_debias, Method = 'debias',  error.ll= cerror12.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_weight, Method = 'weight', error.ll= cerror12.ll.marginal, n = n2))
  
  return(result)
}
close(pb)
stopCluster(cl)


Result0 <- Result_NN_null
#Result0 <- rbind(Result0, Result)
pp <- Result0%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant),
                   err=mean(error.ll))
pp

