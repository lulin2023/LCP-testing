


############# case 3 -------------


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


## Load the airfoil data

dat <- read.table("airfoil.txt")
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
N <- nrow(dat.x); p <- ncol(dat.x)

dat.x <- dat.x[order(dat.y),]
dat.y <- sort(dat.y)

n1 <- round(N*0.5); n2 <- N-n1
x1 <- dat.x[1:n1,]; y1 <- dat.y[1:n1]
x2 <- dat.x[-(1:n1),]; y2 <- dat.y[-(1:n1)]

###group flip

flipsize <- round(n1*0.05)
ind1 <- sample(1:n1, flipsize, replace = F)
ind2 <- sample(1:n2, flipsize, replace = F)
x2temp <- x2[ind2,]; y2temp <- y2[ind2]
x1temp <- x1[ind1,]; y1temp <- y1[ind1]
x1[ind1, ]<- x2temp; y1[ind1] <- y2temp
x2[ind2, ] <- x1temp; y2[ind2] <- y1temp


########## Linear logistic -----------

nr <- 20
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_LL_null <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  
  
  n22 <- floor(n2/2); n21 <- n2-n22
  n12 <- floor(n1/2); n11 <- n1-n12
  
  
  rej <- 0
  rej_debias <- 0
  rej_weight <- 0
  T_vec <- rep(0, ns)
  pval_ori <- rep(0,ns)
  pval_debias <- rep(0,ns)
  pval_weight <- rep(0,ns)
  
  for (k in 1:ns) {
    
    h <- (n2/2)^(-1/(2+d))
    
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
    
    g12.orac <- rep(1, n12); g22.orac <- rep(1, n22)
    
    g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
    g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
    cerror12.ll.marginal <- mean(prob.marginal1>0.5)
    cerror22.ll.marginal <- mean(prob.marginal2<0.5)
    centropy.ll.marginal <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, n12), rep(1, n22)))
    error.ll <- gerror(g12.est.ll, g12.orac)
    error.ll
    
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
        classX_cross <- glm(label~., data = rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),])[,-(d+1)], family = "binomial")
        prob <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),], type = 'response')
        prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
        gamma1 <- prob/(1 - prob)
        # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
        # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
        
        gamma[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- gamma1
        astar <- apply(a[-((n0*(i-1)+1):(n0*i)),-((n0*(j-1)+1):(n0*j))], 1, mean)
        alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n0*(i-1)+1):(n0*i)),1:(d)]), ntree = 100)
        alpha1 <- predict(alphamodel, data.frame(x = data1test[(n0*(i-1)+1):(n0*i),1:(d)]))
        alpha2 <- predict(alphamodel, data.frame(x = data2test[(n0*(j-1)+1):(n0*j),1:(d)]))
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
    pval_debias[k] <- pnorm(-abs(T_hat_debias))
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
    pval_ori[k] <- pnorm(-abs(T_hat))
    T_vec[k] <- T_hat
    
    
    #### proposed ----------------
    X1 <- x12
    X2 <- x22
    
    var_vec <- diag(var(x12[1:n22,]-x22[1:n22,])/2)
    H_scale <- function(x){
      return(exp(-sum(x^2/(var_vec*h^2))/2))
    }
    
    
    Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
    for (j in 1:length(Vg1)) {
      Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H_scale)
    }
    
    T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
    Kernel <- Kernel*(1/2-Indicator)
    var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(length(Vg1))
    
    T_weight <- T_weight/sqrt(var_weight)
    pval_weight[k] <- pnorm(-abs(T_weight))
    rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
  }
  
  rate <- rej/ns
  rate_debias <- rej_debias/ns
  rate_weight <- rej_weight/ns
  
  pval_ori.med <- median(pval_ori)
  pval_debias.med <- median(pval_debias)
  pval_weight.med <- median(pval_weight)
  
  result <- rbind(result, data.frame(quant = rate, pval=pval_ori.med, Method = 'ori', error.ll= centropy.ll.marginal,n = n2))
  result <- rbind(result, data.frame(quant = rate_debias, pval=pval_debias.med, Method = 'debias',error.ll= centropy.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_weight, pval=pval_weight.med, Method = 'weight',error.ll= centropy.ll.marginal, n = n2))
  
 
  
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
                   Pval = median(pval), sdPval = sd(pval),
                   err=mean(error.ll))
pp


############# Neural network ---------------



nr <- 5
ns <- 50
cl <- makeCluster(8)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
Result_NN <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "grf","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  result <- data.frame()
  
  
  n22 <- floor(n2/2); n21 <- n2-n22
  n12 <- floor(n1/2); n11 <- n1-n12
  
  
  rej <- 0
  rej_debias <- 0
  rej_weight <- 0
  T_vec <- rep(0, ns)
  pval_ori <- rep(0,ns)
  pval_debias <- rep(0,ns)
  pval_weight <- rep(0,ns)
  
  for (k in 1:ns) {
    
    h <- (n2/2)^(-1/(2+d))
    
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
    
    
    # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
    # prob.joint1 <- predict(class, data1test, type = 'response')
    # prob.joint1[prob.joint1<0.01] <- 0.01; prob.joint1[prob.joint1>0.99] <- 0.99
    # V1 <- (1-prob.joint1)/prob.joint1
    # 
    # 
    # prob.joint2 <- predict(class, data2test, type = 'response')
    # prob.joint2[prob.joint2<0.01] <- 0.01; prob.joint2[prob.joint2>0.99] <- 0.99
    # V2 <- (1-prob.joint2)/prob.joint2
    # 
    # 
    # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d)], family = "binomial")
    # 
    # prob.marginal1 <- predict(classX, data1test, type = 'response')
    # prob.marginal1[prob.marginal1<0.01] <- 0.01; prob.marginal1[prob.marginal1>0.99] <- 0.99
    # g1 <- prob.marginal1/(1-prob.marginal1)
    # 
    # 
    # prob.marginal2 <- predict(classX, data2test, type = 'response')
    # prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
    # g2 <- prob.marginal2/(1-prob.marginal2)
    
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
    g1 <- prob.marginal1/(1-prob.marginal1)*n11/n12
    
    prob.marginal2 <- predict(classX, data2test)
    prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
    g2 <- prob.marginal2/(1-prob.marginal2)*n11/n12
    
    g12.orac <- rep(1, n12); g22.orac <- rep(1, n22)
    
    g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
    g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
    cerror12.ll.marginal <- mean(prob.marginal1>0.5)
    cerror22.ll.marginal <- mean(prob.marginal2<0.5)
    centropy.ll.marginal <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, nrow(data1test)), rep(1, nrow(data2test))))
    centropy.ll.marginal
    error.ll <- gerror(g12.est.ll, g12.orac)
    error.ll
    
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
        classX_cross <- nnet(label~., data = data.frame(rbind(data1test[-((n0*(i-1)+1):(n0*i)),], data2test[-((n0*(j-1)+1):(n0*j)),]))[,-(d+1)],size = 20, maxit = 200, linout = FALSE)
        prob <- predict(classX_cross, data1test[(n0*(i-1)+1):(n0*i),])
        prob[prob<0.01] <- 0.01; prob[prob>0.99] <- 0.99
        gamma1 <- prob/(1 - prob)
        # classX_cross <- randomForest(label~., data = rbind(data1test[-((n1*(i-1)+1):(n1*i)),], data2test[-((n1*(j-1)+1):(n1*j)),])[,-(d+1)], ntree = 100)
        # gamma1 <- predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2]/(1 - predict(classX_cross, data1test[(n1*(i-1)+1):(n1*i),], type = 'prob')[, 2])
        
        gamma[(n0*(i-1)+1):(n0*i),(n0*(j-1)+1):(n0*j)] <- gamma1
        astar <- apply(a[-((n0*(i-1)+1):(n0*i)),-((n0*(j-1)+1):(n0*j))], 1, mean)
        alphamodel <- randomForest(a~., data = data.frame(a = astar, x = data1test[-((n0*(i-1)+1):(n0*i)),1:(d)]), ntree = 100)
        alpha1 <- predict(alphamodel, data.frame(x = data1test[(n0*(i-1)+1):(n0*i),1:(d)]))
        alpha2 <- predict(alphamodel, data.frame(x = data2test[(n0*(j-1)+1):(n0*j),1:(d)]))
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
    pval_debias[k] <- 1-pnorm((T_hat_debias))
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
    pval_ori[k] <- 1-pnorm((T_hat))
    T_vec[k] <- T_hat
    
    
    #### proposed ----------------
    X1 <- x12
    X2 <- x22
    
    var_vec <- diag(var(x12[1:n22,]-x22[1:n22,])/2)
    H_scale <- function(x){
      return(exp(-sum(x^2/(var_vec*h^2))/2))
    }
    
    
    Kernel <- matrix(0, nrow = length(Vg1), ncol = length(Vg2))
    for (j in 1:length(Vg1)) {
      Kernel[j, ] <- apply(t(t(X2) - X1[j, ]), 1, H_scale)
    }
    
    T_weight <- sum(Kernel)/2 - sum(Kernel*Indicator)
    Kernel <- Kernel*(1/2-Indicator)
    var_weight <- -sum(Kernel^2) + sum(apply(Kernel, 1, sum)^2) + sum(apply(Kernel, 2, sum)^2) - 2*sum(Kernel)^2/(length(Vg1))
    
    T_weight <- T_weight/sqrt(var_weight)
    pval_weight[k] <- 1-pnorm((T_weight))
    rej_weight <- rej_weight + as.numeric(pnorm(T_weight)>0.95)
  }
  
  rate <- rej/ns
  rate_debias <- rej_debias/ns
  rate_weight <- rej_weight/ns
  
  pval_ori.med <- median(pval_ori)
  pval_debias.med <- median(pval_debias)
  pval_weight.med <- median(pval_weight)
  
  result <- rbind(result, data.frame(quant = rate, pval=pval_ori.med, Method = 'ori', error.ll= centropy.ll.marginal,n = n2))
  result <- rbind(result, data.frame(quant = rate_debias, pval=pval_debias.med, Method = 'debias',error.ll= centropy.ll.marginal, n = n2))
  result <- rbind(result, data.frame(quant = rate_weight, pval=pval_weight.med, Method = 'weight',error.ll= centropy.ll.marginal, n = n2))
  
  
  return(result)
}
close(pb)
stopCluster(cl)


#### Null
Result0 <- Result_NN
#Result0 <- rbind(Result0, Result)
pp <- Result0%>%
  group_by(Method, n)%>%
  dplyr::summarize(Quant = mean(quant), sdQuant = sd(quant),
                   Pval = median(pval), sdPval = sd(pval))
pp


centropy.ll.marginal <- rep(0,150)

for(i in 1:150){
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
  
  
  # class <- glm(label~., rbind(data1train, data2train), family = 'binomial')
  # prob.joint1 <- predict(class, data1test, type = 'response')
  # prob.joint1[prob.joint1<0.01] <- 0.01; prob.joint1[prob.joint1>0.99] <- 0.99
  # V1 <- (1-prob.joint1)/prob.joint1
  # 
  # 
  # prob.joint2 <- predict(class, data2test, type = 'response')
  # prob.joint2[prob.joint2<0.01] <- 0.01; prob.joint2[prob.joint2>0.99] <- 0.99
  # V2 <- (1-prob.joint2)/prob.joint2
  # 
  # 
  # classX <- glm(label~., data = rbind(data1train, data2train)[,-(d)], family = "binomial")
  # 
  # prob.marginal1 <- predict(classX, data1test, type = 'response')
  # prob.marginal1[prob.marginal1<0.01] <- 0.01; prob.marginal1[prob.marginal1>0.99] <- 0.99
  # g1 <- prob.marginal1/(1-prob.marginal1)
  # 
  # 
  # prob.marginal2 <- predict(classX, data2test, type = 'response')
  # prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
  # g2 <- prob.marginal2/(1-prob.marginal2)
  
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
  g1 <- prob.marginal1/(1-prob.marginal1)*n11/n12
  
  prob.marginal2 <- predict(classX, data2test)
  prob.marginal2[prob.marginal2<0.01] <- 0.01; prob.marginal2[prob.marginal2>0.99] <- 0.99
  g2 <- prob.marginal2/(1-prob.marginal2)*n11/n12
  
  g12.orac <- rep(1, n12); g22.orac <- rep(1, n22)
  
  g12.est.ll <- prob.marginal1/(1-prob.marginal1)*n11/n21
  g22.est.ll <- prob.marginal2/(1-prob.marginal2)*n11/n21
  cerror12.ll.marginal <- mean(prob.marginal1>0.5)
  cerror22.ll.marginal <- mean(prob.marginal2<0.5)
  centropy.ll.marginal[i] <- centropy(c(prob.marginal1,prob.marginal2), c(rep(0, nrow(data1test)), rep(1, nrow(data2test))))
  
}

mean(centropy.ll.marginal)
