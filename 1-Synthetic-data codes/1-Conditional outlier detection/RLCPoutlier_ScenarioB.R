
# RLCP_outlier_ScenarioB

library(kernlab)
library(MASS)
library("ks")
library(foreach)
library(randomForest)
library(doParallel)
require(ggplot2)
library(ggpubr)
library(caret)
library(grf)
library(isotree)
library(doSNOW)
library(e1071)
library(nnet)
library(kknn)
library(dbscan)
library(FNN)
library(class)


setwd(dirname(rstudioapi::getSourceEditorContext()$path))
#setwd("/home/ll2120210104/RLCP")


#############  different alpha -------------------


d <- 50
n <- 2000
N <- 2000
#h <- 0.2
h <- (n)^(-1/(2+1))



#### isolation forest ----------


nr <- 500
numcores <- 8
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_IOF <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    model <- isolation.forest(data = data.frame(X = X[1:n,]))
    
    data <- data.frame(s = s, X = X, V = predict(model, data.frame(X = X)))
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
   
    
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', alpha=alpha))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', alpha))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_IOF
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Result$Alg <- rep('IOF',nrow(Result))
#write.csv(Result,"Outlier_ScenarioA_alpha_IOF.csv")
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
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


#### one-class SVM ----------

nr <- 500
numcores <- 24
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_oneSVM <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree","e1071"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    #model <- isolation.forest(data = data.frame(X = X[1:n,]))
    model <- svm(x = data.frame(X = X[1:n,]), type="one-classification", kernel="radial",nu=0.1,scale=TRUE)
    
    predictions <- predict(model, data.frame(X = X))
    
    decision_values <- attributes(predict(model, data.frame(X = X), decision.values = TRUE))$decision.values
    
    probabilities <- 1-1 / (1 + exp(-decision_values))
    
    
    data <- data.frame(s = s, X = X, V = probabilities)
    colnames(data)[51] <- 'V'
    head(data)
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', alpha=alpha))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', alpha=alpha))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_oneSVM
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
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


###########  K Nearest Neighbor ---------------------


nr <- 500
numcores <- 24
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_KNN <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree","e1071","dbscan","class","FNN"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (alpha in c(0.05, 0.1, 0.15, 0.2)) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    #model <- isolation.forest(data = data.frame(X = X[1:n,]))
    #model <- svm(x = data.frame(X = X[1:n,]), type="one-classification", kernel="radial",nu=0.1,scale=TRUE)
    
    
    training_data <- data.frame(X=X[1:n,])
    
    # Define a function to compute k-NN distances
    compute_knn_distances <- function(train_data, test_data, k) {
      distances <- knnx.dist(train_data, test_data, k = k)
      return(distances)
    }
    
    # Fit the k-NN model on training data
    k <- 5  # Number of neighbors
    knn_distances_train <- compute_knn_distances(training_data, training_data, k)
    
    # Compute anomaly scores for the training data
    mean_knn_distances_train <- rowMeans(knn_distances_train)
    
    # Normalize the anomaly scores to probabilities
    min_score <- min(mean_knn_distances_train)
    max_score <- max(mean_knn_distances_train)
    probabilities_train <- (mean_knn_distances_train - min_score) / (max_score - min_score)
    
    # Create new test data
    test_data <- data.frame(X=X)
    
    # Compute k-NN distances for the new data
    knn_distances_test <- compute_knn_distances(training_data, test_data, k)
    
    # Compute anomaly scores for the new data
    mean_knn_distances_test <- rowMeans(knn_distances_test)
    
    
    # Normalize the anomaly scores to probabilities for the new data
    min_score <- min(mean_knn_distances_test)
    max_score <- max(mean_knn_distances_test)
    probabilities_test <- (mean_knn_distances_test - min_score) / (max_score - min_score)
    
    
    data <- data.frame(s = s, X = X, V = probabilities_test)
    colnames(data)[51] <- 'V'
    head(data)
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', alpha=alpha))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', alpha=alpha))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_KNN
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


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

Result_all <- data.frame(rbind(Result_IOF,Result_oneSVM,Result_KNN))

Result_all$Alg <- c(rep('Isolation Forest',nrow(Result_KNN)),
                    rep('one-class SVM',nrow(Result_KNN)),
                    rep('KNN',nrow(Result_KNN)))
head(Result_all)

#Result_all <- read.csv("results/Outlier_ScenarioB_alpha_500times.csv")[,-1]
Result_all$alphadraw <- factor(Result_all$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
Result_all$Method <- factor(Result_all$Method, levels = c('RLCP', 'CP'))

#write.csv(Result_all,"results/Outlier_ScenarioB_alpha.csv")

P1 <- ggplot(Result_all, aes(x = alphadraw, y = POWER, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(0,1) +
  scale_x_discrete(name = "alpha") +
  ylab("POWER") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)

P1

P2 <- ggplot(Result_all, aes(x = alphadraw, y = FDP-alpha, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(-0.15,0.25)+
  scale_x_discrete(name = "alpha") +
  ylab("FDP above nominal") +
  geom_hline(aes(yintercept = 0), colour = "#AA0000", na.rm = T) +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)
P2

dev.off()

pdf(file="results/ScenarioB_alpha.pdf",
    width=10,height=7)
P <- ggarrange(P2, P1, ncol=1, nrow=2, common.legend = TRUE, legend="bottom",
               font.label = list(size = 20, face = "bold"))
P
dev.off()















############ different n --------------
d <- 50
n <- 2000
N <- 2000
#h <- 0.2
h <- (n)^(-1/(2+1))
alpha <- 0.1



#### isolation forest ----------


nr <- 500
numcores <- 8
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_IOF_n <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (n in 800*1:5) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    model <- isolation.forest(data = data.frame(X = X[1:n,]))
    
    data <- data.frame(s = s, X = X, V = predict(model, data.frame(X = X)))
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', n = n))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_IOF_n
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$ndraw <- factor(Result$n, levels = c('800', '1600', '2400', '3200', '4000'))
Result$Method <- factor(Result$Method, levels = c('RLCP', 'CP'))
Result$Alg <- rep('IOF',nrow(Result))
#write.csv(Result,"Outlier_ScenarioB_n_IOF_500times.csv")
Resultdraw <- data.frame(Value = c(Result$FDP-alpha, Result$POWER), Method = c(Result$Method, Result$Method), ndraw = c(Result$ndraw, Result$ndraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
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


#### one-class SVM ----------

nr <- 500
numcores <- 24
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_oneSVM <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree","e1071"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (n in 800*1:5) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    #model <- isolation.forest(data = data.frame(X = X[1:n,]))
    model <- svm(x = data.frame(X = X[1:n,]), type="one-classification", kernel="radial",nu=0.1,scale=TRUE)
    
    predictions <- predict(model, data.frame(X = X))
    
    decision_values <- attributes(predict(model, data.frame(X = X), decision.values = TRUE))$decision.values
    
    probabilities <- 1-1 / (1 + exp(-decision_values))
    
    
    data <- data.frame(s = s, X = X, V = probabilities)
    colnames(data)[51] <- 'V'
    head(data)
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', n = n))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_oneSVM
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$ndraw <- factor(Result$n, levels = c('800', '1600', '2400', '3200', '4000'))
Result$Method <- factor(Result$Method, levels = c('RLCP', 'CP'))
Resultdraw <- data.frame(Value = c(Result$FDP-alpha, Result$POWER), Method = c(Result$Method, Result$Method), ndraw = c(Result$ndraw, Result$ndraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
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

Result_all <- data.frame(rbind(Result_IOF,Result_oneSVM))
dim(Result_all)
head(Result_all)
Result_all$Alg <- c(rep('IOF',nrow(Result_IOF)),rep('one-class SVM',nrow(Result_oneSVM)))
head(Result_all)


Result_all$ndraw <- factor(Result$n, levels = c('800', '1600', '2400', '3200', '4000'))


P1 <- ggplot(Result_all, aes(x = ndraw, y = POWER, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(0,1) +
  scale_x_discrete(name = "n") +
  ylab("POWER") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)

P1

P2 <- ggplot(Result_all, aes(x = ndraw, y = FDP-alpha, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(-0.15,0.25)+
  scale_x_discrete(name = "n") +
  ylab("FDP above nominal") +
  geom_hline(aes(yintercept = 0), colour = "#AA0000", na.rm = T) +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)
P2

dev.off()

pdf(file="results/ScenarioB.pdf",
    width=9,height=7)
P <- ggarrange(P2, P1, ncol=1, nrow=2, common.legend = TRUE, legend="bottom",
               font.label = list(size = 20, face = "bold"))
P
dev.off()

#write.csv(Result_all,file = "Result_outlier_ScenarioB_different_n.csv")


###########  K Nearest Neighbor ---------------------


nr <- 500
numcores <- 24
cl=makeCluster(numcores)
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

Result_KNN <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks", "mvtnorm", "isotree","e1071","dbscan","class","FNN"), .errorhandling = "remove",.options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  for (n in 800*1:5) {
    
    s <- matrix(runif((2*n+N)*2, -1, 1), ncol = 2)
    X <- mvrnorm(2*n+N, rep(0, d-2), diag(d-2))*sqrt(0.2+0.9*(s[, 1]^2+s[, 2]^2))
    outlier <- sample(1:N, 0.1*N)
    X[outlier+2*n,] <- 2*X[outlier+2*n,]
    
    #model_ada <- randomForest(y~., data = data.frame(s = s, X = X, y = as.factor(c(rep(0, n), rep(1, n+N)))))
    #model <- isolation.forest(data = data.frame(X = X[1:n,]))
    #model <- svm(x = data.frame(X = X[1:n,]), type="one-classification", kernel="radial",nu=0.1,scale=TRUE)
    
    
    training_data <- data.frame(X=X[1:n,])
    
    # Define a function to compute k-NN distances
    compute_knn_distances <- function(train_data, test_data, k) {
      distances <- knnx.dist(train_data, test_data, k = k)
      return(distances)
    }
    
    # Fit the k-NN model on training data
    k <- 5  # Number of neighbors
    knn_distances_train <- compute_knn_distances(training_data, training_data, k)
    
    # Compute anomaly scores for the training data
    mean_knn_distances_train <- rowMeans(knn_distances_train)
    
    # Normalize the anomaly scores to probabilities
    min_score <- min(mean_knn_distances_train)
    max_score <- max(mean_knn_distances_train)
    probabilities_train <- (mean_knn_distances_train - min_score) / (max_score - min_score)
    
    # Create new test data
    test_data <- data.frame(X=X)
    
    # Compute k-NN distances for the new data
    knn_distances_test <- compute_knn_distances(training_data, test_data, k)
    
    # Compute anomaly scores for the new data
    mean_knn_distances_test <- rowMeans(knn_distances_test)
    
    
    # Normalize the anomaly scores to probabilities for the new data
    min_score <- min(mean_knn_distances_test)
    max_score <- max(mean_knn_distances_test)
    probabilities_test <- (mean_knn_distances_test - min_score) / (max_score - min_score)
    
    
    data <- data.frame(s = s, X = X, V = probabilities_test)
    colnames(data)[51] <- 'V'
    head(data)
    summary(data$V[outlier+2*n])
    summary(data$V[-(outlier+2*n)])
    
    datacal <- data[(n+1):(2*n),]
    datatest <- data[(2*n+1):(2*n+N),]
    datatrain <- data[1:n,]
    
    s_sam <- matrix(0, ncol = 2, nrow = N)
    for (i in 1:N) {
      s_sam[i,] <- mvrnorm(1, c(datatest$s.1[i], datatest$s.2[i]), (h^2)*diag(2))
    }
    
    weight <- matrix(0, nrow = N, ncol = n+1)
    for (j in 1:n) {
      #weight[, j] <- ifelse(apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)<=h^2, 1, 0)
      #weight[, j] <- exp(-apply((cbind(datatest$s.1-datacal$s.1[j], datatest$s.2-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
      weight[, j] <- exp(-apply((cbind(s_sam[, 1]-datacal$s.1[j], s_sam[, 2]-datacal$s.2[j]))^2, 1, sum)/(2*h^2))
    }
    weight[, n+1] <- exp(-apply((cbind(s_sam[, 1]-datatest$s.1, s_sam[, 2]-datatest$s.2))^2, 1, sum)/(2*h^2))
    
    
    IndMat <- matrix(1, nrow = N, ncol = n+1)
    for (j in 1:n) {
      IndMat[,j] <- ifelse(datatest$V<=datacal$V[j], 1, 0)
    }
    IndMat[, n+1] <- runif(N)
    W <- weight*IndMat
    
    
    ###---Weighted_CP---###
    pvalues <- (apply(W, 1, sum))/(apply(weight, 1, sum))
    
    Rtild <- rep(0, N)
    unnorm_p <- apply(W, 1, sum)
    sum_weight <- apply(weight, 1, sum)
    for (j in 1:N) {
      pvalues_j <- (unnorm_p - W[, n+1] + weight[, n+1]*ifelse(datatest$V<=datatest$V[j], 1, 0))/sum_weight
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
    result <- rbind(result, data.frame(FDP = FDP, POWER = POWER, Method = 'RLCP', n = n))
    
    ###---CP---###
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
    result <- rbind(result, data.frame(FDP = FDP_CP, POWER = POWER_CP, Method = 'CP', n = n))
  }
  
  return(result)
}
close(pb)
stopCluster(cl)


Result <- Result_KNN
#Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
#Resultdraw <- data.frame(Value = c(Result$FDP-Result$alpha, Result$POWER), Method = c(Result$Method, Result$Method), alphadraw = c(Result$alphadraw, Result$alphadraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))
Result$ndraw <- factor(Result$n, levels = c('800', '1600', '2400', '3200', '4000'))
Resultdraw <- data.frame(Value = c(Result$FDP-alpha, Result$POWER), Method = c(Result$Method, Result$Method), ndraw = c(Result$ndraw, Result$ndraw), Type = c(rep('FDP above nominal', nrow(Result)), rep('Power', nrow(Result))), hline = c(rep(0, nrow(Result)), rep(NA, nrow(Result))))


P1 <- ggplot(data = Resultdraw, aes(x = ndraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "") +
  scale_x_discrete(name = "n") +
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

Result_all <- data.frame(rbind(Result_IOF,Result_oneSVM,Result_KNN))
#Result1 <- read.csv("results/Result_outlier_ScenarioB_different_n.csv")[,-1]
#Result1 <- Result1[,1:4]
#head(Result1)
head(Result_KNN)
dim(Result_all)
head(Result_all)
Result_all$Alg <- c(rep('Isolation Forest',nrow(Result_IOF)),
                    rep('one-class SVM',nrow(Result_oneSVM)),
                    rep('KNN',nrow(Result_KNN)))
head(Result_all)

#Result_all <- Result1
Result_all <- read.csv("results/Outlier_ScenarioB_n_500times.csv")[,-1]
Result_all$ndraw <- factor(Result_all$n, levels = c('800', '1600', '2400', '3200', '4000'))
Result_all$Method <- factor(Result_all$Method, levels = c('RLCP', 'CP'))

#write.csv(Result_all,"results/Outlier_ScenarioB_n.csv")

P1 <- ggplot(Result_all, aes(x = ndraw, y = POWER, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(0,1) +
  scale_x_discrete(name = "n") +
  ylab("POWER") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)

P1

P2 <- ggplot(Result_all, aes(x = ndraw, y = FDP-alpha, fill=Method)) +
  geom_boxplot(alpha=0.7)  +
  ylim(-0.15,0.25)+
  scale_x_discrete(name = "n") +
  ylab("FDP above nominal") +
  geom_hline(aes(yintercept = 0), colour = "#AA0000", na.rm = T) +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme(plot.title = element_text(size = 14, face = "bold"),
        text = element_text(size = 20),
        axis.title = element_text(face = "bold"),
        axis.text.x = element_text()) +
  facet_grid(~Alg)
P2

dev.off()

pdf(file="results/ScenarioB.pdf",
    width=10,height=7)
P <- ggarrange(P2, P1, ncol=1, nrow=2, common.legend = TRUE, legend="bottom",
               font.label = list(size = 20, face = "bold"))
P
dev.off()

