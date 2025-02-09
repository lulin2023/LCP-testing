


######### conditional label screening ------------

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
library(ggsci)

######## CDC Diabetes Health Indicators dataset -----------

setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# import the data----

data <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
head(data)
colnames(data)

summary(data)
dim(data)

y1 <- as.factor(data$Diabetes_binary)
y2 <- as.factor(data$HeartDiseaseorAttack)
y3 <- as.factor(data$Stroke)

names(data)


data$y1=y1
data$y2=y2
data$y3=y3
table(data$y1)
table(data$y2)
table(data$y3)
head(data)
summary(data)
names(data)

data <- data[,-c(1,7,8)]

summary(data)
# simulation setting-----

d <- dim(data)[2]-3 # dimension of covariates


N <- 500
n <- 2000
alpha_array <- c(0.05)

h <- (n/2)^(-1/(2+d))

table(data$y1)




nr <- 500

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)





Result_RF <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  # Null=NullIndex(data$y1,Value)
  # Alter=setdiff(1:Number,Null)
  # 
  # IndexSample=c(sample(Null,round((n+N)*sr),replace = FALSE),sample(Alter,n+N-round((n+N)*sr),replace = FALSE))
  data$y2 <- as.numeric(data$y2)
  data_scale <- data
  data_scale[,1:d] <- scale(data[,1:d])
  summary(data)
  
  IndexSample <- sample(1:nrow(data),n+N,replace = FALSE)
  IndexSample1 <- sample(IndexSample,n+N,replace = FALSE)
  newdata=data[IndexSample1,]
  
  newdata.scale <- data_scale[IndexSample1,]
  # X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  # Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  # Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  # 
  # Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  # Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  datanew <- newdata
  datanew$y1 <- as.factor(datanew$y1)
  datanew$y2 <- as.factor(datanew$y2)
  datanew$y3 <- as.factor(datanew$y3)
  datatrain <- datanew[1:(n/2),]
  model1 <- randomForest(y1~., data = datatrain[,-c(d+2,d+3)])
  model2 <- randomForest(y2~., data = datatrain[,-c(d+1,d+3)])
  model3 <- randomForest(y3~., data = datatrain[,-c(d+1,d+2)])
  
  datacal <- datanew[(n/2+1):n,]
  datatest <- datanew[(n+1):(n+N),]
  
  datacal.scale <- newdata.scale[(n/2+1):n,]
  datatest.scale <- newdata.scale[(n+1):(n+N),]
  
  summary(datatest)
  
  Index1 <- datatest$Sex==0 & datatest$BMI > 30
  Index2 <- datatest$Sex==1 & datatest$BMI > 30
  Index3 <- datatest$Sex==0 & datatest$BMI <= 30
  Index4 <- datatest$Sex==1 & datatest$BMI <= 30
  
  V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  V3cal <- predict(model3, datacal, type = 'prob')[, 2]
  
  V1test <- predict(model1, datatest, type = 'prob')[, 2]
  V2test <- predict(model2, datatest, type = 'prob')[, 2]
  V3test <- predict(model3, datatest, type = 'prob')[, 2]
  
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0),ifelse(datacal$y3==0, V3cal, 0)), 1, max)
  
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest.scale[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  
  
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal.scale[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest.scale[, k]
  }
  weight[, n/2+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  weight <- weight/apply(weight, 1, sum)
  
  Indicator1 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator1[i, j] <- ifelse(V1test[i]<=V[j], 1, 0)
    }
  }
  Indicator2 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator2[i, j] <- ifelse(V2test[i]<=V[j], 1, 0)
    }
  }
  Indicator3 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator3[i, j] <- ifelse(V3test[i]<=V[j], 1, 0)
    }
  }
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  Indicator3[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  pvalue3 <- apply(weight*Indicator3, 1, sum)
  
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    s6 <- ifelse(V3test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0)|(s6==1&datatest$y3==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1)|(s6==1&datatest$y3==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2)|(s6==1&datatest$y3==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3)|(s6==1&datatest$y3==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4)|(s6==1&datatest$y3==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    s5 <- ifelse(pvalue3<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0)|(s5==1&datatest$y3==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1)|(s5==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2)|(s5==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3)|(s5==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4)|(s5==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
    result <- rbind(result, data.frame(mFWER = FWER_w, 
                                       cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
                                       POWER1 = POWER3, POWER2 = POWER4,
                                       Method = 'RLCP', alpha = alpha))
  }
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_RF
Result$alphadraw <- factor(Result$alpha, levels = c('0.05'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha,
                                   Result$POWER1,
                                   Result$POWER2), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method,Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result)),
                                  rep('Power1', nrow(Result)),
                                  rep('Power2', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)), 
                                   rep(NA, nrow(Result)),
                                   rep(NA, nrow(Result))))

P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(limits = c(-0.2, 1),
                     breaks = seq(0,1,0.25))+
  scale_x_discrete(name = "alpha") +
  ylab("") +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF"))+
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
  facet_grid(~Type,scales = "free_y")
P1


pp2 <- Result_RF%>%
  group_by(Method, alpha)%>%
  dplyr::summarize(m.FWER = mean(mFWER), sdmFWER = sd(mFWER),
                   cFWER = mean(cFWER1), sdcFWER = sd(cFWER1),
                   power1 = mean(POWER1), sdpower1 = sd(POWER1),
                   power2 = mean(POWER2), sdpower2 = sd(POWER2))
pp2



############  Linear logistic ---------------

nr <- 50

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_LL <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  
  # Null=NullIndex(data$y1,Value)
  # Alter=setdiff(1:Number,Null)
  # 
  # IndexSample=c(sample(Null,round((n+N)*sr),replace = FALSE),sample(Alter,n+N-round((n+N)*sr),replace = FALSE))
  data$y2 <- as.numeric(data$y2)
  data_scale <- data
  data_scale[,1:d] <- scale(data[,1:d])
  summary(data)
  
  IndexSample <- sample(1:nrow(data),n+N,replace = FALSE)
  IndexSample1 <- sample(IndexSample,n+N,replace = FALSE)
  newdata=data[IndexSample1,]
  
  newdata.scale <- data_scale[IndexSample1,]
  # X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  # Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  # Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  # 
  # Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  # Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  datanew <- newdata
  datanew$y1 <- as.factor(datanew$y1)
  datanew$y2 <- as.factor(datanew$y2)
  datanew$y3 <- as.factor(datanew$y3)
  datatrain <- datanew[1:(n/2),]
  # model1 <- randomForest(y1~., data = datatrain[,-c(d+2,d+3)])
  # model2 <- randomForest(y2~., data = datatrain[,-c(d+1,d+3)])
  # model3 <- randomForest(y3~., data = datatrain[,-c(d+1,d+2)])
  model1 <- glm(y1~., data = data.frame(datatrain[,-c(d+2,d+3)]),family = 'binomial')
  model2 <- glm(y2~., data = data.frame(datatrain[,-c(d+1,d+3)]),family = 'binomial')
  model3 <- glm(y3~., data = data.frame(datatrain[,-c(d+1,d+2)]),family = 'binomial')
  
  datacal <- datanew[(n/2+1):n,]
  datatest <- datanew[(n+1):(n+N),]
  
  datacal.scale <- newdata.scale[(n/2+1):n,]
  datatest.scale <- newdata.scale[(n+1):(n+N),]
  
  summary(datatest)
  
  Index1 <- datatest$Sex==0 & datatest$BMI > 30
  Index2 <- datatest$Sex==1 & datatest$BMI > 30
  Index3 <- datatest$Sex==0 & datatest$BMI <= 30
  Index4 <- datatest$Sex==1 & datatest$BMI <= 30
  
  # V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  # V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  # V3cal <- predict(model3, datacal, type = 'prob')[, 2]
  # 
  # V1test <- predict(model1, datatest, type = 'prob')[, 2]
  # V2test <- predict(model2, datatest, type = 'prob')[, 2]
  # V3test <- predict(model3, datatest, type = 'prob')[, 2]
  
  V1cal <- predict(model1, datacal,  type = 'response')
  V2cal <- predict(model2, datacal,  type = 'response')
  V3cal <- predict(model3, datacal,  type = 'response')
  
  V1test <- predict(model1, datatest,  type = 'response')
  V2test <- predict(model2, datatest,  type = 'response')
  V3test <- predict(model3, datatest,  type = 'response')
  
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0),ifelse(datacal$y3==0, V3cal, 0)), 1, max)
  
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest.scale[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  
  
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal.scale[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest.scale[, k]
  }
  weight[, n/2+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  weight <- weight/apply(weight, 1, sum)
  
  Indicator1 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator1[i, j] <- ifelse(V1test[i]<=V[j], 1, 0)
    }
  }
  Indicator2 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator2[i, j] <- ifelse(V2test[i]<=V[j], 1, 0)
    }
  }
  Indicator3 <- matrix(0, nrow = N, ncol = n/2+1)
  for (i in 1:N) {
    for (j in 1:(n/2)) {
      Indicator3[i, j] <- ifelse(V3test[i]<=V[j], 1, 0)
    }
  }
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  Indicator3[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  pvalue3 <- apply(weight*Indicator3, 1, sum)
  
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    s6 <- ifelse(V3test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0)|(s6==1&datatest$y3==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1)|(s6==1&datatest$y3==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2)|(s6==1&datatest$y3==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3)|(s6==1&datatest$y3==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4)|(s6==1&datatest$y3==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    s5 <- ifelse(pvalue3<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0)|(s5==1&datatest$y3==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1)|(s5==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2)|(s5==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3)|(s5==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4)|(s5==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
    result <- rbind(result, data.frame(mFWER = FWER_w, 
                                       cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
                                       POWER1 = POWER3, POWER2 = POWER4,
                                       Method = 'RLCP', alpha = alpha))
  }
  return(result)
}



# Result_LL <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","glmnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
#   
#   result <- data.frame()
#   
#   # Null=NullIndex(data$y1,Value)
#   # Alter=setdiff(1:Number,Null)
#   # 
#   # IndexSample=c(sample(Null,round((n+N)*sr),replace = FALSE),sample(Alter,n+N-round((n+N)*sr),replace = FALSE))
#   data$y2 <- as.numeric(data$y2)
#   data_scale <- data
#   data_scale[,1:d] <- scale(data[,1:d])
#   summary(data)
#   
#   IndexSample <- sample(1:nrow(data),n+N,replace = FALSE)
#   IndexSample1 <- sample(IndexSample,n+N,replace = FALSE)
#   newdata=data[IndexSample1,]
#   
#   newdata.scale <- data_scale[IndexSample1,]
#   # X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
#   # Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
#   # Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
#   # 
#   # Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
#   # Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
#   
#   datanew <- newdata
#   datanew$y1 <- as.factor(datanew$y1)
#   datanew$y2 <- as.factor(datanew$y2)
#   datatrain <- datanew[1:(n/2),]
#   # model1 <- randomForest(y1~., data = datatrain[,-(d+2)])
#   # model2 <- randomForest(y2~., data = datatrain[,-(d+1)])
#   model1 <- glm(y1~., data = data.frame(datatrain[,-(d+2)]),family = 'binomial')
#   model2 <- glm(y2~., data = data.frame(datatrain[,-(d+1)]),family = 'binomial')
#   
#   datacal <- datanew[(n/2+1):n,]
#   datatest <- datanew[(n+1):(n+N),]
#   
#   datacal.scale <- newdata.scale[(n/2+1):n,]
#   datatest.scale <- newdata.scale[(n+1):(n+N),]
#   
#   summary(datatest)
#   # Index1 <- datatest$x.1<0&datatest$x.3<0
#   # Index2 <- datatest$x.1>0&datatest$x.3<0
#   # Index3 <- datatest$x.1<0&datatest$x.3>0
#   # Index4 <- datatest$x.1>0&datatest$x.3>0
#   
#   Index1 <- datatest$Sex==0 & datatest$BMI > 30
#   Index2 <- datatest$Sex==1 & datatest$BMI > 30
#   Index3 <- datatest$Sex==0 & datatest$BMI <= 30
#   Index4 <- datatest$Sex==1 & datatest$BMI <= 30
#   
#   # V1cal <- predict(model1, datacal, type = 'prob')[, 2]
#   # V2cal <- predict(model2, datacal, type = 'prob')[, 2]
#   # V1test <- predict(model1, datatest, type = 'prob')[, 2]
#   # V2test <- predict(model2, datatest, type = 'prob')[, 2]
#   V1cal <- predict(model1, datacal,  type = 'response')
#   V2cal <- predict(model2, datacal,  type = 'response')
#   V1test <- predict(model1, datatest,  type = 'response')
#   V2test <- predict(model2, datatest,  type = 'response')
#   V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0)), 1, max)
#   
#   
#   s_sam <- matrix(0, ncol = d, nrow = N)
#   for (i in 1:N) {
#     s_sam[i,] <- mvrnorm(1, as.numeric(datatest.scale[i, 1:d]), (h^2)*diag(d))
#   }
#   
#   weight <- matrix(0, nrow = N, ncol = n/2+1)
#   
#   
#   for (j in 1:(n/2)) {
#     diffmat <- matrix(0, nrow = N, ncol = d)
#     for (k in 1:d) {
#       diffmat[, k] <- s_sam[, k] - datacal.scale[j, k]
#     }
#     weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
#   }
#   diffmat <- matrix(0, nrow = N, ncol = d)
#   for (k in 1:d) {
#     diffmat[, k] <- s_sam[, k] - datatest.scale[, k]
#   }
#   weight[, n/2+1] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
#   weight <- weight/apply(weight, 1, sum)
#   
#   Indicator1 <- matrix(0, nrow = N, ncol = n/2+1)
#   for (i in 1:N) {
#     for (j in 1:(n/2)) {
#       Indicator1[i, j] <- ifelse(V1test[i]<=V[j], 1, 0)
#     }
#   }
#   Indicator2 <- matrix(0, nrow = N, ncol = n/2+1)
#   for (i in 1:N) {
#     for (j in 1:(n/2)) {
#       Indicator2[i, j] <- ifelse(V2test[i]<=V[j], 1, 0)
#     }
#   }
#   rand <- runif(N)
#   Indicator1[, n/2+1] <- rand
#   Indicator2[, n/2+1] <- rand
#   
#   pvalue1 <- apply(weight*Indicator1, 1, sum)
#   pvalue2 <- apply(weight*Indicator2, 1, sum)
#   
#   
#   
#   for(alpha in alpha_array){
#     
#     # thresholding rule without weighting
#     thr <- quantile(V, 1-alpha)
#     
#     s1 <- ifelse(V1test>thr, 1, 0)
#     s2 <- ifelse(V2test>thr, 1, 0)
#     FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0), 1, 0))
#     POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
#     POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
#     
#     FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
#     FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
#     FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
#     FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
#     result <- rbind(result, data.frame(mFWER = FWER, 
#                                        cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
#                                        POWER1 = POWER1, POWER2=POWER2,
#                                        Method = 'THR', alpha = alpha))
#     
#     # thresholding rule with RLCP
#     s3 <- ifelse(pvalue1<alpha, 1, 0)
#     s4 <- ifelse(pvalue2<alpha, 1, 0)
#     FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0), 1, 0))
#     POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
#     POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
#     
#     FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
#     FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
#     FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
#     FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
#     
#     result <- rbind(result, data.frame(mFWER = FWER_w, 
#                                        cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
#                                        POWER1 = POWER3, POWER2 = POWER4,
#                                        Method = 'RLCP', alpha = alpha))
#   }
#   return(result)
# }

close(pb)
stopCluster(cl)

Result <- Result_LL
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha,
                                   Result$POWER1,
                                   Result$POWER2), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method,Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result)),
                                  rep('Power1', nrow(Result)),
                                  rep('Power2', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)), 
                                   rep(NA, nrow(Result)),
                                   rep(NA, nrow(Result))))

P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(limits = c(-0.2, 1),
                     breaks = seq(0,1,0.25))+
  scale_x_discrete(name = "alpha") +
  ylab("") +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF"))+
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
  facet_grid(~Type,scales = "free_y")
P1


################# final results --------------------

head(Result_LL)
pp1 <- Result_LL%>%
  group_by(Method, alpha)%>%
  dplyr::summarize(m.FWER = mean(mFWER), sdmFWER = sd(mFWER,na.rm=T),
                   cFWER = mean(cFWER1), sdcFWER = sd(cFWER1,na.rm=T))
pp1

pp2 <- Result_RF%>%
  group_by(Method, alpha)%>%
  dplyr::summarize(m.FWER = mean(mFWER), sdmFWER = sd(mFWER),
                   cFWER = mean(cFWER1), sdcFWER = sd(cFWER1))
pp2



Result_all <- rbind(Result_RF,Result_LL)
Result_all$Alg <- c(rep('RF',nrow(Result_RF)),rep('LL',nrow(Result_LL)))
Result <- Result_all
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result_all)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))),
                         Alg = c(Result$Alg,Result$Alg,Result$Alg))
pdf(file="Labelscreen_Census_FWER.pdf",
    width=10,height=6)
P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, color = Method)) +
  geom_boxplot(alpha = 0.8) +
  scale_y_continuous(limits = c(-0.2, 0.25),
                     breaks = seq(0,1,0.1))+
  scale_x_discrete(name = "alpha") +
  ylab("") +
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme_bw()  +
  stat_summary(mapping = aes(group = Method),
               fun = "mean",
               geom = "point", shape = 23, size = 1.1, fill = "red",
               position = position_dodge(0.8)) +
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF"))+
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
  facet_grid(Alg~Type,scales = "free_y")
P1
dev.off()
dev.new()
