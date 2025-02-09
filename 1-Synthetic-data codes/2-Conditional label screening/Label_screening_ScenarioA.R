


######### conditional label screening -----------
setwd(dirname(rstudioapi::getSourceEditorContext()$path))


N <- 500
n <- 2000
alpha_array <- c(0.05,0.1,0.15,0.2)
d <- 4
h <- (n/2)^(-1/(2+d))


######## random forest -----------

nr <- 100

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_RF <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  
  Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  data <- data.frame(x = X, y1  = Y1, y2 = Y2)
  datatrain <- data[1:(n/2),]
  model1 <- randomForest(y1~., data = datatrain[,-(d+2)])
  model2 <- randomForest(y2~., data = datatrain[,-(d+1)])
  
  datacal <- data[(n/2+1):n,]
  datatest <- data[(n+1):(n+N),]
  Index1 <- datatest$x.1<0&datatest$x.3<0
  Index2 <- datatest$x.1>0&datatest$x.3<0
  Index3 <- datatest$x.1<0&datatest$x.3>0
  Index4 <- datatest$x.1>0&datatest$x.3>0
  
  
 
  V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  V1test <- predict(model1, datatest, type = 'prob')[, 2]
  V2test <- predict(model2, datatest, type = 'prob')[, 2]
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0)), 1, max)
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest[, k]
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
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
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


Result <- Result_RF
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))))

pdf(file="Labelscreen_ScenarioA_FWER.pdf",
    width=10,height=4)
P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, color = Method)) +
  geom_boxplot(alpha = 0.8) +
  scale_y_continuous(limits = c(-0.2, 0.2),
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
  facet_grid(~Type,scales = "free_y")
P1
dev.off()
dev.new()


######  neural network -----------

nr <- 100

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_NN <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","nnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  
  Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  data <- data.frame(x = X, y1  = Y1, y2 = Y2)
  datatrain <- data[1:(n/2),]
  # model1 <- randomForest(y1~., data = datatrain[,-(d+2)])
  # model2 <- randomForest(y2~., data = datatrain[,-(d+1)])
  model1 <- nnet(y1~., data = data.frame(datatrain[,-(d+2)]),size = 10, maxit = 100, linout = FALSE)
  model2 <- nnet(y2~., data = data.frame(datatrain[,-(d+1)]),size = 10, maxit = 100, linout = FALSE)
  
  datacal <- data[(n/2+1):n,]
  datatest <- data[(n+1):(n+N),]
  Index1 <- datatest$x.1<0&datatest$x.3<0
  Index2 <- datatest$x.1>0&datatest$x.3<0
  Index3 <- datatest$x.1<0&datatest$x.3>0
  Index4 <- datatest$x.1>0&datatest$x.3>0
  
  
  
  # V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  # V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  # V1test <- predict(model1, datatest, type = 'prob')[, 2]
  # V2test <- predict(model2, datatest, type = 'prob')[, 2]
  V1cal <- predict(model1, datacal)
  V2cal <- predict(model2, datacal)
  V1test <- predict(model1, datatest)
  V2test <- predict(model2, datatest)
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0)), 1, max)
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest[, k]
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
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
    result <- rbind(result, data.frame(mFWER = FWER_w, 
                                       cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
                                       POWER1 = POWER3, POWER2 = POWER4,
                                       Method = 'RLCP', alpha = alpha))
  }
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_NN
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))))
pdf(file="Labelscreen_ScenarioA_FWER_NN.pdf",
    width=10,height=4)
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
  facet_grid(~Type,scales = "free_y")
P1
dev.off()


############## linear logistic--------------

nr <- 100

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_LL <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","nnet","glmnet"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  
  Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  data <- data.frame(x = X, y1  = Y1, y2 = Y2)
  datatrain <- data[1:(n/2),]
  # model1 <- randomForest(y1~., data = datatrain[,-(d+2)])
  # model2 <- randomForest(y2~., data = datatrain[,-(d+1)])
  model1 <- glm(y1~., data = data.frame(datatrain[,-(d+2)]),family = 'binomial')
  model2 <- glm(y2~., data = data.frame(datatrain[,-(d+1)]),family = 'binomial')
  
  datacal <- data[(n/2+1):n,]
  datatest <- data[(n+1):(n+N),]
  Index1 <- datatest$x.1<0&datatest$x.3<0
  Index2 <- datatest$x.1>0&datatest$x.3<0
  Index3 <- datatest$x.1<0&datatest$x.3>0
  Index4 <- datatest$x.1>0&datatest$x.3>0
  
  
  
  # V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  # V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  # V1test <- predict(model1, datatest, type = 'prob')[, 2]
  # V2test <- predict(model2, datatest, type = 'prob')[, 2]
  V1cal <- predict(model1, datacal,  type = 'response')
  V2cal <- predict(model2, datacal,  type = 'response')
  V1test <- predict(model1, datatest,  type = 'response')
  V2test <- predict(model2, datatest,  type = 'response')
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0)), 1, max)
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest[, k]
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
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
    result <- rbind(result, data.frame(mFWER = FWER_w, 
                                       cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
                                       POWER1 = POWER3, POWER2 = POWER4,
                                       Method = 'RLCP', alpha = alpha))
  }
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_LL
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))))

P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, color = Method)) +
  geom_boxplot(alpha = 0.8) +
  scale_y_continuous(limits = c(-0.2, 0.3),
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
  facet_grid(~Type,scales = "free_y")
P1



############# Support vector machine -------------

nr <- 100

cl <- makeCluster(8)
#cl <- detectCores()
registerDoSNOW(cl)
pb <- txtProgressBar(max = nr, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)


Result_SVM <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "grf", "kernlab", "randomForest", "ks","nnet","e1071"), .errorhandling = "remove", .options.snow = opts)%dopar% {
  
  result <- data.frame()
  X <- matrix(runif((n+N)*d, -1, 1), nrow = n+N, ncol = d)
  Y1 <- -2*X[, 1] + 7*X[, 2]^3 + 3*exp(X[, 3] + 2*X[, 4]^2) + rnorm(n+N, sd = 1)
  Y2 <- -6*X[, 1] + 5*X[, 2]^3 + 2*exp(2*X[, 3] + X[, 4]^2) + rnorm(n+N, sd = 1)
  
  Y1 <- as.factor(ifelse(Y1>quantile(Y1, 0.7), 1, 0))
  Y2 <- as.factor(ifelse(Y2>quantile(Y2, 0.7), 1, 0))
  
  data <- data.frame(x = X, y1  = Y1, y2 = Y2)
  datatrain <- data[1:(n/2),]
  # model1 <- randomForest(y1~., data = datatrain[,-(d+2)])
  # model2 <- randomForest(y2~., data = datatrain[,-(d+1)])
  model1 <- svm(y1~., data = data.frame(datatrain[,-(d+2)]), probability = TRUE)
  model2 <- svm(y2~., data = data.frame(datatrain[,-(d+1)]), probability = TRUE)
  
  datacal <- data[(n/2+1):n,]
  datatest <- data[(n+1):(n+N),]
  Index1 <- datatest$x.1<0&datatest$x.3<0
  Index2 <- datatest$x.1>0&datatest$x.3<0
  Index3 <- datatest$x.1<0&datatest$x.3>0
  Index4 <- datatest$x.1>0&datatest$x.3>0
  
  
  
  # V1cal <- predict(model1, datacal, type = 'prob')[, 2]
  # V2cal <- predict(model2, datacal, type = 'prob')[, 2]
  # V1test <- predict(model1, datatest, type = 'prob')[, 2]
  # V2test <- predict(model2, datatest, type = 'prob')[, 2]
  V1cal <- attr(predict(model1, datacal, probability = TRUE), "probabilities")[, 2]
  V2cal <- attr(predict(model2, datacal, probability = TRUE), "probabilities")[, 2]
  V1test <- attr(predict(model1, datatest, probability = TRUE), "probabilities")[, 2]
  V2test <- attr(predict(model2, datatest, probability = TRUE), "probabilities")[, 2]
  
  V <- apply(cbind(ifelse(datacal$y1==0, V1cal, 0), ifelse(datacal$y2==0, V2cal, 0)), 1, max)
  
  s_sam <- matrix(0, ncol = d, nrow = N)
  for (i in 1:N) {
    s_sam[i,] <- mvrnorm(1, as.numeric(datatest[i, 1:d]), (h^2)*diag(d))
  }
  
  weight <- matrix(0, nrow = N, ncol = n/2+1)
  for (j in 1:(n/2)) {
    diffmat <- matrix(0, nrow = N, ncol = d)
    for (k in 1:d) {
      diffmat[, k] <- s_sam[, k] - datacal[j, k]
    }
    weight[, j] <- exp(-apply(diffmat^2, 1, sum)/(2*h^2))
  }
  diffmat <- matrix(0, nrow = N, ncol = d)
  for (k in 1:d) {
    diffmat[, k] <- s_sam[, k] - datatest[, k]
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
  rand <- runif(N)
  Indicator1[, n/2+1] <- rand
  Indicator2[, n/2+1] <- rand
  
  pvalue1 <- apply(weight*Indicator1, 1, sum)
  pvalue2 <- apply(weight*Indicator2, 1, sum)
  
  for(alpha in alpha_array){
    
    # thresholding rule without weighting
    thr <- quantile(V, 1-alpha)
    
    s1 <- ifelse(V1test>thr, 1, 0)
    s2 <- ifelse(V2test>thr, 1, 0)
    FWER <- mean(ifelse((s1==1&datatest$y1==0)|(s2==1&datatest$y2==0), 1, 0))
    POWER1 <- sum(s1==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER2 <- sum(s2==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1 <- sum(ifelse((s1==1&datatest$y1==0&Index1)|(s2==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2 <- sum(ifelse((s1==1&datatest$y1==0&Index2)|(s2==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3 <- sum(ifelse((s1==1&datatest$y1==0&Index3)|(s2==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4 <- sum(ifelse((s1==1&datatest$y1==0&Index4)|(s2==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    result <- rbind(result, data.frame(mFWER = FWER, 
                                       cFWER1 = FWER1, cFWER2 = FWER2, cFWER3 = FWER3, cFWER4 = FWER4,
                                       POWER1 = POWER1, POWER2=POWER2,
                                       Method = 'THR', alpha = alpha))
    
    # thresholding rule with RLCP
    s3 <- ifelse(pvalue1<alpha, 1, 0)
    s4 <- ifelse(pvalue2<alpha, 1, 0)
    FWER_w <- mean(ifelse((s3==1&datatest$y1==0)|(s4==1&datatest$y2==0), 1, 0))
    POWER3 <- sum(s3==1&datatest$y1==1)/sum(datatest$y1==1)
    POWER4 <- sum(s4==1&datatest$y2==1)/sum(datatest$y2==1)
    
    FWER1_w <- sum(ifelse((s3==1&datatest$y1==0&Index1)|(s4==1&datatest$y2==0&Index1), 1, 0))/sum(ifelse(Index1, 1, 0))
    FWER2_w <- sum(ifelse((s3==1&datatest$y1==0&Index2)|(s4==1&datatest$y2==0&Index2), 1, 0))/sum(ifelse(Index2, 1, 0))
    FWER3_w <- sum(ifelse((s3==1&datatest$y1==0&Index3)|(s4==1&datatest$y2==0&Index3), 1, 0))/sum(ifelse(Index3, 1, 0))
    FWER4_w <- sum(ifelse((s3==1&datatest$y1==0&Index4)|(s4==1&datatest$y2==0&Index4), 1, 0))/sum(ifelse(Index4, 1, 0))
    
    result <- rbind(result, data.frame(mFWER = FWER_w, 
                                       cFWER1 = FWER1_w, cFWER2 = FWER2_w, cFWER3 = FWER3_w, cFWER4 = FWER4_w,
                                       POWER1 = POWER3, POWER2 = POWER4,
                                       Method = 'RLCP', alpha = alpha))
  }
  return(result)
}

close(pb)
stopCluster(cl)

Result <- Result_SVM
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))))

P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, color = Method)) +
  geom_boxplot(alpha = 0.8) +
  scale_y_continuous(limits = c(-0.2, 0.3),
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
  facet_grid(~Type,scales = "free_y")
P1










########### plots ---------------
Result_all <- rbind(Result_RF,Result_NN,Result_LL)
Result_all$Alg <- c(rep('RF',nrow(Result_RF)),rep('NN',nrow(Result_NN)),rep('LL',nrow(Result_LL)))
Result <- Result_all
Result$alphadraw <- factor(Result$alpha, levels = c('0.05', '0.1', '0.15', '0.2'))
attributes(Result_all)
Resultdraw <- data.frame(Value = c(Result$mFWER-Result$alpha, 
                                   Result$cFWER1-Result$alpha,
                                   Result$cFWER2-Result$alpha,
                                   Result$cFWER3-Result$alpha,
                                   Result$cFWER4-Result$alpha), 
                         Method = c(Result$Method, Result$Method, Result$Method, Result$Method, Result$Method), 
                         alphadraw = c(Result$alphadraw, Result$alphadraw,Result$alphadraw, Result$alphadraw,Result$alphadraw), 
                         Type = c(rep('mFWER above nominal', nrow(Result)),
                                  rep('cFWER1 above nominal', nrow(Result)),
                                  rep('cFWER2 above nominal', nrow(Result)),
                                  rep('cFWER3 above nominal', nrow(Result)),
                                  rep('cFWER4 above nominal', nrow(Result))), 
                         hline = c(rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result)),
                                   rep(0, nrow(Result))),
                         Alg = c(Result$Alg,Result$Alg,Result$Alg,Result$Alg,Result$Alg))
pdf(file="Labelscreen_ScenarioA_FWER.pdf",
    width=12,height=8)
P1 <- ggplot(data = Resultdraw, aes(x = alphadraw, y = Value, color = Method)) +
  geom_boxplot(alpha = 0.8) +
  scale_y_continuous(limits = c(-0.2, 0.35),
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

#write.csv(Result_all,"Labelscreening_ScenarioA.csv")

#Result_all <- read.csv("Labelscreening_ScenarioA.csv")[,-1]
Result_all$Method[which(Result_all$Method=='RLCP')] = 'LCP-ls'
