CBAdaboost <- function(train, test, iter, noise, l) {
  feat <- ncol(train) - 1
  train$label <- train$true
  test$label <- test$true
  
  knn <- 5
  ## add noise to train data
  set.seed(66)
  noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
  if (noise != 0)
    train[noise_point,]$label <- -train[noise_point,]$label
  
  ## filter noise
  KNN <- knn(train[,c(1:feat)], train[,c(1:feat)], train[, feat + 2], k = 20, prob = TRUE)
  train$knn <- KNN
  train$prob <- attr(KNN,"prob")
  
  ## Input
  KNN <- knn(train[,c(1:feat)], train[,c(1:feat)], train[, feat + 2], k = knn, prob = TRUE)
  train$knn <- KNN
  train$prob <- attr(KNN,"prob")
  train$r <- ifelse(train$label == train$knn, train$prob, 1 - train$prob)
  
  ## Initialization
  n <- length(train[,1])
  fit <- list()
  beta <- 0
  train$w1 <- train$r
  train$w2 <- 1 - train$r
  train$D <- abs(train$w1 - train$w2)
  train$D <- train$D/(sum(train$D) + 1e-19)
  train$h <- rep(0, n)
  train$fits <- 0
  
  ## For t= 1...T
  for (t in 1:iter) {
    # 1 Relabel
    train$z <- sign((train$w1 - train$w2)*train$label + 1e-19) ## 0 -> 1
    # 2+3 Draw instance and train
    fit[[t]] <- rpart(z~., data = train[,c(1:feat, ncol(train))], 
                      weights = train$D)
    # 4 beta
    train$h <- sign(predict(fit[[t]], train[,c(1:feat, ncol(train))]) + 1e-19)
    beta[t] <- 0.5 * log((sum(train[which(train$h == train$label),]$w1) + 
                            sum(train[which(train$h != train$label),]$w2)) / 
                           (sum(train[which(train$h != train$label),]$w1) + 
                              sum(train[which(train$h == train$label),]$w2)))
    if (beta[t] < 0) {
      t = t - 1
      break
    }
    # 5 Update w
    train$w1 <- train$w1*exp(-train$label*beta[t]*train$h)
    train$w2 <- train$w2*exp(train$label*beta[t]*train$h)
    train$D <- abs(train$w1 - train$w2)
    train$D <- train$D/(sum(train$D) + 1e-19)
  }
  for (j in 1:t) {
    train$fits <- train$fits + beta[j]*predict(fit[[j]], train[,1:feat])
  }
  train$pred <- sign(train$fits)
  
  ## Test
  test$fits = rep(0, length(test[,1]))
  for (j in 1:t) {
    test$fits <- test$fits + beta[j]*predict(fit[[j]], test[,1:feat])
  }
  test$pred <- sign(test$fits)
  
  return(sum(test$pred == test$label)/nrow(test))
}

#### test ####
SingelCore <- function(noise_num, data) {
  ex <- rep(0,20)
  noise <- (noise_num - 1)*5
  for (t in 1:20) {
    set.seed(77*t)
    flds_out <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    ex[t] <- CBAdaboost(train,test,50,noise)
  }
  mean(ex)
}

cl <- makeCluster(30)
registerDoParallel(cl)
n_set <- length(dataSet)
n_nos <- 7
parallel_num <- n_set*n_nos
Ex_result <- 1
Sys.time()
Ex_result <- foreach(iter = 1:parallel_num , .combine = 'c', 
                     .packages = c('ada','rpart','caret','caTools','class')) %dopar% 
  SingelCore(data = dataSet[[(floor((iter - 1)/n_nos) + 1)]], 
             noise_num = (floor((iter - 1) %% n_nos) + 1))
Sys.time()
stopCluster(cl)

alg_result <- matrix(nrow = n_set, ncol = n_nos)
for (i in 1:n_set) {
  alg_result[i,] <- 1 - Ex_result[((i - 1)*n_nos + 1):((i - 1)*n_nos + n_nos)]
}
print(alg_result)

par(mfrow = c(3,4))
for (i in 1:10) {
  plot(alg_result[i,])
}

CBAdaboostResult <- alg_result
save(CBAdaboostResult, file = "/home/lxs/ownCloud/SPGBDT/KBS/CBAdaboost-0411.RData")
