RBoost1 <- function(train, test, iter, noise, l) {
  feat <- ncol(train) - 1
  z_limit <- l
  train$label <- train$true
  test$label <- test$true
  
  ## add noise to train data
  set.seed(66)
  noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
  if (noise != 0)
    train[noise_point,]$label <- -train[noise_point,]$label
  
  ## Initialization
  n <- length(train[,1])
  fit <- list()
  train$niu <- rep(0.5, n)
  train$w <- train$niu*(1 - train$niu)
  train$weight <- train$w/(sum(train$w) + 1e-19)
  train$fits <- rep(0, n)
  train$z <- ifelse(train$label == 1, 1/(3*train$niu - 1 + 1e-19),
                    1/(3*train$niu - 2 + 1e-19))
  ## For t= 1...T
  for (t in 1:iter) {
    fit[[t]] <- rpart(z~., data = train[,c(1:feat, ncol(train))], 
                      weights = train$weight, method = "anova")
    train$fits <- train$fits + predict(fit[[t]], train[,c(1:feat)])
    train$niu <- 1/(1 + exp(-train$fits))
    train$z <- ifelse(train$label == 1, 1/(3*train$niu - 1 + 1e-19), 
                      1/(3*train$niu - 2 + 1e-19))
    train$w <- train$niu*(1 - train$niu) + 1e-19
    train$weight <- train$w/(sum(train$w))
    train$z[which(train$z > z_limit)] <- z_limit
    train$z[which(train$z < -z_limit)] <- -z_limit
  }
  train$pred <- sign(train$fits)
  
  ## Test
  test$fits = 0
  for (j in 1:iter) {
    test$fits <- test$fits + predict(fit[[j]], test[,1:feat])
  }
  test$pred <- sign(test$fits)
  
  return(sum(test$pred == test$label)/nrow(test))
}

APRBoost1 <- function(data, iter, noise) {
  feat <- ncol(data) - 1
  best_test <- 0
  for (l in seq(0.25,2,0.25)) {
    z_limit <- l
    lambda_test <- numeric()
    for (times in 1:1) {
      train_right <- 0;test_right <- 0
      set.seed(77*times)
      flds <- createFolds(data$true, k = 5, list = TRUE, returnTrain = FALSE)
      for (f in 1:length(flds)) {
        train <- data[-flds[[f]],]
        test <- data[flds[[f]],]
        train$label <- train$true
        test$label <- test$true
        
        ## add noise to train data
        set.seed(66)
        noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
        if (noise != 0)
          train[noise_point,]$label <- -train[noise_point,]$label
        
        ## Initialization
        n <- length(train[,1])
        fit <- list()
        train$niu <- rep(0.5, n)
        train$w <- train$niu*(1 - train$niu)
        train$weight <- train$w/(sum(train$w) + 1e-19)
        train$fits <- rep(0, n)
        train$z <- ifelse(train$label == 1, 1/(3*train$niu - 1 + 1e-19),
                          1/(3*train$niu - 2 + 1e-19))
        ## For t= 1...T
        for (t in 1:iter) {
          fit[[t]] <- rpart(z~., data = train[,c(1:feat, ncol(train))], 
                            weights = train$weight, method = "anova")
          train$fits <- train$fits + predict(fit[[t]], train[,c(1:feat)])
          train$niu <- 1/(1 + exp(-train$fits))
          train$z <- ifelse(train$label == 1, 1/(3*train$niu - 1 + 1e-19), 
                            1/(3*train$niu - 2 + 1e-19))
          train$w <- train$niu*(1 - train$niu) + 1e-19
          train$weight <- train$w/(sum(train$w))
          train$z[which(train$z > z_limit)] <- z_limit
          train$z[which(train$z < -z_limit)] <- -z_limit
        }
        train$pred <- sign(train$fits)
        
        ## Test
        test$fits = 0
        for (j in 1:iter) {
          test$fits <- test$fits + predict(fit[[j]], test[,1:feat])
        }
        test$pred <- sign(test$fits)
        
        ## cross validation result
        train_right <- train_right + sum(train$pred == train$label)
        test_right <- test_right + sum(test$pred == test$label)
      }
      lambda_test[times] <- test_right/nrow(data)
    }
    
    lambda_test <- mean(lambda_test)
    
    if (lambda_test > best_test) {
      best_test <- lambda_test
      best_l <- l
      cat('better lambda:',best_l,' test:',best_test,'\n')
    }
  }
  return(best_l)
}

#### test ####
SingelCore <- function(noise_num, data) {
  noise <- (noise_num - 1)*5
  best_l <- matrix(nrow = 5,ncol = 1)
  for (p in 1:5) {
    set.seed(77*p)
    flds_out <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    best_l[p,] <- APRBoost1(train, 50, noise)
  }
  l <- as.numeric(rownames(as.matrix(which(table(best_l[,1]) == max(table(best_l[,1])))))[1])
  
  r <- numeric()
  for (t in 1:20) {
    set.seed(77*t)
    flds_out <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    r[t] <- RBoost1(train, test, 50, noise, l)
    cat('Time:',t,' Result:',r[t],'\n',sep = '')
  }
  return(mean(r))
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

RBoostResult <- alg_result
save(RBoostResult, file = "/home/lxs/ownCloud/SPGBDT/KBS/RBoost-0411.RData")
