SSAdaBoost <- function(train, test, iter, noise, l, a, b) {
  feat <- ncol(train) - 1

  lambda <- l

  train$label <- train$true
  test$label <- test$true
  
  train$F <- 0
  train$loss1 <- exp(1)
  alpha <- a
  ## add noise to train data
  set.seed(66)
  noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
  if (noise != 0)
    train[noise_point,]$label <- -train[noise_point,]$label
  
  ## 1.Start with weights w_i=1/N, i=1,...,N
  N <- nrow(train)
  train$w <- 1/N
  ## 2.Repeat for m=1,...,M
  M <- iter;fit <- list();err <- rep(0,M);c <- rep(0,M)
  for (m in 1:M) {
    # (a) Fit the classifier f_m(x) using weights w_i on the training data.
    fit[[m]] <- rpart(label~., data = train[,c(1:feat,(feat + 2))], 
                      weights = train$w, method = "class",
                      control = rpart.control(maxdepth = 2))
    train$f <- predict(fit[[m]],train,type = 'class')
    # (b) Computer err_m and c_m
    err[m] <- sum(train[which(train$f != train$label),]$w)
    if (err[m] == 0) {
      c[m] <- 7
    } else {
      c[m] <- log((1 - err[m])/err[m])
    }
    # (c) Set w_i and renormalize so that sum(w) = 1
    train$w <- ifelse(train$f != train$label, train$w*exp(c[m]), train$w)
    
    ### add 
    train$F <- train$F + c[m]*as.numeric(as.character(predict(fit[[m]],train,type = 'class')))
    
    train$loss2 <- exp(-train$label*train$F)
    train$detaloss <- train$loss2 - train$loss1
    
    # add at 11.20
    # train$loss <- train$loss2
    # train$detaloss <- train$detaloss * (mean(train$loss)/mean(train$detaloss))
    # train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
    
    # add at 11.22
    train$loss <- rank(train$loss2)/max(rank(train$loss2))
    train$detaloss <- rank(train$detaloss)/max(rank(train$detaloss))
    train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
    
    keep_max <- order(train$balance, 
                      decreasing = TRUE)[1:as.integer(nrow(train)*lambda)]
    train[keep_max,'w'] <- 0
    
    alpha <- alpha * b
    train$loss1 <- train$loss2
    
    ### end add
    
    train$w <- train$w/sum(train$w)
  }
  ## 3. Output the classifier sign
  train$pred <- 0
  for (m in 1:M) {
    train$pred <- train$pred + 
      c[m]*as.numeric(as.character(predict(fit[[m]],train,type = 'class')))
  }
  train$pred <- sign(train$pred)
  
  ## Test
  test$pred <- 0
  for (m in 1:M) {
    test$pred <- test$pred + 
      c[m]*as.numeric(as.character(predict(fit[[m]],test,type = 'class')))
  }
  test$pred <- sign(test$pred)
  
  return(sum(test$pred == test$label)/nrow(test))
}

APCSPAD <- function(data, iter, noise) {
  feat <- ncol(data) - 1
  best_test <- 0
  for (l in seq(0.0,0.35,0.05)) {
    for (a in seq(0.3,0.7,0.1)) {
      for (b in seq(0.91,0.99,0.02)) {
        lambda <- l
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
            train$F <- 0
            train$loss1 <- exp(1)
            alpha <- a
            ## add noise to train data
            set.seed(66)
            noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
            if (noise != 0)
              train[noise_point,]$label <- -train[noise_point,]$label
            
            ## 1.Start with weights w_i=1/N, i=1,...,N
            N <- nrow(train)
            train$w <- 1/N
            ## 2.Repeat for m=1,...,M
            M <- iter;fit <- list();err <- rep(0,M);c <- rep(0,M)
            for (m in 1:M) {
              # (a) Fit the classifier f_m(x) using weights w_i on the training data.
              fit[[m]] <- rpart(label~., data = train[,c(1:feat,(feat + 2))], 
                                weights = train$w, method = "class",
                                control = rpart.control(maxdepth = 2))
              train$f <- predict(fit[[m]],train,type = 'class')
              # (b) Computer err_m and c_m
              err[m] <- sum(train[which(train$f != train$label),]$w)
              if (err[m] == 0) {
                c[m] <- 7
              } else {
                c[m] <- log((1 - err[m])/err[m])
              }
              # (c) Set w_i and renormalize so that sum(w) = 1
              train$w <- ifelse(train$f != train$label, train$w*exp(c[m]), train$w)
              
              ### add 
              train$F <- train$F + c[m]*as.numeric(as.character(predict(fit[[m]],train,type = 'class')))
              
              train$loss2 <- exp(-train$label*train$F)
              train$detaloss <- train$loss2 - train$loss1
              
              # add at 11.20
              # train$loss <- train$loss2
              # train$detaloss <- train$detaloss * (mean(train$loss)/mean(train$detaloss))
              # train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
              
              # add at 11.22
              train$loss <- rank(train$loss2)/max(rank(train$loss2))
              train$detaloss <- rank(train$detaloss)/max(rank(train$detaloss))
              train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
              
              keep_max <- order(train$balance, 
                                decreasing = TRUE)[1:as.integer(nrow(train)*lambda)]
              train[keep_max,'w'] <- 0
              
              alpha <- alpha * b
              train$loss1 <- train$loss2
              
              ### end add
              
              train$w <- train$w/sum(train$w)
            }
            ## 3. Output the classifier sign
            train$pred <- 0
            for (m in 1:M) {
              train$pred <- train$pred + 
                c[m]*as.numeric(as.character(predict(fit[[m]],train,type = 'class')))
            }
            train$pred <- sign(train$pred)
            
            ## Test
            test$pred <- 0
            for (m in 1:M) {
              test$pred <- test$pred + 
                c[m]*as.numeric(as.character(predict(fit[[m]],test,type = 'class')))
            }
            test$pred <- sign(test$pred)
          
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
          best_a <- a
          best_b <- b
          cat('better lambda:',best_l,best_a,best_b,' test:',best_test,'\n')
        }
        else {
          cat('better lambda:',l,a,b,' test:',lambda_test,'\n')
        }
      }
    }
  }
  return(c(best_l,best_a,best_b))
}

#### test ####
SingelCore <- function(noise_num, data) {
  noise <- (noise_num - 1)*5
  best_l <- matrix(nrow = 5,ncol = 3)
  for (p in 1:5) {
    set.seed(77*p)
    flds_out <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    best_l[p,] <- APCSPAD(train, 50, noise)
  }
  l <- as.numeric(rownames(as.matrix(which(table(best_l[,1]) == max(table(best_l[,1])))))[1])
  a <- as.numeric(rownames(as.matrix(which(table(best_l[,2]) == max(table(best_l[,2])))))[1])
  b <- as.numeric(rownames(as.matrix(which(table(best_l[,3]) == max(table(best_l[,3])))))[1])
  
  r <- numeric()
  for (t in 1:20) {
    set.seed(77*t)
    flds_out <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    r[t] <- CSPAD(train, test, 50, noise, l, a, b)
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

CSPADResult <- alg_result
save(CSPADResult, file = "/home/lxs/ownCloud/SPGBDT/KBS/CSPAD-0410.RData")
