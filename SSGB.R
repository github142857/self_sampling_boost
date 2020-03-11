SSGB <- function(train, test, iter, noise, l, a, b, shrink=0.1, depth=2) {
  feat <- ncol(train) - 1
  lambda <- l
  train$label <- train$true
  test$label <- test$true

  ## need to adjust
  alpha <- a
  
  ## add noise to train data
  set.seed(66)
  noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
  if (noise != 0) {
    train[noise_point,]$label <- -train[noise_point,]$label
  }
  
  ## 1.Initialize model with a constant value
  f0 <- as.numeric(names(table(train$label))
                   [which(table(train$label) == max(table(train$label)))][1])
  
  ## 2.Repeat for m=1,...,M
  fit <- list(); gamma <- rep(0,iter)
  train$F <- f0
  train$w <- 1
  train$loss1 <- 0.5*((train$label - train$F)^2)
  
  for (m in 1:iter) {
    # a.Compute so-called pseudo-residuals
    train$pr <- train$label - train$F
    
    # b.Fit a base learner (e.g. tree) to pseudo-residuals
    fit[[m]] <- rpart(pr~., data = train[,c(1:feat,(feat + 6))],
                      weights = train$w, 
                      method = "anova",
                      control = rpart.control(maxdepth = depth))
    # c.Compute multiplier gamma_m by solving the one-dimensional optimization problem
    train$f <- predict(fit[[m]],train)
    #### weighted loss
    #gamma[m] <- sum(train$pr*train$w)/sum(train$f*train$w)
    gamma[m] <- mean(train$pr/(train$f + 1e-19)*train$w)
    # shrink
    gamma[m] <- gamma[m] * shrink
    # d.Update the model
    train$F <- train$F + gamma[m]*train$f
    
    # Update v
    train$w <- 1
    train$loss2 <- 0.5*((train$label - train$F)^2)
    train$detaloss <- train$loss2 - train$loss1
    
    # add at 11.20
    # train$loss <- train$loss2
    # train$detaloss <- train$detaloss * (mean(train$loss)/mean(train$detaloss))
    # 
    # train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss2
    
    # add at 11.22
    train$loss <- rank(train$loss2)/max(rank(train$loss2))
    train$detaloss <- rank(train$detaloss)/max(rank(train$detaloss))
    train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
    
    keep_max <- order(train$balance, 
                      decreasing = TRUE)[1:as.integer(nrow(train)*lambda)]
    train[keep_max,'w'] <- 0
    
    alpha <- alpha * b
    train$loss1 <- train$loss2
    
  }
  train$pred <- ifelse(train$F >= 0, 1, -1)
  
  ## Test
  test$pred <- f0
  for (m in 1:iter) {
    test$pred <- test$pred + gamma[m]*predict(fit[[m]],test)
  }
  test$pred <- sign(test$pred)
  
  return(sum(test$pred == test$label)/nrow(test))
}

APCSPGB <- function(data, iter, noise, shrink=0.1, depth=2) {
  feat <- ncol(data) - 1
  best_test <- 0
  for (l in seq(0.0,0.35,0.05)) {
    for (a in seq(0.3,0.7,0.1)) {
      for (b in seq(0.91,0.99,0.02)) {
        lambda <- l
        lambda_test <- numeric()
        for (times in 1:2) {
          train_right <- 0;test_right <- 0
          set.seed(77*times)
          flds <- createFolds(data$true, k = 5, list = TRUE, returnTrain = FALSE)
          for (f in 1:length(flds)) {
            train <- data[-flds[[f]],]
            test <- data[flds[[f]],]
            train$label <- train$true
            test$label <- test$true
            ## need to adjust
            alpha <- a
            
            ## add noise to train data
            set.seed(66)
            noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
            if (noise != 0) {
              train[noise_point,]$label <- -train[noise_point,]$label
            }
            
            ## 1.Initialize model with a constant value
            f0 <- as.numeric(names(table(train$label))
                             [which(table(train$label) == max(table(train$label)))][1])
            
            ## 2.Repeat for m=1,...,M
            fit <- list(); gamma <- rep(0,iter)
            train$F <- f0
            train$w <- 1
            train$loss1 <- 0.5*((train$label - train$F)^2)
            
            for (m in 1:iter) {
              # a.Compute so-called pseudo-residuals
              train$pr <- train$label - train$F
              
              # b.Fit a base learner (e.g. tree) to pseudo-residuals
              fit[[m]] <- rpart(pr~., data = train[,c(1:feat,(feat + 6))],
                                weights = train$w, 
                                method = "anova",
                                control = rpart.control(maxdepth = depth))
              # c.Compute multiplier gamma_m by solving the one-dimensional optimization problem
              train$f <- predict(fit[[m]],train)
              #### weighted loss
              #gamma[m] <- sum(train$pr*train$w)/sum(train$f*train$w)
              gamma[m] <- mean(train$pr/(train$f + 1e-19)*train$w)
              # shrink
              gamma[m] <- gamma[m] * shrink
              # d.Update the model
              train$F <- train$F + gamma[m]*train$f
              
              # Update v
              train$w <- 1
              train$loss2 <- 0.5*((train$label - train$F)^2)
              train$detaloss <- train$loss2 - train$loss1
              
              # add at 11.20
              # train$loss <- train$loss2
              # train$detaloss <- train$detaloss * (mean(train$loss)/mean(train$detaloss))
              # 
              # train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss2
              
              # add at 11.22
              train$loss <- rank(train$loss2)/max(rank(train$loss2))
              train$detaloss <- rank(train$detaloss)/max(rank(train$detaloss))
              train$balance <- alpha * train$detaloss + (1 - alpha) * train$loss
              
              keep_max <- order(train$balance, 
                                decreasing = TRUE)[1:as.integer(nrow(train)*lambda)]
              train[keep_max,'w'] <- 0
              
              alpha <- alpha * b
              train$loss1 <- train$loss2
              
            }
            train$pred <- ifelse(train$F >= 0, 1, -1)
            
            ## Test
            test$pred <- f0
            for (m in 1:iter) {
              test$pred <- test$pred + gamma[m]*predict(fit[[m]],test)
            }
            test$pred <- ifelse(test$pred >= 0, 1, -1)
            
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
    best_l[p,] <- APCSPGB(train, 50, noise, shrink=0.1, depth=2)
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
    r[t] <- CSPGB(train, test, 50, noise, l, a, b, shrink = 0.1, depth = 2)
    cat('Time:',t,' Result:',r[t],'\n',sep='')
  }
  return(mean(r))
}

cl <- makeCluster(45)
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

CSPGBResult <- alg_result
save(CSPGBResult, file = "/home/lxs/ownCloud/SPGBDT/KBS/CSPGB-0411.RData")

Core_APCSPAD <- function(number) {
  set.seed(77*number)
  flds_out <- createFolds(Ionosphere$true, k = 10, list = TRUE, returnTrain = FALSE)
  train <- Ionosphere[-flds_out[[1]],]
  test <- Ionosphere[flds_out[[1]],]
  best_l <- APCSPAD(train, 50, 30)
}
