GradientBoosting <- function(train, test, iter, noise, shrink=0.1, depth=2) {
  feat <- ncol(train) - 1
  train$label <- train$true
  test$label <- test$true
  
  ## add noise to train data
  set.seed(66)
  noise_point <- sample(1:length(train[,1]), (noise*length(train[,1])) %/% 100)
  if (noise != 0)
    train[noise_point,]$label <- -train[noise_point,]$label

  ## 1.Initialize model with a constant value
  f0 <- as.numeric(names(table(train$label))
                   [which(table(train$label) == max(table(train$label)))][1])
  ## 2.Repeat for m=1,...,M
  M <- iter
  fit <- list(); gamma <- rep(0,M)
  train$F <- f0
  for (m in 1:M) {
    # computer the loss
    #loss[,m] <- 0.5*((train$label - train$F)^2)
    
    # a.Compute so-called pseudo-residuals
    train$pr <- train$label - train$F
    # b.Fit a base learner (e.g. tree) to pseudo-residuals
    fit[[m]] <- rpart(pr~., data = train[,c(1:feat,(feat + 4))],
                      #weights = train$w, 
                      method = "anova",
                      control = rpart.control(maxdepth = depth))
    # c.Compute multiplier gamma_m by solving the one-dimensional optimization problem
    train$f <- predict(fit[[m]],train)
    gamma[m] <- mean(train$pr/(train$f + 1e-19))
    # shrink
    gamma[m] <- gamma[m] * shrink
    # d.Update the model
    train$F <- train$F + gamma[m]*train$f
    
    # computer the loss
    #loss[,m] <- 0.5*((train$label - train$F)^2)
  }
  train$pred <- ifelse(train$F >= 0, 1, -1)
  
  ## Test
  test$pred <- f0
  for (m in 1:M) {
    test$pred <- test$pred + gamma[m]*predict(fit[[m]],test)
  }
  test$pred <- sign(test$pred)
  
  return(sum(test$pred == test$label)/nrow(test))
}


#### test ####
SingelCore <- function(noise_num, data) {
  ex <- rep(0,10)
  noise <- (noise_num - 1)*5
  for (t in 1:20) {
    set.seed(77*t)
    flds_out <- createFolds(data$true, k = 5, list = TRUE, returnTrain = FALSE)
    train <- data[-flds_out[[1]],]
    test <- data[flds_out[[1]],]
    ex[t] <- GradientBoosting(train,test,50,noise)
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

GradientBoostingResult <- alg_result
save(alg_result, file = "/home/lxs/ownCloud/SPGBDT/KBS/GradientBoosting-0409.RData")
