DiscreteAdaBoost <- function(train, test, iter, noise) {
  feat <- ncol(train) - 1
  train$label <- train$true
  test$label <- test$true
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
      c[m] <- 100
    } else {
      c[m] <- log((1 - err[m])/err[m])
    }
    # (c) Set w_i and renormalize so that sum(w) = 1
    train$w <- ifelse(train$f != train$label, train$w*exp(c[m]), train$w)
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