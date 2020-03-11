GentleAdaboost <- function(train, test, iter, noise) {
  feat <- ncol(train) - 1
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
  train$w <- 1/n
  train$Fits <- 0
  ## For t= 1...T
  for (t in 1:iter) {
    ## (a) Fit the function f by a weighted least-squares regression of y to x using w
    fit[[t]] <- rpart(label~., data = train[,c(1:feat, which(names(train) == 'label'))], 
                      weights = train$w, method = "anova",
                      control = rpart.control(maxdepth = 2))
    ## (b) Update F
    train$f <- predict(fit[[t]], train[,c(1:feat)])
    train$Fits <- train$Fits + train$f
    ## (c) Update w
    train$w <- train$w*exp(-train$label*train$f)
    train$w <- train$w/sum(train$w)
  }
  train$pred <- sign(train$Fits)
  
  ## Test
  test$Fits <- rep(0, length(test[,1]))
  for (j in 1:iter) {
    test$Fits <- test$Fits + predict(fit[[j]], test[,1:feat])
  }
  test$pred <- sign(test$Fits)
  
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
    ex[t] <- GentleAdaboost(train,test,50,noise)
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

GentleAdaboostResult <- alg_result
save(alg_result, file = "/home/lxs/ownCloud/SPGBDT/KBS/GentleAdaboost-0409.RData")
