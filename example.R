library(caret)
source('AdaBoost.R')

#### Prepare UCI Dateset ####
Chess <- read.table("kr-vs-kp.data.txt",sep = ',')
Chess$true <- ifelse(Chess[, 37] == 'won', 1, -1)
Chess <- Chess[, -37]
for (i in 1:ncol(Chess)) {
  Chess[i] <- as.numeric(Chess[,i])
}

#### Example ####
data <- Chess
iter <- 50
noise <- 15 #15%
# data to (train, test)
# flds <- createFolds(data$true, k = 10, list = TRUE, returnTrain = FALSE)
test_Result <- DiscreteAdaBoost(train, test, iter, noise)