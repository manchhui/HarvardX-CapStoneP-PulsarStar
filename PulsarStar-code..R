#Load Packaages
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")

#Load Data
MyData <- read.csv(file = "pulsar_stars.csv", header = TRUE, sep = ",")

#Dimension Check
dim(MyData)[1]
dim(MyData)[2]
mean(MyData$target_class == "0")

#Shortening Column Names
colnames(MyData) <- c('mean_ip',
                      'sd_ip',
                      'exk_ip',
                      'skew_ip',
                      'mean_ds_c',
                      'sd_ds_c',
                      'exk_ds_c',
                      'skew_ds_c',
                      'target')

#Dataset Exploration Graph 1
MyData$target <- ifelse(MyData$target == 1, "P", "NP")
MyData %>% gather(predictors, value, -target) %>%
  ggplot(aes(target, value, fill = target)) +
  geom_boxplot() +
  facet_wrap(~predictors, scales = "free", ncol = 4) +
  theme(axis.text.x = element_blank(), legend.position="bottom")

# Scaling and Preparing Data
startype <- as.factor(MyData$target)
pstarpred <- as.matrix(select(MyData, -c(length(MyData))))
x_centered <- sweep(pstarpred, 2, colMeans(pstarpred))
x_scaled <- sweep(x_centered, 2, colSds(pstarpred), FUN = "/")

# Create Test and Train set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(startype, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled [test_index,]
test_y <- startype[test_index]
train_x <- x_scaled [-test_index,]
train_y <- startype[-test_index]

#training method
tr <- trainControl(method = "repeatedcv",
                   number = 5,
                   repeats = 3)

# Logistic regression algorithm
set.seed(1, sample.kind = "Rounding")
train_glm <- train(train_x, 
                   train_y,
                   method = "glm",
                   tuneLength=10,
                   trControl = tr)
glm_preds <- predict(train_glm, test_x)
train_glm$bestTune
glm_cm <- confusionMatrix(glm_preds, test_y)
glm_cm$overall["Accuracy"]

# Naive bayes algorithm
set.seed(1, sample.kind = "Rounding")
train_nb <- train(train_x, 
                  train_y,
                  method = "naive_bayes",
                  tuneLength=10,
                  trControl = tr)
nb_preds <- predict(train_nb, test_x)
train_nb$bestTune
nb_cm <- confusionMatrix(nb_preds, test_y)
nb_cm$overall["Accuracy"]

# knn algorithm
set.seed(1, sample.kind = "Rounding")
train_knn <- train(train_x, 
                   train_y,
                   method = "knn",
                   tuneLength=10,
                   trControl = tr)
knn_preds <- predict(train_knn, test_x)
train_knn$bestTune
ggplot(train_knn, highlight = TRUE)
knn_cm <- confusionMatrix(knn_preds, test_y)
knn_cm$overall["Accuracy"]

# rpart algorithm
set.seed(1, sample.kind = "Rounding")
train_rpart <- train(train_x, 
                     train_y,
                     method = "rpart",
                     tuneLength=10,
                     trControl = tr)
rpart_preds <- predict(train_rpart, test_x)
train_rpart$bestTune
ggplot(train_rpart, highlight = TRUE)
rpart.plot(train_rpart$finalModel)
rpart_cm <- confusionMatrix(rpart_preds, test_y)
rpart_cm$overall["Accuracy"]

# Random forest algorithm
set.seed(1, sample.kind = "Rounding")
train_rf <- train(train_x, 
                  train_y,
                  method = "rf",
                  tuneLength=2,
                  trControl = tr,
                  ntree= 200,
                  importance = TRUE)
rf_preds <- predict(train_rf, test_x)
train_rf$bestTune
ggplot(train_rf, highlight = TRUE)
plot(train_rf$finalModel)
rf_cm <- confusionMatrix(rf_preds, test_y)
rf_cm$overall["Accuracy"]

# Final Table
models <- c("Logistic Regression", 
            "Naive Bayes", 
            "K-Nearest Neighbours", 
            "Descision Trees", 
            "Random forest")
accuracy <- c(mean(glm_preds == test_y),
              mean(nb_preds == test_y),
              mean(knn_preds == test_y),
              mean(rpart_preds == test_y),
              mean(rf_preds == test_y))
sensi <- c(glm_cm$byClass[1],
           nb_cm$byClass[1],
           knn_cm$byClass[1],
           rpart_cm$byClass[1],
           rf_cm$byClass[1])
specif<- c(glm_cm$byClass[2],
           nb_cm$byClass[2],
           knn_cm$byClass[2],
           rpart_cm$byClass[2],
           rf_cm$byClass[2])
data.frame(Model = models, 
           Accuracy = accuracy, 
           F1_Score = ifelse(sensi + specif == 0, 0, 2 * sensi * specif / (sensi + specif)))
