####################################################################
# APA Pràctica (Grau FIB)
# Josep de Cid & Gonzalo Recio
#
# Breast Cancer Diagnostic
# Q1 2017-2018
####################################################################

# install.packages('corrplot')
# install.packages('caTools')
# install.packages('ggplot2')
# install.packages('rpart')
# install.packages('randomForest') 
# install.packages('caret')
# install.packages('e1071')
set.seed(42)

####################################################################
# SECTION 1: Data Preprocessing
####################################################################

dataset <- read.csv('data.csv')
# Remove unnecessary column 'X'
dataset <- subset(dataset, select = -c(X, id))

summary(dataset)
# We see that there are no NA values.
# There are also no strange or out of range values.

# Converting diagnosis into factor variables
diagnosis <- as.factor(dataset$diagnosis)
dataset$diagnosis <- diagnosis
table(dataset$diagnosis)  # Unbalanced observations

# Take a look at heavy corrlations among different variables
library(corrplot)
correlation <- cor(dataset[, 2:31])
corrplot(correlation, order = 'hclust', tl.cex = 1, addrect = 8)

# Split dataset into training and test set
library(caTools)
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
training.set = subset(dataset, split == TRUE)
test.set = subset(dataset, split == FALSE)

# Feature Scaling
training.set[, -1] <- scale(training.set[, -1])
test.set[, -1] <- scale(test.set[, -1])

# As we have a lot of dimensions and very correlated data, let's apply a PCA
pca <- prcomp(dataset[, 2:31], center = TRUE, scale = TRUE)
plot(pca, type="l")

# Take a look at the variance proportion given by each component
summary(pca)

# Seting a Statistical significance level SL = 0.05
# We obserbe that with 10 components we get over 0.95
# We would need 20 extra components only to obtain an extra 0.05, so we discard this ones.
pca.df <- as.data.frame(pca$x)

library(ggplot2)
ggplot(pca.df) +
  geom_point(aes(x = PC1, y = PC2, col = diagnosis)) +
  ggtitle('Diagnosis distribution over first two Principal Components')

####################################################################
# SECTION 2: Model Building
####################################################################

### K-NN
library(class)
pred.knn <- knn(train = training.set[, -1],
                test = test.set[, -1],
                cl = training.set[, 1],
                k = 5)

# K-NN Confusion matrix and accuracy
(conf.knn <- table(test.set[, 1], pred.knn))
(acc.knn <- (conf.knn[1, 1] + conf.knn[2, 2]) / dim(test.set)[1])

### Logistic
classifier.log <- glm(formula = diagnosis ~ .,
                      family = binomial,
                      data = training.set)

prob.log <- predict(classifier.log,
                    type = 'response',
                    newdata = test.set[-1])
pred.log <- ifelse(prob.log > 0.5, 'M', 'B')

# Logistic Confusion matrix and accuracy
(conf.log <- table(test.set[, 1], pred.log))
(acc.log <- (conf.log[1, 1] + conf.log[2, 2]) / dim(test.set)[1])

### Decision Tree
library(rpart)
classifier.dt <- rpart(formula = diagnosis ~ .,
                       data = training.set)

pred.dt <- predict(classifier.dt,
                   newdata = test.set[-1],
                   type = 'class')

# Decision Tree Confusion matrix and Accuracy
(conf.dt <- table(test.set[, 1], pred.dt))
(acc.dt <- (conf.dt[1, 1] + conf.dt[2, 2]) / dim(test.set)[1])

### Random Forest Regression
library(randomForest)
regressor.rf <- randomForest(formula = diagnosis ~ ., 
                             data = training.set,
                             importance = TRUE,  # Most significant variables
                             ntree = 100)

pred.rf <- predict(regressor.rf, newdata = test.set)

# Random Forest Confusion matrix, Accuracy and Cross-Validation
(conf.rf <- table(test.set[, 1], pred.rf))
(acc.rf <- (conf.rf[1, 1] + conf.rf[2, 2]) / dim(test.set)[1])
(cv.rf = rfcv(trainx = dataset[-1],
              trainy = dataset$diagnosis,
              cv.fold = 5, scale = 'log', step = 0.5))

### SVM
library(e1071)
classifier.svm <- svm(formula = diagnosis ~ .,
                      data = training.set,
                      type = 'C-classification',
                      kernel = 'linear')

pred.svm <- predict(classifier.svm, newdata = test.set[-1])

# SVM Confusion matrix and accuracy
(conf.svm <- table(test.set[, 1], pred.svm))
(acc.svm <- (conf.svm[1, 1] + conf.svm[2, 2]) / dim(test.set)[1])

### Naive Bayes
model.nb <- naiveBayes(diagnosis ~ .,
                       data = training.set)

pred.nb <- predict(model.nb, newdata = test.set)

# Naive Bayes Confusion matrix and accuracy
(conf.nb <- table(pred.nb, test.set$diagnosis))
(acc.nb <- (conf.nb[1, 1] + conf.nb[2, 2]) / dim(test.set)[1])

# we obtain 87% accuracy, let's try with cross-validation method for training the model

#-------------- Dona el putu mateix-------------------------
library(MASS)
library(caret)
library(klaR)
train_control <- trainControl(method = 'LOOCV') #method="repeatedcv", number=10, repeats=3) # method="cv", number=10)
#create model
model.nbcv <- train(form = diagnosis ~ ., data = training.set, method = "nb", trControl=train_control)
pred.nbcv <- predict(model.nbcv, newdata = test.set)

(conf.nbcv <- table(pred.nbcv, test.set$diagnosis))
(acc.nbcv <- (conf.nbcv[1, 1] + conf.nbcv[2, 2]) / dim(test.set)[1])
#------------------------------------------------------------

### Neural Networks
library(nnet)
classifier.nn <- nnet(diagnosis ~ .,
                      data = training.set,
                      size = 3, maxit = 500, decay = 0)

pred.nn <- as.factor(predict(classifier.nn, newdata = test.set, type = 'class'))
(conf.nn <- table(pred.nn, test.set$diagnosis))
(acc.nn <- (conf.nn[1, 1] + conf.nn[2, 2]) / dim(test.set)[1])
