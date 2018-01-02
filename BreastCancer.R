####################################################################
# APA Pràctica (Grau FIB)
# Josep de Cid & Gonzalo Recio

# Breast Cancer Diagnostic
# Q1 2017-2018
####################################################################

# install.packages('corrplot')
# install.packages('caTools')
# install.packages('ggplot2')
# install.packages('randomForest') 
# install.packages('caret')
# install.packages('e1071')

library(corrplot)
library(caTools)
library(ggplot2)
library(randomForest)
library(e1071)
library(class)

set.seed(422)

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
correlation <- cor(dataset[, 2:31])
corrplot(correlation, order = 'hclust', tl.cex = 1, addrect = 8)

# Split dataset into training and test set
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
training.set = subset(dataset, split == TRUE)
test.set = subset(dataset, split == FALSE)

# Feature Scaling
training.set[, 2:31] <- scale(training.set[, 2:31])
test.set[, 2:31] <- scale(test.set[, 2:31])

# As we have a lot of dimensions and very correlated data, let's apply a PCA
pca <- prcomp(dataset[, 2:31], center = TRUE, scale = TRUE)
plot(pca, type="l")

# Take a look at the variance proportion given by each component
summary(pca)

# Seting a Statistical significance level SL = 0.05
# We obserbe that with 10 components we get over 0.95
# We would need 20 extra components only to obtain an extra 0.05, so we discard this ones.
pca.df <- as.data.frame(pca$x)

ggplot(pca.df) +
  geom_point(aes(x = PC1, y = PC2, col = diagnosis)) +
  ggtitle('Diagnosis distribution over first two Principal Components')

####################################################################
# SECTION 2: Model Building
####################################################################

### Random Forest Regression
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
              cv.fold = 5))

### K-NN
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

### SVM

### Naive Bayes

### Neural Networks

### Decision Tree