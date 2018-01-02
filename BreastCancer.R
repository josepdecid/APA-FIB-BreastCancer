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
library(caret)
library(e1071)

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
table(dataset$diagnosis) # Unbalanced observations

# Take a look at heavy corrlations among different variables
correlation <- cor(dataset[, 2:31])
corrplot(correlation, order = 'hclust', tl.cex = 1, addrect = 8)

# Split dataset into training and test set
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
trainingSet = subset(dataset, split = TRUE)
testSet = subset(dataset, split = FALSE)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# Feature Scaling
trainingSet[, 2:31] <- scale(trainingSet[, 2:31])

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

# Random Forest Regression
regressor.rf <- randomForest(formula = diagnosis ~ .,
                             data = trainingSet,
                             ntree = 100)
pred.rf <- predict(regressor.rf, newdata = testSet)
(conf.rf <- confusionMatrix(pred.rf, testSet$diagnosis, positive = 'M'))
