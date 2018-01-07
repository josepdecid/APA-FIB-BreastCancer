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
# install.packages('lfda')
set.seed(42)

####################################################################
# SECTION 1: Data Preprocessing
####################################################################

# Dataset lecture ----

dataset <- read.csv('data.csv')
# Remove unnecessary NaN column 'X' and useless 'id'.
dataset <- subset(dataset, select = -c(X, id))

# Let's check that there are no NA or out-of-range values.
# So that, it's not necessary to deal with missing or invalid data.
summary(dataset)

# Convert diagnosis into factor variables.
diagnosis <- as.factor(dataset$diagnosis)
dataset$diagnosis <- diagnosis

# Check that diagnosis distribution is very unbalanced.
plot(x = diagnosis, main = 'Diagnosis distribution',
     col = c('#F8766D', '#00BFC4'),
     xlab = 'Diagnosis', ylab = 'Count')

# Check that there are heavy corrlations among different variables.
library(corrplot)
correlation <- cor(dataset[, -1])
corrplot(correlation, order = 'hclust', tl.cex = 1, addrect = 8)

# Training and Test set spliting ----
library(caTools)
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
training.set = subset(dataset, split == TRUE)
test.set = subset(dataset, split == FALSE)

# Feature Scaling ----
dataset[, -1] <- scale(dataset[, -1])
training.set[, -1] <- scale(training.set[, -1])
test.set[, -1] <- scale(test.set[, -1])

# PCA ----
pca <- prcomp(dataset[, 2:31])

# Take a look at the variance proportion given by each component
summary(pca)
plot(x = pca, main = 'Variance / PC', type = 'l')

# Seting an arbitrary statistical Significance Level SL = 0.05
# Observe that with 10 components we get over 0.95 of variance explained.
# 20 extra components would only explain an extra 0.05, so we discard these ones.
pca.df <- as.data.frame(pca$x)

# Display diagnosis over 2D (First two PC).
library(ggplot2)
ggplot(pca.df, aes(x = PC1, y = PC2, col = diagnosis)) +
  ggtitle('Diagnosis distribution over first two PC') +
  geom_point(alpha = 0.8)

# We can see that data is easily separable.

# Create PCA for training and test set.
training.set.pca <- subset(pca.df, split == TRUE)
test.set.pca <- subset(pca.df, split == FALSE)

# LCA ----
library(MASS)
lda <- lda(formula = diagnosis ~ ., data = dataset)
lda.df <- as.data.frame(predict(lda, dataset))
lda.df$diagnosis <- dataset$diagnosis
                        
# Display diagnosis over 1D (First LD).
ggplot(lda.df, aes(x = LD1, y = 0, col = diagnosis)) +
  ggtitle('Diagnosis distribution over first LD') +
  geom_point(alpha = 0.8)

# Display diagnosis over 2D (First two LD).
ggplot(lda.df, aes(x = LD1, fill = diagnosis)) +
  ggtitle('Diagnosis density over first LD') +
  geom_density(alpha = 0.8)

# We can also see that data is easily separable.

# Create LDA for training and test set.
training.set.lda <- subset(lda.df, split == TRUE)
test.set.lda <- subset(lda.df, split == FALSE)

####################################################################
# SECTION 2: Model Building
####################################################################

# K-NN ----
library(class)
pred.knn <- knn(train = training.set[, -1],
                test = test.set[, -1],
                cl = training.set[, 1],
                k = 5)

# K-NN Confusion matrix and accuracy
(conf.knn <- table(test.set[, 1], pred.knn))
(acc.knn <- (conf.knn[1, 1] + conf.knn[2, 2]) / dim(test.set)[1])

# Logistic ----
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

# Decision Tree ----
library(rpart)
classifier.dt <- rpart(formula = diagnosis ~ .,
                       data = training.set)

pred.dt <- predict(classifier.dt,
                   newdata = test.set[-1],
                   type = 'class')

# Decision Tree Confusion matrix and Accuracy
(conf.dt <- table(test.set[, 1], pred.dt))
(acc.dt <- (conf.dt[1, 1] + conf.dt[2, 2]) / dim(test.set)[1])

# Random Forest Regression ----
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

# SVM ----
library(e1071)
classifier.svm <- svm(formula = diagnosis ~ .,
                      data = training.set,
                      type = 'C-classification',
                      kernel = 'linear')

pred.svm <- predict(classifier.svm, newdata = test.set[-1])

# SVM Confusion matrix and accuracy
(conf.svm <- table(test.set[, 1], pred.svm))
(acc.svm <- (conf.svm[1, 1] + conf.svm[2, 2]) / dim(test.set)[1])

# Naive Bayes ----
model.nb <- naiveBayes(diagnosis ~ .,
                       data = training.set)

pred.nb <- predict(model.nb, newdata = test.set)

# Naive Bayes Confusion matrix and accuracy
(conf.nb <- table(pred.nb, test.set$diagnosis))
(acc.nb <- (conf.nb[1, 1] + conf.nb[2, 2]) / dim(test.set)[1])

# we obtain 94% accuracy, let's try with cross-validation method for training the model


# Dona el putu mateix
library(MASS)
library(caret)
library(klaR)
train_control <- trainControl(method="cv",
                              number = 5,
                              # preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)
#create model
model.nbcv <- train(form = diagnosis ~ ., data = training.set, method = "nb", trControl=train_control)
pred.nbcv <- predict(model.nbcv, newdata = test.set)

(conf.nbcv <- table(pred.nbcv, test.set$diagnosis))
(acc.nbcv <- (conf.nbcv[1, 1] + conf.nbcv[2, 2]) / dim(test.set)[1])
# Now we get 94-96% accuracy with cross-validation
#------------------------------------------------------------

# Neural Networks ----
library(nnet)
set.seed(42)
classifier.nn <- nnet(formula = diagnosis ~ .,
                      data = training.set,
                      size = 10, maxit = 2000, decay = 0)

pred.nn <- as.factor(predict(classifier.nn, newdata = test.set, type = 'class'))
(conf.nn <- table(pred.nn, test.set$diagnosis))
(acc.nn <- (conf.nn[1, 1] + conf.nn[2, 2]) / dim(test.set)[1])

par(mfrow=c(3,2))
for (i in 1:3)
{
  set.seed(42)
  nn1 <- nnet(formula=diagnosis ~ ., data=training.set, size=i, decay=0, maxit=2000,trace=T)
  pred.nn1 <- as.numeric(as.factor(predict(nn1, type='class')))
 
  pca.tr <- prcomp(training.set[, 2:31])
  pca.tr.df <- as.data.frame(pca.tr$x)
  plot(pca.tr.df$PC2 ~ pca.tr.df$PC1,pch=20,col=c('red','green')[pred.nn1])
  title(main=paste(i,'hidden unit(s)'))
  plot(pca.tr.df$PC2 ~ pca.tr.df$PC1, pch=20,col=c('red','green')[as.numeric(training.set$diagnosis)])
  title(main='Real Diagnosis')
}

par(mfrow=c(3,2))
for (i in 1:3)
{
  set.seed(42)
  nn1 <- nnet(formula=diagnosis ~ ., data=training.set, size=i, decay=0, maxit=2000,trace=T)
  pred.nn1 <- as.numeric(as.factor(predict(nn1, type='class')))

  fda.tr <- lfda(training.set[-1], training.set[1], r = 3, metric='plain')
  fda.tr.df <- as.data.frame(fda.tr$Z)
  plot(fda.tr.df$V3 ~ fda.tr.df$V1, pch=20,col=c('red','green')[pred.nn1])
  title(main=paste(i,'hidden unit(s)'))
  plot(fda.tr.df$V3 ~ fda.tr.df$V1, pch=20,col=c('red','green')[as.numeric(training.set$diagnosis)])
  title(main='Real Diagnosis')
}

par(mfrow=c(1,1))

# With 3 hidden units, que NN learns quite perfectly with normalized data

# This method finds that best number of hidden units is 5 and decay weight value 0.1
set.seed(42)
nnet <- train(form = diagnosis ~ ., data=training.set, method = 'nnet', metric = 'Accuracy', maxit=2000,trace=T, linout = F)
pred.nnet <- as.numeric(as.factor(predict(nnet, newdata = test.set)))
(conf.nnet <- table(pred.nnet, test.set$diagnosis))
(acc.nnet <- (conf.nnet[1, 1] + conf.nnet[2, 2]) / dim(test.set)[1])

#coef(nnet)
train_control <- trainControl(method="LOOCV", number = 10)
grid <- expand.grid(.decay = c(0, 0.0001, 0.001, 0.01, 0.1), .size = c(1,2,3,4))
set.seed(42)
nnet <- train(form = diagnosis ~.,
              data = training.set,
              method = 'nnet', 
              maxit = 2000,
              metric="Accuracy", 
              tuneGrid = grid, 
              trControl = train_control)

pred.nn <- as.factor(predict(nnet, newdata = test.set, type = 'raw'))
(conf.nn <- table(pred.nn, test.set$diagnosis))
(acc.nn <- (conf.nn[1, 1] + conf.nn[2, 2]) / dim(test.set)[1])
# 98.2% accuracy