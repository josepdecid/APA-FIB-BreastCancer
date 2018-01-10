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
# install.packages('xgboost')
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

# Feature Scaling ----
dataset[, -1] <- scale(dataset[, -1],
                       scale = TRUE,
                       center = TRUE)

# Correlation ----
library(corrplot)
correlation <- cor(dataset[, -1])
corrplot(corr = correlation, order = 'hclust',
         tl.col = 'black', tl.cex = 0.8)

# There are some variables with almost a correlation of 1
# Let's apply a feature selection removing very correlated variables.

# Feature selection ----

# area_se, radius_se, perimeter_se -> area_se
dataset$radius_se <- NULL
dataset$perimeter_se <- NULL
# area_mean, radius_mean, perimeter_mean -> area_mean
dataset$radius_mean <- NULL
dataset$perimeter_mean <- NULL
# area_worst, radius_worst, perimeter_worst -> area_worst
dataset$radius_worst <- NULL
dataset$area_worst <- NULL

# Now let's apply feature extraction (PCA or LDA) because we
# can easily reduce dimensionality without losing so much information.

# Training and Test set spliting ----
library(caTools)
split = sample.split(dataset$diagnosis, SplitRatio = 0.8)
training.set = subset(dataset, split == TRUE)
test.set = subset(dataset, split == FALSE)

# PCA ----
library(caret)
(pca <- preProcess(x = training.set[-1],
                   method = 'pca',
                   thresh = 0.95))

# Create PCA for training and test set.
training.set.pca <- predict(pca, newdata = training.set)
test.set.pca <- predict(pca, newdata = test.set)

# Observe that with 10 components we get over 0.95 of variance explained.
# 20 extra components would only explain an extra 0.05, so we discard these ones.

# TODO: Falta plot amb la variància que explica cada component

# Display diagnosis over 2D (First two PC).
# We can see that data is easily separable.
library(ggplot2)
ggplot(training.set.pca, aes(x = PC1, y = PC2, col = diagnosis)) +
  ggtitle('Diagnosis distribution over first two PC (training)') +
  geom_point(alpha = 0.8)

# LCA ----
library(MASS)
lda <- lda(formula = diagnosis ~ .,
           data = dataset)

# Create LDA for training and test set.
training.set.lda <- as.data.frame(predict(lda, training.set))
training.set.lda <- training.set.lda[c(1, 4)]
colnames(training.set.lda) <- c('diagnosis', 'LD1')

test.set.lda <- as.data.frame(predict(lda, test.set))
test.set.lda <- test.set.lda[c(1, 4)]
colnames(test.set.lda) <- c('diagnosis', 'LD1')
                        
# Display diagnosis over 1D.
ggplot(training.set.lda, aes(x = LD1, y = 0, col = diagnosis)) +
  ggtitle('Diagnosis distribution over LD1 (training)') +
  geom_point(alpha = 0.8)

# Display diagnosis over 2D (Density).
ggplot(training.set.lda, aes(x = LD1, fill = diagnosis)) +
  ggtitle('Diagnosis density over first LD (training)') +
  geom_density(alpha = 0.8)

# We can conclude that data is easily separable.

####################################################################
# SECTION 2: Model Building
####################################################################

# Let's define a common trainControl to set the same train validation in all methods
trc <- trainControl(method = 'repeatedcv',
                    number = 10,
                    repeats = 5)

# K-NN ----
classifier.knn <- train(form = diagnosis ~ .,
                        data = training.set,
                        method = 'knn',
                        trControl = trc,
                        tuneLength = 20,
                        preProcess = c('center', 'scale'))
classifier.knn

pred.knn <- predict(classifier.knn,
                    newdata = test.set)

# K plot, Confusion matrix and Accuracy.
plot(classifier.knn)
(conf.knn <- confusionMatrix(data = pred.knn,
                             reference = test.set$diagnosis,
                             positive = 'M'))
(acc.knn <- mean(pred.knn == test.set$diagnosis),)

# Logistic ----
classifier.log <- glm(formula = diagnosis ~ .,
                      family = binomial,
                      data = training.set)

prob.log <- predict(classifier.log,
                    type = 'response',
                    newdata = test.set[-1])
pred.log <- ifelse(prob.log > 0.5, 'M', 'B')

# Logistic Confusion matrix and accuracy.
(conf.log <- table(test.set[, 1], pred.log))
(acc.log <- (conf.log[1, 1] + conf.log[2, 2]) / dim(test.set)[1])

# Random Forest Regression ----
classifier.rf <- train(form = diagnosis ~ .,
                       data = training.set,
                       method = 'ranger',
                       trControl = trc,
                       tuneLength = 20,
                       preProcess = c('center', 'scale'))
classifier.rf

pred.rf <- predict(classifier.rf,
                   newdata = test.set)

# nTree plot, Confusion matrix, Accuracy.
plot(classifier.rf)
(conf.rf <- confusionMatrix(data = pred.rf,
                            reference = test.set$diagnosis,
                            positive = 'M'))
(acc.rf <- mean(pred.rf == test.set$diagnosis))

# SVM ----
classifier.svm.line <- train(form = diagnosis ~ .,
                             data = training.set,
                             method = 'svmLinear',
                             trControl = trc,
                             trace = FALSE)

classifier.svm.poly <- train(form = diagnosis ~ .,
                             data = training.set,
                             method = 'svmPoly',
                             trControl = trc,
                             trace = FALSE)

classifier.svm.gaus <- train(form = diagnosis ~ .,
                             data = training.set,
                             method = 'svmRadial',
                             trControl = trc,
                             trace = FALSE)

pred.svm.line <- predict(classifier.svm.line,
                         newdata = test.set)
pred.svm.poly <- predict(classifier.svm.poly,
                         newdata = test.set)
pred.svm.gaus <- predict(classifier.svm.gaus,
                         newdata = test.set)

# Cost plots, SVM Confusion matrix and accuracy.
plot(classifier.svm.poly)
plot(classifier.svm.gaus)
(conf.svm.line <- confusionMatrix(data = classifier.svm.line,
                                  reference = test.set$diagnosis,
                                  positive = 'M'))
(conf.svm.poly <- confusionMatrix(data = classifier.svm.poly,
                                  reference = test.set$diagnosis,
                                  positive = 'M'))
(conf.svm.gaus <- confusionMatrix(data = classifier.svm.gaus,
                                  reference = test.set$diagnosis,
                                  positive = 'M'))
(acc.svm.line <- mean(pred.svm.line == test.set$diagnosis))
(acc.svm.poly <- mean(pred.svm.poly == test.set$diagnosis))
(acc.svm.gaus <- mean(pred.svm.gaus == test.set$diagnosis))

# Naive Bayes ----
classifier.nb <- train(form = diagnosis ~ .,
                       data = training.set,
                       method = 'nb',
                       trControl = trc,
                       trace = FALSE)

pred.nb <- predict(classifier.nb, newdata = test.set)

# Distribution type plot, Bayes Confusion matrix and accuracy.
plot(classifier.nb, type = 'p', col = 'darkred')
(conf.nb <- confusionMatrix(data = classifier.nb,
                            reference = test.set$diagnosis,
                            positive = 'M'))
(acc.nb <- mean(pred.nb == test.set$diagnosis))

# Neural Networks ----
library(nnet)
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
grid <- expand.grid(decay = 10^seq(-3, 0, 0.3),
                    size = seq(1, 10))
nnet <- train(form = diagnosis ~.,
              data = training.set,
              method = 'nnet', 
              tuneGrid = grid, 
              trControl = trc,
              trace = FALSE)

pred.nn <- as.factor(predict(nnet, newdata = test.set, type = 'raw'))
(conf.nn <- table(pred.nn, test.set$diagnosis))
(acc.nn <- (conf.nn[1, 1] + conf.nn[2, 2]) / dim(test.set)[1])
# 98.2% accuracy
# Gradient Boosting ----
tune.xgb <- expand.grid(
  eta = c(0.01, 0.001, 0.0001),
  nrounds = 500,
  lambda = 1,
  alpha = 0
)

classifier.xgb <- train(x = as.matrix(training.set[-1]),
                        y = training.set$diagnosis,
                        method = 'xgbLinear',
                        trControl = trc,
                        tuneGrid = tune.xgb)

pred.xgb <- predict(classifier.xgb,
                    newdata = test.set)

# Cost plots, Confusion matrix and accuracy.
(conf.xgb <- confusionMatrix(data = classifier.xgb,
                             reference = test.set$diagnosis,
                             positive = 'M'))
(acc.xgb <- mean(pred.xgb == test.set$diagnosis))

####################################################################
# SECTION 3: Model Comparaison
####################################################################

classifiers <- list(
  KNN = classifier.knn,
  SVM = classifier.svm.poly,
  RF = classifier.rf,
  NN = nnet,
  GB = classifier.xgb)

models.corr <- modelCor(classifiers)