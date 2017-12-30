####################################################################
# APA Pràctica (Grau FIB)
# Josep de Cid & Gonzalo Recio

# Breast Cancer Diagnostic
# Q1 2017-2018
####################################################################

####################################################################
# SECTION 1: Data Preprocessing
####################################################################

dataset <- read.csv('data.csv')
# Remove unnecessary column 'X'
dataset <- subset(dataset, select = -X )

summary(dataset)
# We see that there are no NA values.
# There are also no strange or out of range values.

# Converting diagnosis into factor variables
dataset$diagnosis <- as.factor(dataset$diagnosis)
table(dataset$diagnosis) # Unbalanced observations

# Feature Scaling ????

# We can see 3 different feature groups: 'mean', 'std error', and 'worst'
# Extract groups into different variables
features.mean <- dataset[, 3:12]
features.sderr <- dataset[, 13:22]
features.worst <- dataset[, 23:32]

# Take a look at heavy corrlations among different variables
library(corrplot)
correlation <- cor(dataset[,3:32])
corrplot(correlation, order = 'hclust', tl.cex = 1, addrect = 8)

# Split dataset into training and test set
library(caTools)
split = sample.split(dataset$id, SplitRatio = 0.8)
training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)

# As we have a lot of dimensions and very correlated data, let's apply a PCA
pca <- prcomp(dataset[, 3:32], center = TRUE, scale = TRUE)
plot(pca, type="l")

# Take a look at the variance proportion given by each component
summary(pca)

# Seting a Statistical significance level SL = 0.05
# We obserbe that with 10 components we get over 0.95
# We would need 20 extra components only to obtain an extra 0.05, so we discard this ones.
pca.df <- as.data.frame(pca$x)

library(ggplot2)
ggplot(pca.df) +
  geom_point(aes(x = PC1, y = PC2, col = dataset$diagnosis)) +
  ggtitle('Diagnosis distribution over first two Principal Components')
