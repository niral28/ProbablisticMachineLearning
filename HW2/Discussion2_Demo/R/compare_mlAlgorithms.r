###
### Purpose: to assess the performances of prediction models for whether a consumer is declined or approved for a credit card.
###

##
## Load the data set
##

# Load the Credit Card data set from the AER package.
# The data set contains a response vector card that indicates whether a consumer was declined (no) or approved for a credit card (yes).
library(AER)
data(CreditCard)
CreditCard$Class <- CreditCard$card
CreditCard <- subset(CreditCard, select=-c(card, expenditure, share))

set.seed(1984)

library(caret)

# 4/5 of data for training, 1/5 of data for evaluation/testing
training <- createDataPartition(CreditCard$Class, p = 0.8, list=FALSE)
trainData <- CreditCard[training,]
testData <- CreditCard[-training,]

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10,
                           # Estimate class probabilities
                           classProbs = TRUE,
                           # Evaluate performance using 
                           # the following function
                           summaryFunction = twoClassSummary)

library(pROC)
						   
# Fit a glm model (logistic regression) and score the test data set.
glmModel <- glm(Class~ . , data=trainData, family=binomial)
pred.glmModel <- predict(glmModel, newdata=testData, type="response")

# Calculate the AUC for the test data set.
roc.glmModel <- pROC::roc(testData$Class, pred.glmModel)
auc.glmModel <- pROC::auc(roc.glmModel)

# Support Vector Machines:
set.seed(2014)

svmModel <- train(Class ~ ., data=trainData, method = "svmRadial", metric="ROC", trControl = fitControl, verbose=FALSE, tuneLength=5)
pred.svmModel <- as.vector(predict(svmModel, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.svmModel <- pROC::roc(testData$Class, pred.svmModel)
auc.svmModel <- pROC::auc(roc.svmModel)

# Boosted Trees
set.seed(2014)

gbmModel <- train(Class ~ ., data=trainData, method = "gbm", metric="ROC", trControl = fitControl, verbose=FALSE, tuneLength=5)
pred.gbmModel <- as.vector(predict(gbmModel, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.gbmModel <- pROC::roc(testData$Class, pred.gbmModel)
auc.gbmModel <- pROC::auc(roc.gbmModel)

# CART
set.seed(2014)

cartModel <- train(Class ~ ., data=trainData, method = "rpart", metric="ROC", trControl = fitControl, tuneLength=5)
# Loading required package: rpart
pred.cartModel <- as.vector(predict(cartModel, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.cartModel <- pROC::roc(testData$Class, pred.cartModel)
auc.cartModel <- pROC::auc(roc.cartModel)

# C4.5
set.seed(2014)

c45Model <- train(Class ~ ., data=trainData, method = "J48", metric="ROC", trControl = fitControl, tuneLength=5)
# Loading required package: J48
pred.c45Model <- as.vector(predict(c45Model, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.c45Model <- pROC::roc(testData$Class, pred.c45Model)
auc.c45Model <- pROC::auc(roc.c45Model)

# C5.0
set.seed(2014)

c50Model <- train(Class ~ ., data=trainData, method = "C5.0", metric="ROC", trControl = fitControl, tuneLength=5)
# Loading required package: C5.0
pred.c50Model <- as.vector(predict(c50Model, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.c50Model <- pROC::roc(testData$Class, pred.c50Model)
auc.c50Model <- pROC::auc(roc.c50Model)

# Random Forest
set.seed(2014)

rfModel <- train(Class ~ ., data=trainData, method = "rf", metric="ROC", trControl = fitControl, verbose=FALSE, tuneLength=5)
# Loading required package: randomForest
# randomForest 4.6-10
# Type rfNews() to see new features/changes/bug fixes.
pred.rfModel <- as.vector(predict(rfModel, newdata=testData, type="prob")[,"yes"])

# Calculate the AUC for the test data set.
roc.rfModel <- pROC::roc(testData$Class, pred.rfModel)
auc.rfModel <- pROC::auc(roc.rfModel)

##
## Choose the best model
##

# Plot AUC, on the test data set, for each model.
test.auc <- data.frame(model=c("glm", "svm", "gbm", "cart", "c45", "c50", "rForest"),auc=c(auc.glmModel, auc.svmModel, auc.gbmModel, auc.cartModel, auc.c45Model, auc.c50Model, auc.rfModel))
test.auc <- test.auc[order(test.auc$auc, decreasing=TRUE),]
test.auc$model <- factor(test.auc$model, levels=test.auc$model)
test.auc

library(ggplot2)
theme_set(theme_gray(base_size = 18))
qplot(x=model, y=auc, data=test.auc, geom="bar", stat="identity", position = "dodge")+ geom_bar(fill = "light blue", stat="identity")

# Plot tuning parameters that were chosen by repeated CV.

# plot(glmModel)
plot(roc.glmModel, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.3, print.auc.col="blue", type="l", col='blue', lwd=1, lty=1)
# plot(svm.rocCurve)
plot(roc.svmModel, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.25, print.auc.col="purple", type="l", add=TRUE, col='purple', lwd=1, lty=1)
# plot(gbmModel)
plot(roc.gbmModel, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.2, print.auc.col="red", type="l", add=TRUE, col='red', lwd=1, lty=1)
# plot(cartModel)
plot(roc.cartModel, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.15, print.auc.col="green", type="l", add=TRUE, col='green', lwd=1, lty=1)
# plot(c45Model)
plot(roc.c45Model, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.1, print.auc.col="orange", type="l", add=TRUE, col='orange', lwd=1, lty=1)
# plot(c50Model)
plot(roc.c50Model, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.05, print.auc.col="navy", type="l", add=TRUE, col='navy', lwd=1, lty=1)
# plot(rfModel)
plot(roc.rfModel, print.auc=TRUE, print.auc.x=0.7, print.auc.y=0.0, print.auc.col="black", type="l", add=TRUE, col='black', lwd=1, lty=1)

legend("bottomright", legend=c("logistic regression", "SVM", "Boosted Trees", "CART", "C45", "C50", "rForest"), col=c("blue", "purple", "red", "green", "orange", "navy", "black"), lwd=1)

# uncomment the following lines to see the models generated
 #print(glmModel)
 #print(svmModel)
 #print(gbmModel)
 #print(cartModel)
 #print(c45Model)
 #print(c50Model)
 #print(rfModel)

