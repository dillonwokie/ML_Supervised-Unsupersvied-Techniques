# ========================================================================================================
# Purpose:      ST3189 Coursework part 3 (Bank Marketing dataset)
# Name:         Dillon Yew (10196936)
# Task:         Build a classification model to predict if client will subscribe to a term deposit
# Dataset:      bank.csv
# Packages:     caTools,caret,class,dplyr,pROC,randomForest,e1071,rpart,rpart.plot
#=========================================================================================================

# Loading packages
library(caTools)
library(caret)
library(class)
library(dplyr)
library(pROC)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)

# Set working directory
setwd("D:/ST3189 Machine Learning/Coursework/Part 3")

#R code to import the bank marketing dataset
bank=read.table("bank.csv",sep=";",header=TRUE)
bank=subset(bank, select=-duration)

# Verifying and examining dataset
bank[sapply(bank,is.character)]=lapply(bank[sapply(bank,is.character )],as.factor)
sum(is.na(bank)) 
sum(is.na(bank.scaled))
summary(bank)


# Logistic Regression(LR)-----------------------------------------------------------

# Train-Test split
set.seed(1)
train=sample(1:nrow(bank), nrow(bank)*0.7)
trainset=bank[train,]
testset=bank[-train,]

# LR Model
lr.bank=glm(y ~., data=bank, family="binomial", subset=train)
summary(lr.bank)
lr.prob=predict(lr.bank, newdata=testset, type="response")
lr.pred=as.factor(ifelse(lr.prob > 0.5, "yes", "no"))
confusionMatrix(lr.pred, testset[,16])

# ROC Curve for Logistic Regression
par(pty="s")
roc(testset$y ~ lr.prob, plot=TRUE, col="#377eb8", lwd=4, print.auc=TRUE)
par(pty='m')


# K-Nearest Neighbors-----------------------------------------------------------

# Normalize function
normalize= function(x){
  (x-min(x))/(max(x)-min(x))
}

# Train-Test Split
set.seed(1)
bank.n= bank %>% 
  mutate_if(is.numeric, scale)
train.knn=sample(1:nrow(bank), nrow(bank)*0.7)
trainset.knn=bank.n[train.knn,]
testset.knn=bank.n[-train.knn,]
train.labels=bank[train.knn, 16]
test.labels=bank[-train.knn, 16]
trainset.knn[sapply(trainset.knn,is.factor)]=lapply(trainset.knn[sapply(trainset.knn,is.factor )],as.numeric)
testset.knn[sapply(testset.knn,is.factor)]=lapply(testset.knn[sapply(testset.knn,is.factor )],as.numeric)

# KNN Model
knn.pred=knn(train=trainset.knn,test=testset.knn,
             cl=train.labels,
             k=round(sqrt(nrow(trainset.knn))), prob=TRUE )
confusionMatrix(knn.pred,testset[,16])

# ROC Curve
par(pty="s")
roc(testset.knn$y, attributes(knn.pred)$prob, plot=TRUE, col="#377eb8", lwd=4, print.auc=TRUE)
par(pty="m")


# Classification Tree-----------------------------------------------------------
dt=rpart(y ~ ., data=trainset, method="class")
rpart.plot(dt, main="Maximal Decision Tree for bank")
summary(dt)

# Optimal Classification Tree via Cross-Validation and Pruning 
CVerror.cap=dt$cptable[which.min(dt$cptable[,"xerror"]), "xerror"] + 
dt$cptable[which.min(dt$cptable[,"xerror"]), "xstd"]
i=1; j=4
while (dt$cptable[i,j] > CVerror.cap) {
  i=i+1
}
cp.opt = ifelse(i > 1, sqrt(dt$cptable[i,1] * dt$cptable[i-1,1]), 1)

dt.1se=prune(dt, cp=cp.opt)
rpart.plot(dt.1se, main="Optimal Decision Tree for Bank")
dt.pred=predict(dt.1se, testset, type="class")
dt.prob=predict(dt.1se, testset, type="prob")
confusionMatrix(dt.pred, testset[,16])

# ROC Curve for Classification Tree
par(pty="s")
roc(testset$y ~ dt.prob[,2], plot=TRUE, col="#377eb8", lwd=4, print.auc=TRUE)
par(pty="m")


# RandomForest------------------------------------------------------------------

# RF Model
rf=randomForest(y ~., data=bank, subset=train, mtry=round(sqrt(15)))
rf
plot(rf, main="RandomForest for bank")
rf.pred=predict(rf, testset, type="response")
confusionMatrix(rf.pred, testset[,16])

# ROC Curve
par(pty="s")
roc(rf$y, rf$votes[,1], plot=TRUE, col="#377eb8", lwd=4, print.auc=TRUE)
par(pty='m')

# Summary of Results------------------------------------------------------------

# Combined ROC Curve
roc(testset$y ~ lr.prob, plot=TRUE, col="#377eb8", lwd=4, print.auc=TRUE)
plot.roc(testset.knn$y, attributes(knn.pred)$prob, add=TRUE, col="#afb837", lwd=4, print.auc=TRUE, print.auc.y=0.4)
plot.roc(testset$y ~ dt.prob[,2], add=TRUE, col="#b8376b", lwd=4, print.auc=TRUE, print.auc.y=0.3)
plot.roc(rf$y, rf$votes[,1], add=TRUE, col="#37b857", lwd=4, print.auc=TRUE, print.auc.y=0.2)
legend(0,0.4, legend=c("LR", "KNN","DT(Class)","RF"), col=c("#377eb8", "#afb837","#b8376b","#37b857"), lwd=4)



