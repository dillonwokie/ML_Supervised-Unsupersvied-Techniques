# ========================================================================================================
# Purpose:      ST3189 Coursework part 2 (Student Performance dataset)
# Name:         Dillon Yew (10196936)
# Task:         Build a regression model and assess its predictive performance
# Dataset:      student-mat.csv, student-por.csv
# Packages:     caTools,ISLR,randomForest,rpart,rpart.plot,gbm
#=========================================================================================================

# Loading packages
library(caTools)
library(ISLR)
library(randomForest)
library(rpart)
library(rpart.plot)
library(gbm)

# Set working directory
setwd("D:/ST3189 Machine Learning/Coursework/Part 2")

# R code to import and prepare the student performance dataset
school1=read.table("student-mat.csv",sep=";",header=TRUE)
school2=read.table("student-por.csv",sep=";",header=TRUE)
schools=merge(school1,school2,by=c("school","sex","age","address","famsize",
        "Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))

# Express variables as factors
school1[sapply(school1,is.character)]=lapply(school1[sapply(school1,is.character )],as.factor)
school2[sapply(school2,is.character)]=lapply(school2[sapply(school2,is.character )],as.factor)
schools[sapply(schools,is.character)]=lapply(schools[sapply(schools,is.character )],as.factor)

# Verifying and examining dataset
sum(is.na(school1)) 
sum(is.na(school2)) 
sum(is.na(schools)) 
summary(school1)
summary(school2)
summary(schools)


## School1 (Student's final grade in Mathematics)--------------------------------

# Train-test split
set.seed(1)
train1=sample.split(Y=school1$G3, SplitRatio = 0.7)
trainset1=subset(school1, train1==T)[,-c(31,32)]
testset1=subset(school1, train1==F)[,-c(31,32)]

# Linear Regression
lm1=lm(G3 ~ ., data=trainset1)
summary(lm1)
model=("Linear Regression")
RMSE.train1=sqrt(mean((trainset1$G3 - predict(lm1))^2))
RMSE.test1=sqrt(mean((testset1$G3 - predict(lm1, newdata = testset1))^2))

# Decision Tree
dt1=rpart(G3 ~., data=trainset1, method="anova")
model=c(model, "Maximal Decision tree")
print(dt1)
rpart.plot(dt1, main="Maximal Decision Tree for Mathematics")
RMSE.train1=c(RMSE.train1, sqrt(mean((trainset1$G3 - predict(dt1))^2)))
RMSE.test1=c(RMSE.test1, sqrt(mean((testset1$G3 - predict(dt1, newdata = testset1))^2)))

# Compute optimal decision tree via CV error within 1SE of the minimum CV tree
CVerror.cap1=dt1$cptable[which.min(dt1$cptable[,"xerror"]), "xerror"] + 
dt1$cptable[which.min(dt1$cptable[,"xerror"]), "xstd"]
i=1; j=4
while (dt1$cptable[i,j] > CVerror.cap1) {
  i=i+1
}
cp.opt1 = ifelse(i > 1, sqrt(dt1$cptable[i,1] * dt1$cptable[i-1,1]), 1)

# Optimal decision tree via pruning
dt1.1se=prune(dt1, cp = cp.opt1)
model=c(model, "Optimal Decision Tree")
rpart.plot(dt1.1se, main="Optimal Decision Tree for Mathematics")
RMSE.train1=c(RMSE.train1, sqrt(mean((trainset1$G3 - predict(dt1.1se))^2)))
RMSE.test1=c(RMSE.test1, sqrt(mean((testset1$G3 - predict(dt1.1se, newdata = testset1))^2)))

# Bagging 
bag1=randomForest(G3 ~., data=trainset1, mtry=30)
bag1
model=c(model,"Bagging")
plot(bag1, main="BAG-School1 (Mathematics)")
RMSE.train1=c(RMSE.train1, sqrt(mean((trainset1$G3 - predict(bag1, newdata = trainset1))^2)))
RMSE.test1=c(RMSE.test1, sqrt(mean((testset1$G3 - predict(bag1, newdata = testset1))^2)))

# RandomForest 
rf1=randomForest(G3 ~., data=trainset1, mtry=round(sqrt(30)))
rf1
model=c(model,"Randomforest")
plot(rf1, main="RF-School1 (Mathematics)")
RMSE.train1=c(RMSE.train1, sqrt(mean((trainset1$G3 - predict(rf1, newdata = trainset1))^2)))
RMSE.test1=c(RMSE.test1, sqrt(mean((testset1$G3 - predict(rf1, newdata = testset1))^2)))

# Boosting
boost1=gbm(G3 ~., data = trainset1, distribution="gaussian",
           n.trees=5000, interaction.depth=4)
model=c(model,"Boosting")
summary(boost1)
RMSE.train1=c(RMSE.train1, sqrt(mean((trainset1$G3 - predict(boost1, newdata = trainset1))^2)))
RMSE.test1=c(RMSE.test1, sqrt(mean((testset1$G3 - predict(boost1, newdata = testset1))^2)))

# Results for school1 (Student's final grade in Mathematics)
results1=data.frame(model, RMSE.train1, RMSE.test1)


## School2 (Student's final grade in Portuguese)--------------------------------

# Train-test split
set.seed(1)
train2=sample.split(Y=school2$G3, SplitRatio = 0.7)
trainset2=subset(school2, train2==T)[,-c(31,32)]
testset2=subset(school2, train2==F)[,-c(31,32)]

# Linear Regression
lm2=lm(G3 ~ ., data=trainset2)
summary(lm2)
model=("Linear Regression")
RMSE.train2=sqrt(mean((trainset2$G3 - predict(lm2))^2))
RMSE.test2=sqrt(mean((testset2$G3 - predict(lm2, newdata = testset2))^2))

# Decision Tree
dt2=rpart(G3 ~., data=trainset2, method="anova")
model=c(model, "Maximal Decision tree")
print(dt2)
rpart.plot(dt2, main="Maximal Decision Tree for Portuguese")
RMSE.train2=c(RMSE.train2, sqrt(mean((trainset2$G3 - predict(dt2))^2)))
RMSE.test2=c(RMSE.test2, sqrt(mean((testset2$G3 - predict(dt2, newdata = testset2))^2)))

# Compute optimal decision tree via CV error within 1SE of the minimum CV tree
CVerror.cap2=dt2$cptable[which.min(dt2$cptable[,"xerror"]), "xerror"] + 
  dt2$cptable[which.min(dt2$cptable[,"xerror"]), "xstd"]
i=1; j=4
while (dt2$cptable[i,j] > CVerror.cap2) {
  i=i + 1
}
cp.opt2 = ifelse(i > 1, sqrt(dt2$cptable[i,1] * dt2$cptable[i-1,1]), 1)

# Optimal decision tree via pruning
dt2.1se=prune(dt2, cp = cp.opt2)
model=c(model, "Optimal Decision Tree")
rpart.plot(dt2.1se, main="Optimal Decision Tree for Portuguese")
RMSE.train2=c(RMSE.train2, sqrt(mean((trainset2$G3 - predict(dt2.1se))^2)))
RMSE.test2=c(RMSE.test2, sqrt(mean((testset2$G3 - predict(dt2.1se, newdata = testset2))^2)))

# Bagging 
bag2=randomForest(G3 ~., data=trainset2, mtry=30, importance=TRUE)
bag2
model=c(model,"Bagging")
plot(bag2, main="BAG-School2 (Portuguese)")
RMSE.train2=c(RMSE.train2, sqrt(mean((trainset2$G3 - predict(bag2, newdata = trainset2))^2)))
RMSE.test2=c(RMSE.test2, sqrt(mean((testset2$G3 - predict(bag2, newdata = testset2))^2)))

# RandomForest 
rf2=randomForest(G3 ~., data=trainset2, mtry=round(sqrt(30)), importance=TRUE)
rf2
model=c(model,"Randomforest")
plot(rf2, main="RF-School2 (Portuguese)")
RMSE.train2=c(RMSE.train2, sqrt(mean((trainset2$G3 - predict(rf2, newdata = trainset2))^2)))
RMSE.test2=c(RMSE.test2, sqrt(mean((testset2$G3 - predict(rf2, newdata = testset2))^2)))

# Boosting
boost2=gbm(G3 ~., data = trainset2, distribution="gaussian",
           n.trees=5000, interaction.depth=4)
model=c(model,"Boosting")
summary(boost2)
RMSE.train2=c(RMSE.train2, sqrt(mean((trainset2$G3 - predict(boost2, newdata = trainset2))^2)))
RMSE.test2=c(RMSE.test2, sqrt(mean((testset2$G3 - predict(boost2, newdata = testset2))^2)))

# Results for school2 (Student's final grade in Portuguese)
results2=data.frame(model, RMSE.train2, RMSE.test2)


## Combined Results (Mathematics and Portuguese)--------------------------------
results1
results2








