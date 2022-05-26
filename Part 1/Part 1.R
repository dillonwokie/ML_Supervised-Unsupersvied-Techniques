# ========================================================================================================
# Purpose:      ST3189 Coursework part 1 (EWCS dataset)
# Name:         Dillon Yew (10196936)
# Task:         Visualize and describe the data via unsupervised learning methods
# Dataset:      EWCS_2016.csv
# Packages:     ggplot2,dplyr,ISLR,factoextra
#=========================================================================================================

## Data Cleaning/Extraction -------------------------------------------------

# Loading packages
library(ggplot2)
library(dplyr)
library(ISLR)
library(factoextra)

# Set working directory
setwd("D:/ST3189 Machine Learning/Coursework/Part 1")

# Loading the EWCS dataset
ewcs=read.table("EWCS_2016.csv",sep=",",header=TRUE)
ewcs[,][ewcs[, ,] == -999] <- NA
kk=complete.cases(ewcs)
ewcs=ewcs[kk,]

# Examining EWCS dataset
sum(is.na(ewcs)) 
ggcorr(ewcs, label=TRUE, label_size=4, label_alpha=TRUE) # Correlation between variables
apply(ewcs,2, mean)
apply(ewcs,2, var)
summary(ewcs)
# Age have the largest mean and variance.It would dominate the principal components.
# We need to standardize the variables when performing PCA

## Principal Component Analysis (PCA)-------------------------------------------

pca=prcomp(ewcs, scale=TRUE)
names(pca)
summary(pca) # First 4 Principal Components capture about 70% of the variance
pca.sdev=pca$sdev
pca.loadings=pca$rotation
biplot(pca, scale=0, cex=0.8, xlab="PC1 (40.0% of variance explained)", 
       ylab="PC2 (12.8% of variance explained)")

# Proportion of Variance Explained (PVE)
pca.var=pca$sdev^2
pve=pca.var/sum(pca.var)
par(mfrow=c(1,2))
plot(pve, col="blue", type="o", xlab="Principal Component",
     ylab="PVE", ylim=c(0,1), xlim=c(0,10))
plot(cumsum(pve), col='blue', type="o", xlab="Principal Component",
     ylab="Cumulative PVE", ylim=c(0,1), xlim=c(0,10))
par(mfrow=c(1,1))


## K Means Clustering-----------------------------------------------------------

# Clustering the data
set.seed(20)
ewcs.scaled=scale(ewcs) # Scaling the data

# Determining optimal number of clusters
wss = sapply(1:20, function(x){
  kmeans(ewcs.scaled, x, iter.max=15)$tot.withinss}
)
fviz_nbclust(ewcs.scaled, kmeans, method = "wss")

# Checking for patterns in clusters
km2=kmeans(ewcs.scaled, centers=2, nstart=20) # K = 2 clusters 
table(km2$cluster)
km2.results=data.frame(ewcs,km2$cluster)
km2.cluster1=subset(km2.results, km2$cluster==1)
km2.cluster2=subset(km2.results, km2$cluster==2)
ewcs %>% 
  mutate(Clusters = km2$cluster) %>% 
  group_by(Clusters) %>% 
  summarise_all("mean")


## Hierarchical Clustering------------------------------------------------------

ewcs.scaled=scale(ewcs) # Scaling the data
data.dist=dist(ewcs.scaled)

# Linkages
hc.complete =hclust(data.dist, method ="complete")
hc.average =hclust(data.dist, method ="average")

# Plotting the deprograms
par(mfrow = c(1,2))
plot(hc.complete, main="Complete Linkage", xlab="", sub="", cex=0.9)
abline(h=13, col='red')
sum(cutree(hc.complete , 2)==2)
sum(cutree(hc.complete , 2)==1)
plot(hc.average, main="Average Linkage", xlab="", sub="", cex=0.9)
abline(h=8, col='red')
sum(cutree(hc.average , 2)==2)
sum(cutree(hc.average , 2)==1)

# Checking for patterns in clusters
hc.results=data.frame(ewcs, cutree(hc.complete,2))
hc.cluster1=subset(hc.results, cutree(hc.complete,2)==1)
hc.cluster2=subset(hc.results, cutree(hc.complete,2)==2)
ewcs %>% 
  mutate(Clusters = cutree(hc.complete,2)) %>% 
  group_by(Clusters) %>% 
  summarise_all("mean")













