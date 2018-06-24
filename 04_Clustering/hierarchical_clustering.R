# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 6/4/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv")

X <- dataset[4:5]#Age, Salary and Purchased columns


# ------------------------------------- Build the dendogram and visualize it------------------------------ #

dendrogram = hclust(dist(X, method="euclidean"), method="ward.D")
plot(dendrogram,
    main = paste("Dendogram"),
    xlab="Customers",
    y_lab= "Euclidean distances")


# ---------------------------------- Fit the hierarchical cluster and visualize it -----------------------#

hc = hclust(dist(X, method="euclidean"), method="ward.D")
y_hc = cutree(hc, 5, )

library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Clusters of clients"),
         xlab = "Annual income",
         ylab = "Spending score"
)

