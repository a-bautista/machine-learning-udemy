# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 6/4/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Mall_Customers.csv")

X <- dataset[4:5]#Age, Salary and Purchased columns


# ---------------------------------------------- Using the Elbow method ------------------------------------ #

set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10,wcss, type="b", main = paste("Clusters of clients"),xlab="Number of clusters", ylab="WCSS")


# ---------------------------------------------- Applying K-Means  ------------------------------------ #

set.seed(29)
kmeans <- kmeans(X,5,iter.max=300,nstart=10)
y_kmeans = kmeans$cluster

# ---------------------------------------------- Plot results  ------------------------------------ #

library(cluster)
clusplot(X,
         y_kmeans,
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





