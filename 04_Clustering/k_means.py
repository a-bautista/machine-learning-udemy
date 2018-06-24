# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    # ---------------------------------------- Retrieve the dataset ------------------------------------------------ #
    dataset = pd.read_csv(
        "C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Mall_Customers.csv")
    X = dataset.iloc[:,[3,4]].values

    # ---------------------------------- using the Elbow method to find the optional number of clusters ------------ #
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
    plt.title("The Elbow method")
    plt.ylabel("WCSS")
    plt.show()

    # ---------------------------------- fit the number of clusters into our dataset ------------------------------ #

    kmeans_defined_clusters = KMeans(n_clusters=5,init='k-means++',max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans_defined_clusters.fit_predict(X)

    # ---------------------------------- visualize the clusters ------------------------------ #
    plt.scatter(X[y_kmeans == 0,0],
                X[y_kmeans == 0,1],
                s = 100, c= "red", label = "Cluster 1"
                )

    plt.scatter(X[y_kmeans == 1, 0],
                X[y_kmeans == 1, 1],
                s=100, c="blue", label="Cluster 2"
                )

    plt.scatter(X[y_kmeans == 2, 0],
                X[y_kmeans == 2, 1],
                s=100, c="green", label="Cluster 3"
                )

    plt.scatter(X[y_kmeans == 3, 0],
                X[y_kmeans == 3, 1],
                s=100, c="yellow", label="Cluster 4"
                )

    plt.scatter(X[y_kmeans == 4, 0],
                X[y_kmeans == 4, 1],
                s=100, c="magenta", label="Cluster 5"
                )

    plt.scatter(kmeans_defined_clusters.cluster_centers_[:, 0], kmeans_defined_clusters.cluster_centers_[:,1], s = 300,
                c = "black", label="Centroids")

    plt.title("Clusters of clients")
    plt.xlabel("Annual income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()