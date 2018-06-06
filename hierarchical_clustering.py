# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

def main():
    # ---------------------------------------- Retrieve the dataset ------------------------------------------------ #
    dataset = pd.read_csv(
        "C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv")
    X = dataset.iloc[:, [3, 4]].values

    # ---------------------------------------- Build and visualize the dendogram ------------------------------------------------ #
    dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
    # ward minimizes the variance within each cluster
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Euclidean distances")
    plt.show()

    # ---------------------------------------- Fit the hierarchical cluster ----------------------------------------- #

    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_predict_hc = hc.fit_predict(X)

    # ---------------------------------------- Visualize the clusters in 2D ----------------------------------------- #

    plt.scatter(X[y_predict_hc == 0, 0],
                X[y_predict_hc== 0, 1],
                s=100, c="red", label="Cluster 1"
                )

    plt.scatter(X[y_predict_hc == 1, 0],
                X[y_predict_hc == 1, 1],
                s=100, c="blue", label="Cluster 2"
                )

    plt.scatter(X[y_predict_hc == 2, 0],
                X[y_predict_hc == 2, 1],
                s=100, c="green", label="Cluster 3"
                )

    plt.scatter(X[y_predict_hc == 3, 0],
                X[y_predict_hc == 3, 1],
                s=100, c="yellow", label="Cluster 4"
                )

    plt.scatter(X[y_predict_hc == 4, 0],
                X[y_predict_hc == 4, 1],
                s=100, c="magenta", label="Cluster 5"
                )

    plt.title("Clusters of clients")
    plt.xlabel("Annual income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()