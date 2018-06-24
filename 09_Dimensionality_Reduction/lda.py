# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

def main():
    data_processing()


def data_processing():
    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 9 - Dimensionality Reduction\\Section 44 - Linear Discriminant Analysis (LDA)\\Wine.csv")
    X = dataset.iloc[:, 0:13].values
    y = dataset.iloc[:, 13].values

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ------------------------------------------- Apply LDA ---------------------------------------------------------- #

    # there are two independent variables that separate the most the classes of our dependent variable - we knew thise from PCA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train = lda.fit_transform(X_train,y_train) # we include the y_train because this is a supervised model
    X_test = lda.transform(X_test)

    # ------------------------------------- Fitting Logistic Regression ---------------------------------------------- #

    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    # ----------------------------------- Predicting the Test Set Results -------------------------------------------- #
    y_pred = classifier.predict(X_test)
    print("Results: \n", y_pred)

    # ----------------------------------- Making the confusion matrix ----------------------------------------------- #

    cm = confusion_matrix(y_test, y_pred)
    print("Test confusion matrix: \n",cm)

    # ---------------------------------- Visualizing the Training set results ---------------------------------------- #

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.show()

    # ---------------------------------- Visualizing the Test set results ---------------------------------------- #

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()