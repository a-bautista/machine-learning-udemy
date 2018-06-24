# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

def main():
    data_processing()


def data_processing():

    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 3 - Classification\\Section 19 - Decision Tree Classification\\Social_Network_Ads.csv")

    # take all the columns except the last one for your matrix of features
    X = dataset.iloc[:, [2,3]].values
    y = dataset.iloc[: ,4].values

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #
    # feature scaling is not necessary in this algorithm but if you do not scale it then your code my break
    sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    X_test = sc_X.transform(X_test)

    # ------------------------------------- Fitting Classifier ---------------------------------------------- #

    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # ----------------------------------- Predicting the Test Set Results -------------------------------------------- #

    y_pred = classifier.predict(X_test)
    print("Results test: \n", y_test)
    print("Results prediction: \n", y_pred)

    # ----------------------------------- Making the confusion matrix ----------------------------------------------- #

    # 7 incorrect predictions
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n",cm)

    # ---------------------------------- Visualizing the Training set results ---------------------------------------- #

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1,  stop = X_set[:, 0].max() + 1, step = 0.01),\
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random forest classification (Training Set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    # ---------------------------------- Visualizing the Test set results ---------------------------------------- #

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1,  stop = X_set[:, 0].max() + 1, step = 0.01),\
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random forest classification (Test Set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
