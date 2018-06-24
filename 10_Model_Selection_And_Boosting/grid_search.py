# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def main():
    data_processing()


def data_processing():
    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Social_Network_Ads.csv")
    X = dataset.iloc[:, [2,3]].values
    y = dataset.iloc[:, 4].values

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ------------------------------------------- Apply SVM ---------------------------------------------------------- #
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)


    # ----------------------------------- Predicting the Test Set Results -------------------------------------------- #
    y_pred = classifier.predict(X_test)
    print("Results: \n", y_pred)

    # ----------------------------------- Making the confusion matrix ----------------------------------------------- #

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n",cm)

    # ------------------------------------- Apply K-Fold cross validation -------------------------------------------- #

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10) #supervise learning because we use the dependent variable
    print("Vector of accuracies: ", accuracies)
    print("Mean value of the accuracies: ", accuracies.mean())
    print(accuracies.std())

    # ----------------------------- Apply Grid search for finding the best model with the best parameters------------- #
    #tune the parameters of the SVC model
    parameters = [{'C':[1, 10, 100, 1000], 'kernel':['linear']}, #linear model
                  {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]} #non-linear
                ]
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = "accuracy",
                               cv= 10, # 10 fold cross validations will be applied to the grid search, k-cross validation
                               n_jobs = -1 # use all cpus in a large dataset
    )

    grid_search = grid_search.fit(X_train, y_train)

    best_accuracy = grid_search.best_score_
    print("Best accuracy prediction result: ",best_accuracy)

    best_parameters = grid_search.best_params_
    print("Best parameters on your model: ", best_parameters)
    
    # ---------------------------------- Visualizing the Training set results ---------------------------------------- #

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Kernel PCA (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

    # ---------------------------------- Visualizing the Test set results ---------------------------------------- #

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Kernel PCA(Test set)')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()