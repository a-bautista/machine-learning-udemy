# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def main():
    data_processing()
    #regressor = build_LR_model(X_train, y_train)
    #y_predict = predict_values(regressor, X_test)
    #build_optimal_linear_regression_model(X_no_column_zero, y)
    #visualize_results(X,y,lin_reg, lin_reg_two, poly_reg)


def data_processing():

    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv")

    # take all the columns except the last one for your matrix of features
    # this is a vector
    #X = dataset.iloc[:, 1].values

    # this is a matrix
    X = dataset.iloc[:, 1:2].values

    # define your dependent variable vector
    y = dataset.iloc[:, 2].values

    # --------------------------------------- Fitting decision tree regression to the data set   ------------------------------------ #

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    y_predict = regressor.predict(6.6)
    print(y_predict)

    # -------------------------------------- visualize the decision tree regression  Results ----------------------------- #

    # You will see that the values in the graph are not adjusted to the curve because the level 10 which is the CEO
    # is an outlier.
    plt.scatter(X, y, color='red')
    # The line from below contains y_predictor which is a vector and this will not be graphed against a matrix.
    # plt.plot(X, y_predictor, color='blue')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title("Truth or Bluff? (Decision Tree Regression results)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()

    # -------------------------------------- visualize the decision tree regression high resolution ----------------------------- #
    # this how to see a non-continuous model
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X, y, color='red')
    # The line from below contains y_predictor which is a vector and this will not be graphed against a matrix.
    # plt.plot(X, y_predictor, color='blue')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title("Truth or Bluff? (Decision Tree Regression results)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    '''
    # you split the data into a training set and testing set but only if you have at least two columns with many rows of observations.
    # test_size = 0.2 indicates that 20% of data will go to TEST and 80% will go to TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X_no_column_zero, y, test_size = 0.2, random_state=0)
    print("Matrix of features training set:\n", X_train.astype(int), "\nMatrix of features test set:\n", X_test.astype(int),
          "\nDependent variable training set:\n", y_train, "\nDependent variable test set:\n", y_test)'''

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    #Feature scaling does not apply in Linear Regression due to the libraries that we are going to use.
    # sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    # X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    # X_test = sc_X.transform(X_test)'''




def build_LR_model(X_train,y_train):

    # build the Linear Regression model
    regressor = LinearRegression()

    # train the model
    regressor.fit(X_train, y_train)

    return regressor

def predict_values(regressor, X_test):

    y_predict = regressor.predict(X_test)
    # this result throws the profit column based on the multiple columns that we took as input
    print("\nPredicted values:\n ", y_predict.astype(int))
    return y_predict

def visualize_results(X,y, lin_reg, lin_reg_two, poly_reg, regressor):

    plt.scatter(X, y, color='Red')
    plt.plot(X, regressor.predict(X), color='Blue')
    plt.title("Truth or Bluff? (Polynomial Regression)")
    plt.xlabel('POsition Level')
    plt.ylabel('Salary')
    plt.show()

    # visualize the Polynomial Regression Results
    # make a smooth graph
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid),1))

    plt.scatter(X, y, color='red')
    plt.plot(X_grid, lin_reg_two.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title("Truth or Bluff? (Polynomial Regression)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()

    # predicting a new result with Linear Regression
    # predict the salary of level 6.5
    print(lin_reg.predict(6.5))

    # predicting a new result with Polynomial Regression
    # predict the salary of level 6.5
    print(lin_reg_two.predict(poly_reg.fit_transform(6.5)))


if __name__ == "__main__":
    main()