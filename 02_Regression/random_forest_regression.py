# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    dataset = pd.read_csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\02_Regression\\Position_Salaries.csv")

    X = dataset.iloc[:, 1:2].values

    # define your dependent variable vector
    y = dataset.iloc[:, 2].values

    # --------------------------------------- Fitting the random forest regression to the data set   ------------------------------------ #

    regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
    regressor.fit(X, y)
    y_predict = regressor.predict(6.5)
    print(y_predict)


    # -------------------------------------- visualize the random forest regression high resolution ----------------------------- #
    # this how to see a non-continuous model
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X, y, color='red')
    # The line from below contains y_predictor which is a vector and this will not be graphed against a matrix.
    # plt.plot(X, y_predictor, color='blue')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title("Truth or Bluff? (Random Forest Regression results)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()



if __name__ == "__main__":
    main()