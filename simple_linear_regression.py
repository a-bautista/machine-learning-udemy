# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    X_train, X_test, y_train, y_test = data_processing()
    regressor = build_LR_model(X_train, y_train)
    y_predict = predict_values(regressor, X_test)
    visualize_results(X_train, X_test, y_train, y_test, y_predict, regressor)


def data_processing():

    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv(
    "C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 2 - Regression\\Section 3 "
    "-------------------- Part 2 - Regression --------------------\\Simple_Linear_Regression\\Salary_Data.csv")

    # take all the columns except the last one for your matrix of features
    X = dataset.iloc[:, :-1].values

    # define your dependent variable vector
    y = dataset.iloc[:, 1].values

    print("Matrix of features", X, "dependent variable", y)

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    # you split the data into a training set and testing set
    # test_size = 0.2 indicates that 20% of data will go to TEST and 80% will go to TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)
    print("Matrix of features training set",X_train, "Matrix of features test set" ,X_test, "Dependent variable training set",
          y_train, "Dependent variable test set" ,y_test)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    '''Feature scaling does not apply in Linear Regression due to the libraries that we are going to use.'''
    # sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    # X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    # X_test = sc_X.transform(X_test)

    # print(X_train)
    # print(X_test)

    return X_train,X_test, y_train, y_test


def build_LR_model(X_train,y_train ):

    # build the Linear Regression model
    regressor = LinearRegression()

    # train the model
    regressor.fit(X_train, y_train)

    return regressor

def predict_values(regressor, X_test):

    y_predict = regressor.predict(X_test)
    print(y_predict)
    return y_predict

def visualize_results(X_train, X_test, y_train, y_test, y_predict, regressor):

    # visualize the training set results
    plt.scatter(X_train, y_train, color='red')
    # below you will plot the predicted values but you won't use y_predict because this variable already contains predicted values
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training set'')')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()

    # visualize the test set results
    plt.scatter(X_test, y_test, color='red')
    # no need to change the X_train values to X_test because the regressor contains the linear model formula trained
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Test set'')')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()


if __name__ == "__main__":
    main()