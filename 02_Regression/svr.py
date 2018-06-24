# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# SVR stands for Support Vector Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import statsmodels.formula.api as sm


def main():
    data_processing()
    # regressor = build_LR_model(X_train, y_train)
    # y_predict = predict_values(regressor, X_test)
    # build_optimal_linear_regression_model(X_no_column_zero, y)
    # visualize_results(X,y,lin_reg, lin_reg_two, poly_reg)


def data_processing():

    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\02_Regression\\Position_Salaries.csv")

    # take all the columns except the last one for your matrix of features

    # matrix of features
    X = dataset.iloc[:, 1:2].values

    # define your dependent variable vector
    y = dataset.iloc[:, 2].values

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    # Feature scaling does not apply in Linear Regression due to the libraries that we are going to use.
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply
    # the changes with fit_transform
    X = sc_X.fit_transform(X)
    #convert the 1 vector to matrix
    y = sc_y.fit_transform(np.reshape(y, (-1,1)))


    # --------------------------------------- Fitting SVR to the data set   ------------------------------------ #

    # 'rbf' --> Gaussian Kernel
    regressor = SVR(kernel='rbf')
    regressor.fit(X,y)

    # we had to add the inverse_transform to get the original scale of the salary because our values are scaled
    y_prediction = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.reshape(6.5, (-1,1)))))

    print(y_prediction)

    # -------------------------------------- visualize the SVR  Results ----------------------------- #
    # You will see that the values in the graph are not adjusted to the curve because the level 10 which is the CEO
    # is an outlier.
    plt.scatter(X, y, color='red')
    # The line from below contains y_predictor which is a vector and this will not be graphed against a matrix.
    # plt.plot(X, y_predictor, color='blue')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title("Truth or Bluff? (SVR)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()



    # --------------------------------------- Encode the non numerical data  ------------------------------------ #

    '''
    # Encode the categorical data of column 0
    labelEncoder_X = LabelEncoder()

    # Use the OneHotEncoder to separate the categorical data into multiple columns
    oneHotEncoder_X = OneHotEncoder(categorical_features=[3])

    # Encode the categorical data of column 3 to numerical data
    X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

    # Convert the categorical data into multiple columns
    X = oneHotEncoder_X.fit_transform(X).toarray()
    print("Matrix of features:\n", X.astype(int))
    print("Dependent variable:\n", y)

    # print the matrix of features as floats with fixed decimal places
    #print("Matrix of features:\n %0.2f"% (np.vectorize(np.float32(X[0]))))
    #print("Dependent variable:\n %0.2f"% (y[0]))

    # avoid the dummy variable trap, you do not take the column of index 0
    X_no_column_zero = X[:, 1:]

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    # you split the data into a training set and testing set but only if you have at least two columns with many rows of observations.
    # test_size = 0.2 indicates that 20% of data will go to TEST and 80% will go to TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X_no_column_zero, y, test_size = 0.2, random_state=0)
    print("Matrix of features training set:\n", X_train.astype(int), "\nMatrix of features test set:\n", X_test.astype(int),
          "\nDependent variable training set:\n", y_train, "\nDependent variable test set:\n", y_test)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    #Feature scaling does not apply in Linear Regression due to the libraries that we are going to use.
    # sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    # X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    # X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test, X_no_column_zero, y'''


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

def visualize_results(X,y, lin_reg, lin_reg_two, poly_reg):

    # visualize the Linear Regression Results
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title("Truth or Bluff? (Linear Regression)")
    plt.xlabel("Position Label")
    plt.ylabel("Salary")
    plt.show()

    # visualize the Polynomial Regression Results

    #make a smooth graph
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


def build_optimal_linear_regression_model(X_matrix_of_features_no_column_zero, y):
    '''Building the optimal model using backward elimination.'''

    # The line of code from below helps you to create a column of fifty 1s at the end of the matrix of features but we want
    # this new column at the beginning.
    # X = np.append(arr = X_train, values = np.ones((50,1)).astype(int), axis = 1)

    # We create an array of fifty 1s and we add the matrix of features to this new array.
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X_matrix_of_features_no_column_zero, axis = 1)

    # The variable from below will contain the variables that are highly statistical positive for determining a
    # good prediction  y dependant variable.
    # Select all the variables from your matrix of features
    numVars = len(X[0])

    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > 0.05:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X = np.delete(X, j, 1)
        print(regressor_OLS.summary())


if __name__ == "__main__":
    main()