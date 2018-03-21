# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split


def main():

    # ------------------------ Retrieve the dataset and replace the missing values ------------------------------------ #
    # Get the data in a variable
    dataset = pd.read_csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data.csv")

    # Upper case letters represent Features
    # take all the columns except the last one for your matrix of features
    X = dataset.iloc[:, :-1].values

    # lower case letters represent Labels
    # define your dependent variable vector
    y = dataset.iloc[:,3].values

    ''' In Machine Learning you have your matrix of features and your dependent variables, that is, the variable that you want 
        to predict. It is common that your matrix of features may have missing data, so in order to avoid this problem you should 
        either compute the mean value of other records and insert these mean values in the blank cells or use outliers for those 
        values such as -99999. '''

    # defining outliers where X should be a pandas df
    # X.fillna(-99999, inplace=True)

    # define this variable to take care of the missing values
    imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)

    # take the values of your dataset from columns 1 and 2
    imputer = imputer.fit(X[:, 1:3])

    # insert the new computed values in your dataset
    X[:, 1:3] = imputer.transform(X[:, 1:3])

    # display results
    print(X)

    # ------------------------ Encode the text data (categorical data) to numerical data ------------------------------------ #

    # Encode the categorical data of column 0
    labelEncoder_X = LabelEncoder()

    # Use the OneHotEncoder to separate the categorical data into multiple columns because your categorical data is defined
    # in 3 categories: France, Germany and Spain
    oneHotEncoder_X = OneHotEncoder(categorical_features=[0])

    # Encode the categorical data of column 0 to numerical data
    X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

    # Convert the categorical data into multiple columns
    X = oneHotEncoder_X.fit_transform(X).toarray()

    print(X)

    # Encode the categorical data of column 3, no need to use oneHotEncoder because we are not separating this column into
    # multiple ones
    labelEncoder_y = LabelEncoder()
    y = labelEncoder_y.fit_transform(y)

    print(y)

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    # you split the data into a training set and testing set
    # test_size = 0.2 indicates that 20% of data will go to TEST and 80% will go to TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print(X_train, X_test, y_train, y_test)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    X_test  = sc_X.transform(X_test)

    print(X_train)
    print(X_test)


if __name__ == "__main__":
    main()