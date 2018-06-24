# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def main():
    data_processing()


def data_processing():
    dataset = pd.read_csv(
        "C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Churn_Modelling.csv")

    # take all the columns except the last one for your matrix of features = independent variables
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # ------------------------ Encode the text data (categorical data) to numerical data ------------------------------------ #

    # Encode the categorical columns 1 and 2 from the X dataset
    labelEncoder_X_country = LabelEncoder()
    labelEncoder_X_gender = LabelEncoder()

    # Encode the categorical columns 1 and 2 from the X dataset
    X[:, 1] = labelEncoder_X_country.fit_transform(X[:, 1])
    X[:, 2] = labelEncoder_X_gender.fit_transform(X[:, 2])

    # Use the OneHotEncoder to separate the categorical data into multiple columns because your categorical data is defined
    # in various countries : France, Germany and Spain
    oneHotEncoder_X_country = OneHotEncoder(categorical_features=[1])

    # Use the OneHotEncoder to separate the categorical data into multiple columns because your categorical data is defined
    # in various Female and Male == NOT NECESSARY BECAUSE THIS IS A BINARY RESULT, WE APPLY THE ONE HOT ENCODE WHEN WE HAVE
    # MORE THAN BINARY CATEGORIES
    # oneHotEncoder_X_gender = OneHotEncoder(categorical_features=[2])

    # Convert the categorical data into multiple columns
    X = oneHotEncoder_X_country.fit_transform(X).toarray()

    # Avoid the dummy variable trap - that is, select only 2 dummy variables from the country instead of all the variables
    X = X[:, 1:]

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    X_test = sc_X.transform(X_test)

    # ------------------------------------- Fitting Classifier ---------------------------------------------- #

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # ----------------------------------- Predicting the Test Set Results -------------------------------------------- #

    y_pred = classifier.predict(X_test)
    print("Results test: \n", y_test)
    print("Results prediction: \n", y_pred)

    # ----------------------------------- Making the confusion matrix ----------------------------------------------- #

    # 7 incorrect predictions
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n",cm)

    

if __name__ == "__main__":
    main()
