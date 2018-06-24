from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def main():
    X_train, X_test, y_train, y_test = data_processing()
    neural_network(X_train, X_test, y_train, y_test)

def data_processing():

    # ---------------------------------------- Retrieve the dataset --------------------------------------------- #
    dataset = pd.read_csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Churn_Modelling.csv")

    # take all the columns except the last one for your matrix of features = independent variables
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[: ,13].values

    # ------------------------ Encode the text data (categorical data) to numerical data ------------------------------------ #

    # Encode the categorical columns 1 and 2 from the X dataset
    labelEncoder_X_country = LabelEncoder()
    labelEncoder_X_gender  = LabelEncoder()

    # Encode the categorical columns 1 and 2 from the X dataset
    X[:, 1] = labelEncoder_X_country.fit_transform(X[:, 1])
    X[:, 2] = labelEncoder_X_gender.fit_transform(X[:, 2])

    # Use the OneHotEncoder to separate the categorical data into multiple columns because your categorical data is defined
    # in various countries : France, Germany and Spain
    oneHotEncoder_X_country = OneHotEncoder(categorical_features=[1])

    # Use the OneHotEncoder to separate the categorical data into multiple columns because your categorical data is defined
    # in various Female and Male == NOT NECESSARY BECAUSE THIS IS A BINARY RESULT, WE APPLY THE ONE HOT ENCODE WHEN WE HAVE
    # MORE THAN BINARY CATEGORIES
    #oneHotEncoder_X_gender = OneHotEncoder(categorical_features=[2])

    # Convert the categorical data into multiple columns
    X = oneHotEncoder_X_country.fit_transform(X).toarray()

    # Avoid the dummy variable trap - that is, select only 2 dummy variables from the country instead of all the variables
    X = X[:,1:]

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #

    sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    X_test  = sc_X.transform(X_test)


    return X_train, X_test, y_train, y_test

def neural_network(X_train, X_test, y_train, y_test):

    # initialize the neural network
    classifier = Sequential()

    # adding the input layer and the first hidden layer
    # Dense(output_dim = #average number (independent variables + dependent variables, input_dim = independent variables)
    ''' Original Independent variables 
    1 - France  = 0
    2 - Germany = 1
    3 - Spain   = 2
    4 - Female  = 1
    5 - Male    = 0
    6 - Age
    7 - Tenure 
    8 - Balance
    9 - NumOfProducts
    10 - HasCreditCard
    11 - Active Member
    12 - Estimated Salary   
    
    Because we avoid the dummy variable trap we have the following independent variables
    1 - Germany = 1
    2 - Spain   = 2
    3 - Female  = 1
    4 - Male    = 0
    5 - Age
    6 - Tenure 
    7 - Balance
    8 - NumOfProducts
    9 - HasCreditCard
    10 - Active Member
    11 - Estimated Salary
    
    Output variable layer is only 1 or 0 for Exited but that is considered to be one variable
    1- Exited = 0 or 1
    
    Thus, the Dense number is equal to (11 independent variables + 1 dependent variable)/2 = 6 nodes in the hidden layer
    init starts the numbers to small values
    relu is the rectifier function
    '''

    classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=11))

    # adding the second hidden layer
    # we remove the input_dim layer because we already defined the initial input values in the first layer

    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

    # adding the output layer
    # because the output value only throws if the customer stays or leaves (0 or 1), we only type in 1 output_dim value
    # and we need the probability of how likely is the customer going to be or stay in the bank, therefore, we use the
    # sigmoid function.
    # If we have an output variable with more than 2 categories then we type output_dim = 3 and the activation function
    # is softmax which is the sigmoid function but applied to more than 2 categories
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


    # compile the Artificial Neural Network

    # loss - if the output variable has a binary outcome then you type binary_crossentropy
    # loss - if the output variable has a more than 2 outcomes the you use the categorial_cross_entropy
    # metrics indicates to improve little by little the accuracy
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    # fitting the Artificial Neural Network to the training set
    classifier.fit(X_train,y_train,batch_size=10, epochs=100)

    # predict the values
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(y_pred)

    # prepare the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # X_test contains 2000 observations based on the 20% of the 10,000 observations from the dataset
    # accuracy of the classification is number of correct predictions of customers who leave the bank + number
    # of correct predictions of customers who do not leave the bank / number of test observations
    print(cm)


if __name__ == "__main__":
    main()