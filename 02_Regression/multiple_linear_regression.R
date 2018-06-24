# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/19/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\50_Startups.csv")

# ---------------------------------------------- Encode the categorical data ------------------------------------ #

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))


# ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

#install the caTools library for doing the split of training and testing set
install.packages('caTools') #hit yes and yes when you see the library prompt

#load the library
library(caTools)

# set the seed for random data
set.seed(123)

# split is a variable that divided the data into training and testing sets, SplitRatio = 0.8 indicates that 80% of data will go to training
split = sample.split(dataset$Profit, SplitRatio = 0.8)

# split the data into training and testing sets
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ------------------------ Feature scaling ------------------------------------ #

#training_set[,2:3] = scale(training_set[,2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])
#View(training_set)
#View(test_set)

# Fiting Multiple_Linear_Regression to training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.State + State, data = training_set)
# below is a trick to set the relationship between the independent variable
# and dependent variables. The '.' indicates to use all the independent variables.
regressor = lm(formula = Profit ~ ., data = training_set)

# get more information about your model with the below formula
summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Building the optimal data using Backward Elimination

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regressor)


# remove another variable
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regressor)


backwardElimination <- function(x, sl) {
    numVars = length(x)
    for (i in c(1:numVars)){
      regressor = lm(formula = Profit ~ ., data = x)
      maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
      if (maxVar > sl){
        j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
        x = x[, -j]
      }
      numVars = numVars - 1
    }
    return(summary(regressor))
  }

  SL = 0.05
  dataset = dataset[, c(1,2,3,4,5)]
  backwardElimination(training_set, SL)