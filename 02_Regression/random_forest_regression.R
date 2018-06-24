# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 5/2/2018


# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Position_Salaries.csv")
dataset = dataset[2:3]

# ----------------------- Fitting random forest regression to the dataset ------------------------------ #

install.packages('randomForest')
library(randomForest)
set.seed(1234)

# in our regressor we are using a support vector machine instead of a support vector regressor because our model is non-linear,
# we have linear, polynomial or gaussian and we are using polynomial
# eps-regression already contains the gaussian kernel

#dataset[1] will give you a dataframe and dataset$Salary will give you a vector
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree=300)

# --------------------- predicting a new result with random forest ------------------------- #

y_pred = predict(regressor, data.frame(Level = 6.5))


# ------------------- visualize a better graph of the random forest regression results ------------------------- #

install.packages('ggplot2')
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

# red flag in this graph:
ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = 'blue') + ggtitle('Truth or Bluff? (Random Forest Regression) ') + xlab('Level') + ylab('Salary')

