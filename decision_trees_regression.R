# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/26/2018


# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv")
dataset = dataset[2:3]

# ----------------------- Fiting decision trees regression to the dataset ------------------------------ #

install.packages('rpart')
library(rpart)

# in our regressor we are using a support vector machine instead of a support vector regressor because our model is non-linear,
# we have linear, polynomial or gaussian and we are using polynomial
# eps-regression already contains the gaussian kernel

regressor = rpart(formula = Salary ~ ., data=dataset, control = rpart.control(minsplit = 1))

# --------------------- predicting a new result with decision tree  ------------------------- #

y_pred = predict(regressor, data.frame(Level = 6.5))


# ------------------- visualize a better graph of the decision tree regression results ------------------------- #
install.packages('ggplot2')
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

# red flag in this graph:
ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = 'blue') + ggtitle('Truth or Bluff? (Decision Tree Regression) ') + xlab('Level') + ylab('Salary')
