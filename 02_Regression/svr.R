# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/26/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Position_Salaries.csv")
dataset = dataset[2:3]

# ----------------------- Fiting SVR to the dataset ------------------------------ #

install.packages('e1071')
library(e1071)

# in our regressor we are using a support vector machine instead of a support vector regressor because our model is non-linear,
# we have linear, polynomial or gaussian and we are using polynomial
# eps-regression already contains the gaussian kernel

regressor = svm(formula = Salary ~ ., data=dataset, type = 'eps-regression')

# --------------------- predicting a new result with SVR ------------------------- #

y_pred = predict(regressor, data.frame(Level = 6.5))

# -------------------- visualizing the Polynomial regression results ---------------------------- #
install.packages('ggplot2')
library(ggplot2)

ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), color = 'blue') + ggtitle('Truth or Bluff? (SVR Regression) ') + xlab('Level') + ylab('Salary')

# ------------------- visualize a better graph in the Polynomial regression results ------------------------- #

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid))), color = 'blue') + ggtitle('Truth or Bluff? (Polynomial Regression) ') + xlab('Level') + ylab('Salary')