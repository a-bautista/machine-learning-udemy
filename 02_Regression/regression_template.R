# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/26/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\02_Regression\\Position_Salaries.csv")

# ---------------------------------------------- Encode the categorical data ------------------------------------ #



# ----------------------- Fiting Polynomial_Regression to the dataset ------------------ #

# Polynomial regression is a multiple linear regression in which the independent variables are the polynomial features
#  of one independendent variables. Polynomial features are the independent variables equal to squared or cubic values.

# Polynomial features
dataset$LevelTwo = dataset$Level^2
dataset$LevelThree = dataset$Level^3

# if you add a new polynomial feature you need to re-run the polynomial regressor
dataset$LevelFour = dataset$Level^4

poly_reg = lm(formula = Salary ~ ., data = dataset)

# --------------------- predicting a new result with Polynomial Regression ------------------------- #

y_pred_poly = predict(poly_reg, data.frame(Level = 6.5, LevelTwo = 6.5 ^ 2, LevelThree = 6.5 ^ 3, LevelFour = 6.5 ^ 4))


# -------------------- visualizing the Polynomial regression results ---------------------------- #
install.packages('ggplot2')
library(ggplot2)

ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), color = 'blue') + ggtitle('Truth or Bluff? (Polynomial Regression) ') + xlab('Level') + ylab('Salary')

# ------------------- visualize a better graph in the Polynomial regression results ------------------------- #

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid))), color = 'blue') + ggtitle('Truth or Bluff? (Polynomial Regression) ') + xlab('Level') + ylab('Salary')