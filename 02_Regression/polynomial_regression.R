# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/26/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\02_Regression\\Position_Salaries.csv")


# ---------------------------------------------- Get columns from the dataset ------------------------------------ #
dataset = dataset[2:3]


# ----------------------- Fiting Linear_Regression to the dataset ------------------ #

lin_reg = lm(formula = Salary ~ ., data = dataset)

# ----------------------- Fiting Polynomial_Regression to the dataset ------------------ #
# Polynomial regression is a multiple linear regression in which the independent variables are the polynomial features
#  of one independendent variables. Polynomial features are the independent variables equal to squared or cubic values.

# Polynomial features

dataset$LevelTwo = dataset$Level^2
dataset$LevelThree = dataset$Level^3

# if you add a new polynomial feature you need to re-run the polynomial regressor
dataset$LevelFour = dataset$Level^4

poly_reg = lm(formula = Salary ~ ., data = dataset)

# -------------------- visualizing the Linear regression results ---------------------------- #

install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = 'blue') + ggtitle('Truth or Bluff? (Linear Regression) ') + xlab('Level') + ylab('Salary')

# -------------------- visualizing the Polynomial regression results ---------------------------- #

ggplot() + geom_point(aes(x = dataset$Level , y = dataset$Salary), color='red') + geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), color = 'blue') + ggtitle('Truth or Bluff? (Polynomial Regression) ') + xlab('Level') + ylab('Salary')

# --------------------- predicting a new result with Linear Regression ------------------------- #

# you need to create a new level in the dataset, so that is why we used data.frame(Level = 6.5)
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# --------------------- predicting a new result with Polynomial Regression ------------------------- #

y_pred_poly = predict(poly_reg, data.frame(Level = 6.5, LevelTwo = 6.5 ^ 2, LevelThree = 6.5 ^ 3, LevelFour = 6.5 ^ 4))