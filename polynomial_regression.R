# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/26/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv")


# ---------------------------------------------- Get columns from the dataset ------------------------------------ #
dataset = dataset[2:3]

# ---------------------------------------------- Encode the categorical data ------------------------------------ #
'''
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
'''

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