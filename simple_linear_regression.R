# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 4/13/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 2 - Regression\\Section 3 -------------------- Part 2 - Regression --------------------\\Simple_Linear_Regression\\Salary_Data.csv")

# ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

#install the caTools library for doing the split of training and testing set
install.packages('caTools') #hit yes and yes when you see the library prompt

#load the library
library(caTools)

# set the seed for random data
set.seed(123)

# split is a variable that divided the data into training and testing sets, SplitRatio = 0.8 indicates that 80% of data will go to training
split = sample.split(dataset$Salary, SplitRatio = 2/3)

# split the data into training and testing sets
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ------------------------ Feature scaling ------------------------------------ #

#training_set[,2:3] = scale(training_set[,2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])
#View(training_set)
#View(test_set)

# Fiting Simple_Linear_Regression
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# get more information about your model with the below formula
# summary(regressor)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing the training set results

install.packages('ggplot2')
library(ggplot2)

# training set
ggplot() + geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour = 'red') + geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set )), colour = 'blue') + ggtitle('Salary vs Experience (Training set)') + xlab('Years of experience') + ylab('Salary')

# test set
ggplot() + geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), colour = 'red') + geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set )), colour = 'blue') + ggtitle('Salary vs Experience (Test set)') + xlab('Years of experience') + ylab('Salary')



