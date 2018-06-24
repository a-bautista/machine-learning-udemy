# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 3/20/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\01_Data_Preprocessing\\Data.csv")

# ---------------------------------------------- Handle the missing values ------------------------------------ #

# replace the missing values in column Age by computing the mean value
dataset$Age = ifelse(is.na(dataset$Age),ave(dataset$Age,FUN = function(x) mean(x, na.rm = TRUE)),dataset$Age)

# replace the missing values in column Salary by computing the mean value
dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Salary,FUN = function(x) mean(x, na.rm = TRUE)),dataset$Salary)

# ---------------------------------------------- Encode the categorical data ------------------------------------ #

dataset$Country = factor(dataset$Country, levels = c('France', 'Spain', 'Germany'), labels = c(1,2,3))
dataset$Purchase = factor(dataset$Purchase, levels = c('No', 'Yes'), labels = c(0,1))

# ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

#install the caTools library for doing the split of training and testing set
install.packages('caTools') #hit yes and yes when you see the library prompt

#load the library
library(caTools)

# set the seed for random data
set.seed(123)

# split is a variable that divided the data into training and testing sets, SplitRatio = 0.8 indicates that 80% of data will go to training
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

# split the data into training and testing sets
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ------------------------ Feature scaling ------------------------------------ #

training_set[,2:3] = scale(training_set[,2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
View(training_set)
View(test_set)







print(dataset)

