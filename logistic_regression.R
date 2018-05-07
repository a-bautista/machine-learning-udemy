# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 5/7/2018


# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 3 - Classification\\Section 14 - Logistic Regression\\Social_Network_Ads.csv")

dataset = dataset[,3:5]#Age, Salary and Purchased columns

library(caTools) # necessary library for doing the split
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ------------------------ Feature scaling ------------------------------------ #

training_set[,1:2] = scale(training_set[,1:2])
test_set[, 1:2] = scale(test_set[, 1:2])
View(training_set)
View(test_set)

# ---------------- Fit logistic regression to the training set ------------------- #

classifier = glm(formula = Purchased ~ .,family = binomial, data = training_set)

# ----------------------- Predicting the Test set results ----------------------- #

prob_pred = predict(classifier, type = 'response', newdata = test_set[-3]) #remove the last column which is the dependent variable
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# ----------------------- Making the confusion matrix -------------------------- #

cm = table(test_set[, 3], y_pred) # 57 and 26 are the correct predictions


# ----------------------- Visualizing the Training set results -------------------- #

install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[,1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) -1, max(set[,2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type='response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim =  range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#This lines must be executed at the very end, so you can see the different points
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# ----------------------- Visualizing the Test set results -------------------- #

install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) -1, max(set[,1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) -1, max(set[,2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type='response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim =  range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#This lines must be executed at the very end, so you can see the different points
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
