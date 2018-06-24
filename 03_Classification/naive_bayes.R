# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 5/11/2018


# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 3 - Classification\\Section 18 - Naive Bayes\\Social_Network_Ads.csv")

dataset = dataset[,3:5]#Age, Salary and Purchased columns


# --------------------------------------- encoding the target feature ------------------------------------ #

dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))

library(caTools) # necessary library for doing the split
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ------------------------ Feature scaling ------------------------------------ #

training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])
View(training_set)
View(test_set)

# ---------------- Fit classifier to the training set ------------------- #

install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-3], y = training_set$Purchased)


# ----------------------- Predicting the Test set results ----------------------- #

y_pred = predict(classifier, newdata = test_set[-3]) #remove the last column which is the dependent variable

# ----------------------- Making the confusion matrix -------------------------- #

cm = table(test_set[, 3], y_pred) # 10 incorrect predictions

# ----------------------- Visualizing the Training set results -------------------- #

install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[,1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) -1, max(set[,2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Naive Bayes (Training set)',
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
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Naive Bayes (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim =  range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#This lines must be executed at the very end, so you can see the different points
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))




