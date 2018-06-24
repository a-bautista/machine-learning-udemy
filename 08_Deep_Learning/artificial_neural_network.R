# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 6/18/2018


# --------------------------------------- Importing the data ------------------------------------ #

dataset = read.csv("C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Churn_Modelling.csv")
View(dataset)
dataset = dataset[4:14]
View(dataset)

# --------------------------------------- encoding the categorical data ------------------------------------ #

dataset$Geography = as.numeric(factor(dataset$Geography,
                    levels = c('France','Spain','Germany'),
                    labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender,
                    levels = c('Female','Male'),
                    labels = c(1,2)))

# ------------------------------------- split intro training and test sets ------------------------------------ #

library(caTools) # necessary library for doing the split
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# ------------------------ Feature scaling ------------------------------------ #
# feature scaling is not necessary in this algorithm but if you do not scale it then your code my break
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])
View(training_set)
View(test_set)

# ---------------- Use an Artificial Neural Network package  ------------------- #
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1) #make an instance of h2o to run your ANN model instead of using your cpu or gpu
classifier = h2o.deeplearning(y='Exited',training_frame = as.h2o(training_set), activation = 'Rectifier', hidden = c(6,6), epochs = 100, train_samples_per_iteration = -2) #vector and neurons. 6 represents the number of independent variables = 11 / outcome results = 2 = 5.5 approx 6


# ----------------------- Predicting the Test set results ----------------------- #

prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11])) #remove the last column which is the dependent variable

#y_pred = ifelse(prob_pred > 0.5, 1, 0)

y_pred = (prob_pred > 0.5)

y_pred = as.vector(y_pred)
# ----------------------- Making the confusion matrix -------------------------- #

cm = table(test_set[, 11], y_pred)

h2o.shutdown() # disconnect from the server
