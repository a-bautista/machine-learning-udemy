# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 6/5/2018

# ---------------------------------------------- Retrieve the dataset ------------------------------------ #

install.packages("arules")
library("arules")
dataset = read.csv("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Market_Basket_Optimisation.csv", header = FALSE)

# ------------------------------------- Build the sparse matrix -------------------------------------- #

dataset = read.transactions("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Market_Basket_Optimisation.csv", sep=',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN = 100)

#minlen is the minimum of items purchased together

rules = eclat(data = dataset, parameter = list(support = ((4*7)/7500), minlen= 2))

# ------------------------------------- Visualizing results -------------------------------------- #
inspect(sort(rules,by = 'support')[1:10])
