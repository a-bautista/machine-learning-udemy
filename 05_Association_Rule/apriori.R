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

# ------------------------------------- Training Apriori -------------------------------------- #
# support and confidence is not a fixed parameter
# we buy a product 3 times a day in a 7 days week / total number of transactions
# we just trype random 0.8
# check video 159

rules = apriori(data = dataset, parameter = list(support = ((4*7)/7500) , confidence = 0.4 ))

# ------------------------------------- Visualizing results -------------------------------------- #
inspect(sort(rules,by = 'lift')[1:10])


