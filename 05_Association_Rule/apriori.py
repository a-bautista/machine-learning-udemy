# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from .apyori import apriori
#from .modules import apyori
from modules import apyori

def main():

    # ---------------------------------------- Retrieve the dataset ------------------------------------------------ #

    output = open("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\results.txt","w+")
    dataset = pd.read_csv(
        "C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Market_Basket_Optimisation.csv", header=None)

    transactions = []
    for i in range(0, len(dataset)):
        transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

    # ---------------------------------------- Training Apriori on the dataset --------------------------------------- #

    rules = apyori.apriori(transactions, min_support = (3*7)/7500, min_confidence = 0.2, min_lift = 3, min_length=2)

    # ---------------------------------------- Visualizing the results --------------------------------------- #
    results = list(rules)
    for i in range(0, len(results)):
        output.write(str(results[i]))
        output.write("\n")

    output.close()

if __name__ == "__main__":
    main()