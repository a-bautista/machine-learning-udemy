# !Python 3.5.2
# Author: Alejandro Bautista Ramos

# Importing the libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from math import sqrt, log

def main():

    # ---------------------------------------- Retrieve the dataset ------------------------------------------------ #

    dataset = pd.read_csv(
        "C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 6 - Reinforcement Learning\\Section 32 - Upper Confidence Bound (UCB)\\Ads_CTR_Optimisation.csv")
    random_selection_algorithm(dataset)
    ucb_algorithm(dataset)
    print("End of UCB")


def ucb_algorithm(dataset):
    N = 10000  # number of rounds, users will select one ad. You need the same number of values as in your dataset.
    d = 10 #number of ads
    ads_selected = []
    numbers_of_selections = [0]  * d
    sums_of_rewards = [0] * d
    total_reward = 0
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = sqrt(3 / 2 * log(n + 1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                # you set the value of each ad for the first round
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] = numbers_of_selections[ad] + 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward

    plt.hist(ads_selected)
    plt.title("Histogram of ads selections")
    plt.xlabel("Ads")
    plt.ylabel("Number of times each ad was selected")
    plt.show()


def random_selection_algorithm(dataset):
    N = 10000
    d = 10
    ads_selected = []
    total_reward = 0
    for n in range(0, N):
        ad = random.randrange(d)
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        total_reward = total_reward + reward

    plt.hist(ads_selected)
    plt.title("Histogram of ads selections")
    plt.xlabel("Ads")
    plt.ylabel("Number of times each ad was selected")
    plt.show()

if __name__ == "__main__":
    main()
