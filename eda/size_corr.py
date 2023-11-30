import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader

dl = DataLoader("data/test.csv", modelType="gnn", pad=False)
output = dl.getData()
x_train = output[0]
y_train = output[1]
x_test = output[2]
y_test = output[3]
# Remerge the training and testing data
x_train.extend(x_test)
y_train.extend(y_test)

# Create a boxplot that shows whether the length of the adjacency matrix is correlated with the size of the molecule
# Create a list of the lengths of the adjacency matrices
adj_lengths = []
for i in range(len(x_train)):
    adj_lengths.append(len(x_train[i]))

# Create a list of adjacency matrix lengths where y is 1 and another where y is 0
adj_lengths_1 = []
adj_lengths_0 = []
for i in range(len(adj_lengths)):
    if y_train[i] == 1:
        adj_lengths_1.append(adj_lengths[i])
    else:
        adj_lengths_0.append(adj_lengths[i])

# Create a plot with both boxplots
plt.boxplot([adj_lengths_1, adj_lengths_0])
plt.title("Boxplot of Adjacency Matrix Lengths")
plt.xlabel("p_np")
plt.ylabel("Adjacency Matrix Length")
plt.xticks([1,2], ["p_np = 1", "p_np = 0"])
plt.show()