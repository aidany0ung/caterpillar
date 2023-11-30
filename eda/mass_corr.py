import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader

dl = DataLoader("data/test.csv", modelType="spectral", addChemFeatures=True, pad=False)
output = dl.getData()
x_train = output[0]
y_train = output[1]
x_test = output[2]
y_test = output[3]
# Remerge the training and testing data
x_train.extend(x_test)
y_train.extend(y_test)

mass = []
for i in x_train:
    mass.append(i[-3])

#Split mass_list based on p_np
mass_1 = []
mass_0 = []
for i in range(len(mass)):
    if y_train[i] == 1:
        mass_1.append(mass[i])
    else:
        mass_0.append(mass[i])

# Create a plot with both boxplots
plt.boxplot([mass_1, mass_0])
plt.title("Boxplot of Mass")
plt.xlabel("p_np")
plt.ylabel("Mass")
plt.xticks([1,2], ["p_np = 1", "p_np = 0"])
plt.show()