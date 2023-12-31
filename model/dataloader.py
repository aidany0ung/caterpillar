import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumHeavyAtoms
from scipy.sparse.csgraph import laplacian
import scipy.linalg as la
import pandas as pd
from collections import defaultdict
import numpy as np

class DataLoader():
    def __init__(self, filename,fiftyfifty=False, modelType='gnn', addChemFeatures=False, flatten = False, pad = True, twoSTD = True):
        self.filename = filename
        self.fiftyfifty = fiftyfifty
        self.modelType = modelType
        self.addChemFeatures = addChemFeatures
        self.flatten = flatten
        self.pad = pad
        self.twoSTD = twoSTD
        self.dataset = self.load_data()
        


    def load_data(self):
        # load in data from dataset path as df
        print(os.getcwd())
        df = pd.read_csv(self.filename)
        # Check thatt the columns are correct
        if 'smile' not in df.columns and 'p_np' not in df.columns:
            raise ValueError('Columns are not correct')
        # Convert the SMILES strings into RDKit molecules
        df['mol'] = df['smile'].apply(Chem.MolFromSmiles)
        # Drop NA molecules
        df = df.dropna(subset=['mol'])
        df['adj'] = df['mol'].apply(Chem.rdmolops.GetAdjacencyMatrix)            
        return df
    
    def getData(self):
        if self.modelType == 'smile':
            self.checkfiftyfifty()
            #print(self.dataset.shape)
            X = np.array(self.dataset['smile'])
            y = np.array(self.dataset['p_np'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
            print(X_train)
            return X_train, y_train, X_test, y_test
        if self.modelType =='baseline':
            self.checkfiftyfifty()
            X = np.array(self.dataset['ExactMass'])
            y = np.array(self.dataset['p_np'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
            return X_train, y_train, X_test, y_test
        if self.modelType == 'gnn':
            self.checkfiftyfifty()
            if self.twoSTD:
                self.dataset['adj_len'] = self.dataset['adj'].apply(lambda x: len(x))
                self.dataset = self.dataset[self.dataset['adj_len'] < 45.2]
                self.dataset = self.dataset[self.dataset['adj_len'] > 2.9]
            # Convert the adjacency matrices into a list of adjacency matrices and a list of p_np values
            # Pad the adjacency matrices so that they are all the same size as the largest one
            # Get the longest graph
            longest_graph = max([len(x) for x in self.dataset['adj'].tolist()])
            # Pad the adjacency matrices
            if self.pad:
                self.dataset['adj'] = self.dataset['adj'].apply(lambda x: np.pad(x, ((0,longest_graph-len(x)),(0,longest_graph-len(x))), 'constant'))
            # If self.flatten is true, flatten the adjacency matrices
            if self.flatten:
                self.dataset['adj'] = self.dataset['adj'].apply(lambda x: x.flatten())
            x = self.dataset['adj'].tolist()
            y = self.dataset['p_np'].tolist()
            # Split the data into train and test
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            # Get the longest graph
            longest_graph = max([len(x) for x in x_train])
            # Reset self.dataset to original
            self.dataset = self.load_data()
            return x_train, y_train, x_test, y_test, longest_graph
        if self.modelType == 'spectral':
            default = ['mol','adj','p_np']
            if self.addChemFeatures:
                default.extend(['num_atoms', 'num_bonds', 'num_carbon_atoms', 'num_oxygen_atoms', 'num_nitrogen_atoms', 'num_sulfur_atoms', 'num_fluorine_atoms', 'num_chlorine_atoms', 'num_bromine_atoms', 'num_iodine_atoms', 'num_phosphorus_atoms', "ExactMass", "HBondDonorCount","HBondAcceptorCount"])
                # Filter out all in self.dataset where ExactMass is -1
                self.dataset = self.dataset[self.dataset['ExactMass'] != -1]
            # get only the columns we want
            self.dataset = self.dataset[default]
            self.checkfiftyfifty()
            print(self.dataset['p_np'].value_counts())
            x = []
            y = []
            for index in range(len(self.dataset)):
                print(index)
                # Calculate the laplacian matrix
                lap = laplacian(self.dataset['adj'].iloc[index])
                # Calculate the eigenvectors
                eigenvalues, eigenvectors = la.eigh(lap)
                # Find max and min eigenvalues
                max_eig = max(eigenvalues)
                min_eig = min(eigenvalues)
                #Find the indices of the max and min eigenvalues
                max_eig_index = np.where(eigenvalues == max_eig)
                min_eig_index = np.where(eigenvalues == min_eig)
                # Get the eigenvectors corresponding to the max and min eigenvalues
                max_eig_vec = eigenvectors[:,max_eig_index]
                min_eig_vec = eigenvectors[:,min_eig_index]
                eigs = [0] * 90
                if len(max_eig_vec) > 45:
                    continue
                #print(min_eig_vec.shape)
                #print(max_eig_vec.shape)
                if max_eig_vec.shape[2] > 1:
                    for i in range(len(max_eig_vec)):
                        eigs[i] = int(max_eig_vec[i][0][0])
                else:
                    for i in range(len(max_eig_vec)):
                        eigs[i] = int(max_eig_vec[i])
                if min_eig_vec.shape[2] > 1:
                    for i in range(len(min_eig_vec)):
                        eigs[i+45] = int(min_eig_vec[i][0][0])
                else:
                    for i in range(len(min_eig_vec)):
                        eigs[i+45] = int(min_eig_vec[i])
                eigs.append(max_eig)
                eigs.append(min_eig)
                # Append the chemical features
                if self.addChemFeatures:
                    eigs.append(self.dataset['num_atoms'].iloc[index])
                    eigs.append(self.dataset['num_bonds'].iloc[index])
                    eigs.append(self.dataset['num_carbon_atoms'].iloc[index])
                    eigs.append(self.dataset['num_oxygen_atoms'].iloc[index])
                    eigs.append(self.dataset['num_nitrogen_atoms'].iloc[index])
                    eigs.append(self.dataset['num_sulfur_atoms'].iloc[index])
                    eigs.append(self.dataset['num_fluorine_atoms'].iloc[index])
                    eigs.append(self.dataset['num_chlorine_atoms'].iloc[index])
                    eigs.append(self.dataset['num_bromine_atoms'].iloc[index])
                    eigs.append(self.dataset['num_iodine_atoms'].iloc[index])
                    eigs.append(self.dataset['num_phosphorus_atoms'].iloc[index])
                    eigs.append(self.dataset['ExactMass'].iloc[index])
                    eigs.append(self.dataset['HBondDonorCount'].iloc[index])
                    eigs.append(self.dataset['HBondAcceptorCount'].iloc[index])
                # Append the list to x
                x.append(eigs)
                print(index,self.dataset['p_np'].iloc[index])
                y.append(self.dataset['p_np'].iloc[index])
            print(self.dataset['p_np'].value_counts())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            self.dataset = self.load_data()
            # Plot a histogram of the length of eigenvectors using matplotlib
            return x_train, y_train, x_test, y_test
            

    def checkfiftyfifty(self):
        if self.fiftyfifty:
            df = self.dataset
            df_1 = df[df['p_np'] == 1]
            df_0 = df[df['p_np'] == 0]
            if len(df_1) > len(df_0):
                l = len(df_0)
            else:
                l = len(df_1)
            df_new = pd.concat([df_1.sample(l, replace=False), df_0.sample(l, replace=False)])
            self.dataset = df_new

#dl = DataLoader('data/test.csv', modelType='smile', fiftyfifty=True)
#x_train, y_train, x_test, y_test = dl.getData()
#print(y_train.count(1),y_train.count(0))
'''
dl = DataLoader('data/test.csv', modelType='gnn', pad=False, twoSTD=False)
x_train, y_train, x_test, y_test, longest_graph = dl.getData()
x_train.extend(x_test)
y_train.extend(y_test)

hist = []
for i in x_train:
    hist.append(i.shape[0])

hist = np.asarray(hist)
print(np.std(hist))
print(np.mean(hist))

ctr = 0
cant = 0
for i in range(len(hist)):
    item = hist[i]
    if item > 45:
        ctr += 1
        if y_train[i] == 0:
            cant += 1
print(ctr/len(hist))
print(max(hist))
'''