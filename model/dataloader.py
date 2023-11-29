import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, fiftyfifty=False, modelType='gnn', addChemFeatures=False):
        self.fiftyfifty = fiftyfifty
        self.modelType = modelType
        self.addChemFeatures = addChemFeatures
        self.dataset = self.load_data()


    def load_data(self):
        # load in data from dataset path as df
        print(os.getcwd())
        df = pd.read_csv('../data/test.csv')
        return df
    
    def getData(modelType):
        if modelType == 'gnn':
            return self.getGNNData()
        elif modelType == 'spectral':
            return self.getSpectralData()
        elif modelType == 'autoencoder':
            return self.getGNNData()
        

    # Return x_train, y_train, x_test, y_test. the x's are a list of adjacency matrices, the y's are p_np
    def getGNNData(self):
        if self.fiftyfifty:
            df = self.dataset
            # Get a list of all rows with p_np == 0
            p_np_0 = df[df.p_np == 0].index.tolist()
            # Get a list of all rows with p_np == 1
            p_np_1 = df[df.p_np == 1].index.tolist()
            # Randomly sample from p_np_1 to get a list of indices
            p_np_1_sample = np.random.choice(p_np_1, size=len(p_np_0), replace=False)
            print(max(p_np_0), max(p_np_1), max(p_np_1_sample))
            p_np_0.extend(p_np_1_sample)
            # Filter df by the indices
            df = df.iloc[p_np_0]
            # Reset the index
            df = df.reset_index(drop=True)
            self.dataset = df
        # Get the adjacency matrices
        x = self.dataset['adjacency_matrix'].tolist()
        # Get the p_np values
        y = self.dataset['p_np'].tolist()
        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # Get the longest graph
        longest_graph = max([len(x) for x in x_train])
        # Reset self.dataset to original
        self.dataset = self.load_data()
        return x_train, y_train, x_test, y_test, longest_graph

    # Return x_train, y_train, x_test, y_test. The x's are the eig_small, eig_large, and eig_small_1, eig_small_2 columns, the y's are p_np
    def getSpectralData(self):
        # Get the adjacency matrices
        default = ['eig_small', 'eig_large', 'eig_small_1', 'eig_small_2']
        if self.addChemFeatures:
            default.extend(['n_atoms', 'n_bonds', 'n_carbon', 'n_oxygen', 'n_nitrogen', 'n_sulfur', 'n_fluorine', 'n_chlorine', 'n_bromine', 'n_iodine', 'n_phosphorus', "ExactMass", "HBondDonorCount","HBondAcceptorCount"])
        # Filter out all in self.dataset where ExactMass is -1
        self.dataset = self.dataset[self.dataset['ExactMass'] != -1]
        if self.fiftyfifty:
            df = self.dataset
            # Get a list of all rows with p_np == 0
            p_np_0 = df[df.p_np == 0].index.tolist()
            # Get a list of all rows with p_np == 1
            p_np_1 = df[df.p_np == 1].index.tolist()
            # Randomly sample from p_np_1 to get a list of indices
            p_np_1_sample = np.random.choice(p_np_1, size=len(p_np_0), replace=False)
            print(max(p_np_0), max(p_np_1), max(p_np_1_sample))
            p_np_0.extend(p_np_1_sample)
            # Filter df by the indices
            df = df.iloc[p_np_0]
            # Reset the index
            df = df.reset_index(drop=True)
            self.dataset = df
        x = self.dataset[default].values.tolist()
        # Flatten the list of lists
        x = [item for sublist in x for item in sublist]
        # Get the p_np values
        y = self.dataset['p_np'].tolist()
        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        self.dataset = self.load_data()
        return x_train, y_train, x_test, y_test
    
