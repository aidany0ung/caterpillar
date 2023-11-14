from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumHeavyAtoms
from scipy.sparse.csgraph import laplacian
import pandas as pd
from collections import defaultdict
import numpy as np

def countAtomsOf(mol, atom):
    count = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() == atom:
            count += 1
    return count

def countCarbon(mol):
    return countAtomsOf(mol, 'C')

def countOxygen(mol):
    return countAtomsOf(mol, 'O')

def countNitrogen(mol):
    return countAtomsOf(mol, 'N')

def countSulfur(mol):
    return countAtomsOf(mol, 'S')

def countFluorine(mol):
    return countAtomsOf(mol, 'F')

def countChlorine(mol):
    return countAtomsOf(mol, 'Cl')

def countBromine(mol):
    return countAtomsOf(mol, 'Br')

def countIodine(mol):
    return countAtomsOf(mol, 'I')

def countPhosphorus(mol):
    return countAtomsOf(mol, 'P')

df = pd.read_csv('../data/bppp_dummy.csv', sep=',', header=0,index_col=0)

# Convert the SMILES strings into RDKit molecules
df['mol'] = df['smile'].apply(Chem.MolFromSmiles)

# Drop NA molecules
df = df.dropna(subset=['mol'])

# Convert the RDKit molecules into adjacency matrices
df['adj'] = df['mol'].apply(Chem.rdmolops.GetAdjacencyMatrix)

# Convert the adjacency matrices into Laplacian matrices
df['lap'] = df['adj'].apply(laplacian)

# Create a column with the number of eigenvalues
df['eig'] = df['lap'].apply(np.linalg.eigvals)

# Create columns with eigenvectors (NOT eigenvalues) of the 2 smallest eigenvalues and the 2 largest eigenvalues
df['eig_small'] = df['lap'].apply(lambda x: np.linalg.eigh(x)[1][:, :2])
df['eig_large'] = df['lap'].apply(lambda x: np.linalg.eigh(x)[1][:, -2:])
df['eig_small_1'] = df['eig_small'].apply(lambda x: x[0][0])
df['eig_small_2'] = df['eig_small'].apply(lambda x: x[0][1])
df['eig_large_1'] = df['eig_large'].apply(lambda x: x[0][0])
df['eig_large_2'] = df['eig_large'].apply(lambda x: x[0][1])
df['eig_large_2'] = df['eig_large'].apply(lambda x: x[1])

# Create a column with the number of atoms
df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())

# Create a column with the number of bonds
df['num_bonds'] = df['mol'].apply(lambda x: x.GetNumBonds())

# Create a column with the number of rings
df['num_rings'] = df['mol'].apply(lambda x: x.GetRingInfo().NumRings())

# Create a column with the number of rotatable bonds
df['num_rotatable_bonds'] = df['mol'].apply(lambda x: Chem.rdMolDescriptors.CalcNumRotatableBonds(x))

# Create a column with the number of heavy atoms
df['num_heavy_atoms'] = df['mol'].apply(lambda x: Chem.rdMolDescriptors.CalcNumHeavyAtoms(x))

# Create a column with the number of carbon atoms
df['num_carbon_atoms'] = df['mol'].apply(lambda x: countCarbon(x))

# Create a column with the number of oxygen atoms
df['num_oxygen_atoms'] = df['mol'].apply(lambda x: countOxygen(x))

# Create a column with the number of nitrogen atoms
df['num_nitrogen_atoms'] = df['mol'].apply(lambda x: countNitrogen(x))

# Create a column with the number of sulfur atoms
df['num_sulfur_atoms'] = df['mol'].apply(lambda x: countSulfur(x))

# Create a column with the number of fluorine atoms
df['num_fluorine_atoms'] = df['mol'].apply(lambda x: countFluorine(x))

# Create a column with the number of chlorine atoms
df['num_chlorine_atoms'] = df['mol'].apply(lambda x: countChlorine(x))

# Create a column with the number of bromine atoms
df['num_bromine_atoms'] = df['mol'].apply(lambda x: countBromine(x))

# Create a column with the number of iodine atoms
df['num_iodine_atoms'] = df['mol'].apply(lambda x: countIodine(x))

# Create a column with the number of phosphorus atoms
df['num_phosphorus_atoms'] = df['mol'].apply(lambda x: countPhosphorus(x))

print(df.head())

# Save the dataframe to a CSV file
df.to_csv('../data/test.csv')