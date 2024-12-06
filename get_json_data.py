import numpy as np
import pandas as pd
import json
import os
import sys

np.random.seed(2) # set random seed generator

if sys.platform == 'win32':
    path = 'C:/Users/Garren/Documents/Article - Pure PMF/Pure_MC' # path to save files
else:
    path = '/home/garren/Article - Pure PMF/Pure_MC'

os.makedirs(f'{path}/Pure RK PMF', exist_ok=True) # create folder to save files for varying temperatures
os.makedirs(f'{path}/Pure RK PMF - 298', exist_ok=True) # create folder to save files for 298.15 K

# read data
df = pd.read_excel(f'{path}/AllData.xlsx', sheet_name='Data')
comps = pd.read_excel(f'{path}/AllData.xlsx', sheet_name='Components')

# Extract all indices across all data
idx1 = np.sum((df['Component 1'].to_numpy().astype(str)[:, np.newaxis] == comps['IUPAC'].to_numpy().astype(str)[np.newaxis, :]) * np.arange(len(comps))[np.newaxis,:], axis=1) # index of component 1
idx2 = np.sum((df['Component 2'].to_numpy().astype(str)[:, np.newaxis] == comps['IUPAC'].to_numpy().astype(str)[np.newaxis, :]) * np.arange(len(comps))[np.newaxis,:], axis=1) # index of component 2
idx_all = np.char.add(np.char.add(idx1.astype(str), ' + '), idx2.astype(str)) # all indices
idx_all, idx = np.unique(idx_all, return_index=True) # unique indices
idx_all = idx_all[np.argsort(idx)] # sort unique indices
Idx_all = np.array([idx_all[i].split(' + ') for i in range(len(idx_all))]).astype(int) # convert unique indices to array

# Extract functional groups
fg = comps['Functional Group'].to_numpy().astype(str)

# Extract functional groups associated with each compound: Used to determine testing dataset
fg1 = fg[Idx_all[:,0]]
fg2 = fg[Idx_all[:,1]]

# Extract the unique combinations, and their counts for each mixtures of functional groups
all_fg = np.char.add(np.char.add(fg1, ' + '), fg2)
unique_fg, idx, counts_fg = np.unique(all_fg, return_index=True, return_counts=True)
unique_fg = unique_fg[np.argsort(idx)]
counts_fg = counts_fg[np.argsort(idx)]

# Determine the number of testing mixtures for each functional group
min_entries_per_fg = 2 # minimum number of mixtures for each combination of functional groups
sparse = 0.25 # Use approx 25% for training mixtures
num_test = np.max([np.ones(np.sum((counts_fg >= min_entries_per_fg))).astype(int), 
                    (counts_fg[(counts_fg >= min_entries_per_fg)]*sparse).astype(int)], 
                    axis=0) # number of testing mixtures, minimum of 1
fg_test = unique_fg[(counts_fg >= min_entries_per_fg)] # combination of functional groups for testing mixtures

# Randomly chose testing mictures from each type of combination of functional groups and assign them an index. 
# Using this index extract the neccassary data for testing
test_idx = np.unique(np.concatenate([np.random.choice(np.where(all_fg == fg_test[i])[0], num_test[i]) for i in range(len(fg_test))])) # randomly chose testing mixtures per functional group combination, concatenate the indicices across all combinations, and sort using unique
train_idx = np.setdiff1d(np.arange(len(Idx_all)), test_idx) # remaining mixtures are used for training
testing_indices = Idx_all[test_idx,:] # testing indices
Idx_known = Idx_all[train_idx,:] # training indices

# Check that each compound has at least one training mixture. If not, add a random mixture to the training set
N = len(comps) # number of compounds
compounds_with_no_training_data = np.array(comps['IUPAC'][np.nonzero([(np.sum((Idx_known[:,0] == i).astype(int)+(Idx_known[:,1] == i).astype(int))==0).astype(int)*(i+1) for i in range(N)])[0]].tolist()) # compounds with no training data
idx_comps_no_training_data =  np.sum((compounds_with_no_training_data[:,np.newaxis] == comps['IUPAC'].to_numpy().astype(str)[np.newaxis,:]).astype(int)*np.arange(N)[np.newaxis,:], axis=1) # index of compounds with no training data
idx_move_to_train = np.array([np.random.choice(np.where(((testing_indices[:,0] == ii).astype(int) + (testing_indices[:,1] == ii).astype(int)) > 0)[0]) for ii in idx_comps_no_training_data]) # randomly chose a testing mixture for each compound with no training data
train_idx = np.sort(np.append(train_idx, test_idx[idx_move_to_train])) # add the randomly chosen testing mixtures to the training set
test_idx = np.sort(np.delete(test_idx, idx_move_to_train)) # remove the randomly chosen testing mixtures from the testing set

# update the training and testing indices
testing_indices = Idx_all[test_idx,:]
Idx_known = Idx_all[train_idx,:]



# Extracting data for model at 298.15 K
# Extract all indices 
Idx_all_298 = np.column_stack([df[np.abs(df['Temperature [K]'] - 298.15)<=0.5]['Component 1 - Index'].to_numpy().astype(int), 
                             df[np.abs(df['Temperature [K]'] - 298.15)<=0.5]['Component 2 - Index'].to_numpy().astype(int)]) # all indices at 298.15 K +- 0.5 K
_, idx_298 = np.unique(np.char.add(np.char.add(Idx_all_298[:,0].astype(str), ' + '), Idx_all_298[:,1].astype(str)), return_index=True) # unique combinations of functional groups
Idx_all_298 = Idx_all_298[idx_298] # unique combinations of functional groups

# Extracting testing data based on data across temperature
testing_indices_298 = []
for ii in range(len(testing_indices)):
    i,j = testing_indices[ii,:]
    idx = ((Idx_all_298[:,0] == i).astype(int) + (Idx_all_298[:,1] == j).astype(int)) == 2
    if np.sum(idx) > 0: # if i,j indicies are in testing data at 298.15 K
        testing_indices_298 += [[i,j]] # add to testing data at 298.15 K
testing_indices_298 = np.array(testing_indices_298) 

# Extracting training data based on data across temperature
idx_298 = np.char.add(np.char.add(Idx_all_298[:,0].astype(str), ' + '), Idx_all_298[:,1].astype(str))[:,np.newaxis] == np.char.add(np.char.add(testing_indices_298[:,0].astype(str), ' + '), testing_indices_298[:,1].astype(str))[np.newaxis,:] # matrix of indices where we have testing data
idx_298 = np.sum(idx_298, axis=1) == 0 # remove testing data
Idx_known_298 = Idx_all_298[idx_298,:] # training data at 298.15 K
# Note: Due to the conctruction of these testing data, the training data at 298.15 K may not have any training data for certain compounds



## Save json data across temperatures
mix_df = np.char.add(np.char.add(df['Component 1'].to_numpy().astype(str), ' + '), df['Component 2'].to_numpy().astype(str))
mix_train = np.char.add(np.char.add(comps['IUPAC'].to_numpy().astype(str)[Idx_known[:,0]], ' + '), comps['IUPAC'].to_numpy().astype(str)[Idx_known[:,1]])
CA = comps['Self Cluster assignment'].to_numpy().astype(int)
unique_CA = np.unique(CA)
C = (unique_CA[:,np.newaxis] == CA[np.newaxis,:]).astype(int)
K = len(unique_CA)
x = np.concatenate([df['Composition component 1 [mol/mol]'][mix_df == m].to_numpy().astype(float) for m in mix_train])
T = np.concatenate([df['Temperature [K]'][mix_df == m].to_numpy().astype(float) for m in mix_train])
y = np.concatenate([df['Excess Enthalpy [J/mol]'][mix_df == m].to_numpy().astype(float) for m in mix_train])
N_known = Idx_known.shape[0]
N_points = np.array([np.sum(mix_df == m) for m in mix_train])
Idx_unknown = np.array([[i, j] for i in range(N) for j in range(i+1,N)])
idx = np.sum(np.char.add(np.char.add(Idx_unknown[:,0].astype(str), ' + '), Idx_unknown[:,1].astype(str))[:,np.newaxis] ==
             np.char.add(np.char.add(Idx_known[:,0].astype(str), ' + '), Idx_known[:,1].astype(str))[np.newaxis,:], axis=1) == 0
Idx_unknown = Idx_unknown[idx,:]
N_unknown = int((N**2-N)/2 - N_known)
v = 1e-3*np.ones(N_known)
v_MC = 0.2
x2_int = np.concatenate([np.append(np.linspace(0,0.45, 10)[1:], [0.495, 1-0.495]), np.linspace(0.55, 1, 10)[:-1]])
T2_int = [288.15, 298.15, 308.15]
N_C = x2_int.shape[0]
N_T = len(T2_int)
order = 3
jitter = 1e-7

# save data in dictionary
data = {'N_known': int(N_known),
        'N_unknown': int(N_unknown),
        'N_points': N_points.tolist(),
        'order': int(order),
        'x1': x.tolist(),
        'T1': T.tolist(),
        'y1': y.tolist(),
        'N_C': int(N_C),
        'N_T': int(N_T),
        'T2_int': T2_int,
        'x2_int': x2_int.tolist(),
        'v_MC': v_MC,
        'N': int(N),
        'Idx_known': (Idx_known+1).tolist(),
        'Idx_unknown': (Idx_unknown+1).tolist(),
        'jitter': jitter,
        'v': v.tolist(),
        'C': C.tolist(),
        'K': int(K),
        'v_cluster': [(0.1)**2 for _ in range(K)],
        'sigma_refT': [0.1, 1e-5, 0.1],
        'testing_indices': testing_indices.tolist(),}

# save data
with open(f'{path}/Pure RK PMF/data.json', 'w') as f:
    json.dump(data, f)



## Save json data at 298.15 K
idx_298 = np.abs(df['Temperature [K]'] - 298.15) <= 0.5
mix_df = np.char.add(np.char.add(df['Component 1'][idx_298].to_numpy().astype(str), ' + '), df['Component 2'][idx_298].to_numpy().astype(str))
mix_train = np.char.add(np.char.add(comps['IUPAC'].to_numpy().astype(str)[Idx_known_298[:,0]], ' + '), comps['IUPAC'].to_numpy().astype(str)[Idx_known_298[:,1]])
x = np.concatenate([df['Composition component 1 [mol/mol]'][idx_298].to_numpy().astype(float)[mix_df==m] for m in mix_train])
T = np.concatenate([df['Temperature [K]'][idx_298].to_numpy().astype(float)[mix_df==m] for m in mix_train])
y = np.concatenate([df['Excess Enthalpy [J/mol]'][idx_298].to_numpy().astype(float)[mix_df==m] for m in mix_train])
N_known = Idx_known_298.shape[0]
N_points = np.array([np.sum(mix_df == m) for m in mix_train])
Idx_unknown = np.array([[i, j] for i in range(N) for j in range(i+1,N)])
idx = np.sum(np.char.add(np.char.add(Idx_unknown[:,0].astype(str), ' + '), Idx_unknown[:,1].astype(str))[:,np.newaxis] ==
             np.char.add(np.char.add(Idx_known_298[:,0].astype(str), ' + '), Idx_known_298[:,1].astype(str))[np.newaxis,:], axis=1) == 0
Idx_unknown = Idx_unknown[idx,:]
N_unknown = int((N**2-N)/2 - N_known)
x2_int = np.concatenate([np.append(np.linspace(0,0.45, 10)[1:], [0.495, 1-0.495]), np.linspace(0.55, 1, 10)[:-1]])
N_C = x2_int.shape[0]
T2_int = [298.15]
N_T = len(T2_int)
v = 1e-3*np.ones(N_known)

data = {'N_known': int(N_known),
        'N_unknown': int(N_unknown),
        'N_points': N_points.tolist(),
        'order': int(order),
        'x1': x.tolist(),
        'T1': T.tolist(),
        'y1': y.tolist(),
        'N_C': int(N_C),
        'N_T': int(N_T),
        'T2_int': T2_int,
        'x2_int': x2_int.tolist(),
        'v_MC': v_MC,
        'N': int(N),
        'Idx_known': (Idx_known_298+1).tolist(),
        'Idx_unknown': (Idx_unknown+1).tolist(),
        'jitter': jitter,
        'v': v.tolist(),
        'C': C.tolist(),
        'K': int(K),
        'v_cluster': [(0.1)**2 for _ in range(K)],
        'sigma_refT': [0.1, 1e-5, 0.1],
        'testing_indices': testing_indices_298.tolist(),}

# save data
with open(f'{path}/Pure RK PMF - 298/data.json', 'w') as f:
    json.dump(data, f)



