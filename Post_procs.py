"""
Note to user:
The code was written with a specific directory structure in mind, see the class description for more details.
The <home_path> variable should be changed depending on the user's directory structure.
"""

import cmdstanpy
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import json
from IPython.display import clear_output

if sys.platform == 'win32':
    home_path = 'C:/Users/Garren/Documents/Article - Pure PMF/Pure_MC/'
    
else:
    home_path = '/home/garren/Article - Pure PMF/Pure_MC/'

# Use Agg for saving a lot of plots without opening figure window
matplotlib.use('Agg')

#initialize by closing all open figures
plt.clf()
plt.close()

class Post_process:
    """
    Post processing class for Pure MC.
    The class takes Stan output and performs the necessary post processing steps to obtain the final results.
    Methods:
        get_tensors                 : Get the A tensor from the Stan output. The size of the tensor is ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds.
        extract_interps             : Extract the interpolated values from the A tensors into a 3D array with ranks x num_conditions x num_compounds.
        get_reconstructed_values    : Computes the excess enthalpy predictions for the known (training) mixtures.
        get_testing_values          : Computes the excess enthalpy predictions for the testing mixtures.
        get_testing_metrics         : Computes the MAE, RMSE and MARE for the testing mixtures.
        get_testing_metrics_T_dep   : Computes the MAE, RMSE and MARE for the testing mixtures at each temperature.
        plot_2D_plots               : Plots the 2D plots for the testing or training data

    The class expects the following directory structure:
    home_path
        ├── Stan Models
            ├── Pure_PMF_include_clusters_<bool>_zeros_<bool>_refT_<bool>.stan (Stan files)
        ├── AllData.xlsx (Excel file with all data)
                The 'Data' sheet should contain the following columns: 'Component 1'                        -> (IUPAC name of component 1),
                                                                       'Component 2'                        -> (IUPAC name of component 2),
                                                                       'Composition component 1 [mol/mol]'  -> (Composition of component 1),
                                                                       'Temperature [K]'                    -> (Temperature),
                                                                       'Excess Enthalpy [J/mol]'            -> (Experimental excess enthalpy),
                                                                       'UNIFAC_DMD [J/mol]'                 -> (mod. UNIFAC Dortmund excess enthalpy),
                The 'Components' sheet should contain the following columns: 'IUPAC'                        -> (IUPAC name of component),
                                                                             'Functional Group'             -> (Functional group of component) 
                                                                             'Self Cluster assignment'      -> (Cluster assignment of component, integer values starting from 0 indicating the first cluster)
        ├── UNIFAC_Plots.xlsx (Excel file with known data plots)
                A single sheet containing UNIFAC predictions across tetsing and training mixtures across conditions. 
                The sheet should contain the following columns: 'Component 1'                               -> (IUPAC name of component 1),
                                                                'Component 2'                               -> (IUPAC name of component 2),
                                                                'Composition component 1 [mol/mol]'         -> (Composition of component 1),
                                                                'Temperature [K]'                           -> (Temperature),
                                                                'UNIFAC_DMD [J/mol]'                        -> (mod. UNIFAC Dortmund excess enthalpy)
        ├── Thermo_UNIFAC_DMD_unknown.xlsx (Excel file with unknown data plots)
                A single sheet containing UNIFAC predictions across unknown mixtures across across compositions at 288.15, 298.15 and 308.15 K.
                The sheet should contain the following columns: 'Component 1'                               -> (IUPAC name of component 1),
                                                                'Component 2'                               -> (IUPAC name of component 2),
                                                                'Composition component 1 [mol/mol]'         -> (Composition of component 1),
                                                                'Temperature [K]'                           -> (Temperature),
                                                                'UNIFAC_DMD [J/mol]'                        -> (mod. UNIFAC Dortmund excess enthalpy)
        ├── Pure RK PMF - 298 (Directory containing results for 298 K)
            ├── data.json (Data file)
            ├── Results
                ├── Include_clusters_<bool>
                    ├── Add_zeros_<bool>
                        ├── <int> (Rank) 
                            ├── <csv file> (stan output)
                            ├── <txt file> (stan output)
                            ├── inits.json (inits file not needed if csv file is good)
        ├── Pure RK PMF (Directory containing results for all temperatures)
            ├── data.json (Data file)
            ├── Results
                ├── Include_clusters_<bool>
                    ├── Add_zeros_<bool>
                        ├── RefT_<bool>
                            ├── <int> (Rank) 
                                ├── <csv file> (stan output)
                                ├── <txt file> (stan output)
                                ├── inits.json (inits file not needed if csv file is good)
    """

    def __init__(self, include_clusters: bool, include_zeros: bool, refT: bool, T: str, home_path: str = home_path):
        """
        Inputs:
            include_clusters: bool -> Whether to include clusters in the model.
            include_zeros: bool -> Whether to include zeros in the model.
            refT: bool -> Whether to include reference temperature in the model.
            T: str -> Temperature at which to extract results. Either '298' or 'all'.
            If T == '298', the class will look for results at 298 K. 
            If T == 'all', the class will look for results at all temperatures.
            If T == '298', refT is ignored, but still required as input.

        Variables initialized:
            stan_path               : str               -> Path to Stan files.
            stan_file               : str               -> Path to Stan file.
            excel_data              : str               -> Path to excel data file.
            excel_plots_known       : str               -> Path to mod. UNIFAC Dortmund predictions (for testing and training mixtures) to generate plots.
            excel_unknown_vs_uni    : str               -> Path to mod. UNIFAC Dortmund predictions for unknown mixtures to generate plots.
            include_clusters        : bool              -> Whether to include clusters in the model.
            include_zeros           : bool              -> Whether to include zeros in the model.
            refT                    : bool              -> Whether to include reference temperature in the model.
            T                       : str               -> Temperature at which to extract results.
            path                    : str               -> Path to results directory up until ranks.
            data_file               : str               -> Path to data file.
            ranks                   : list              -> List of ranks.
            fg                      : np.array          -> Array of functional group assignments.
            c_all                   : np.array          -> Array of all compound names (IUPAC).
            Kx                      : lambda function   -> Kernel function for composition.
            KT                      : lambda function   -> Kernel function for temperature.
            K                       : lambda function   -> Kernel function for composition and temperature.
            Idx_known               : np.array          -> Array of indices for known (testing) mixtures.
            testing_indices         : np.array          -> Array of indices for testing mixtures.
            indices_df_298          : np.array          -> Array of indices for testing mixtures at 298 K.
                                                           Only used if T == '298'.
        """                                                

        # Stan files
        self.stan_path = f'{home_path}/Stan Models'
        self.stan_file = f'{self.stan_path}/Pure_PMF_include_clusters_{include_clusters}_zeros_{include_zeros}_refT_'
        # paths to excel files
        self.excel_data = f'{home_path}/AllData.xlsx' # Path to testing data file
        self.excel_plots_known = f'{home_path}/UNIFAC_Plots.xlsx' # Path to known data plots
        self.excel_unknown_vs_uni = f'{home_path}/Thermo_UNIFAC_DMD_unknown.xlsx' # Path to unknown data plots
        # Ensure T is either '298' or 'all'
        assert T in ['298', 'all'], "T must be either '298' or 'all'"
        if T == '298':
            home_path += 'Pure RK PMF - 298'
            self.stan_file += 'False' # always exclude reference temperature
        else:
            home_path += 'Pure RK PMF'
            self.stan_file += f'{refT}'
        self.stan_file += '.stan' # append stan file extension
        
        self.include_clusters = include_clusters    # Add clusters
        self.include_zeros = include_zeros          # Add zeros         
        self.refT = refT                            # Reference temperature
        self.T = T                                  # Temperature index

        self.path = f'{home_path}/Results' # path to where files are stored
        self.data_file = f'{home_path}/data.json'                # data file
        self.path += f'/Include_clusters_{self.include_clusters}/Add_zeros_{self.include_zeros}'
        if T == 'all':
            self.path += f'/RefT_{self.refT}'

        # get all ranks
        self.ranks = [i for i in os.listdir(self.path) if i.isdigit()] # list of all ranks
        self.ranks = np.array(self.ranks).astype(int)
        self.ranks = np.sort(self.ranks)

        # save all compounds nanes in cluster along with functional group assignments
        with pd.ExcelFile(self.excel_data) as f:
            self.fg = pd.read_excel(f, sheet_name='Components')['Functional Group'].to_numpy().astype(str)
            self.c_all = pd.read_excel(f, sheet_name='Components')['IUPAC'].to_numpy().astype(str)

        # Kernel functions
        self.Kx = lambda x1, x2: np.column_stack([np.column_stack([x1**(i+2)-x1 for i in range(3)]), 
                                     1e-1*x1*np.sqrt(1-x1)*np.exp(x1)]) @ np.column_stack([np.column_stack([x2**(i+2)-x2 for i in range(3)]), 
                                                                                           1e-1*x2*np.sqrt(1-x2)*np.exp(x2)]).T
        self.KT = lambda T1, T2: np.column_stack([np.ones_like(T1), T1, T1**2, 1e-3*T1**3]) @ np.column_stack([np.ones_like(T2), T2, T2**2, 1e-3*T2**3]).T
        self.K = lambda x1, x2, T1, T2: self.Kx(x1, x2) * self.KT(T1, T2)

        # Training and testing Indices
        self.Idx_known = np.array(json.load(open(self.data_file, 'r'))['Idx_known'])-1
        self.testing_indices = np.array(json.load(open(self.data_file, 'r'))['testing_indices'])

        # index for testing data only at 298.15 K
        if T == '298':
            self.indices_df_298 = np.abs(pd.read_excel(self.excel_data, sheet_name='Data')['Temperature [K]'].to_numpy().astype(float) - 298.15) <= 0.5

    def get_tensors(self) -> np.array:
        """
        Description: Extracts the A tensor from the Stan output.
                     The size of the tensor is ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds.

        Inputs: 
            None

        Outputs: 
            A: np.array -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds. 
        """

        self.log_prob = []
        self.log_obj = []
        
        # stan model for log_prob calculations
        model = cmdstanpy.CmdStanModel(stan_file=self.stan_file)
        A = [] # list of all A tensors to be converted to strings
        
        v_cluster = np.array(json.load(open(self.data_file, 'r'))['v_cluster'])
        C = np.array(json.load(open(self.data_file, 'r'))['C'])
        sigma_refT = np.array(json.load(open(self.data_file, 'r'))['sigma_refT'])[:,np.newaxis,np.newaxis,np.newaxis]

        for rank in self.ranks:
            m = {}
            try:
                csv_file = [f'{self.path}/{rank}/{f}' for f in os.listdir(f'{self.path}/{rank}') if f.endswith('.csv')][0]
                MAP = cmdstanpy.from_csv(csv_file)
                keys = list(MAP.stan_variables().keys())
                for key in keys:
                    m[key] = MAP.stan_variables()[key]
                del MAP, csv_file
            except:
                inits_file = f'{self.path}/{rank}/inits.json'
                m = json.load(open(inits_file, 'r'))
                keys = list(m.keys())
                for key in keys:
                    m[key] = np.array(m[key])
                del inits_file
            data = json.load(open(self.data_file, 'r'))
            data['D'] = rank
            data['v_features'] = 100*np.ones(rank)
            self.log_prob += [model.log_prob(data=data, params=m).iloc[0,0]]
            self.log_obj += [np.log(-self.log_prob[-1])]

            sigma_cluster = np.sqrt(v_cluster)[np.newaxis,:] * np.ones((rank,1))
            sigma_cluster_mat = sigma_cluster @ C

            if self.include_clusters and self.refT and self.T == 'all':
                U = m["U_raw"] * sigma_refT + (m["U_raw_refT"] * sigma_cluster_mat[np.newaxis,:,:] + m["U_raw_means"] @ C[np.newaxis,:,:])
                V = m["V_raw"] * sigma_refT + (m["V_raw_refT"] * sigma_cluster_mat[np.newaxis,:,:] + m["V_raw_means"] @ C[np.newaxis,:,:])
            elif self.include_clusters and (not self.refT or self.T == '298'): # ignore refT if T == '298'; refT is always False
                U = m["U_raw"] * sigma_cluster_mat[np.newaxis,:,:] + m["U_raw_means"] @ C[np.newaxis,np.newaxis,:,:]
                V = m["V_raw"] * sigma_cluster_mat[np.newaxis,:,:] + m["V_raw_means"] @ C[np.newaxis,np.newaxis,:,:]
            elif not self.include_clusters and self.refT and self.T == 'all':
                U = m["U_raw"] * sigma_refT + m["U_raw_refT"]
                V = m["V_raw"] * sigma_refT + m["V_raw_refT"]
            else:
                U = m["U_raw"]
                V = m["V_raw"]

            v_features = np.diag(data['v_features'])[np.newaxis,np.newaxis,:,:]
            
            A += [U.transpose(0,1,3,2) @ v_features @ V]
        
        return np.array(A)
    
    def extract_interps(self, Idx: np.array, A=None) -> np.array:
        """
        Description: Extracts the interpolated values from the A tensors into a 3D array with ranks x num_conditions x num_compounds.

        Inputs:
            Idx         : np.array -> Array of indices for known (testing) mixtures.
            A           : np.array -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)
                                      If not provided, the method will call get_tensors to get the A tensor.
        
        Outputs:
            y_MC_pred   : np.array -> Array of interpolated values of size ranks x num_conditions x Idx.shape[0].
        """

        if A is None:
            A = self.get_tensors()
        R = A.shape[0]
        N_T = A.shape[1]
        M = A.shape[2]
        N = A.shape[3]
        y_MC_pred = np.stack([np.column_stack([np.concatenate([np.concatenate([A[r,t,:, idx[0], idx[1]], A[r,t,::-1,idx[1],idx[0]]]) for t in range(N_T)])
                            for idx in Idx]) for r in range(R)])
        return y_MC_pred
    
    def get_reconstructed_values(self, A=None) -> dict:
        """
        Description: Computes the excess enthalpy predictions for the known (training) mixtures.

        Inputs:
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)
                                       If not provided, the method will call get_tensors to get the A tensor.
        
        Outputs:
            data_dict   : dict      -> Dictionary containing the following: Component 1                         : np.array -> IUPAC name of component 1,
                                                                            Component 2                         : np.array -> IUPAC name of component 2,
                                                                            Composition component 1 [mol/mol]   : np.array -> Composition of component 1,
                                                                            Temperature [K]                     : np.array -> Temperature,
                                                                            Excess Enthalpy [J/mol]             : np.array -> Experimental excess enthalpy,
                                                                            UNIFAC_DMD [J/mol]                  : np.array -> mod. UNIFAC Dortmund excess enthalpy,
                                                                            MC [J/mol]                          : np.array -> Predicted excess enthalpy. Size of Idx_known.shape[0] x ranks.
        """

        if A is None:
            A = self.get_tensors()

        y_MC_interp = self.extract_interps(A=A, Idx=self.Idx_known)

        del A

        # Extraxt testing data from excel
        data_df = pd.read_excel(self.excel_data)
        if self.T == '298':
            data_df = data_df[self.indices_df_298]
        mix_all = np.char.add(np.char.add(data_df['Component 1'].to_numpy().astype(str), ' + '), data_df['Component 2'].to_numpy().astype(str))
        mix_known = np.char.add(np.char.add(self.c_all[self.Idx_known[:,0]], ' + '), self.c_all[self.Idx_known[:,1]])

        x = [data_df['Composition component 1 [mol/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known]
        T = [data_df['Temperature [K]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known]
        y_exp = np.concatenate([data_df['Excess Enthalpy [J/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known])
        y_UNIFAC = np.concatenate([data_df['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known])
        c1 = np.concatenate([data_df['Component 1'].to_numpy().astype(str)[mix_all == mix] for mix in mix_known])
        c2 = np.concatenate([data_df['Component 2'].to_numpy().astype(str)[mix_all == mix] for mix in mix_known])

        del data_df

        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])

        x_MC = np.concatenate([x2_int for _ in T2_int])
        T_MC = np.concatenate([t*np.ones_like(x2_int) for t in T2_int])
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC, L_MC_inv

        y_MC = [self.K(x[i], x_MC, T[i], T_MC) @ K_MC_inv @ y_MC_interp[:,:,i].T for i in range(len(x))]

        y_MC = np.concatenate(y_MC, axis=0)
        x = np.concatenate(x)
        T = np.concatenate(T)

        data_dict = {'Component 1': c1,
                     'Component 2': c2,
                     'Composition component 1 [mol/mol]': x,
                     'Temperature [K]': T,
                     'Excess Enthalpy [J/mol]': y_exp,
                     'UNIFAC_DMD [J/mol]': y_UNIFAC,
                     'MC [J/mol]': y_MC}

        return data_dict
    
    def get_testing_values(self, A=None) -> dict:
        """
        Description: Computes the excess enthalpy predictions for the testing mixtures.

        Inputs:
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)
                                       If not provided, the method will call get_tensors to get the A tensor.
        
        Outputs:
            data_dict   : dict      -> Dictionary containing the following: Component 1                         : np.array -> IUPAC name of component 1,
                                                                            Component 2                         : np.array -> IUPAC name of component 2,
                                                                            Composition component 1 [mol/mol]   : np.array -> Composition of component 1,
                                                                            Temperature [K]                     : np.array -> Temperature,
                                                                            Excess Enthalpy [J/mol]             : np.array -> Experimental excess enthalpy,
                                                                            UNIFAC_DMD [J/mol]                  : np.array -> mod. UNIFAC Dortmund excess enthalpy,
                                                                            MC [J/mol]                          : np.array -> Predicted excess enthalpy. Size of testing_indices.shape[0] x ranks.
        """

        # Get matrices
        if A is None:
            A = self.get_tensors()

        # Extraxt testing data from excel
        data_df = pd.read_excel(self.excel_data)
        if self.T == '298':
            data_df = data_df[self.indices_df_298]
        mix_all = np.char.add(np.char.add(data_df['Component 1'].to_numpy().astype(str), ' + '), data_df['Component 2'].to_numpy().astype(str))
        mix_known = np.char.add(np.char.add(self.c_all[self.testing_indices[:,0]], ' + '), self.c_all[self.testing_indices[:,1]])

        x = [data_df['Composition component 1 [mol/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known]
        T = [data_df['Temperature [K]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known]
        y_exp = np.concatenate([data_df['Excess Enthalpy [J/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known])
        y_UNIFAC = np.concatenate([data_df['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[mix_all == mix] for mix in mix_known])
        c1 = np.concatenate([data_df['Component 1'].to_numpy().astype(str)[mix_all == mix] for mix in mix_known])
        c2 = np.concatenate([data_df['Component 2'].to_numpy().astype(str)[mix_all == mix] for mix in mix_known])

        del data_df

        # get interpolated values
        y_MC_interp = self.extract_interps(A=A, Idx=self.testing_indices)

        # All mixtures
        mix_all = np.char.add(np.char.add('c1', ' + '), 'c2')

        # Extract data from file
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])

        x_MC = np.concatenate([x2_int for _ in T2_int])
        T_MC = np.concatenate([t*np.ones_like(x2_int) for t in T2_int])
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC, L_MC_inv

        y_MC = [self.K(x[i], x_MC, T[i], T_MC) @ K_MC_inv @ y_MC_interp[:,:,i].T for i in range(len(x))]
        
        y_MC = np.concatenate(y_MC, axis=0)
        x = np.concatenate(x)
        T = np.concatenate(T)

        data_dict = {'Component 1': c1,
                     'Component 2': c2,
                     'Composition component 1 [mol/mol]': x,
                     'Temperature [K]': T,
                     'Excess Enthalpy [J/mol]': y_exp,
                     'UNIFAC_DMD [J/mol]': y_UNIFAC,
                     'MC [J/mol]': y_MC}

        return data_dict
    
    def get_testing_metrics(self, A=None, data_dict=None) -> dict:
        """
        Description: Computes the MAE, RMSE and MARE for the testing mixtures.

        Inputs:
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)
                                       If not provided, the method will call get_tensors to get the A tensor.
            data_dict   : dict      -> Dictionary containing the following: Component 1                         : np.array -> IUPAC name of component 1,
                                                                            Component 2                         : np.array -> IUPAC name of component 2,
                                                                            Composition component 1 [mol/mol]   : np.array -> Composition of component 1,
                                                                            Temperature [K]                     : np.array -> Temperature,
                                                                            Excess Enthalpy [J/mol]             : np.array -> Experimental excess enthalpy,
                                                                            UNIFAC_DMD [J/mol]                  : np.array -> mod. UNIFAC Dortmund excess enthalpy,
                                                                            MC [J/mol]                          : np.array -> Predicted excess enthalpy. Size of testing_indices.shape[0] x ranks.
                                       If not provided, the method will call get_testing_values to get the data_dict.
        
        Outputs:
            err_dict    : dict      -> Dictionary containing the following: ['Component 1', '', '']: np.array -> IUPAC name of component 1,
                                                                            ['Component 2', '', '']: np.array -> IUPAC name of component 2,
                                                                            ['UNIFAC', 'MAE', '']: np.array -> Mean Absolute Error for UNIFAC,
                                                                            ['UNIFAC', 'RMSE', '']: np.array -> Root Mean Squared Error for UNIFAC,
                                                                            ['UNIFAC', 'MARE', '']: np.array -> Mean Absolute Relative Error for UNIFAC,
                                                                            ['MC', 'MAE', <rank>]: np.array -> Mean Absolute Error for MC. <rank> is an interger corresponding to the rank
                                                                            ['MC', 'RMSE', <rank>]: np.array -> Root Mean Squared Error for MC. <rank> is an interger corresponding to the rank
                                                                            ['MC', 'MARE', <rank>]: np.array -> Mean Absolute Relative Error for MC. <rank> is an interger corresponding to the rank
                                       The final entry in each array is the overall error.
        """

        if A is None and data_dict is None:
            A = self.get_tensors()
        if A is not None and data_dict is None:
            data_dict = self.get_testing_values(A=A)

        mix_all = np.char.add(np.char.add(data_dict['Component 1'], ' + '), data_dict['Component 2'])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        err_dict = {('Component 1', '', ''): [],
                    ('Component 2', '', ''): [],
                    ('UNIFAC', 'MAE', ''): [],
                    ('UNIFAC', 'RMSE', ''): [],
                    ('UNIFAC', 'MARE', ''): [],}
        metrics = ['MAE', 'RMSE', 'MARE']
        for metric in metrics:
            for m in range(len(self.ranks)):
                err_dict[('MC', metric, self.ranks[m])] = []

        y_exp = data_dict['Excess Enthalpy [J/mol]']
        y_UNIFAC = data_dict['UNIFAC_DMD [J/mol]']
        y_MC = data_dict['MC [J/mol]']

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]
            err_dict['Component 1', '', ''] += [data_dict['Component 1'][idx][0]]
            err_dict['Component 2', '', ''] += [data_dict['Component 2'][idx][0]]
            err_dict['UNIFAC', 'MAE', ''] += [np.mean(np.abs(y_exp[idx] - y_UNIFAC[idx]))]
            err_dict['UNIFAC', 'RMSE', ''] += [np.sqrt(np.mean((y_exp[idx] - y_UNIFAC[idx])**2))]
            err_dict['UNIFAC', 'MARE', ''] += [np.mean(np.abs((y_exp[idx] - y_UNIFAC[idx])/y_exp[idx]))]
            for m in range(len(self.ranks)):
                err_dict['MC', 'MAE', self.ranks[m]] += [np.mean(np.abs(y_exp[idx] - y_MC[idx,m]))]
                err_dict['MC', 'RMSE', self.ranks[m]] += [np.sqrt(np.mean((y_exp[idx] - y_MC[idx,m])**2))]
                err_dict['MC', 'MARE', self.ranks[m]] += [np.mean(np.abs((y_exp[idx] - y_MC[idx,m])/y_exp[idx]))]

        err_dict['Component 1', '', ''] += ['Overall']
        err_dict['Component 2', '', ''] += ['']
        err_dict['UNIFAC', 'MAE', ''] += [np.mean(np.abs(y_exp - y_UNIFAC))]
        err_dict['UNIFAC', 'RMSE', ''] += [np.sqrt(np.mean((y_exp - y_UNIFAC)**2))]
        err_dict['UNIFAC', 'MARE', ''] += [np.mean(np.abs((y_exp - y_UNIFAC)/y_exp))]
        for m in range(len(self.ranks)):
            err_dict['MC', 'MAE', self.ranks[m]] += [np.mean(np.abs(y_exp - y_MC[:,m]))]
            err_dict['MC', 'RMSE', self.ranks[m]] += [np.sqrt(np.mean((y_exp - y_MC[:,m])**2))]
            err_dict['MC', 'MARE', self.ranks[m]] += [np.mean(np.abs((y_exp - y_MC[:,m])/y_exp))]

        keys = list(err_dict.keys())
        for key in keys:
            err_dict[key] = np.array(err_dict[key])

        return err_dict
    
    def get_testing_metrics_T_dep(self, A=None, data_dict=None) -> dict:
        """
        Description: Computes the MAE, RMSE and MARE for the testing mixtures at each temperature.

        Inputs:
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)
                                       If not provided, the method will call get_tensors to get the A tensor.
            data_dict   : dict      -> Dictionary containing the following: Component 1                         : np.array -> IUPAC name of component 1,
                                                                            Component 2                         : np.array -> IUPAC name of component 2,
                                                                            Composition component 1 [mol/mol]   : np.array -> Composition of component 1,
                                                                            Temperature [K]                     : np.array -> Temperature,
                                                                            Excess Enthalpy [J/mol]             : np.array -> Experimental excess enthalpy,
                                                                            UNIFAC_DMD [J/mol]                  : np.array -> mod. UNIFAC Dortmund excess enthalpy,
                                                                            MC [J/mol]                          : np.array -> Predicted excess enthalpy. Size of testing_indices.shape[0] x ranks.
                                       If not provided, the method will call get_testing_values to get the data_dict.
        
        Outputs:
            err_dict    : dict      -> Dictionary containing the following: ['Component 1', '', '']: np.array -> IUPAC name of component 1,
                                                                            ['Component 2', '', '']: np.array -> IUPAC name of component 2,
                                                                            ['Temperature [K]', '', '']: np.array -> Temperature,
                                                                            ['UNIFAC', 'MAE', '']: np.array -> Mean Absolute Error for UNIFAC,
                                                                            ['UNIFAC', 'RMSE', '']: np.array -> Root Mean Squared Error for UNIFAC,
                                                                            ['UNIFAC', 'MARE', '']: np.array -> Mean Absolute Relative Error for UNIFAC,
                                                                            ['MC', 'MAE', <rank>]: np.array -> Mean Absolute Error for MC. <rank> is an interger corresponding to the rank
                                                                            ['MC', 'RMSE', <rank>]: np.array -> Root Mean Squared Error for MC. <rank> is an interger corresponding to the rank
                                                                            ['MC', 'MARE', <rank>]: np.array -> Mean Absolute Relative Error for MC. <rank> is an interger corresponding to the rank
        """

        if A is None and data_dict is None:
            A = self.get_tensors()
        if A is not None and data_dict is None:
            data_dict = self.get_testing_values(A=A)

        mix_all = np.char.add(np.char.add(data_dict['Component 1'], ' + '), data_dict['Component 2'])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]

        err_dict = {('Component 1', '', ''): [],
                    ('Component 2', '', ''): [],
                    ('Temperature [K]', '', ''): [],
                    ('UNIFAC', 'MAE', ''): [],
                    ('UNIFAC', 'RMSE', ''): [],
                    ('UNIFAC', 'MARE', ''): [],}
        metrics = ['MAE', 'RMSE', 'MARE']
        for metric in metrics:
            for m in range(len(self.ranks)):
                err_dict[('MC', metric, self.ranks[m])] = []

        y_exp = data_dict['Excess Enthalpy [J/mol]']
        y_UNIFAC = data_dict['UNIFAC_DMD [J/mol]']
        y_MC = data_dict['MC [J/mol]']

        for j in range(len(unique_mix)):
            idx = mix_all == unique_mix[j]

            T_mix = data_dict['Temperature [K]'][idx]
            T_unique = np.unique(T_mix)
            T_unique = np.concatenate([T_unique.astype(int)+0.15,
                                       T_unique.astype(int)+1.15,])
            T_unique = np.unique(T_unique)
            T_unique = T_unique[np.sum(np.abs(T_unique[:,np.newaxis] - T_mix[np.newaxis,:]) <= 0.5, axis=1)>0]

            for TT in T_unique:    
                T_idx = np.abs(T_mix - TT) <= 0.5
                err_dict['Component 1', '', ''] += [data_dict['Component 1'][idx][0]]
                err_dict['Component 2', '', ''] += [data_dict['Component 2'][idx][0]]
                err_dict['Temperature [K]', '', ''] += [TT]
                err_dict['UNIFAC', 'MAE', ''] += [np.mean(np.abs(y_exp[idx][T_idx] - y_UNIFAC[idx][T_idx]))]
                err_dict['UNIFAC', 'RMSE', ''] += [np.sqrt(np.mean((y_exp[idx][T_idx] - y_UNIFAC[idx][T_idx])**2))]
                err_dict['UNIFAC', 'MARE', ''] += [np.mean(np.abs((y_exp[idx][T_idx] - y_UNIFAC[idx][T_idx])/y_exp[idx][T_idx]))]
                for m in range(len(self.ranks)):
                    err_dict['MC', 'MAE', self.ranks[m]] += [np.mean(np.abs(y_exp[idx][T_idx] - y_MC[idx,m][T_idx]))]
                    err_dict['MC', 'RMSE', self.ranks[m]] += [np.sqrt(np.mean((y_exp[idx][T_idx] - y_MC[idx,m][T_idx])**2))]
                    err_dict['MC', 'MARE', self.ranks[m]] += [np.mean(np.abs((y_exp[idx][T_idx] - y_MC[idx,m][T_idx])/y_exp[idx][T_idx]))]

        keys = list(err_dict.keys())
        for key in keys:
            err_dict[key] = np.array(err_dict[key])

        return err_dict
    
    def plot_2D_plots(self, data_type=None, ranks=None, plot_one=False, A=None) -> None:
        """
        Description: Plots the 2D plots for the testing or training mixtures.

        Inputs:
            data_type   : str       -> Type of data to plot. Must be either 'Testing' or 'Training'.
            ranks       : list      -> List of ranks to plot. If None, all ranks will be plotted.
            plot_one    : bool      -> If True, only one rank will be plotted for each mixture. When True, ranks should contain a single entry of the rank to plot
                                       If False, all ranks will be plotted on the same plot.
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)

        Outputs:
            None. Generates the 2D plots and stores it in the a created directory.
        """

        assert data_type in ['Testing', 'Training'], "data_type must be either 'Testing' or 'Training'"
        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            Idx = self.testing_indices
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_testing_values(A=A)
        elif data_type == 'Training':
            Idx = self.Idx_known
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_reconstructed_values(A=A)
        
        if ranks is None:
            ranks = self.ranks
            ranks_idx = np.arange(len(ranks))
        else:
            ranks = np.array(ranks).astype(int)
            ranks_idx = np.zeros_like(self.ranks).astype(int)
            for r in ranks:
                ranks_idx += (self.ranks == r).astype(int)
            ranks_idx = np.where(ranks_idx.astype(bool))[0]
        
        for r in ranks: # test if rank is in ranks
            if np.sum(self.ranks == r) == 0:
                print(f'Rank {r} not found')
                return
        
        x2_int = np.array(json.load(open(self.data_file, 'r'))['x2_int'])
        T2_int = np.array(json.load(open(self.data_file, 'r'))['T2_int'])
        x_MC = np.concatenate([x2_int for _ in T2_int])
        T_MC = np.concatenate([t*np.ones_like(x2_int) for t in T2_int])
        jitter = json.load(open(self.data_file, 'r'))['jitter']
        v_MC = np.array(json.load(open(self.data_file, 'r'))['v_MC'])

        K_MC = self.K(x_MC, x_MC, T_MC, T_MC) + (jitter+v_MC)*np.eye(x_MC.shape[0])
        L_MC = np.linalg.cholesky(K_MC)
        L_MC_inv = np.linalg.inv(L_MC)
        K_MC_inv = L_MC_inv.T @ L_MC_inv
        del K_MC, L_MC

        mix_all = np.char.add(np.char.add(self.c_all[Idx[:,0]], ' + '), self.c_all[Idx[:,1]])
        unique_mix, idx = np.unique(mix_all, return_index=True)
        unique_mix = unique_mix[np.argsort(idx)]
        df_UNIFAC = pd.read_excel(self.excel_plots_known)
        UNIFAC_mix = np.char.add(np.char.add(df_UNIFAC['Component 1'].to_numpy().astype(str), ' + '), df_UNIFAC['Component 2'].to_numpy().astype(str))

        exp_mix = np.char.add(np.char.add(data_dict['Component 1'], ' + '), data_dict['Component 2'])

        colours = ['r', 'b', 'magenta', 'y', 'saddlebrown', 'k', 'cyan', 'lime']

        if plot_one:
            png_path = f'{self.path}/{ranks[0]}/2D Plots/{data_type}'
        else:
            png_path = f'{self.path}/2D Plots/{data_type}'

        try:
            os.makedirs(png_path)
        except:
            pass

        for j in range(Idx.shape[0]):
            y_idx = exp_mix == unique_mix[j]
            UNIFAC_idx = UNIFAC_mix == unique_mix[j]
            yy = data_dict['Excess Enthalpy [J/mol]'][y_idx]
            yy_UNIFAC = df_UNIFAC['UNIFAC_DMD [J/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            x_y = data_dict['Composition component 1 [mol/mol]'][y_idx]
            T_y = data_dict['Temperature [K]'][y_idx]
            c1 = self.c_all[Idx[j,0]]
            c2 = self.c_all[Idx[j,1]]
            x_UNIFAC = df_UNIFAC['Composition component 1 [mol/mol]'].to_numpy().astype(float)[UNIFAC_idx]
            T_UNIFAC = df_UNIFAC['Temperature [K]'].to_numpy().astype(float)[UNIFAC_idx]

            K_pred_MC = self.K(x_UNIFAC, x_MC, T_UNIFAC, T_MC)

            if plot_one:
                yy_MC_mean = K_pred_MC @ K_MC_inv @ y_MC_interp[ranks_idx,:,j][0,:]    
            else:
                yy_MC_mean = K_pred_MC @ K_MC_inv @ y_MC_interp[ranks_idx,:,j].T

            T_uniq = np.unique(T_y)
            T_uniq = np.concatenate([T_uniq.astype(int)+0.15,T_uniq.astype(int)+1.15])
            T_uniq = np.unique(T_uniq)
            T_uniq = T_uniq[np.sum(np.abs(T_uniq[:,np.newaxis]-T_y[np.newaxis,:]) <= 0.5, axis=1) > 0]
            for i in range(len(T_uniq)):
                fig, ax = plt.subplots()
                TT = T_uniq[i]
                T_y_idx = np.abs(T_y - TT) <= 0.5
                T_UNIFAC_idx = T_UNIFAC == TT

                if plot_one:
                    T_MC_idx = T2_int == TT 
                    if np.sum(T_MC_idx) > 0:
                        T_MC_idx = np.where(T_MC_idx)[0][0]
                        ax.plot(x2_int, y_MC_interp[ranks_idx,T_MC_idx*x2_int.shape[0]:(T_MC_idx+1)*x2_int.shape[0],j][0,:], '.r', markersize=15, label=f'MC Rec Rank {ranks[0]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx], '-r', label=f'MC Smooth Rank {ranks[0]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], '-g', label='UNIFAC')
                else:
                    for m in range(len(ranks)):
                        ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_MC_mean[T_UNIFAC_idx,m], '-', color=f'{colours[ranks_idx[m]]}', label=f'MC Rank {ranks[m]}')
                    ax.plot(x_UNIFAC[T_UNIFAC_idx], yy_UNIFAC[T_UNIFAC_idx], '--g', label='UNIFAC')
                ax.set_xlabel('Composition of Compound 1 [mol/mol]', fontsize=15)
                ax.set_ylabel('Excess Enthalpy [J/mol]', fontsize=18)
                ax.set_title(f'(1) {c1} + (2) {c2} at {T_uniq[i]:.2f} K', fontsize=13)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.plot(x_y[T_y_idx], yy[T_y_idx], '.k', label='Experimental Data', markersize=15)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.tight_layout()

                fig_path = f'{png_path}/{j}_{i}.png'
                fig.savefig(fig_path, dpi=300)
                plt.clf()
                plt.close()

                clear_output(wait=False)
                print(f'{j}_{i}')
        clear_output(wait=False)

    def plot_func_groups_MC_vs_UNIFAC_with_embedded_box(self, data_type=None, A=None) -> None:
        """
        Description: Plots the functional groups for the MC and UNIFAC predictions. The box plots are embedded in the plot.

        Inputs:
            data_type   : str       -> Type of data to plot. Must be either 'Testing' or 'Training'.
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)

        Outputs:
            None. Generates the 2D plots and stores it in the a created directory.
        """

        assert data_type in ['Testing', 'Training'], "data_type must be either 'Testing' or 'Training'"
        png_path = f'{self.path}/2D Plots/Functional_Group_MC_vs_UNIFAC'
        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            Idx = self.testing_indices
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_testing_values(A=A)
            png_path = f'{png_path}/Testing'
        elif data_type == 'Training':
            Idx = self.Idx_known
            y_MC_interp = self.extract_interps(A=A, Idx=Idx)
            data_dict = self.get_reconstructed_values(A=A)
            png_path = f'{png_path}/Training'

        os.makedirs(png_path, exist_ok=True) # Create directory if it does not exist
        
        # Get unique functional groups
        true_unique_fg, idx = np.unique(self.fg, return_index=True) # Unique functional groups
        true_unique_fg = true_unique_fg[np.argsort(idx)] # Sort to keep same format as listed in excel sheet

        # Indices corresponding to the compounds. Used to extract correct functional groups
        c1_idx = np.sum((data_dict['Component 1'][:, np.newaxis] == self.c_all[np.newaxis, :]) * np.arange(self.c_all.shape[0])[np.newaxis,:], axis=1)
        c2_idx = np.sum((data_dict['Component 2'][:, np.newaxis] == self.c_all[np.newaxis, :]) * np.arange(self.c_all.shape[0])[np.newaxis,:], axis=1)
        
        # Extract functional groups
        fg1 = self.fg[c1_idx]   # Functional groups for component 1
        fg2 = self.fg[c2_idx]   # Functional groups for component 2
        fg_mix = np.char.add(np.char.add(fg1, ' + '), fg2)  # Functional groups for the mixtures
        unique_fg_mix, idx = np.unique(fg_mix, return_index=True)   # Unique functional groups for the mixtures
        unique_fg_mix = unique_fg_mix[np.argsort(idx)]  # Sort to keep same format as listed in excel sheet
        unique_fg_mix_split_testing = [fg.split(' + ') for fg in unique_fg_mix] # Obtain the individual functional groups for the mixtures
        fg_indices = np.array([[np.where(true_unique_fg == ffg[0])[0][0], np.where(true_unique_fg == ffg[1])[0][0]] for ffg in unique_fg_mix_split_testing]) # Get the indices of the functional groups

        max_val = 100 # max value for the difference in metrics
        
        # colour schemes for embedded box plots
        MC_colour = 'g'
        UNI_colour = 'r'

        # Generate plots
        for r in range(len(self.ranks)):
            for metrics in ['MAE', 'MARE']:
                if metrics == 'MAE':
                    # absolute error per datapoint for MC and UNIFAC
                    box_plot = np.column_stack([np.abs(data_dict['MC [J/mol]'][:,r]-data_dict['Excess Enthalpy [J/mol]']), 
                                                np.abs(data_dict['UNIFAC_DMD [J/mol]']-data_dict['Excess Enthalpy [J/mol]'])])
                    # MAE for MC and UNIFAC per combination of functional groups
                    all_means = np.array([np.mean(box_plot[((fg1 == unique_fg_mix_split_testing[i][0]).astype(int) + (fg2 == unique_fg_mix_split_testing[i][1]).astype(int)) == 2, :], axis=0) for i in range(len(unique_fg_mix))])
                    cbar_title = 'Difference in MAE [J/mol]'
                elif metrics == 'MARE':
                    # relative error per datapoint for MC and UNIFAC
                    box_plot = np.column_stack([np.abs((data_dict['MC [J/mol]'][:,r]-data_dict['Excess Enthalpy [J/mol]'])/data_dict['Excess Enthalpy [J/mol]']), 
                                                np.abs((data_dict['UNIFAC_DMD [J/mol]']-data_dict['Excess Enthalpy [J/mol]'])/data_dict['Excess Enthalpy [J/mol]'])])*100
                    # Index where the experimental excess enthalpy is non-zero
                    non_zero_idx = data_dict['Excess Enthalpy [J/mol]'] != 0
                    # MARE for MC and UNIFAC per combination of functional groups with non-zero experimental excess enthalpy removed
                    all_means = np.array([np.mean(box_plot[((fg1 == unique_fg_mix_split_testing[i][0]).astype(int) + (fg2 == unique_fg_mix_split_testing[i][1]).astype(int)) == 2, :][non_zero_idx[((fg1 == unique_fg_mix_split_testing[i][0]).astype(int) + (fg2 == unique_fg_mix_split_testing[i][1]).astype(int)) == 2]], axis=0) for i in range(len(unique_fg_mix))])
                    cbar_title = 'Difference in MARE [%]'
                
                # Difference in metrics between MC and UNIFAC
                diff_means = all_means[:,1] - all_means[:,0]
                # Clip the difference in metrics to -max_val and max_val
                diff_means_clip = np.clip(diff_means, -max_val, max_val)
                # Create matrix for the difference in metrics
                A_diff = np.nan*np.ones((len(true_unique_fg), len(true_unique_fg)))
                # Add the difference in metrics to the matrix
                A_diff[fg_indices[:,0], fg_indices[:,1]] = diff_means_clip

                fig, ax = plt.subplots(1,1,figsize=(10,10))
                # Generate layout for the plot
                ax.set_xlim(-0.5, len(true_unique_fg)-0.5)
                ax.set_ylim(-0.5, len(true_unique_fg)-0.5)
                ax.set_xticks(np.arange(len(true_unique_fg)), true_unique_fg, rotation=90)
                ax.set_yticks(np.arange(len(true_unique_fg)), true_unique_fg)
                # Add grid
                minor_ticks = np.arange(-0.5, len(true_unique_fg)-0.5, 1)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_yticks(minor_ticks, minor=True)
                ax.grid(which='minor', color='k', linestyle='--', linewidth=1, alpha=0.5)
                ax.tick_params(axis='both', which='minor', length=0)
                ax.invert_yaxis() # Invert y-axis

                # Plot the difference in metrics
                im = ax.imshow(A_diff, cmap='RdYlGn', vmin=-max_val, vmax=max_val)
                cbar = fig.colorbar(im, shrink=0.8)
                c_ticks = cbar.get_ticks().astype(int)
                c_tick_labels = c_ticks.astype(str)
                c_tick_labels[0] = f'<{c_tick_labels[0]}'
                c_tick_labels[-1] = f'>{c_tick_labels[-1]}'
                cbar.set_ticks(c_ticks)
                cbar.set_ticklabels(c_tick_labels)
                plt.tight_layout()

                # Embed box plots
                fract = 1/len(true_unique_fg) # lenght of total plot
                offset = 0.17*fract  # offset for the box plots
                ax1 = []    # List to store the axes for the box plots
                for i in range(len(unique_fg_mix)):
                    fg1_idx = fg1 == unique_fg_mix_split_testing[i][0]  # Functional groups of component 1 index
                    fg2_idx = fg2 == unique_fg_mix_split_testing[i][1]  # Functional groups of component 2 index
                    fg_idx = (fg1_idx.astype(int) + fg2_idx.astype(int)) == 2   # Functional groups of the mixture index
                    xx, yy = fg_indices[i,1], len(true_unique_fg)-fg_indices[i,0] - 1   # x and y coordinates for the box plot. y coordinate is inverted due to the inverted y-axis
                    rect = [xx*fract+offset, yy*fract+offset, fract-offset*2, fract-offset*2] # get the rectangle for the box plot
                    box = ax.get_position() # get the position of the main plot
                    width = box.width   # width of the main plot
                    height = box.height # height of the main plot
                    inax_position  = ax.transAxes.transform(rect[0:2]) 
                    transFigure = fig.transFigure.inverted()
                    infig_position = transFigure.transform(inax_position)    
                    x = infig_position[0]
                    y = infig_position[1]
                    width *= rect[2]
                    height *= rect[3]
                    ax1 += [fig.add_axes([x,y,width,height])]

                    # Check for nan's if Relative error with exess enthalpy of zero
                    idx_not_nan = data_dict['Excess Enthalpy [J/mol]'][fg_idx] != 0
                    ax1[-1].boxplot(box_plot[fg_idx,0][idx_not_nan], whis=(0,100), 
                                showmeans=True, meanline=True, meanprops=dict(color='k', linewidth=1.5, linestyle=(0, (1, 1))), 
                                medianprops=dict(color=MC_colour, linewidth=1.5), showfliers=False, 
                                boxprops=dict(color=MC_colour, linewidth=1.5), whiskerprops=dict(color=MC_colour, linewidth=1.5, linestyle='--'), 
                                widths=0.9, positions=[1])
                    ax1[-1].boxplot(box_plot[fg_idx,1][idx_not_nan], whis=(0,100), 
                                showmeans=True, meanline=True, meanprops=dict(color='k', linewidth=1.5, linestyle=(0, (1, 1))), 
                                medianprops=dict(color=UNI_colour, linewidth=1.5), showfliers=False, 
                                boxprops=dict(color=UNI_colour, linewidth=1.5), whiskerprops=dict(color=UNI_colour, linewidth=1.5, linestyle='--'), 
                                widths=0.9, positions=[2])
                    
                    # Cap the y-axis for the box plots at the 20th and 80th percentile and disable the ticks
                    minlim = np.min(np.percentile(box_plot[fg_idx,:][idx_not_nan,:], 20, axis=0))
                    maxlim = np.max(np.percentile(box_plot[fg_idx,:][idx_not_nan,:], 80, axis=0))
                    ax1[-1].set_ylim(minlim, maxlim)
                    ax1[-1].set_xticks([])
                    ax1[-1].set_yticks([])
                    ax1[-1].tick_params(axis='both', which='both', length=0)

                # Grey plots for the lower triangle
                for i in range(len(true_unique_fg)):
                    for j in range(i+1, len(true_unique_fg)):
                        ax.fill_between([i-0.5, i+0.5], [j-0.5, j-0.5], [j+0.5, j+0.5], color='k', alpha=0.3)

                # Add text to the color bar
                # Get the position of the colorbar's axis
                cbar_pos = cbar.ax.get_position()  # Returns a Bbox object

                # Extract the bounding box coordinates: (x0, y0) is the bottom-left corner, and (width, height) are its dimensions
                x0, y0, width, height = cbar_pos.x0, cbar_pos.y0, cbar_pos.width, cbar_pos.height

                # Calculate the vertical positions for the text: bottom, middle, and top of the colorbar
                text_positions = [y0, y0 + height / 2, y0 + 0.9999*height]

                # Add text next to these positions using fig.text
                fig.text(x0 + 0.8*width, text_positions[0], '(UNIFAC Best)', ha='left', va='center')  # Bottom text
                fig.text(x0 + 0.8*width, text_positions[1], '(No difference)', ha='left', va='center')  # Middle text
                fig.text(x0 + 0.8*width, text_positions[2], '(MC Best)', ha='left', va='center')     # Top text
                fig.text(x0 - 0.1*width, text_positions[1], cbar_title, ha='center', va='center', rotation=90)  # Middle text
                
                plot_path = f'{png_path}/{metrics}_Rank_{self.ranks[r]}.png'
                fig.savefig(plot_path, dpi=500, bbox_inches='tight')
                plt.clf()
                plt.close()

                clear_output(wait=False)
                print(f'{metrics}_Rank_{self.ranks[r]} saved')

    def plot_best_rank_vs_sparsity(self, data_type=None, A=None) -> None:
        """
        Description: Plots the best rank vs sparsity for the testing or training mixtures.

        Inputs:
            data_type   : str       -> Type of data to plot. Must be either 'Testing' or 'Training'.
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)

        Outputs:
            None. Generates the 2D plots
        """
        assert data_type in ['Testing', 'Training'], "data_type must be either 'Testing' or 'Training'"

        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            data_dict = self.get_testing_values(A=A)
            Idx = self.testing_indices
        elif data_type == 'Training':
            data_dict = self.get_reconstructed_values(A=A)
            Idx = self.Idx_known

        png_path = f'{self.path}/2D Plots/Best_Rank_vs_Sparsity_{data_type}'

        N = len(self.c_all)
        for metric in ['MAE', 'MARE']:
            if metric == 'MAE':
                diff_MC = np.abs(data_dict['MC [J/mol]'] - data_dict['Excess Enthalpy [J/mol]'][:,np.newaxis])
                MAE_all_comps_ranks = np.array([np.mean(diff_MC[( (data_dict['Component 1'] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'] == self.c_all[idx[1]]).astype(int) ) == 2], axis=0) for idx in Idx])
                plot_path = f'{png_path}_MAE.png'
            elif metric == 'MARE':
                diff_MC = np.abs(data_dict['MC [J/mol]'] - data_dict['Excess Enthalpy [J/mol]'][:,np.newaxis])/np.abs(data_dict['Excess Enthalpy [J/mol]'][:,np.newaxis])*100
                idx_non_zero = data_dict['Excess Enthalpy [J/mol]'] != 0
                MAE_all_comps_ranks = np.array([np.mean(diff_MC[idx_non_zero][( (data_dict['Component 1'][idx_non_zero] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'][idx_non_zero] == self.c_all[idx[1]]).astype(int) ) == 2], axis=0) for idx in Idx])
                plot_path = f'{png_path}_MARE.png'
            min_MAE_ranks_idx = np.argmin(MAE_all_comps_ranks, axis=1)
            A_ranks_idx = np.nan*np.ones((N,N))
            A_ranks_idx[Idx[:,0], Idx[:,1]] = min_MAE_ranks_idx + 0.5
            rr, counts = np.unique(min_MAE_ranks_idx, return_counts=True)
            # Set count to zero if rank not encountered
            idx_ranks = (np.sum(self.ranks[rr][:,np.newaxis] == self.ranks[np.newaxis,:], axis=0) == 0)
            for i in range(np.sum(idx_ranks)):
                iidx = np.where(idx_ranks)[0][i]
                rr = np.insert(rr, iidx, np.arange(len(self.ranks))[iidx])
                counts = np.insert(counts, iidx, 0)

            fig,ax = plt.subplots(figsize=(10,10))

            cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'black'])
            norm = BoundaryNorm(np.arange(len(self.ranks)+1), len(self.ranks)+1)
            im = ax.imshow(A_ranks_idx, cmap=cmap, norm=norm)
            cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(self.ranks))+0.5, shrink=0.8)
            cticks = [f'{self.ranks[i]} ({counts[i]})' for i in range(len(self.ranks))]
            cbar.set_ticklabels(cticks, fontsize=10)

            A_grey = np.nan*np.eye(N)
            for i in range(N):
                for j in range(i,N):
                    A_grey[j,i] = 0.25
                
            ax.imshow(A_grey, cmap='Greys',vmin=0,vmax=1)

            unique_fg, idx, counts = np.unique(self.fg, return_index=True, return_counts=True)
            unique_fg = unique_fg[np.argsort(idx)]
            counts = counts[np.argsort(idx)]
            counts[0]=counts[0]-1
            counts = counts

            end_points = [0]
            for count in np.cumsum(counts):
                count += 0.5
                end_points += [count]
                ax.plot([count, count], [0, N-1], '--k', alpha=0.3)
                ax.plot([0, N-1], [count, count], '--k', alpha=0.3)

            if data_type == 'Testing':
                ax.plot(self.Idx_known[:,1], self.Idx_known[:,0], '*k', markersize=5, alpha=0.2, label='Training Data')
                ax.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.07))

            mid_points = (np.array(end_points[:-1])+np.array(end_points[1:]))/2
            ax.set_xticks(mid_points, unique_fg, rotation=90, fontsize=12)
            ax.set_yticks(mid_points, unique_fg, fontsize=12)

            # Add text to the color bar
            # Get the position of the colorbar's axis
            cbar_pos = cbar.ax.get_position()  # Returns a Bbox object

            # Extract the bounding box coordinates: (x0, y0) is the bottom-left corner, and (width, height) are its dimensions
            x0, y0, width, height = cbar_pos.x0, cbar_pos.y0, cbar_pos.width, cbar_pos.height

            # Calculate the vertical positions for the text: bottom, middle, and top of the colorbar
            text_positions = [y0, y0 + height / 2, y0 + 0.9999*height]

            cbar_title = 'Best performing rank (Count)'
            fig.text(x0 - 0.1*width, text_positions[1], cbar_title, ha='center', va='center', rotation=90, fontsize=15)  # Middle text

            fig.savefig(plot_path, dpi=500, bbox_inches='tight')

            plt.clf()
            plt.close()

            clear_output(wait=False)

    def plot_MC_vs_UNIFAC_sparsity(self, data_type=None, A=None) -> None:
        """
        Description: Plots the MC vs UNIFAC metric sparsity plots per compound for the testing or training mixtures.

        Inputs:
            data_type   : str       -> Type of data to plot. Must be either 'Testing' or 'Training'.
            A           : np.array  -> A tensor of size ranks x num_temperatures x num_compositions/2 x num_compounds x num_compounds (Optional)

        Outputs:
            None. Generates the 2D plots
        """
        assert data_type in ['Testing', 'Training'], "data_type must be either 'Testing' or 'Training'"

        if A is None:
            A = self.get_tensors()
        if data_type == 'Testing':
            data_dict = self.get_testing_values(A=A)
            Idx = self.testing_indices
        elif data_type == 'Training':
            data_dict = self.get_reconstructed_values(A=A)
            Idx = self.Idx_known

        png_path = f'{self.path}/2D Plots/MC_vs_UNIFAC/{data_type}'
        os.makedirs(png_path, exist_ok=True) # Create directory if it does not exist
        max_val = 100 # max value for the difference in metrics

        N = len(self.c_all)
        for r in range(len(self.ranks)):
            for metric in ['MAE', 'MARE']:
                if metric == 'MAE':
                    diff_MC = np.abs(data_dict['MC [J/mol]'][:,r] - data_dict['Excess Enthalpy [J/mol]'])
                    diff_UNI = np.abs(data_dict['UNIFAC_DMD [J/mol]'] - data_dict['Excess Enthalpy [J/mol]'])
                    mean_MC_err = np.array([np.mean(diff_MC[( (data_dict['Component 1'] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'] == self.c_all[idx[1]]).astype(int) ) == 2]) for idx in Idx])
                    mean_UNI_err = np.array([np.mean(diff_UNI[( (data_dict['Component 1'] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'] == self.c_all[idx[1]]).astype(int) ) == 2]) for idx in Idx])
                    plot_path = f'{png_path}/MAE_{self.ranks[r]}.png'
                    cbar_title = 'Difference in MAE [J/mol]'
                elif metric == 'MARE':
                    diff_MC = np.abs(data_dict['MC [J/mol]'][:,r] - data_dict['Excess Enthalpy [J/mol]'])/np.abs(data_dict['Excess Enthalpy [J/mol]'])*100
                    diff_UNI = np.abs(data_dict['UNIFAC_DMD [J/mol]'] - data_dict['Excess Enthalpy [J/mol]'])/np.abs(data_dict['Excess Enthalpy [J/mol]'])*100
                    idx_non_zero = data_dict['Excess Enthalpy [J/mol]'] != 0
                    mean_MC_err = np.array([np.mean(diff_MC[idx_non_zero][( (data_dict['Component 1'][idx_non_zero] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'][idx_non_zero] == self.c_all[idx[1]]).astype(int) ) == 2]) for idx in Idx])
                    mean_UNI_err = np.array([np.mean(diff_UNI[idx_non_zero][( (data_dict['Component 1'][idx_non_zero] == self.c_all[idx[0]]).astype(int) + (data_dict['Component 2'][idx_non_zero] == self.c_all[idx[1]]).astype(int) ) == 2]) for idx in Idx])
                    plot_path = f'{png_path}/MARE_{self.ranks[r]}.png'
                    cbar_title = 'Difference in MARE [%]'
                
                mean_diff = mean_UNI_err - mean_MC_err
                mean_diff[mean_diff > max_val] = max_val
                mean_diff[mean_diff < -max_val] = -max_val

                A_diff = np.nan*np.ones((N,N))
                A_diff[Idx[:,0], Idx[:,1]] = mean_diff

                fig,ax = plt.subplots(figsize=(10,10))

                im = ax.imshow(A_diff, cmap='RdYlGn', vmin=-max_val, vmax=max_val)
                cbar = fig.colorbar(im, shrink=0.8)
                c_ticks = cbar.get_ticks().astype(int)
                c_tick_labels = c_ticks.astype(str)
                c_tick_labels[0] = f'<{c_tick_labels[0]}'
                c_tick_labels[-1] = f'>{c_tick_labels[-1]}'
                cbar.set_ticks(c_ticks)
                cbar.set_ticklabels(c_tick_labels)

                A_grey = np.nan*np.eye(N)
                for i in range(N):
                    for j in range(i,N):
                        A_grey[j,i] = 0.25
                    
                ax.imshow(A_grey, cmap='Greys',vmin=0,vmax=1)

                unique_fg, idx, counts = np.unique(self.fg, return_index=True, return_counts=True)
                unique_fg = unique_fg[np.argsort(idx)]
                counts = counts[np.argsort(idx)]
                counts[0]=counts[0]-1
                counts = counts

                end_points = [0]
                for count in np.cumsum(counts):
                    count += 0.5
                    end_points += [count]
                    ax.plot([count, count], [0, N-1], '--k', alpha=0.3)
                    ax.plot([0, N-1], [count, count], '--k', alpha=0.3)

                if data_type == 'Testing':
                    ax.plot(self.Idx_known[:,1], self.Idx_known[:,0], '*k', markersize=5, alpha=0.2, label='Training Data')
                    ax.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.07))

                mid_points = (np.array(end_points[:-1])+np.array(end_points[1:]))/2
                ax.set_xticks(mid_points, unique_fg, rotation=90, fontsize=12)
                ax.set_yticks(mid_points, unique_fg, fontsize=12)

                # Add text to the color bar
                # Get the position of the colorbar's axis
                cbar_pos = cbar.ax.get_position()  # Returns a Bbox object

                # Extract the bounding box coordinates: (x0, y0) is the bottom-left corner, and (width, height) are its dimensions
                x0, y0, width, height = cbar_pos.x0, cbar_pos.y0, cbar_pos.width, cbar_pos.height

                # Calculate the vertical positions for the text: bottom, middle, and top of the colorbar
                text_positions = [y0, y0 + height / 2, y0 + 0.9999*height]

                # Add text next to these positions using fig.text
                fig.text(x0 + 0.8*width, text_positions[0], '(UNIFAC Best)', ha='left', va='center')  # Bottom text
                fig.text(x0 + 0.8*width, text_positions[1], '(No difference)', ha='left', va='center')  # Middle text
                fig.text(x0 + 0.8*width, text_positions[2], '(MC Best)', ha='left', va='center')     # Top text
                fig.text(x0 - 0.1*width, text_positions[1], cbar_title, ha='center', va='center', rotation=90, fontsize=15)  # Middle text

                fig.savefig(plot_path, dpi=500, bbox_inches='tight')

                plt.clf()
                plt.close()

                clear_output(wait=False)
    
