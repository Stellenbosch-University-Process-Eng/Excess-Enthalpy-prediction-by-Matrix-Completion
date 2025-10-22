# Pure_MC
Respository for Article: **Prediction of excess enthalpy in binary mixtures through probabilistic matrix completion, with a GP enforced smoothness constraint** (#insert doi) by 
Garren Hermanus, Tobi Louw and Jamie Cripwell


## Prerequisites
The code here makes use of [Stan](https://github.com/stan-dev) to perform optimization, through the python wrapper [cmstanpy](https://github.com/stan-dev/cmdstanpy). 

### Installation of cmdstanpy
The easiets method of installing stan and cmdstanpy is through anaconda. 
1. Create a new virtual environment:
   
   `conda create -n cmdstanpy`
   
2. Install stan through the conda-forge channel
   
   `conda install -c conda-forge cmdstan`
   
3. Install python, cmdstanpy, numpy, matplotlib, pandas through the conda-forge channel
   
   `conda install -c conda-forge python cmdstanpy numpy matplotlib pandas`

UNIFAC predictions were generated using [thermo](https://github.com/CalebBell/thermo) which can be installed in a different environment using conda 

`conda install -c conda-forge thermo`

## Description of files
1. get_json_data.py converts excess enthalpy data from a .xlsx file into json format compatable with stan
2. compile_stan_models.py writes .stan files to the Stan Models directory and compiles executable stan programs callable from python using cmdstanpy
3. Pure_PMF.py under Pure RK PMF - 298 and Pure RK PMF are the python files used for optimization for the different conditions
4. Post_procs.py contains methods for the post-processing of results given the .csv output from stan
5. All .ipynb files contains examples of usage of the methods in Post_procs.py to obtain intepretable results

## Files/Folders needed to run
```
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
```

