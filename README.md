# Pure_MC
Respository for Article: #insert article name and doi

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
