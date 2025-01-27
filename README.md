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

## Usage
get_json_data.py converts excess enthalpy data from a .xlsx file into json format compatable with stan
compile_stan_models.py writes .stan files to the Stan Models directory and compiles executable stan programs callable from python using cmdstanpy
Pure_PMF.py under Pure RK PMF - 298 and Pure RK PMF are the python files used for optimization for the different conditions
