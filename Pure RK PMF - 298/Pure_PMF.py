import numpy as np # type: ignore
import json
import os

# change stan tmpdir to home. Just a measure added for computations on the HPC which does not 
# like writing to /tmp
old_tmp = os.environ['TMPDIR'] # save previous tmpdir
os.environ['TMPDIR'] = '/home/22796002' # update tmpdir

import cmdstanpy # type: ignore

os.environ['TMPDIR'] = old_tmp # change back to old_tmp

import sys

include_clusters = bool(int(sys.argv[1])) # True if we need to include cluster
add_zeros = bool(int(sys.argv[2])) # True if we need to add zeros
refT = False # always false for 298.15 K
chain_id = int(sys.argv[3]) # chain id
num_non_zero_feat_var = 1+2*(chain_id+1) # number of non-zero feature values

# data file
data_file = f'data.json'
path = f'Results/Include_clusters_{include_clusters}/Add_zeros_{add_zeros}'

os.makedirs(path, exist_ok=True)

# Adjust data to reflect number of non-zero feature variances
data = json.load(open(data_file, 'r'))
data['D'] = int(num_non_zero_feat_var)
data['v_features'] = np.array([100 for _ in range(num_non_zero_feat_var)])

# select stan models with and without ARD
stan_file = f'/home/22796002/Pure_MC/Stan Models/Pure_PMF_include_clusters_{include_clusters}_zeros_{add_zeros}_refT_{refT}.stan'
model = cmdstanpy.CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS': True})

# set number of chains and threads per chain
chains = 1
threads_per_chain = 1

# Total threads for stan to use for parallel computations
os.environ['STAN_NUM_THREADS'] = str(int(threads_per_chain*chains))

# Directories to store output
output_dir = f'{path}/{num_non_zero_feat_var}'

os.makedirs(output_dir, exist_ok=True)

# Directory of init file
inits = f'{output_dir}/inits.json'
init = {}

# Generate inits file
prev_files = [f'{output_dir}/{f}' for f in os.listdir(output_dir) if f.endswith('.csv') or f.endswith('.txt')]
if prev_files: # if files are present, use the last file to generate inits
    csv_file = [f for f in prev_files if f.endswith('.csv')][-1]
    MAP = model.from_csv(csv_file)
    keys = list(MAP.stan_variables().keys())
    for key in keys:
        init[key] = MAP.stan_variables()[key].tolist()
    del MAP # clear memory
    for f in prev_files:
        os.remove(f) # remove files
with open(inits, 'w') as f:
    json.dump(init, f) # save inits to file
del init # clear memory

print('Running Pure PMF Model with the following conditions:')
print(f'Include clusters: {include_clusters}')
print(f'Zeros: {add_zeros}')
print(f'Reference temperature: {refT}')

MAP = model.optimize(data=data, inits=inits, show_console=True,  iter=1000000, refresh=1000, 
                algorithm='lbfgs', jacobian=False, tol_rel_grad=1e-20, tol_rel_obj=1e-20, tol_param=1e-10,
                tol_grad=1e-20, output_dir=output_dir, init_alpha=1e-20)

print('MAP estimate complete')
print(f'Final lp__: {MAP.optimized_params_dict['lp__']}')