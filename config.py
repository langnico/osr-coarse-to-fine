import os
# ----------------------
# PROJECT ROOT DIR
# ----------------------
parent_dir = os.getenv('HOME')
project_root_dir = parent_dir + '/code/osr-coarse-to-fine/'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = parent_dir + '/experiments/osr_coarse_to_fine/'        # directory to store experiment output (checkpoints, logs, etc)

# --- BELOW ARE RELATIVE PATHS ---

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = exp_root + '/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = exp_root + '/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
inat_2021_root = parent_dir + '/data/inat_2021'
inat21_osr_splits = project_root_dir + 'datasets/inat21_osr_splits'

