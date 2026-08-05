import numpy as np

# PQFM circuit parameters
N_QUBITS = 109          # number of qubits
N_CLASSICAL_FEATURES = 216  # input classical feature dimension
N_PQFM_FEATURES = 327  # 3 * N_QUBITS (X,Y,Z per qubit)
N_PAIRS = N_QUBITS - 1  # 108 nearest-neighbor coupling pairs

# Circuit variants
SHORT_CIRCUIT_CONFIG = {'blocks': 1, 'alpha': 1.0, 'fiducial_seed': 1}
LONG_CIRCUIT_CONFIG  = {'blocks': 2, 'alpha': 0.1, 'fiducial_seed': 0}

# Simulation qubit count for quick tests (< full 109)
SIM_N_QUBITS_SMALL = 15  # fast test mode
SIM_N_FEATURES_SMALL = 28  # 2*(15-1)
SIM_N_OUTPUT_SMALL = 45   # 3*15

# Sampling
N_SAMPLE = 16000
ACTIVE_WINDOW_START = '2024-07-24'
ACTIVE_WINDOW_END   = '2024-10-29'
FULL_PERIOD_START   = '2023-09-01'
FULL_PERIOD_END     = '2024-10-29'
RANDOM_SEED_SAMPLING = 42

# Backtesting
TRAINING_SIZES = [500, 1000, 1500, 2000]
BLINDING_DAYS  = [0, 1, 2, 3, 4]
MIN_TEST_EVENTS = 50

# Shots for hardware
N_SHOTS = 4096

# CQEM
N_BINS_LIST = [4, 10, 30, 60]

# Model hyperparameter grids (exact from paper)
LR_PARAM_GRID = {
    'solver': ['lbfgs','sag','saga','newton-cholesky','newton-cg','liblinear'],
    'C': list(10**np.array([-4,-3.2,-2.4,-1.6,-0.8,0,0.8,1.6,2.4,3.2,4.0])),
    'penalty': ['l2'],
    'max_iter': [10000],
}
XGB_PARAM_GRID = {
    'max_depth': [3,5,7,9,11],
    'n_estimators': [80,100,120,140,160,180],
    'learning_rate': [0.15,0.1,0.05,0.01],
}
RF_PARAM_GRID = {
    'criterion': ['gini','entropy','log_loss'],
    'n_estimators': [80,100,120,140,160,180],
}
NN_PARAM_GRID = {
    'hidden_layer_sizes': [(3,),(100,),(164,),(164,65),(262,164,98),(262,196,131,65),(262,196,131,65,3),(196,3)],
    'activation': ['relu','logistic','tanh'],
    'learning_rate': ['constant','invscaling','adaptive'],
    'max_iter': [10000],
}
