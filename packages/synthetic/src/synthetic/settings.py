import math

N = 100
QMAX = 1.0
DATA_SEED = 1
EXPERIMENT_SEED = 2
N_TRIALS = 500

GAMMAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
FIXED_GAMMA = 0.30
BUDGET_SCALES = (1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00)
LINEAR_GAMMA_BUDGET_SCALE = 4.00
POWER_BUDGET_SCALE = 1.00
BETA_TRUE_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
BETA_TRUE = 0.50
BETA = 0.50
DELTAS = (0.05,)
C_CVAR = math.sqrt(15)
REGRET_COMPARISON_BUDGET_SCALE = max(LINEAR_GAMMA_BUDGET_SCALE, POWER_BUDGET_SCALE)

M, DISTRIBUTIONS = 4, [
    {
        "type": "truncated_normal",
        "mu": 0.50,
        "sigma": 0.10,
        "trunc_left": 0.0,
        "trunc_right": QMAX,
    },
    {
        "type": "truncated_normal_mixture",
        "weights": [0.5, 0.5],
        "mus": [0.35, 0.75],
        "sigmas": [0.07, 0.07],
        "trunc_lefts": [0.0, 0.0],
        "trunc_rights": [QMAX, QMAX],
    },
    {
        "type": "truncated_normal_mixture",
        "weights": [0.5, 0.5],
        "mus": [0.45, 0.65],
        "sigmas": [0.07, 0.13],
        "trunc_lefts": [0.0, 0.0],
        "trunc_rights": [QMAX, QMAX],
    },
    {
        "type": "truncated_normal_mixture",
        "weights": [0.5, 0.5],
        "mus": [0.35, 0.65],
        "sigmas": [0.05, 0.08],
        "trunc_lefts": [0.0, 0.0],
        "trunc_rights": [QMAX, QMAX],
    },
]