import numpy as np
from itertools import product
from sunscan.params import SCANNER_PARAMETER_MAP


def get_parameter_lists(parameters_optimize: list, parameter_guess: dict, parameter_bounds: dict, parameter_map: dict):
    """ Generate lists with initial guess and bounds.
        Parameters which should not be optimized will be fixed to the initial guess.
    """
    init_guess_list=[0.0]*len(parameter_map)
    bounds_list=[(0,0)]*len(parameter_map)
    for par, idx in parameter_map.items():
        init_guess_list[idx] = parameter_guess[par]
        if par not in parameters_optimize:
            bounds_list[idx] = (parameter_guess[par], parameter_guess[par])  # Fix parameters not being optimized
        else:
            bounds_list[idx] = (parameter_bounds[par][0]+parameter_guess[par], parameter_bounds[par][1]+parameter_guess[par])
    return init_guess_list, bounds_list



def optimize_brute_force(bounds, optimize_function, optimize_args, init_guess=None, points=3):
    steps=np.linspace(0, 1, points+2)[1:-1]  # Use a finer grid for brute force optimization
    # steps = np.linspace(0,1,8)
    values = []
    for i, (low, high) in enumerate(bounds):
        if low == high:
            values.append([low])
        else:
            values.append(low+steps*(high-low))
    param_combinations = list(product(*values))
    if init_guess is not None:
        param_combinations.append(init_guess)
    best_rmse, best_params = float('inf'), None
    for params in param_combinations:
        rmse = optimize_function(params, *optimize_args)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params, best_rmse