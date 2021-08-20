import numpy as np, pandas as pd
import utils
import sys



def get_initialized_ts_params_for_effect(effect_params, num_time_steps, effect_dim_size=1):
    effect_name = effect_params['Effect_Name'].values[0]
    dist_type = effect_params['Initialization_Dist_Type'].values[0]
    effect_param1 = effect_params['Initialization_Dist_Parameter1'].values[0]
    effect_param2 = effect_params['Initialization_Dist_Parameter2'].values[0]
    rand_walk_fact = effect_params['Root_Monthly_Rand_Walk_Factor'].values[0]
    renormalize_val = effect_params['Re-normalize_value'].values[0]
    # print(rand_walk_fact, effect_param1, effect_param2, renormalize_val); sys.exit()
    
    dist_sampler_func = utils.get_sampling_func(dist_type)
    
    starting_val = dist_sampler_func(effect_dim_size, effect_param1, effect_param2)

    if not np.isnan(renormalize_val) and np.isscalar(renormalize_val):
        starting_val = starting_val * renormalize_val / starting_val.mean()

    effect_vals = get_random_walk_vals(dist_type, num_time_steps, starting_val, rand_walk_fact, renormalize_val)
    # if effect_name == 'Day_of_the_week':
    #     print(effect_name, starting_val, effect_vals)
    #     sys.exit()

    # bit of a hack! need to handle this better. 
    if effect_name == 'Time_of_the_day': 
        # print(effect_name, starting_val, effect_vals.shape); sys.exit()
        effect_vals = modify_time_of_day_factors(effect_vals)
        # print(effect_vals.shape) ; sys.exit()
    return effect_vals



def get_random_walk_vals(noise_dist_type, num_rows, starting_val, rand_walk_fact, renormalize_val): 

    dist_sampler_func = utils.get_sampling_func(noise_dist_type)
    D = starting_val.shape[0]
    current_val = starting_val
    random_walk_vals = []
    for _ in range(num_rows):
        random_walk_vals.append(np.array(current_val))
        
        if noise_dist_type == 'normal':  
            noise_vals = dist_sampler_func(D)   
            noise_vals_scaled = starting_val * rand_walk_fact * noise_vals / 3.
        elif noise_dist_type == 'uniform':
            noise_vals = dist_sampler_func(D, mean=0., half_range=0.5)
            noise_vals_scaled = starting_val * rand_walk_fact * noise_vals 
        else: 
            raise Exception(f"Sorry, cannot recognize distribution type {noise_dist_type} for ts param initialization.") 
        
        current_val = current_val + noise_vals_scaled
        if not np.isnan(renormalize_val) and np.isscalar(renormalize_val):
            current_val = current_val * renormalize_val / current_val.mean()

    random_walk_vals = np.vstack(random_walk_vals)
    # print(f"random_walk_vals:{random_walk_vals.shape}")
    return np.round(random_walk_vals, 4)


def get_shifted_mean(mean, max_child_mean_shift_factor):
    # half_range = mean * max_child_mean_shift_factor
    # rand_shifted_mean = mean - half_range  + 2 * half_range * (1. - np.random.rand()) 
    half_range = max_child_mean_shift_factor
    rand_shifted_mean = utils.get_random_normal(1, mean, half_range)
    return rand_shifted_mean[0]


def get_correlated_series(x, correl, mu_y, sigma_y):
        mu_x = x.mean()
        sigma_x = x.std()
        
        # reference: http://home.iitk.ac.in/~zeeshan/pdf/The%20Bivariate%20Normal%20Distribution.pdf
        if sigma_x < 1e-5:
            mu_y_given_x = mu_y + correl * (x - mu_x)
        else:
            mu_y_given_x = mu_y + correl * (sigma_y / sigma_x) * (x - mu_x)

        if sigma_y < 1e-5:
            sigma_y_sq_given_x = ( 1 - correl * correl) * 1e-5
        else:
            sigma_y_sq_given_x = ( 1 - correl * correl) * sigma_y * sigma_y
        sigma_y_given_x = np.sqrt(sigma_y_sq_given_x)
        y = np.random.normal(loc = mu_y_given_x, scale = sigma_y_given_x, size=len(x))   
        return y


def modify_time_of_day_factors(effect_vals):
    def tod_correlator(x):
        mu_x = x.mean()
        sigma_x = x.std()
        mu_y = mu_x
        sigma_y = sigma_x
        y = get_correlated_series( x, correl=0.9, mu_y = mu_y, sigma_y = sigma_y)
        y = np.clip(y, a_min = 1e-4, a_max = None)
        return y
    tod_factors = [effect_vals]
    for _ in range(6):
        tod_factors.append(np.apply_along_axis(tod_correlator, axis = 1, arr = effect_vals))
    tod_factors = np.concatenate(tod_factors, axis=1)
    # print(tod_factors.shape) ; sys.exit()
    return tod_factors


def get_correlated_effect(effect, effect_params, parent_ts_params, correl):
    max_child_mean_shift_factor = effect_params['Max_Child_Mean_Shift_Factor'].values[0]
    renormalize_val = effect_params['Re-normalize_value'].values[0]
    lower_bound = effect_params['Lower_Bound'].values[0]
    # print(parent_ts_params) ;  sys.exit()

    x = parent_ts_params
    if parent_ts_params.shape[1] == 1: # 1d array reshaped to be 2d
        x = x.flatten().reshape((-1,1))
    
    def wrap_correlator(x):
        mu_x = x.mean()
        sigma_x = x.std()

        mu_y = get_shifted_mean(mu_x, max_child_mean_shift_factor)
        sigma_y_mult = 1.1
        sigma_y =  sigma_x * sigma_y_mult
        
        y = get_correlated_series( x, correl, mu_y, sigma_y)
        y = np.clip(y, a_min = lower_bound, a_max = None)
        return y
    
    y = np.apply_along_axis(wrap_correlator, axis = 0, arr = x)
    if not np.isnan(renormalize_val) and np.isscalar(renormalize_val):
        y = y * renormalize_val
        y = y / y.mean(axis=1)[:, None]

    # print(np.corrcoef(x.flatten(), y.flatten()))
    # sys.exit()
    return y
