
import numpy as np, copy, pandas as pd, random
import copy, sys, os
import math
import variables as vars
import utils 
import time_scale2 as ts
from tree_traverse import TreeTraverser
import ts_functions as ts_params_f
from datetime import datetime



class TS_Generator_Tree():
    
    IRREGULAR_EFFECT_TYPES = ['anomaly', 'level_shift', 'missing_data']

    def __init__(self, rand_seed = None):
        super().__init__()
        self._org_tree = None
        self._act_tree_template = None
        self._org_node_count = None
        self._act_node_count = None
        self._history_dates = None
        self._special_day_ids = None
        self._history_months = None
        self._hierarchy_correls = None
        self._effects = vars.ts_regular_effect_types
        self._magnitudes = {}
        self._daily_time_intervals = None  

        if type(rand_seed) == int or type(rand_seed) == float: self._rand_seed = rand_seed
        else:  self._rand_seed = 42

        self._set_seeds()


    @property
    def org_tree(self): return self._org_tree

    @property
    def act_tree_template(self): return self._act_tree_template

    @property
    def history_dates(self): return self._history_dates

    @property
    def history_months(self): return self._history_months

    @property
    def special_day_ids(self): return self._special_day_ids

    @property
    def effects(self): return self._effects

    @property
    def magnitudes(self): return self._magnitudes

    @property
    def org_node_count(self): return self._org_node_count

    @property
    def act_node_count(self): return self._act_node_count


    def _set_seeds(self):        
        os.environ['PYTHONHASHSEED']=str(self._rand_seed)
        random.seed(self._rand_seed)
        np.random.seed(self._rand_seed)


    def generate_ts_params(self, data_gen_params):  
        if vars.DEBUG: print("Starting data generation...")   
        if vars.DEBUG: print("Generating org and act trees...")
           
        self._data_gen_params = data_gen_params
        self._org_tree, self._act_tree_template = utils.get_org_and_act_trees( data_gen_params['hierarchy_params'] )
        
        self._org_node_count = self._org_tree.node_count
        self._act_node_count = self._act_tree_template.node_count
        # print(self._org_node_count, self._act_node_count) ; sys.exit()

        if vars.DEBUG: print("Generating relative volume magnitudes at org and act nodes...")
        self.generate_magnitudes_for_org_and_act_nodes()

        if vars.DEBUG: print("Generating history time scale...")
        self._history_dates, self._special_day_ids = ts.generate_time_scale(data_gen_params, self._rand_seed)   
        self._history_months = ts.get_history_months(self._history_dates) 
        # if vars.DEBUG: print(self._history_months); sys.exit()

        self._effects['Time_of_the_day']['len']  = ts.get_num_time_of_day_intervals(
            self._data_gen_params['main_params']['Data_granularity_in_minutes'] )
        # if vars.DEBUG: print(self._effects['Time_of_the_day']['len'])
        
        self._daily_time_intervals = ts.get_daily_time_intervals(
            self._data_gen_params['main_params']['Data_granularity_in_minutes'] )

        if vars.DEBUG: print("Generating open hours...")  
        self._generate_open_hours_for_org_nodes()

        if vars.DEBUG: print("Assigning activity trees to each org node...")  
        self._assign_activity_tree_to_org_nodes()  

        self._effects['Special_Days']['len'] = ts.get_num_special_days( self._history_dates )
        # if vars.DEBUG: print(self._effects['Special_Days']['len'])

        # for node in TreeTraverser.traverse_pre_order(self._act_tree_template): if vars.DEBUG: print(node)
        # sys.exit()
        
        self.initialize_ts_factors_for_root()
        self._cache_hierarchy_correlations()
        self.propagate_ts_factors() 

    
    def generate_magnitudes_for_org_and_act_nodes(self):
        
        def get_dist_params(df, param_name):
            idx = df['Parameter_Type'] == param_name
            dist_type = df.loc[idx, 'Distribution_Type'].values[0]
            param1 = df.loc[idx, 'Parameter1'].values[0]
            param2 = df.loc[idx, 'Parameter2'].values[0]
            return dist_type, param1, param2

        # set relative activity magnitude factors
        dist_type, param1, param2 = get_dist_params(self._data_gen_params['magnitude'], 'Activity_Relative_Scaler')
        # if vars.DEBUG: print(dist_type, param1, param2)
        self._set_magnitudes_for_nodes('act', dist_type, param1, param2)

        # set relative org magnitudes
        dist_type, param1, param2 = get_dist_params(self._data_gen_params['magnitude'], 'Org_Magnitude')
        self._set_magnitudes_for_nodes('org', dist_type, param1, param2)

        # set relative org + actmagnitudes
        dist_type, param1, param2 = get_dist_params(self._data_gen_params['magnitude'], 'Org_and_Activity_Scaler')
        self._set_magnitudes_for_nodes('org_and_act', dist_type, param1, param2)        
        
        
    def _set_magnitudes_for_nodes(self, magnitude_fact_type, dist_type, param1, param2):

        if magnitude_fact_type == 'org':
            nodes = [ node.id for node in TreeTraverser.traverse_pre_order_leaves_only(self._org_tree) ]
        elif magnitude_fact_type == 'act':
            nodes = [ node.id for node in TreeTraverser.traverse_pre_order_leaves_only(self._act_tree_template) ]
        elif magnitude_fact_type == 'org_and_act':
            nodes = [ (org_node.id, act_node.id) 
                for org_node in TreeTraverser.traverse_pre_order_leaves_only(self._org_tree)
                for act_node in TreeTraverser.traverse_pre_order_leaves_only(self._act_tree_template)   ]
        else: 
            raise Exception(f"Sorry can't recognize magnitude_fact_type: {magnitude_fact_type}")

        dist_sampler_func = utils.get_sampling_func(dist_type)
        rel_factors = dist_sampler_func(len(nodes), param1, param2)
        rel_factors = np.clip(rel_factors, a_min=0.001, a_max=1000000.)

        self._magnitudes[magnitude_fact_type] = { 
            node:  rel_factors[i] for i, node in enumerate(nodes) }

    
    def _get_magnitude_for_org_and_act(self, org_id, act_id):
        # org magnitude
        org_factor = self._magnitudes['org'][org_id]
        # org magnitude
        act_factor = self._magnitudes['act'][act_id]
        # org magnitude
        org_and_act_factor = self._magnitudes['org_and_act'][(org_id, act_id) ]
        # if vars.DEBUG: print(org_factor, act_factor, org_and_act_factor)
        return org_factor * act_factor * org_and_act_factor


    def initialize_ts_factors_for_root(self):
        org_root = next(TreeTraverser.traverse_pre_order(self._org_tree))
        act_root = next(TreeTraverser.traverse_pre_order( org_root.properties['act_tree']))

        num_months = self._history_months.shape[0]
        
        # initialize the activity root parameters
        for effect in self._effects:
            idx = self._data_gen_params['regular_effects']['Effect_Name'] == effect
            if sum(idx) == 0:
                if vars.DEBUG: print(f"No parameters found for effect type: {effect}.")
                continue
            param_row = self._data_gen_params['regular_effects'].loc[idx]

            initialized_vals = ts_params_f.get_initialized_ts_params_for_effect(
                param_row, num_months, self._effects[effect]['len'] )
                
            act_root.properties[effect] = initialized_vals
        act_root.properties['ts_params_generated'] = True
        # if vars.DEBUG: print(act_root.properties['Level'])
        # if vars.DEBUG: print(act_root.properties['Time_of_the_day'].mean(axis=0))
        # sys.exit()

    
    def _get_parent_ts_params(self, org_node, act_node):
        if org_node.parent_node is None: 
            # this is org root node. Get parent activity node's ts params within this same org node
            ts_params = act_node.parent_node.properties
            parent_type = act_node.parent_node.type
        else:
            # get the parent org's ts params for the same activity
            act_node_from_parent_org = org_node.parent_node.properties['act_tree']\
                .get_node_by_id_from_tree(act_node.id)
            ts_params = act_node_from_parent_org.properties
            parent_type = org_node.type

        # if vars.DEBUG: print(act_node.id, ts_params.keys()) ; sys.exit()
        return ts_params, parent_type


    def _cache_hierarchy_correlations(self):
        self._hierarchy_correls = {}
        for k, v in self._data_gen_params['hierarchy_params']['org_hier_params'].items():
            self._hierarchy_correls[ ( 'Org', k+1 ) ] = float(v['level_down_correl'])
        for k, v in self._data_gen_params['hierarchy_params']['act_hier_params'].items():
            self._hierarchy_correls[ ( 'Act', k+1 ) ] = float(v['level_down_correl'])


    def propagate_ts_factors(self):
        org_num=0
        if vars.DEBUG: print_period = 1
        for org_node in TreeTraverser.traverse_pre_order(self._org_tree):   
            if vars.DEBUG and org_num % print_period == 0: 
                print(f"Generating data for org node {org_num+1} out of {self._org_node_count} ...", end='\r')    

            for act_node in TreeTraverser.traverse_pre_order(org_node.properties['act_tree']):
                if act_node.properties['ts_params_generated'] == False:
                    parent_ts_params, parent_type = self._get_parent_ts_params(org_node, act_node)

                    level = act_node.level if parent_type == 'Act' else org_node.level
                    target_correl = self._hierarchy_correls[(parent_type, level)]

                    for effect in self._effects:
                        # if effect not in [ 'Level', 'Trend_Linear', 'Trend_Multiplicative', 'Month_of_the_year']: continue
                        # if effect != 'Level': continue

                        idx = self._data_gen_params['regular_effects']['Effect_Name'] == effect
                        if sum(idx) == 0: continue

                        param_row = self._data_gen_params['regular_effects'].loc[idx]

                        ts_params = ts_params_f.get_correlated_effect( effect, 
                            param_row, parent_ts_params[effect], target_correl)

                        act_node.properties[effect] = ts_params
                
                    act_node.properties['ts_params_generated'] == True
                    # if vars.DEBUG: print(f"done generating ts factors for {org_node.id}__{act_node.id}")

                # else:  if vars.DEBUG: print(f"factors already set for: {act_node}") 
            org_num += 1
            # if vars.DEBUG: print(f"org node complete: {org_node}") #; sys.exit()
        if vars.DEBUG:  print('-'*100)  


    
    def _generate_open_hours_for_org_nodes(self):
        allowed_oh_type = self._data_gen_params['main_params']['Open_hours_allowed']
        day_str = { 1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun' }
         
        for node in TreeTraverser.traverse_pre_order_leaves_only(self._org_tree):
            sampled_open_hrs = utils.sample_open_hours(self._data_gen_params['weekly_open_hours'], allowed_oh_type)
            node.properties['open_hours'] = sampled_open_hrs
            
            time_intervals_by_day = []  
            for day_num in day_str.keys():

                open_time = sampled_open_hrs[f'{day_str[day_num]}_open']
                close_time = sampled_open_hrs[f'{day_str[day_num]}_close']

                if isinstance(open_time, datetime): open_time = open_time.time()
                if isinstance(close_time, datetime): close_time = close_time.time()

                if isinstance(open_time, str): open_time = utils.parse_time_from_string(open_time)
                if isinstance(close_time, str): close_time = utils.parse_time_from_string(close_time)
                

                idx = self._daily_time_intervals['time_interval']>=open_time
                start_int = self._daily_time_intervals.loc[idx]['time_int_num'].min()

                idx = self._daily_time_intervals['time_interval']<close_time
                end_int = self._daily_time_intervals.loc[idx]['time_int_num'].max()

                if start_int == end_int: continue
                
                idx = ((self._daily_time_intervals['time_int_num'] >= start_int) &
                    (self._daily_time_intervals['time_int_num'] <= end_int) )
                day_intervals = self._daily_time_intervals.loc[idx][['time_int_num','time_interval']]
                day_intervals.insert(0, 'weekday', day_num)

                time_intervals_by_day.append(day_intervals)                
            
            time_intervals_by_day = pd.concat(time_intervals_by_day)
            node.properties['time_intervals_by_day'] = time_intervals_by_day


    def _assign_activity_tree_to_org_nodes(self):
        for org_node in TreeTraverser.traverse_pre_order(self.org_tree):
            act_tree_prototype = copy.deepcopy(self._act_tree_template)

            #keep reference to org node in the act_nodes for traversal later on
            for act_node in TreeTraverser.traverse_pre_order(act_tree_prototype):
                act_node.properties['org_node'] = org_node
                act_node.properties['ts_params_generated'] = False

            #assign the act_node to the org_node 
            org_node.properties['act_tree'] = act_tree_prototype


    def _get_monthly_history(self, org_node, act_node, join_cols_for_effects ):        

        # initialize
        monthly_data = self._history_months.copy()

        # join level, additive trend, and mult. trend, and monthly seasonality factors
        for effect in ['Level', 'Trend_Linear', 'Trend_Multiplicative', 'Month_of_the_year']: 
            if effect != 'Month_of_the_year':                
                monthly_data[effect] = act_node.properties[effect]     
            else:                
                df2 = pd.DataFrame(act_node.properties[effect])
                effect_col_names = [ i+1 for i in range(df2.shape[1])]
                df2.columns = effect_col_names
                # if vars.DEBUG: print(df2.head())

                df2 = pd.concat([monthly_data[['cumu_month_num']], df2], axis=1)

                # unpivot the seasonality factors
                id_columns = ['cumu_month_num']
                df2 = pd.melt(df2, id_vars = id_columns, value_vars = effect_col_names, 
                    var_name=join_cols_for_effects[effect][-1], value_name=effect)                
                # if vars.DEBUG: print(df2.head(30))

                # merge factors with the monthly df
                monthly_data = monthly_data.merge(df2, 
                    on = join_cols_for_effects[effect], how = 'left')
                # if vars.DEBUG: print(monthly_data.head(30))
        
        # get magnitude
        magnitude = self._get_magnitude_for_org_and_act(org_node.id, act_node.id) #; if vars.DEBUG: print(f"magnitude: {magnitude}")
        monthly_data['magnitude'] = magnitude

        # apply level 
        true_val_col = 'true_monthly_val'
        val_col_w_noise = 'obs_monthly_val'
        monthly_data[true_val_col] = monthly_data['magnitude'] * monthly_data['Level']

        # apply additive trend
        monthly_data['cum_sum_trend'] = monthly_data['Trend_Linear'].cumsum() / 12.
        monthly_data[true_val_col] = monthly_data['cum_sum_trend']\
            .add(1.).mul(monthly_data[true_val_col])

        # apply multiplicative trend  -- the way this is applied, it looks identical to additive trend, hence not applying
        # monthly_data[true_val_col] = monthly_data['Trend_Multiplicative']\
        #     .mul(monthly_data[true_val_col])

        # apply monthly seasonality
        monthly_data[true_val_col] = monthly_data['Month_of_the_year']\
            .mul(monthly_data[true_val_col])

        
        monthly_data = self._apply_noise(monthly_data, true_val_col, val_col_w_noise, 'monthly')
        
        # monthly_data.to_csv("monthly_data.csv", index=False)
        return monthly_data


    def _apply_noise(self, data, val_col, val_w_noise_col, noise_granularity): 
        dist_type, param1, param2 = self._get_noise_dist_params(noise_granularity)
        sampling_func = utils.get_sampling_func(dist_type)
        data[val_w_noise_col] = sampling_func(len(data), param1, param2 )
        data[val_w_noise_col] *= data[val_col]
        return data


    def _get_noise_dist_params(self, noise_granularity):
        noise_params_df = self._data_gen_params['noise']
        if noise_granularity == 'monthly': noise_param_type = 'Monthly_Noise'
        elif noise_granularity == 'daily': noise_param_type = 'Daily_Noise'
        elif noise_granularity == 'time_interval': noise_param_type = 'Time_Interval_Noise'
        else: 
            noise_types = ['Monthly_Noise', 'Daily_Noise', 'Time_Interval_Noise']
            raise Exception(f"Sorry, do not recognize noise type {noise_granularity}. \nMust be one of {noise_types}")

        idx = noise_params_df['Parameter_Type'] == noise_param_type
        if sum(idx) != 1: raise Exception(f"Sorry, cannot find noise params for {noise_param_type}.")

        dist_type = noise_params_df.loc[idx, 'Distribution_Type'].values[0]
        param1 = noise_params_df.loc[idx, 'Parameter1'].values[0]
        param2 = noise_params_df.loc[idx, 'Parameter2'].values[0]
        
        return dist_type, param1, param2 


    def _generate_daily_history(self, org_node, act_node, monthly_data, join_cols_for_effects):

        # initialize daily df
        daily_data = self._history_dates.copy()   
        daily_data = daily_data.merge(monthly_data[['year', 'month', 'cumu_month_num']],
            on= ['year', 'month'], how='left')   

        for effect in ['Day_of_the_week', 'Day_of_the_month', 'Special_Days']: 

            df2 = pd.DataFrame(act_node.properties[effect])
            if effect == 'Special_Days':
                effect_col_names = list(self._special_day_ids['special_day'])
            else:
                effect_col_names = [ i+1 for i in range(df2.shape[1])]
            df2.columns = effect_col_names

            df2 = pd.concat([monthly_data[['cumu_month_num']], df2], axis=1)
            #if vars.DEBUG: print(df2.head()) #; sys.exit()

            # unpivot the seasonality factors
            id_columns = ['cumu_month_num']
            df2 = pd.melt(df2, id_vars = id_columns, value_vars = effect_col_names, 
                var_name=join_cols_for_effects[effect][-1], value_name=effect)                
            # if vars.DEBUG: print(df2.head(10))

            # merge factors with the monthly df
            daily_data = daily_data.merge(df2, 
                on = join_cols_for_effects[effect], how = 'left')

            daily_data[effect] = daily_data[effect].fillna(1.0)
            # if vars.DEBUG: print(daily_data.head(30)) ; sys.exit()

        # join monthly magnitudes
        daily_data = daily_data.merge(monthly_data[['cumu_month_num', 'true_monthly_val', 'obs_monthly_val']],
            on=['cumu_month_num'])

        # apply smoothing
        daily_data = self._apply_smoothing_to_ts_factors(daily_data, smoothing_col='true_monthly_val', window=30)
        daily_data = self._apply_smoothing_to_ts_factors(daily_data, smoothing_col='obs_monthly_val', window=30)
        daily_data = self._apply_smoothing_to_ts_factors(daily_data, smoothing_col='Day_of_the_month', window=8)

        true_val_col = 'true_daily_val'
        val_col_w_noise = 'obs_daily_val'
        # apply DoW, DoM, and Special day factors
        daily_data[true_val_col] = daily_data['true_monthly_val']\
            .mul(daily_data['Day_of_the_week'])\
                .mul(daily_data['Day_of_the_month'])\
                    .mul(daily_data['Special_Days'])\

        daily_data[val_col_w_noise] = daily_data['obs_monthly_val']\
            .mul(daily_data['Day_of_the_week'])\
                .mul(daily_data['Day_of_the_month'])\
                    .mul(daily_data['Special_Days'])\

        # apply daily level noise
        daily_data = self._apply_noise(daily_data, true_val_col, val_col_w_noise, 'daily')
        
        # apply holiday adjustment 
        idx = daily_data['is_holiday'] == 1
        daily_data.loc[idx, 'true_daily_val'] = 0
        daily_data.loc[idx, 'obs_daily_val'] = 0

        # daily_data.to_csv("daily_data.csv", index=False) ; sys.exit()
        return daily_data


    def _apply_smoothing_to_ts_factors(self, history_data, smoothing_col, window):
        history_data[smoothing_col] = history_data[smoothing_col].rolling(window=window, min_periods=1).mean()
        return history_data


    def _generate_intra_daily_history(self, org_node, act_node, daily_data, join_cols_for_effects):

        effect = 'Time_of_the_day'
        # if vars.DEBUG: print(daily_data.head())

        intra_daily_data = daily_data.merge(org_node.properties['time_intervals_by_day'], on='weekday')
        
        num_ints_in_day = self._effects[effect]['len']
        tod_factors = act_node.properties[effect]
        
        tod_factors_df = []
        for d in range(7):
            df = pd.DataFrame(tod_factors[:, d*num_ints_in_day: (d+1)*num_ints_in_day])
            df.columns = [i  for i in range(num_ints_in_day)]
            df.insert(0, 'weekday', d+1)
            df = pd.concat([self._history_months[['cumu_month_num']], df], axis=1)
            tod_factors_df.append(df)
        tod_factors_df = pd.concat(tod_factors_df)
        # if vars.DEBUG: print(tod_factors_df.head()) #; if vars.DEBUG: print(tod_factors_df.shape) ; sys.exit()

        # unpivot the tod factors
        id_columns = ['cumu_month_num','weekday']
        tod_factors_df = pd.melt(tod_factors_df, id_vars=id_columns, 
            value_vars=[i for i in range(num_ints_in_day)],
            var_name=join_cols_for_effects[effect][-1], value_name=effect )

        intra_daily_data = intra_daily_data.merge(tod_factors_df,
            on=['cumu_month_num', 'weekday', 'time_int_num'], how='left' )
        
        # apply smoothing
        intra_daily_data = self._apply_smoothing_to_ts_factors(intra_daily_data, smoothing_col='Time_of_the_day', window=4)

        # final intra-daily volume
        true_val_col = 'true_intra_daily_val'
        val_col_w_noise = 'obs_intra_daily_val'

        # orig value
        intra_daily_data[true_val_col] = intra_daily_data['true_daily_val'] * intra_daily_data['Time_of_the_day']

        # observed value with noise
        intra_daily_data[val_col_w_noise] = intra_daily_data['obs_daily_val'] * intra_daily_data['Time_of_the_day']
        intra_daily_data = self._apply_noise(intra_daily_data, true_val_col, val_col_w_noise, 'time_interval')
        # ----------------------------------------------------------------------------------

        # intra_daily_data.to_csv("intra_daily_data.csv", index=False)
        
        return  intra_daily_data


    def generate_history_volumes(self, org_node, act_node):
        # if act_node.properties['ts_params_generated'] == False: return None, None, None

        # now join the ts_factors
        join_cols_for_effects = {
            'Level': ['cumu_month_num'],
            'Trend_Linear': ['cumu_month_num'],
            'Trend_Multiplicative': ['cumu_month_num'],
            'Month_of_the_year': ['cumu_month_num', 'month'],
            'Day_of_the_week': ['cumu_month_num', 'weekday'],
            'Day_of_the_month': ['cumu_month_num', 'adj_day_in_month'],
            'Special_Days': ['cumu_month_num', 'special_day'],
            'Time_of_the_day': ['cumu_month_num', 'weekday', 'time_int_num'],
        }

        monthly_history = self._get_monthly_history(org_node, act_node, join_cols_for_effects)
        # if vars.DEBUG: print(monthly_history.head())

        daily_history = self._generate_daily_history(org_node, act_node, monthly_history, join_cols_for_effects)
        # if vars.DEBUG: print(daily_history.head())

        intra_daily_history = self._generate_intra_daily_history(org_node, act_node, daily_history, join_cols_for_effects)
        # if vars.DEBUG: print(intra_daily_history.head())

        for df in [monthly_history, daily_history, intra_daily_history]:
            df.insert(0, 'org_id', org_node.id)
            df.insert(1, 'act_id', act_node.id)

        # if vars.DEBUG: print(intra_daily_history.loc[0])
        # sys.exit()

        return  monthly_history, daily_history, intra_daily_history

 
    def _get_freq_dur_and_impact(self, effect_row, freq_mult):        
        irr_effect_specs = []   
        freq = np.random.poisson(effect_row['Poisson_Rate_Per_Year'] * freq_mult)
        for i in range(freq):           
            if not effect_row['Effect_Type'] == 'level_shift':
                dur = random.randint(effect_row['Min_Effect_Duration'], effect_row['Max_Effect_Duration'])
            else:
                dur = 1000000   #some big int number (infinity)
            if effect_row['Min_Effect_Multiplier'] is not np.nan:
                mult = np.round(random.uniform(effect_row['Min_Effect_Multiplier'], 
                    effect_row['Max_Effect_Multiplier']), 4)
            else: mult = None
            irr_effect_specs.append({ 'dur': dur, 'mult': mult, 'effect_type': effect_row['Effect_Type']})
        return irr_effect_specs



    def apply_anomalies_and_level_shift(self, intra_daily_history):

        sort_cols = ['org_id', 'act_id', 'date', 'time_interval']
        intra_daily_history = intra_daily_history.sort_values(by=sort_cols)
        effect_params = self._data_gen_params['irregular_effects']
        dates_in_history = intra_daily_history['date'].drop_duplicates().tolist()
        dates_in_history.sort()

        # if vars.DEBUG: print(dates_in_history); sys.exit()
        min_date = self._history_dates['date'].min()
        max_date = self._history_dates['date'].max()
        num_days_in_history = (max_date - min_date).days + 1
        freq_mult = num_days_in_history / 365.
        # if vars.DEBUG: print('num_days_in_history', num_days_in_history, freq_mult)

        # if vars.DEBUG: print('effect_params', effect_params)

        intra_daily_history['irreg_effect'] = ''
        intra_daily_history['irreg_effect_mult'] = 1.
        intra_daily_history['missing_data'] = 0

        for i, row in effect_params.iterrows():
            if row['Apply_Effect?'] == 0: continue

            effect_type = row['Effect_Type']
            if effect_type not in self.IRREGULAR_EFFECT_TYPES:
                raise Exception(f"Invalid irregular effect: must be one of {self.IRREGULAR_EFFECT_TYPES}")

            irr_effect_specs = self._get_freq_dur_and_impact(row, freq_mult)
            # if vars.DEBUG: print(row['Effect_Name'], irr_effect_specs )

            intra_daily_history = self._apply_irreg_effect( row['Effect_Name'],  intra_daily_history, 
                dates_in_history, irr_effect_specs, row['Effect_Duration_Time_Unit'])
        
        # impacted rows 
        idx = intra_daily_history['irreg_effect'] != ''
        irreg_effect_marker_df = intra_daily_history.loc[idx].copy()

        # apply irreg effect multipliers to volumes
        intra_daily_history['obs_intra_daily_val'] *= intra_daily_history['irreg_effect_mult']

        # retain rows without missing data marker
        idx = intra_daily_history['missing_data'] == 0
        intra_daily_history = intra_daily_history.loc[idx].reset_index(drop=True)
        
        return intra_daily_history, irreg_effect_marker_df



    def _apply_irreg_effect(self, effect_name, intra_daily_history, dates_in_history, irr_effect_specs, dur_unit):
        # if vars.DEBUG: print(effect_name, irr_effect_specs, dur_unit); sys.exit()
        for i, effect_spec in enumerate(irr_effect_specs):
            if dur_unit == 'days':
                rand_idx = random.randint(0, len(dates_in_history)-1)
                impacted_dates = dates_in_history[rand_idx: rand_idx+effect_spec['dur']]
                impacted_idx = intra_daily_history['date'].isin(impacted_dates)
            else:
                N = len(intra_daily_history)
                rand_idx = random.randint(0, N-1)
                idx_max = rand_idx + effect_spec['dur']
                if idx_max > N: idx_max = N
                idx_range = np.arange(rand_idx, rand_idx + effect_spec['dur'])
                impacted_idx = intra_daily_history.index.isin(idx_range)
            
            intra_daily_history.loc[impacted_idx, 'irreg_effect'] += f'{effect_name}_{i}; '
            if math.isnan(effect_spec['mult']):
                intra_daily_history.loc[impacted_idx, 'missing_data'] = 1
            else:
                intra_daily_history.loc[impacted_idx, 'irreg_effect_mult'] *= effect_spec['mult']
        
        # print("after")
        # print(intra_daily_history.head())
        return intra_daily_history


