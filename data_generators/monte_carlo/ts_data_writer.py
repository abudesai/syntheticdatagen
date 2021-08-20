

import numpy as np, copy, pandas as pd, sys
import os, shutil
import multiprocessing
import variables as vars
import utils
from tree_traverse import TreeTraverser
from batched_results_writer import BatchResultsWriter
from datetime import datetime



class TS_DataWriter: 
    def __init__(self, dataset_name, ts_tree, output_dir, run_multi_processing, debug=False):
        super().__init__()
        self._dataset_name = dataset_name
        self._ts_tree = ts_tree
        self._output_dir = output_dir
        self._run_multi_processing = run_multi_processing
        self._debug = debug

    
    def write_ts_params(self):
        self._write_org_and_act_trees()
        self._write_magnitudes_for_nodes()
        self._write_time_scale()
        self._write_open_hours()
        self._write_special_days()
        self._write_ts_factors()
        #### self._write_history()
        self._write_history_multi()


    @staticmethod
    def accumulate_files(input_dir_path, input_file_name_prefix, 
        output_dir_path, outputfile_name, input_files_have_headers): 
        
        input_file_list = [i for i in os.listdir(input_dir_path) if os.path.isfile(os.path.join(input_dir_path,i)) and input_file_name_prefix in i]
        # print(input_file_list) 

        if len(input_file_list) == 0:
            print(f"No output files to accumulate for {input_file_name_prefix} prefix.")
            return False
        
        #First check if the file can be found
        file = input_file_list[0]
        pathandfilename = os.path.join(input_dir_path, file)
        assert os.path.isfile(pathandfilename), 'File not found:' + pathandfilename    
        
        #Now create the outputfile 
        outf = open(os.path.join(output_dir_path, outputfile_name), 'w') 
        
        #If input files have headers
        if input_files_have_headers:
            #Now get header row from first file
            with open(pathandfilename) as f:
                header_row = f.readline()    
        
            #write header to output file
            outf.write(header_row)    
        
        
        #read files row by row and write to the outputfile
        for file in input_file_list:    
            pathandfilename = os.path.join(input_dir_path, file)        
            #check if file exists
            if not os.path.isfile(pathandfilename): 
                print('Dir path or files not found')        
                continue         
            
            with open(pathandfilename) as f:            
                if input_files_have_headers:
                    #read and ignore the header row 
                    next(f)
                
                #now read and write the lines
                for line in f:
                    if line != '':
                        outf.write(line)
                        # sys.exit()
                    else:
                        continue
        
        #Close the consolidated file        
        outf.close()

        return True
        
    
    def _write_history_for_org_list(self, org_list, split_num):
        
        batch_writer = BatchResultsWriter(
            split_num=split_num, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.intra_daily_history_file_prefix,
            batch_size = vars.write_batch_size
        )

        batch_writer2 = BatchResultsWriter(
            split_num=split_num, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.daily_history_file_prefix,
            batch_size = vars.write_batch_size
        )

        batch_writer3 = BatchResultsWriter(
            split_num=split_num, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.monthly_history_file_prefix,
            batch_size = vars.write_batch_size
        )

        batch_writer4 = BatchResultsWriter(
            split_num=split_num, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.irreg_effects_history_file_prefix,
            batch_size = vars.write_batch_size
        )
        
        org_num=0
        for org_node in org_list:   
            if self._debug: 
                print(f"Writing history -- split/inc/total {split_num}/{org_num+1}/{len(org_list)}, org_node: {org_node.id}", end='\r')                     
                     
            org_num += 1

            m, d, i, r = [], [], [], []
            for act_node in TreeTraverser.traverse_pre_order_leaves_only(org_node.properties['act_tree']):  
                
                monthly_history, daily_history, intra_daily_history = self._ts_tree.generate_history_volumes(org_node, act_node)
 
                if monthly_history.empty: 
                    if self._debug: print(f"No time-series factors for org_node {org_node} and act_node {act_node}")
                    continue

                adj_intra_daily_history, irreg_effect_marker_df = self._ts_tree.apply_anomalies_and_level_shift(intra_daily_history) 


                monthly_history, daily_history, intra_daily_history, irreg_effect_marker_df = self.get_cleaned_history_data(adj_intra_daily_history, irreg_effect_marker_df)   

                m.append(monthly_history)
                d.append(daily_history)
                i.append(intra_daily_history)
                r.append(irreg_effect_marker_df)

            
            intra_daily_history = pd.concat(i)
            daily_history = pd.concat(d)
            monthly_history = pd.concat(m)
            irreg_effect_marker_df = pd.concat(r)

            dfs = [intra_daily_history, daily_history, monthly_history ]
            
            dfs = self.pivot_dfs(dfs)
            # print(dfs[0].shape, dfs[1].shape, dfs[2].shape)

            batch_writer.append_and_maybe_write(dfs[0])
            batch_writer2.append_and_maybe_write(dfs[1])
            batch_writer3.append_and_maybe_write(dfs[2])
            # batch_writer4.append_and_maybe_write(irreg_effect_marker_df)
            
        batch_writer.write_results()
        batch_writer2.write_results()
        batch_writer3.write_results()    
        # batch_writer4.write_results()

        # sys.exit()        
        # print(f"Writing history for org node {org_num+1} out of {self._ts_tree.org_tree.leaf_count} ...")


    def pivot_dfs(self, dfs):
        distinct_acts = list(dfs[0]['act_id'].drop_duplicates().sort_values())
        # print(distinct_acts)

        for i, df in enumerate(dfs): 
            all_cols = list(df.columns)
            non_pivoted_columns = [c for c in all_cols if c not in [vars.act_id, vars.value ] ]
            pivoted_columns = [vars.value] 
            pivoting_column = vars.act_id

            df = df.pivot_table(index = non_pivoted_columns, 
                                            aggfunc=sum,
                                            columns=pivoting_column, 
                                            values=pivoted_columns).reset_index() 

            new_column_names = [ col[0] if col[1] == '' else col[1]  for col in df.columns ]
            df.columns = new_column_names

            df = df[non_pivoted_columns + distinct_acts]
            dfs[i] = df            

        return dfs


    def get_cleaned_history_data(self, adj_intra_daily_history, irreg_effect_marker_df):
        
        # monthly data
        groupby_cols = [ 'org_id', 'act_id', 'year', 'month' ]
        monthly_history = adj_intra_daily_history.groupby(groupby_cols, as_index=False)['obs_intra_daily_val'].sum()
        monthly_history.rename(columns={'obs_intra_daily_val': 'value'}, inplace=True)
        monthly_history.insert(0, 'dataset_name', self._dataset_name)        
        # print(monthly_history.head())

        # daily data
        groupby_cols = [ 'org_id', 'act_id', 'date' ]
        daily_history = adj_intra_daily_history.groupby(groupby_cols, as_index=False)['obs_intra_daily_val'].sum()
        daily_history.rename(columns={'obs_intra_daily_val': 'value'}, inplace=True)
        daily_history.insert(0, 'dataset_name', self._dataset_name)
        # print(daily_history.head())

        
        # intra-daily data
        cols = [ 'org_id', 'act_id', 'date', 'time_interval', 'obs_intra_daily_val' ]
        intra_daily_history = adj_intra_daily_history[cols].copy()
        intra_daily_history.rename(columns={'obs_intra_daily_val': 'value'}, inplace=True)
        intra_daily_history['value'] = intra_daily_history['value'].round(2)
        intra_daily_history.insert(0, 'dataset_name', self._dataset_name)   
        intra_daily_history['date'] = intra_daily_history.apply(
            lambda row: datetime.combine(row['date'], row['time_interval']), axis=1 )
        del intra_daily_history['time_interval']
        # print('intra_daily_history', intra_daily_history.shape)

        cols = [ 'org_id', 'act_id', 'date', 'time_interval', 'irreg_effect', 'missing_data', 'irreg_effect_mult' ]
        irreg_effect_marker_df = irreg_effect_marker_df[cols].copy()     
        # print(irreg_effect_marker_df.head())        

        return monthly_history, daily_history, intra_daily_history, irreg_effect_marker_df


    def _write_history_multi(self):

        org_list = [org_node for org_node in TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree) ]
        # print(org_list); sys.exit()

        if self._run_multi_processing:
            # Get cpu_count and use all but one for resource calculations
            num_cpus_to_use = multiprocessing.cpu_count() - 2

            # num_cpus_to_use = 1
            if num_cpus_to_use > 8: num_cpus_to_use = 8

            org_leaf_count = self._ts_tree.org_tree.leaf_count
            if org_leaf_count < num_cpus_to_use: num_cpus_to_use = org_leaf_count
            if self._debug: print(f"num_cpus_to_use: {num_cpus_to_use}")

            if num_cpus_to_use == 1: 
                self._write_history_for_org_list(org_list, 0)
            
            else: 

                # split input files into parts to run on each thread
                split_org_lists = utils.split_list_into_lists(org_list, num_cpus_to_use)

                pool = multiprocessing.Pool(num_cpus_to_use)

                # run forecasts on each thread
                for split_num in range(num_cpus_to_use):
                    pool.apply_async(self._write_history_for_org_list, 
                    args=( split_org_lists[split_num], split_num) )

                pool.close()
                pool.join() 
        else:
            self._write_history_for_org_list(org_list, 0)

        
        if vars.DEBUG:  print('-'*100)   
        self._accumulate_all_split_files()


    def _write_history(self):
        
        if self._debug: print("Writing history files...\nThis may take a minute or two...")

        batch_writer = BatchResultsWriter(
            split_num=0, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.intra_daily_history_file_prefix,
            batch_size = 10
        )

        batch_writer2 = BatchResultsWriter(
            split_num=0, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.daily_history_file_prefix,
            batch_size = 10
        )

        batch_writer3 = BatchResultsWriter(
            split_num=0, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.monthly_history_file_prefix,
            batch_size = 10
        )

        batch_writer4 = BatchResultsWriter(
            split_num=0, 
            output_dir = vars.split_files_output_dir,
            output_file_name_prefix = vars.irreg_effects_history_file_prefix,
            batch_size = 10
        )

        org_num=0
        print_period = 1
        for org_node in TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree):   
            if self._debug and org_num % print_period == 0:  
                print(f"Writing history for org node {org_num+1} out of {self._ts_tree.org_tree.leaf_count} ...")
            org_num += 1
            for act_node in TreeTraverser.traverse_pre_order_leaves_only(org_node.properties['act_tree']):  
                
                monthly_history, daily_history, intra_daily_history = self._ts_tree.generate_history_volumes(org_node, act_node)

                # print(monthly_history.shape, daily_history.shape, intra_daily_history.shape)

                if self._debug and monthly_history.empty: 
                    print(f"No time-series factors for org_node {org_node} and act_node {act_node}")
                    continue

                adj_intra_daily_history, irreg_effect_marker_df = self._ts_tree.apply_anomalies_and_level_shift(intra_daily_history)                

                batch_writer.append_and_maybe_write(adj_intra_daily_history)
                batch_writer2.append_and_maybe_write(daily_history)
                batch_writer3.append_and_maybe_write(monthly_history)
                # batch_writer4.append_and_maybe_write(irreg_effect_marker_df)

            
        batch_writer.write_results()
        batch_writer2.write_results()
        batch_writer3.write_results()
        # batch_writer4.write_results()

        self._accumulate_all_split_files()

        
    def _accumulate_all_split_files(self):
        if self._debug: print("Accumulating split files into single files...")
        # intra-daily files
        self.accumulate_files(
                input_dir_path = vars.split_files_output_dir, 
                input_file_name_prefix = vars.intra_daily_history_file_prefix, 
                output_dir_path = vars.output_dir, 
                outputfile_name = vars.intra_daily_history_file_prefix + '.csv', 
                input_files_have_headers = True)

        # daily files
        self.accumulate_files(
                input_dir_path = vars.split_files_output_dir, 
                input_file_name_prefix = vars.daily_history_file_prefix, 
                output_dir_path = vars.output_dir, 
                outputfile_name = vars.daily_history_file_prefix + '.csv', 
                input_files_have_headers = True)

        # monthly files
        self.accumulate_files(
                input_dir_path = vars.split_files_output_dir, 
                input_file_name_prefix = vars.monthly_history_file_prefix, 
                output_dir_path = vars.output_dir, 
                outputfile_name = vars.monthly_history_file_prefix + '.csv',  
                input_files_have_headers = True)

        # irregular effects files
        self.accumulate_files(
                input_dir_path = vars.split_files_output_dir, 
                input_file_name_prefix = vars.irreg_effects_history_file_prefix, 
                output_dir_path = vars.output_dir, 
                outputfile_name = vars.irreg_effects_history_file_prefix + '.csv',  
                input_files_have_headers = True)


    def _write_magnitudes_for_nodes(self):
        node_magnitudes = []
        for org_node in TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree):   
            for act_node in TreeTraverser.traverse_pre_order_leaves_only(org_node.properties['act_tree']):  
                node_magnitudes.append([
                    org_node.id,
                    act_node.id,
                    self._ts_tree.magnitudes['org'][org_node.id],
                    self._ts_tree.magnitudes['act'][act_node.id],
                    self._ts_tree.magnitudes['org_and_act'][(org_node.id, act_node.id)],                    
                    ])
        df_cols = ['org_id', 'act_id', 'org_magnitude', 'act_rel_scaler', 'org_and_act_scaler']
        node_magnitudes = pd.DataFrame(node_magnitudes, columns=df_cols)
        node_magnitudes['baseline_vol'] = node_magnitudes['org_magnitude'] \
            * node_magnitudes['act_rel_scaler'] * node_magnitudes['org_and_act_scaler']
        node_magnitudes.to_csv(f"./{self._output_dir}/ts_factor_magnitudes.csv", index=False, float_format='%.4f')


    def _write_org_and_act_trees(self):

        if self._ts_tree.org_tree is None: return
        if self._debug: print("Writing org and act trees...")

        # org org tree in flat format
        org_tree_flat = self._ts_tree.org_tree.get_flat_hierarchy()

        # convert to df
        org_tree_flat_df = pd.DataFrame(org_tree_flat)
        
        # name columns
        org_tree_flat_df.columns = [f'Org_Level_{i}' for i in range(org_tree_flat_df.shape[1])]
        org_tree_flat_df.to_csv(f"./{self._output_dir}/org_tree_flat_df.csv", index=False)

        # org act tree in flat format
        act_tree_flat = self._ts_tree.act_tree_template.get_flat_hierarchy()

        # convert to df
        act_tree_flat_df = pd.DataFrame(act_tree_flat)
        
        # name columns
        act_tree_flat_df.columns = [f'Act_Level_{i}' for i in range(act_tree_flat_df.shape[1])]
        act_tree_flat_df.to_csv(f"./{self._output_dir}/act_tree_flat_df.csv", index=False)

    
    def _write_time_scale(self):
        if self._ts_tree.history_dates is None: return
        if self._debug: print("Writing history time scale...")
        self._ts_tree.history_dates.to_csv(f"./{self._output_dir}/history_dates.csv", index=False)


    def _write_special_days(self):
        if self._ts_tree.special_day_ids is None: return
        if self._debug: print("Writing special day ids...")
        self._ts_tree.special_day_ids.to_csv(f"./{self._output_dir}/special_day_ids.csv", index=False)


    def _write_open_hours(self):
        open_hrs_list = []
        org_ids = []
        for node in TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree):
            if node.properties.get('open_hours') is not None:
                open_hrs_list.append(node.properties['open_hours'])
                org_ids.append(node.id)
        
        if len(open_hrs_list) > 0: 
            # get index columns, only one time
            col_names = next(TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree))\
                .properties['open_hours'].index
            org_open_hours = pd.concat(open_hrs_list, axis=1, ignore_index=True).T
            org_open_hours.columns = col_names
            org_open_hours.insert(0, 'Org_Id', org_ids)
            org_open_hours.drop(['Weekly_Hours_Configuration_Num'], axis=1, inplace=True)
            if self._debug: print("Writing open hours...")
            org_open_hours.to_csv(f"./{self._output_dir}/org_open_hours.csv", index=False)

    
    def _write_effect_params(self, org_node_id, act_node_id, effect, effect_params, history_months):
        df = history_months.copy()
        df.insert(0, 'org_id', org_node_id )
        df.insert(1, 'act_id', act_node_id )
        # print(org_node_id, act_node_id, effect, effect_params.shape)
        if effect_params.shape[1] == 1: 
            df[effect.lower()] = np.round(effect_params, 4)
        else:
            for i in range(effect_params.shape[1]):
                df[f'{effect.lower()}_{i+1}'] = np.round(effect_params[:, i], 4)

        # print(df) ; sys.exit() 
        return df 


    def _write_ts_factors(self):
        
        for effect in self._ts_tree.effects:
            # if effect not in [ 'Level', 'Trend_Linear', 'Trend_Multiplicative', 'Month_of_the_year' ]: continue
            if effect == 'Time_of_the_day': continue
            if self._debug: print(f"Writing time series factors for {effect}...")
            
            cumu_results = []
            for org_node in TreeTraverser.traverse_pre_order_leaves_only(self._ts_tree.org_tree):   
                for act_node in TreeTraverser.traverse_pre_order_leaves_only(org_node.properties['act_tree']):  
                        
                    node_results = self._write_effect_params(org_node.id, act_node.id, 
                        effect, act_node.properties[effect], self._ts_tree.history_months)

                    cumu_results.append(node_results)
                
            if len(cumu_results) > 0:
                cumu_results = pd.concat(cumu_results)
                cumu_results.to_csv(f"./{self._output_dir}/ts_factor_{effect.lower()}.csv", index=False)
            # sys.exit()
