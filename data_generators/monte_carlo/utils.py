
import numpy as np, pandas as pd
import random, sys, os, shutil
from datetime import datetime 
import json
import variables as vars 
from node import OrgTree, ActivityTree
from tree_traverse import TreeTraverser
from shutil import copyfile




def get_data_gen_config2(config_file_name):
    config_file = vars.input_dir + config_file_name
    with open(config_file, 'r') as in_f:
        json_obj = json.load(in_f)
    # print(json_obj)
    all_params = process_json_config_file(json_obj)

    # process org and activity hierarchy params into dictionaries
    all_params['hierarchy_params'] = process_org_and_act_hier_params(all_params['hierarchy_params'])
    # print(f['hierarchy_params'] )

    # process main params into dictionary
    all_params['main_params'] = process_main_params(all_params['main_params'])
    # print(f['main_params'] )

    all_params['special_days_dates'] = process_sp_day_dates_from_str(all_params['special_days_dates'])

    # print(all_params['weekly_open_hours'].head())

    return all_params



def process_json_config_file(json_obj):
    
    all_params = {}
    for params in vars.config_params:
        p = json_obj[params['name']]
        df = pd.DataFrame(p)
        all_params[params['name']] = df
    
    return all_params


def process_sp_day_dates_from_str(df):
    df['Special_Day_Date'] = pd.to_datetime(df['Special_Day_Date'], format = '%m/%d/%Y')
    return df


def get_data_gen_config(config_file_name):
    config_file = vars.input_dir + config_file_name
    f = pd.read_excel(config_file, sheet_name=None)

    # process org and activity hierarchy params into dictionaries
    f['hierarchy_params'] = process_org_and_act_hier_params(f['hierarchy_params'])
    # print(f['hierarchy_params'] )

    # process main params into dictionary
    f['main_params'] = process_main_params(f['main_params'])
    # print(f['main_params'] )
    return f


def get_num_runs(main_params):
    return main_params['Num_of_Runs']


def get_dataset_name(main_params):
    return main_params['Dataset_Name']


def process_main_params(main_params):
    main_params_dict = {  k: v for k, v in 
        zip(main_params['Parameter_Name'], main_params['Parameter_Value']) }
    return main_params_dict


def process_org_and_act_hier_params(hier_params):
    org_hier_params, act_hier_params = {}, {}
    for _, row in hier_params.iterrows():
        level = int(row['Level_Num'])
        params = {
                'node_count_bounds': (int(row['Min_Count']), int(row['Max_Count'])),
                'level_down_correl': float(row['Level_Down_Correlation'])
            }
        if row['Tree_Type'] == 'Organization_hierarchy':
            org_hier_params[level] = params
        elif row['Tree_Type'] == 'Activity_hierarchy':
            act_hier_params[level] = params
        else:
            raise Exception(f"Sorry, couldn't recognize {row['Tree_Type'] } as tree type")
    
    # print(org_hier_params); print(act_hier_params)
    hier_params = { 'org_hier_params': org_hier_params, 'act_hier_params': act_hier_params }
    return hier_params


def get_org_and_act_trees(hier_params):
    # create org tree
    org_hier_params = hier_params['org_hier_params']
    num_children_per_gen = [ org_hier_params[lvl]['node_count_bounds']
        for lvl in range(len(org_hier_params.keys())) ] 
    org_tree = OrgTree()
    org_tree.create_tree(num_children_per_gen)   
    # print(num_children_per_gen)
    
    act_hier_params = hier_params['act_hier_params']
    num_children_per_gen = [ act_hier_params[lvl]['node_count_bounds']
        for lvl in range(len(act_hier_params.keys())) ]    
    # print(num_children_per_gen)
    act_tree = ActivityTree()
    act_tree.create_tree(num_children_per_gen)

    return org_tree, act_tree


def sample_open_hours(open_hrs_configs, allowed_oh_type):

    idx = open_hrs_configs['Use_Configuration'] == 1
    open_hrs_configs_filtered = open_hrs_configs.loc[idx].copy().reset_index()

    if allowed_oh_type == 'Weekdays_Only':
        idx = open_hrs_configs['Has_Weekend_Hours'] == 0
        open_hrs_configs_filtered = open_hrs_configs_filtered.loc[idx].copy().reset_index()
    elif allowed_oh_type == 'With_Weekends':        
        idx = open_hrs_configs['Has_Weekend_Hours'] == 1
        open_hrs_configs_filtered = open_hrs_configs_filtered.loc[idx].copy().reset_index()
    else:
        pass

    rand_idx = np.random.randint(open_hrs_configs_filtered.shape[0])
    sampled_open_hours = open_hrs_configs_filtered.iloc[rand_idx].copy()
    return sampled_open_hours



def parse_time_from_string(time_string,  format = '%H:%M:%S'):
    return datetime.strptime(time_string, format).time()


def parse_date_from_string(date_string, format=r'%m/%d/%Y'):
    return datetime.strptime(date_string, format)     # '%m/%d/%Y'


def parse_datetime_from_string(date_string, format='%m/%d/%Y %H:%M:%S'):
    return datetime.strptime(date_string, format)       #'%m/%d/%Y %H:%M:%S'


def get_sampling_func(dist_type):
    if dist_type == 'normal':
        dist_sampler_func = get_random_normal
    elif dist_type == 'uniform':
        dist_sampler_func = get_random_uniform
    else: 
        raise Exception(f"Sorry, cannot recognize distribution type {dist_type} for ts param initialization.")
    return dist_sampler_func

def get_random_normal(size, mean = 0, one_and_half_stddev = 1.5):
    return np.random.randn(size) * (one_and_half_stddev / 1.5) + mean


def get_random_uniform(size, mean=.5, half_range=.5):
    return mean - half_range + (np.random.random(size)) * 2 * half_range



def delete_all_files_in_directory(output_dir):
        print("Clearing previous output files...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else: 
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    raise Exception(f'Failed to delete output file {file_path}. Reason: {e}')


def split_list_into_lists(list_of_items, num_splits):
    num_per_split = (len(list_of_items) // num_splits) + 1

    list_of_split_lists = []
    for i in range(num_splits):
        list_of_split_lists.append(list_of_items[i * num_per_split : (i + 1) * num_per_split ])

    return list_of_split_lists


def del_file():
    file_path = './outputs/irreg_effects.csv'
    os.unlink(file_path)


def test_Poisson():
    s = np.random.poisson((1/15)*3, 1000000)
    import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(s, 14, density=True)
    plt.show()

    (unique, counts) = np.unique(s, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)


def save_run_results(output_dataset_name, run_num):
    from_dir = vars.output_dir
    to_dir = os.path.join(vars.output_dir_multi, f'{output_dataset_name}_run{run_num}')
    # print(from_dir, to_dir)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    else: 
        delete_all_files_in_directory(to_dir)

    file_prefix_to_copy = [
        vars.monthly_history_file_prefix,
        vars.daily_history_file_prefix,
        vars.processed_daily_history_file_prefix,
        vars.intra_daily_history_file_prefix
    ]

    for pref in file_prefix_to_copy: 

        files_to_copy = [f for f in os.listdir(from_dir) if f.startswith(pref)]
        # print(files_to_copy)
        
        for f in files_to_copy: 
            copyfile(os.path.join(from_dir, f), os.path.join(to_dir, f))



def accumulate_files(input_file_list,  output_dir_path, outputfile_name, input_files_have_headers):  

        if len(input_file_list) == 1: 
            base_name = os.path.basename(input_file_list[0])
            to_path = os.path.join(output_dir_path, base_name)
            copyfile(input_file_list[0], to_path)
            return

        
        #Now create the outputfile 
        outf = open(os.path.join(output_dir_path, outputfile_name), 'w') 
        
        #If input files have headers
        if input_files_have_headers:
            #Now get header row from first file
            with open(input_file_list[0]) as f:
                header_row = f.readline()    
        
            #write header to output file
            outf.write(header_row)    
        
        
        #read files row by row and write to the outputfile
        for pathandfilename in input_file_list:    
                 
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



def accumulate_run_files(output_dataset_name):

    print(f"Accumulating multi files for: {output_dataset_name}...")

    file_prefix_to_copy = {
        vars.monthly_history_file_prefix:[],
        vars.daily_history_file_prefix:[],
        vars.processed_daily_history_file_prefix:[],
        vars.intra_daily_history_file_prefix:[],
    }

    root_dir = vars.output_dir_multi

    for root, dirs, files in os.walk(root_dir):
        if output_dataset_name not in root: continue
        print(root)
        if len(files) == 0: continue
        for pref in file_prefix_to_copy.keys(): 
            for f in files: 
                if f.startswith(pref) and not f.endswith('parquet'):
                    file_prefix_to_copy[pref].append(os.path.join(root, f))

    # print(file_prefix_to_copy)

    for pref in file_prefix_to_copy.keys(): 
        if len(file_prefix_to_copy[pref]) == 0: continue
        accumulate_files(
            input_file_list = file_prefix_to_copy[pref], 
            output_dir_path = root_dir, 
            outputfile_name = pref+'.csv', 
            input_files_have_headers = True)



def process_and_write_synthetic_data_multi(file_name,
        series_prefix, feature_prefix, series_col_name, epoch_col_name):

    if vars.DEBUG: print("Creating processed csv and parquet files ...")

    if vars.DEBUG: print("Reading the csv file ...")
    data = pd.read_csv(f'{vars.output_dir_multi}{file_name}.csv', parse_dates=[vars.date_field])
    # print(data.shape); sys.exit()

    print('unique orgs: ', data['org_id'].nunique())
    
    # -------------------------------------------------------------
    # map orgid, actid, and time values into preferred format
    if vars.DEBUG: print("Mapping field names to preferred format ...")

    fields = [vars.org_id]
    
    prefixes = [series_prefix]

    for field, pref in zip(fields, prefixes): 
        field_values = sorted(data[field].drop_duplicates().tolist())
        val_map = { val: f'{pref}{i}' for i, val in enumerate(field_values) }
        # print(val_map)    
        data[field] = data[field].map(val_map)

    # data[vars.date_field] = data[vars.date_field].apply(int)
    # -------------------------------------------------------------
    # pivot the activity columns 
    # if vars.DEBUG: print("Pivoting activity column ...")

    # non_pivoted_columns = [vars.org_id, vars.date_field]
    # pivoted_columns = [vars.value] 
    # pivoting_column = vars.act_id

    # data = data.pivot_table(index = non_pivoted_columns, 
    #                                   aggfunc=sum,
    #                                   columns=pivoting_column, 
    #                                   values=pivoted_columns).reset_index() 

    # new_column_names = [ col[0] if col[1] == '' else col[1] 
    #     for col in data.columns ]
    # data.columns = new_column_names
    
    # -------------------------------------------------------------
    # sort, rename columns, round floats,
    
    if vars.DEBUG: print("Sorting, renaming fields, rounding values ...")

    data = data.sort_values(by=[vars.org_id, vars.date_field])

    data.rename(columns={
        vars.org_id: series_col_name,
        vars.date_field: epoch_col_name,
    }, inplace=True)


    cols_to_round = [col for col in data.columns if col.startswith(feature_prefix)]
    data[cols_to_round] = data[cols_to_round].round(3)
    # print(data.head(10))
    
    # -------------------------------------------------------------
    # write to disk
    
    if vars.DEBUG: print("Writing final file ...")

    data.to_csv(f'{vars.output_dir_multi}processed_{file_name}.csv', index=False)
    data.to_parquet(f'{vars.output_dir_multi}processed_{file_name}.parquet', compression=None)
    
    if vars.DEBUG: print("Done!")   
   
    return 






if __name__ == "__main__":
    # get_data_gen_config()
    # del_file()
    test_Poisson()
