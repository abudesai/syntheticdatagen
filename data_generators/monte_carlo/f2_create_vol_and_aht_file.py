
import numpy as np, pandas as pd
import sys, os
from datetime import datetime as dt
import variables as vars


def parse_time_from_string(time_string,  format = '%H:%M:%S'):
    return dt.strptime(time_string, format).time()


def parse_date_from_string(date_string, format='%Y-%m-%d'):
    return dt.strptime(date_string, format)     # '%Y-%m-%d'


epoch = dt.utcfromtimestamp(0)
def unix_time_micros(t): return (t - epoch).total_seconds() * 1000000.0


def make_parquet_dir():
    if not os.path.exists(vars.parquet_files_output_dir): os.makedirs(vars.parquet_files_output_dir)


def create_processed_csv_and_parquet_files(input_file_name, dataset_name):    
    
    print("Creating processed csv and parquet files ...")

    data = pd.read_csv(f'./outputs/{input_file_name}')

    cols = [ 'org_id', 'date', 'time_interval', 'act_id', 'value' ]
    data = data[cols].copy() 

    # ------------------------------------------
    print("parsing time and sorting values")
    data['date'] = data['date'].apply(lambda d: parse_date_from_string(d))
    data['time_interval'] = data['time_interval'].apply(lambda d: parse_time_from_string(d))

    data['date_time'] = data.apply(
        lambda row: dt.combine(row['date'], row['time_interval']), axis = 1)

    data = data.sort_values(by=['org_id', 'act_id', 'date_time'])
    # print(data.head())
    
    # ------------------------------------------ 
    print("dropping duplicates")
    act_ids = data['act_id'].drop_duplicates().tolist()
    mapped_cols = ['callvolume', 'aht']
    act_map = { k: v for k,v in zip(act_ids, mapped_cols ) }
    data['act_id'] = data['act_id'].map(act_map)

    # print(act_map)
    
    print("Pivoting 2 activities for call volume and aht")
    non_pivoted_columns = ['org_id', 'date_time', 'date', 'time_interval']
    pivoted_columns = ['value'] 
    pivoting_column = 'act_id'

    data = data.pivot_table(index = non_pivoted_columns, 
                                      aggfunc=sum,
                                      columns=pivoting_column, 
                                      values=pivoted_columns).reset_index() 

    # pivot table will result in multi column index. To get a regular column names
    new_colummn_names = [ col[0] if col[1] == '' else col[1] 
        for col in data.columns ]
    data.columns = new_colummn_names
    
    # ------------------------------------------
    # make aht more reasonable 
    
    print("scaling aht values")
    # print(data.head())
    aht_mean = data['aht'].mean()
    rand_num = .9 + np.random.random()*.1
    data['aht'] = data['aht'] * 7 * rand_num / aht_mean
    aht_mean = data['aht'].mean()    
    # ------------------------------------------
    # update column names
    data.rename(columns={'org_id': 'queueid', 'date_time': 'time'}, inplace=True)       
    # ------------------------------------------
    # round cols 
    
    print("rounding to 4 decimals")
    data = data.round({ 'callvolume':4, 'aht':4 })
     
    # print(data.head())    
    
    # ------------------------------------------ 
    file_name_without_ext = os.path.splitext(input_file_name)[0]
    if dataset_name is None:      
        output_file_name = f'{file_name_without_ext}'
    else:
        output_file_name = f'{dataset_name}'
    # ------------------------------------------ 
    make_parquet_dir()
    # ------------------------------------------
    
    print("creating csv and parquet for intra-daily")
    # write time-interval level data
    # write to csv
    data.to_csv(f'{vars.parquet_files_output_dir}{vars.internal_intra_daily_parquet_file_name}.csv', index=False,  float_format='%.4f')
    
    # parquet_file columns
    cols = [ 'time', 'queueid', 'aht', 'callvolume']
    data[cols].to_parquet(f'{vars.parquet_files_output_dir}{vars.internal_intra_daily_parquet_file_name}.parquet', compression=None)
    # data[cols].to_parquet(f'{vars.parquet_files_output_dir}{vars.internal_intra_daily_parquet_file_name}.parquet', compression=None)
    
    # ------------------------------------------
    # write daily level data
    # daily level data 
    print("creating csv and parquet for daily")
    data['date'] = data['time'].apply(lambda t: t.date())
    groupby_cols = ['queueid', 'date']
    daily_data = data.groupby(groupby_cols).agg( { 'callvolume': ['sum'], 'aht': ['mean'] } )
    daily_data.columns = ['_'.join(col).strip() for col in daily_data.columns.values]
    daily_data.reset_index(inplace=True)
    daily_data.rename(columns={'aht_mean': 'aht', 'callvolume_sum': 'callvolume'}, inplace=True)   
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    # print(daily_data.head())

    if dataset_name is None:      
        output_file_name = f'{vars.internal_daily_parquet_file_name}'
    else:
        output_file_name = f'{vars.internal_daily_parquet_file_name}_{dataset_name}'

    daily_data.to_csv(f'{vars.parquet_files_output_dir}{output_file_name}.csv', index=False, float_format='%.4f')
    
    # parquet_file columns
    cols = [ 'date', 'queueid', 'aht', 'callvolume']
    daily_data[cols].to_parquet(f'{vars.parquet_files_output_dir}{output_file_name}.parquet', compression=None)
    # daily_data[cols].to_parquet(f'{vars.parquet_files_output_dir}{vars.internal_daily_parquet_file_name}.parquet', compression=None)
    
    
    # ------------------------------------------

    print("Done creating processed csv and parquet files ...")
    
    # ------------------------------------------


def test_conversion_to_micros():
    print("testing conversion of python date time to microseconds")
    my_date = parse_date_from_string('2021-01-29')
    my_time = parse_time_from_string('23:15:00')
    my_date_time = dt.combine(my_date, my_time)
    time_in_micros = unix_time_micros(my_date_time)
    # print(my_date, my_time, my_date_time)
    # print(my_date_time, time_in_micros)

    exp_time_in_micros = 1611962100000000
    assert np.abs(time_in_micros - exp_time_in_micros) < 1e-5, "uh-oh! something went wrong in conversion to micro-secs."
    print("all good!")




def process_and_write_synthetic_data(series_prefix, feature_prefix, series_col_name, epoch_col_name):

    if vars.DEBUG: print("Creating processed csv and parquet files ...")

    if vars.DEBUG: print("Reading the csv file ...")
    data = pd.read_csv(f'{vars.output_dir}{vars.daily_history_file_prefix}.csv', parse_dates=[vars.date_field])
    # print(data)

    # print(data.head())
    print('unique orgs: ', data['org_id'].nunique())
    # sys.exit()

    
    # -------------------------------------------------------------
    # map orgid, actid, and time values into preferred format
    if vars.DEBUG: print("Mapping field names to preferred format ...")

    # fields = [vars.org_id, vars.act_id, vars.date_field]
    fields = [vars.org_id, vars.act_id]
    
    prefixes = [series_prefix, feature_prefix, '']

    for field, pref in zip(fields, prefixes): 
        field_values = sorted(data[field].drop_duplicates().tolist())
        val_map = { val: f'{pref}{i}' for i, val in enumerate(field_values) }
        # print(val_map)    
        data[field] = data[field].map(val_map)


    # data[vars.date_field] = data[vars.date_field].apply(int)
    # -------------------------------------------------------------
    # pivot the activity columns 
    if vars.DEBUG: print("Pivoting activity column ...")

    non_pivoted_columns = [vars.org_id, vars.date_field]
    pivoted_columns = [vars.value] 
    pivoting_column = vars.act_id

    data = data.pivot_table(index = non_pivoted_columns, 
                                      aggfunc=sum,
                                      columns=pivoting_column, 
                                      values=pivoted_columns).reset_index() 

    new_column_names = [ col[0] if col[1] == '' else col[1] 
        for col in data.columns ]
    data.columns = new_column_names

    
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

    data.to_csv(f'{vars.output_dir}{vars.processed_daily_history_file_prefix}.csv', index=False)
    data.to_parquet(f'{vars.output_dir}{vars.processed_daily_history_file_prefix}.parquet', compression=None)
    
    if vars.DEBUG: print("Done!")   
   
    return 


if __name__ == '__main__':

    # test_conversion_to_micros()

    # file_name = f'{vars.intra_daily_history_file_prefix}.csv'
    # dataset_name = None
    # create_processed_csv_and_parquet_files(file_name, dataset_name)

    series_prefix = 'ser_'
    epoch_prefix = 't_'
    feature_prefix = 'v_'
    series_col_name, epoch_col_name = 'seriesid', 'epoch'

    process_and_write_synthetic_data(series_prefix, feature_prefix, series_col_name, epoch_col_name)




