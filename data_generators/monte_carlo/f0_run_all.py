
import sys, time
import numpy as np
import variables as vars
from f1_generate_synthetic_data import generate_data
from f2_create_vol_and_aht_file import create_processed_csv_and_parquet_files, process_and_write_synthetic_data
from convert_excel_to_json import convert_excel_to_json
from create_charts import  plot_daily_level_data
import utils
import create_charts as charts



def run_generation_multi(data_gen_params):   

    print("Running multi-scenario...") 

    start = time.time()

    utils.delete_all_files_in_directory(vars.output_dir_multi)
    
    num_runs = utils.get_num_runs(data_gen_params['main_params'])

    for run_num in range(num_runs):   
        print('\n'); print('-'*60); 
        print("Run num: ", run_num)
        output_dataset_name = generarate_data_main(data_gen_params, run_num)
        utils.save_run_results(output_dataset_name, run_num)

    output_dataset_name = 'DS_100_Base_scenario'
    utils.accumulate_run_files(output_dataset_name)

    series_prefix = 'ser_'
    feature_prefix = 'v_'
    epoch_prefix = 't_'
    series_col_name, epoch_col_name = 'seriesid', 'epoch'

    utils.process_and_write_synthetic_data_multi(
        file_name = vars.intra_daily_history_file_prefix,
        series_prefix = series_prefix, 
        feature_prefix = feature_prefix, 
        series_col_name = series_col_name, 
        epoch_col_name = epoch_col_name)

    utils.process_and_write_synthetic_data_multi(
        file_name = vars.daily_history_file_prefix,
        series_prefix = series_prefix, 
        feature_prefix = feature_prefix, 
        series_col_name = series_col_name, 
        epoch_col_name = epoch_col_name)

    # charts.plot_daily_level_data(max_num_series=5, max_days = 365, file_dir = vars.output_dir_multi)


    end = time.time()
    print(f"Multi run time: {np.round((end - start)/60.0, 2)} minutes")    


def generarate_data_main(data_gen_params, run_num=0):
    start = time.time()  
   
    # config_file_name = sys.argv[1] if len(sys.argv) > 1 else vars.config_file_name
    output_dataset_name = generate_data(data_gen_params, run_num=run_num) 
    
    end = time.time()
    print(f"Done generating data and writing history files. Total run time: {np.round((end - start)/60.0, 2)} minutes")    

    return output_dataset_name



if __name__ == '__main__':
    convert_excel_to_json()

    config_file_name = sys.argv[1] if len(sys.argv) > 1 else vars.config_file_name
    data_gen_params = utils.get_data_gen_config2(config_file_name)


    # generarate_data_main()data_gen_params

    run_generation_multi(data_gen_params)


    plot_daily_level_data(max_num_series=5, max_days = 365)