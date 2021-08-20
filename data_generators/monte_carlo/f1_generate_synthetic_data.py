
import numpy as np, pandas as pd, math
import random
import time, sys
import time_scale2 as ts  
import utils 
from ts_generator import TS_Generator_Tree
from ts_data_writer import TS_DataWriter
import variables as vars
from create_charts import  plot_daily_level_data



def generate_data(data_gen_params, run_num=0):
    
    start = time.time()     

    # rand_seed = data_gen_params['main_params']['Random_seed_num']
    # if math.isnan(rand_seed) or rand_seed == -1:         
    #     rand_seed = random.randint(0, 100)
    
    volume_tree = TS_Generator_Tree(rand_seed = run_num)
    volume_tree.generate_ts_params(data_gen_params)

    # clear output dir
    utils.delete_all_files_in_directory(vars.output_dir)
    
    dataset_name = utils.get_dataset_name(data_gen_params['main_params'])

    ts_data_writer = TS_DataWriter(
        dataset_name, 
        volume_tree, 
        vars.output_dir, 
        run_multi_processing=True,
        debug = vars.DEBUG)

    ts_data_writer.write_ts_params()


    end = time.time()
    print(f"Done generating data. Run time: {np.round((end - start)/60.0, 2)} minutes")    

    output_dataset_name = f"DS_{data_gen_params['main_params']['Dataset_Number']}_{data_gen_params['main_params']['Dataset_Name']}"
    return output_dataset_name



if __name__ == '__main__':
    
    config_file_name = sys.argv[1] if len(sys.argv) > 1 else vars.config_file_name
    data_gen_params = utils.get_data_gen_config2(config_file_name)
    output_dataset_name = generate_data(data_gen_params) 

    # plot_daily_level_data(max_num_series=5, max_days = 365)

    

