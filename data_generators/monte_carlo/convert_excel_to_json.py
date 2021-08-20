import numpy as np, pandas as pd
import json
import variables as vars 
import pprint



def convert_excel_to_json():    
    print("converting excel to json")
    config_file = vars.input_dir + vars.data_gen_file_name
    f = pd.read_excel(config_file, sheet_name=None)

    f['special_days_dates'] = process_sp_day_dates_to_str(f['special_days_dates'])
    f['weekly_open_hours'] = filter_oh_configs(f['weekly_open_hours'])

    all_params = {}
    for params in vars.config_params:
        df = f[params['name']]
        df_json = df.to_json(orient='records')
        parsed_json = json.loads(df_json)
        # pprint.pprint(parsed_json)
        all_params[params['name']] = parsed_json
        
    with open(f"{vars.input_dir}{vars.config_file_name}", 'w') as out_f:
        json.dump(all_params, out_f, indent=4)


def filter_oh_configs(df): 
    idx = df['Use_Configuration'] == 1
    df = df.loc[idx].reset_index(drop=True).copy()
    return df


def process_sp_day_dates_to_str(df):
    df['Special_Day_Date'] = df['Special_Day_Date'].dt.strftime(date_format='%m/%d/%Y')
    return df


def process_json_config_file():

    with open(f"{vars.input_dir}{vars.config_file_name}", 'r') as in_f:
        json_obj = json.load(in_f)
    # print(json_obj)

    all_params = {}
    for params in vars.config_params:
        p = json_obj[params['name']]
        df = pd.DataFrame(p)
        all_params[params['name']] = df

    
    return all_params



def process_main_params(df):
    df_json = df.to_json(orient='records')
    pprint.pprint(df_json)



if __name__ == '__main__':
    convert_excel_to_json()
    params = process_json_config_file()